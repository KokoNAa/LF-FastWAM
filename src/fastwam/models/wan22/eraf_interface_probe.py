"""Opt-in, same-cache ERAF interface diagnostic. Never constructs an expert.

The four predictions share one actual Video/ERAF computation, incoming memory,
and initial action noise. Only the designated driver's action reaches the env.
Hybrid predictions are NOT hybrid rollout-success measurements.
"""

import copy
from contextlib import contextmanager
import hashlib
from pathlib import Path
import random

import numpy as np
import torch

from .eraf_preservation import INTERFACE_NAMES, tensor_digest, validate_teacher_payload


COMBINATIONS = {
    "old_old": ("old", "old"),
    "new_old": ("new", "old"),
    "old_new": ("old", "new"),
    "new_new": ("new", "new"),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_tensors(first, second, label):
    if not first or set(first) != set(second) or tensor_digest(first) != tensor_digest(second):
        raise ValueError(f"Interface probe requires identical {label} tensors/keys.")


def validate_checkpoints(warm, candidate, warm_sha256):
    """Fail closed: current experiment is V9.28 -> interface-only V9.30."""
    for payload, objective in ((warm, 28), (candidate, 30)):
        if (
            payload.get("format") != "fastwam_policy_guard_v9"
            or payload.get("architecture_metadata", {}).get("eraf_grounding_objective_version") != objective
        ):
            raise ValueError(f"Expected PGC v9 objective {objective} checkpoint.")
    if warm.get("step") != 10000:
        raise ValueError("Expected the V9.28 step10000 warm checkpoint.")
    if not warm.get("base_checkpoint") or warm["base_checkpoint"] != candidate.get("base_checkpoint"):
        raise ValueError("The protected Base checkpoint binding differs.")
    if warm.get("eraf_shared_expert_lora_config") != candidate.get("eraf_shared_expert_lora_config"):
        raise ValueError("Shared LoRA configuration differs.")
    _same_tensors(warm.get("eraf_shared_expert_lora", {}), candidate.get("eraf_shared_expert_lora", {}), "shared LoRA")
    states = [p.get("policy_guard", {}) for p in (warm, candidate)]
    frozen = [{k: v for k, v in s.items() if k.split(".", 1)[0] not in INTERFACE_NAMES} for s in states]
    _same_tensors(*frozen, "frozen guard/ERAF/gate")
    teacher, provenance = validate_teacher_payload(candidate)
    old = {k: v for k, v in states[0].items() if k.split(".", 1)[0] in INTERFACE_NAMES}
    _same_tensors(old, teacher, "fixed teacher / warm interface")
    if provenance["checkpoint_sha256"] != warm_sha256:
        raise ValueError("Fixed teacher does not identify the supplied warm checkpoint SHA256.")
    return {
        "saved_frozen_tensors": "identical", "teacher": "exact_warm",
        "base_checkpoint": warm["base_checkpoint"],
        "frozen_guard_sha256": tensor_digest(frozen[0]),
        "shared_lora_sha256": tensor_digest(warm["eraf_shared_expert_lora"]),
        "warm_interface_sha256": tensor_digest(old),
    }


@contextmanager
def isolated_rng():
    """Preserve caller RNG, including on failure; reset identically per variant."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        cpu = torch.random.get_rng_state()
        cuda = [torch.cuda.get_rng_state(device) for device in devices]

        def reset():
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(cpu)
            for device, state in zip(devices, cuda):
                torch.cuda.set_rng_state(state, device)

        try:
            yield reset
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def _cpu(tensor):
    return tensor.detach().float().cpu().clone()


def delta_metrics(value, reference):
    value, reference = value.float(), reference.float()
    delta = value - reference
    result = {
        "rms": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "value_rms": float(value.square().mean().sqrt()),
        "reference_rms": float(reference.square().mean().sqrt()),
    }
    if delta.shape[-1] <= 16:
        result["per_dimension_rms"] = delta.square().reshape(-1, delta.shape[-1]).mean(0).sqrt().tolist()
    return result


class InterfaceProbe:
    def __init__(self, model, warm_path, candidate_path, driver="new_new", atol=1e-5):
        if driver not in ("old_old", "new_new"):
            raise ValueError("Driver must be old_old or new_new; hybrids are prediction-only.")
        if model.training or getattr(model, "policy_guard_action_expert", None) is not None:
            raise ValueError("Probe requires eval mode and one shared Action Expert.")
        if not getattr(model, "policy_guard_eraf_safe_gain_training", False) or model.policy_guard_gate_mode != "counterfactual":
            raise ValueError("Probe requires forced-ERAF safe-gain inference.")
        if int(model.policy_guard_eraf_grounding_objective_version) != 30:
            raise ValueError("Load the candidate in its V9.30 runtime for both drivers.")
        warm = torch.load(warm_path, map_location="cpu", weights_only=False)
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
        self.provenance = validate_checkpoints(warm, candidate, file_sha256(warm_path))
        self.provenance.update({
            "warm_checkpoint": str(Path(warm_path).resolve()),
            "warm_sha256": file_sha256(warm_path),
            "candidate_checkpoint": str(Path(candidate_path).resolve()),
            "candidate_sha256": file_sha256(candidate_path),
            "driver": driver, "driver_repeat_atol": atol,
        })
        self.model, self.driver, self.atol = model, driver, float(atol)
        self.expert = model.action_expert
        self.banks = {"new": {}, "old": {}}
        for name in INTERFACE_NAMES:
            module = model.policy_guard_modules[name]
            prefix = name + "."
            expected = {k[len(prefix):]: v for k, v in candidate["policy_guard"].items() if k.startswith(prefix)}
            loaded = module.state_dict()
            if set(loaded) != set(expected) or any(
                not torch.equal(v.detach().cpu(), expected[k].to(dtype=v.dtype)) for k, v in loaded.items()
            ):
                raise ValueError(f"Live {name} is not the candidate interface.")
            self.banks["new"][name] = module
            old = copy.deepcopy(module)
            old.load_state_dict({k[len(prefix):]: v for k, v in warm["policy_guard"].items() if k.startswith(prefix)}, strict=True)
            old.requires_grad_(False).eval()
            self.banks["old"][name] = old

    @contextmanager
    def driver_scope(self):
        originals = {name: self.model.policy_guard_modules[name] for name in INTERFACE_NAMES}
        try:
            for name, source in zip(INTERFACE_NAMES, COMBINATIONS[self.driver]):
                self.model.policy_guard_modules[name] = self.banks[source][name]
            yield self
        finally:
            for name, module in originals.items():
                self.model.policy_guard_modules[name] = module

    @torch.no_grad()
    def run(self, *, initial_noise, routed_goal_queries, eraf_outputs, policy_state,
            forward_kwargs, timesteps, deltas, driver_action):
        """Called AFTER ordinary driver prediction; no simulator or state writes."""
        model = self.model
        if model.action_expert is not self.expert or any(m.training for m in model.modules()):
            raise RuntimeError("Expert identity/eval mode changed during the probe.")
        upstream = {"goal_queries": routed_goal_queries, "noise": initial_noise}
        upstream.update({f"eraf/{k}": v for k, v in eraf_outputs.items() if torch.is_tensor(v)})
        upstream.update({f"memory/{k}": v for k, v in (policy_state or {}).items() if torch.is_tensor(v)})
        before = tensor_digest(upstream)
        arrays = {k: _cpu(v).numpy() for k, v in upstream.items()}
        variants = {}
        with isolated_rng() as reset:
            for variant, (compressor_source, injector_source) in COMBINATIONS.items():
                reset()
                tokens, _ = self.banks[compressor_source][INTERFACE_NAMES[0]](routed_goal_queries)
                injector = self.banks[injector_source][INTERFACE_NAMES[1]]
                injected, _, metrics = injector(
                    context=forward_kwargs["context"], context_mask=forward_kwargs["full_context_mask"],
                    goal_queries=tokens, external_scale=model._policy_guard_eraf_context_injection_scale(),
                )
                appended = injected[:, forward_kwargs["context"].shape[1]:]
                action = initial_noise.clone()
                for index, (step_t, delta) in enumerate(zip(timesteps, deltas)):
                    prediction = model._forward_policy_guard_action_from_cache(
                        **forward_kwargs, action_tokens=action,
                        timestep_action=step_t.unsqueeze(0).to(action),
                        routed_goal_queries=tokens, context_injector=injector,
                    )
                    if index == 0:
                        arrays[f"{variant}/first_flow"] = _cpu(prediction).numpy()
                    action = model.infer_action_scheduler.step(prediction, delta, action)
                arrays[f"{variant}/tokens"] = _cpu(tokens).numpy()
                arrays[f"{variant}/injected_tokens"] = _cpu(appended).numpy()
                arrays[f"{variant}/action_normalized"] = _cpu(action[0]).numpy()
                variants[variant] = {k: float(v) for k, v in metrics.items() if v.numel() == 1}
        if tensor_digest(upstream) != before:
            raise RuntimeError("Probe mutated shared upstream tensors or incoming memory.")
        repeat = torch.from_numpy(arrays[f"{self.driver}/action_normalized"])
        if not torch.isfinite(repeat).all() or not torch.allclose(repeat, _cpu(driver_action), atol=self.atol, rtol=0):
            raise RuntimeError("Driver repeat mismatch: stop; same-cache probe is not validated.")
        for variant in COMBINATIONS:
            for suffix in ("tokens", "injected_tokens", "action_normalized", "first_flow"):
                value = torch.from_numpy(arrays[f"{variant}/{suffix}"])
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"Non-finite probe output: {variant}/{suffix}")
                variants[variant][f"{suffix}_vs_old_old"] = delta_metrics(
                    value, torch.from_numpy(arrays[f"old_old/{suffix}"])
                )
        interaction = (arrays["new_new/action_normalized"] - arrays["new_old/action_normalized"]
                       - arrays["old_new/action_normalized"] + arrays["old_old/action_normalized"])
        return {
            "record": {
                "upstream_sha256": before, "shared_video_eraf_cache": True,
                "upstream_tensor_metadata": {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in upstream.items()},
                "same_initial_noise": True, "same_incoming_memory": True,
                "driver": self.driver, "driver_repeat_validated": True,
                "driver_repeat": delta_metrics(repeat, _cpu(driver_action)),
                "variants": variants,
                "factorial_action_interaction_rms": float(np.sqrt(np.mean(interaction ** 2))),
                "interpretation": "same-state predictions only; no hybrid success labels",
            },
            "arrays": arrays,
        }
