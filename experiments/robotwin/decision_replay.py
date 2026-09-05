"""Add audited same-state positives to the unchanged four-pool training loss.

The replay bank contains frozen production VAE/T5/proprio inputs, not Video
K/V: all Video and Action adapter operations are recomputed with gradients.
No model parameters, original samples, or original random draws are changed.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from types import MethodType

import torch

from experiments.robotwin.joint_adapter_repair import build_cache
from experiments.robotwin.same_state_repair import move_cache, noise_tensor


@contextmanager
def deployment_modes(model):
    """Restore every mode, including when checkpoint backward recomputes."""
    modes = [(module, module.training) for module in model.mot.modules()]
    try:
        for module, _ in modes:
            module.training = False
        yield
    finally:
        for module, training in modes:
            module.training = training


def endpoint_loss(model, captured, references, seed, checkpoint=True):
    """Equal positive MSE at sigma=1; C+D, without extra difference gain."""
    from torch.utils.checkpoint import checkpoint as ckpt
    body = getattr(type(model)._predict_action_noise_with_cache, "__wrapped__", None)
    if body is None:
        raise ValueError("Require the production no_grad action predictor.")
    refs = {k: v.to(device=model.device, dtype=model.torch_dtype) for k, v in references.items()}
    if set(refs) != {"source", "target"} or any(v.shape != (1, 32, 14) for v in refs.values()):
        raise ValueError("Expected two complete 32x14 positive action references.")
    noise = noise_tensor((1, 32, 14), seed, model)
    time = torch.tensor([model.train_action_scheduler.num_train_timesteps],
                        device=model.device, dtype=model.torch_dtype)
    losses = []
    for language in ("source", "target"):
        inputs = move_cache(captured[language], model.device)

        def forward(x, t, inputs=inputs):
            # This context belongs INSIDE the checkpoint closure. Otherwise
            # backward would recompute with ordinary-training dropout enabled.
            with deployment_modes(model):
                return body(model, latents_action=x, timestep_action=t, **build_cache(model, inputs))

        prediction = ckpt(forward, noise, time, use_reentrant=False) if checkpoint else forward(noise, time)
        target = model.train_action_scheduler.training_target(refs[language], noise, time)
        losses.append((prediction.float() - target.float()).square().mean())
    return (losses[0] + losses[1]) / 2


def tensor_digest(value):
    """Stable recursive sample digest; reading tensors never consumes RNG."""
    digest = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            y = x.detach().contiguous().cpu()
            digest.update(str((str(y.dtype), tuple(y.shape))).encode())
            digest.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x):
                digest.update(str(key).encode())
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            for child in x:
                visit(child)
        else:
            digest.update(repr(x).encode())
    visit(value)
    return digest.hexdigest()


def rng_digest(model):
    states = {"cpu": torch.get_rng_state()}
    if torch.device(model.device).type == "cuda":
        states["cuda"] = torch.cuda.get_rng_state(model.device)
    return tensor_digest(states)


def balanced_order(rows, seed):
    """One complete pair per draw, interleave every task/domain stratum."""
    groups = {}
    for row in rows:
        groups.setdefault((row["pair_id"], row["task_config"]), []).append(row)
    if not groups or len({len(v) for v in groups.values()}) != 1:
        raise ValueError("Replay strata must be nonempty and equally sized.")
    g = torch.Generator(device="cpu").manual_seed(seed)
    groups = {key: [value[i] for i in torch.randperm(len(value), generator=g).tolist()]
              for key, value in sorted(groups.items())}
    return [groups[key][i] for i in range(len(next(iter(groups.values())))) for key in groups]


class DecisionReplay:
    def __init__(self, manifest, manifest_sha256, weight, seed, log_dir):
        from scripts.probe_robotwin_no_eraf import sha256
        path = Path(manifest)
        if sha256(path) != manifest_sha256:
            raise ValueError("Replay manifest changed.")
        self.manifest = json.loads(path.read_text())
        if self.manifest.get("complete") is not True:
            raise ValueError("Replay bank is incomplete.")
        self.weight = float(weight)
        if not math.isfinite(self.weight) or not 0 <= self.weight <= 1:
            raise ValueError("Replay weight must lie in [0,1].")
        self.seed = int(seed)
        rows = [r for r in self.manifest["states"] if r["replay_split"] == "train"]
        self.order = balanced_order(rows, self.seed)
        self.payloads = {}
        for row in self.order:
            if sha256(row["payload"]) != row["payload_sha256"]:
                raise ValueError("Replay payload changed.")
            self.payloads[row["id"]] = torch.load(row["payload"], map_location="cpu", weights_only=True)
        self.counter = 0
        self.log_dir = Path(log_dir)
        self.handle = None

    def loss(self, model, original, sample, *args, **kwargs):
        distributed = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        world = torch.distributed.get_world_size() if distributed else 1
        if self.handle is None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.handle = (self.log_dir / f"draws_rank{rank}.jsonl").open("x", buffering=1)
            selected = {n: p for n, p in model.mot.named_parameters() if p.requires_grad}
            if not selected or any(not n.endswith((".lora_A", ".lora_B")) for n in selected):
                raise ValueError("Expected only shared Video/Action LoRA to be trainable.")
            if {n.split('.')[1] for n in selected} != {"video", "action"}:
                raise ValueError("Both Video and Action adapters must train.")
            if any(p.requires_grad for p in model.proprio_encoder.parameters()):
                raise ValueError("Captured proprio inputs require a frozen encoder.")
            audit = {"initial_adapter_sha256": tensor_digest(selected), "trainable_tensors": len(selected),
                     "dtypes": sorted({str(p.dtype) for p in selected.values()}),
                     "rank": rank, "world_size": world}
            (self.log_dir / f"initial_rank{rank}.json").write_text(json.dumps(audit, indent=2) + "\n")
        position = self.counter * world + rank
        row = self.order[position % len(self.order)]
        seed = self.seed + 1_000_000 + position
        sample_hash = tensor_digest(sample)
        before = rng_digest(model)
        loss, metrics = original(sample, *args, **kwargs)
        original_value = float(loss.detach())
        after_original = rng_digest(model)
        value = 0.0
        if self.weight:
            payload = self.payloads[row["id"]]
            auxiliary = endpoint_loss(model, payload["captured"], payload["references"], seed)
            value = float(auxiliary.detach())
            if not math.isfinite(value):
                raise ValueError("Nonfinite decision replay loss.")
            loss = loss + self.weight * auxiliary
        if rng_digest(model) != after_original:
            raise ValueError("Replay advanced the ordinary training RNG.")
        self.handle.write(json.dumps({"position": position, "id": row["id"], "seed": seed,
            "sample_sha256": sample_hash, "rng_before": before, "rng_after_original": after_original,
            "original_loss": original_value,
            "replay_mse": value, "weight": self.weight}, allow_nan=False) + "\n")
        self.counter += 1
        return loss, {**metrics, "decision_replay_mse": value, "decision_replay_weight": self.weight}


def create_fastwam(decision_replay, **kwargs):
    """Hydra factory: keep the production model class and checkpoint format."""
    from fastwam.runtime import create_fastwam as original_factory
    model = original_factory(**kwargs)
    replay = DecisionReplay(**dict(decision_replay))
    original = model.training_loss

    def loss(self, sample, *args, **kwargs):
        return replay.loss(self, original, sample, *args, **kwargs)

    model.training_loss = MethodType(loss, model)
    return model
