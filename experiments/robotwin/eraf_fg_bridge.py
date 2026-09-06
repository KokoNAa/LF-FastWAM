"""Differentiable production ERAF path for the warm-policy repair experiment.

Frozen VAE/T5 inputs may be cached. Video LoRA, GoalGraph, ERAF and the action
context are rebuilt from those inputs whenever a student graph is created.
The checkpoint protocol records warm policy plus fresh ERAF explicitly; it
does not claim the historical V9.28/V9.39 training lineage.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping

CHECKPOINT_FORMAT = "robotwin_eraf_fg_adapter_v1"
PROTOCOL = "warm_shared400_fresh_robotwin_eraf_full_goal_repair_v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eraf_guard_config() -> dict[str, Any]:
    # These are architectural constructor settings, not historical-stage
    # completion claims. Our optimizer and checkpoint stage live separately.
    return {
        "enabled": True, "version": 9, "gate_mode": "eraf_only",
        "entity_relation_grounding": {
            "training_stage": "action", "grounding_objective_version": 26,
            "initialization_contract": "released_base_fresh_eraf",
            "fresh_joint_training": True, "pretrained_joint_training": False,
            "action_joint_training": True, "safe_gain_training": False,
            "bidirectional_supervision": True, "completion_only_memory": True,
            "camera_count": 3, "camera_layout": "robotwin_mosaic",
            "visual_aspect_ratio": 5. / 6., "grounding_aux_weight": .1,
            "context_injection_warmup_steps": 0, "context_injection_ramp_steps": 0,
        },
    }


def compatible_lora_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    if not result.get("enabled") or set(result.get("experts", [])) != {"video", "action"}:
        raise ValueError("Repair needs the existing shared Video/Action LoRA.")
    # The baseline's paired-language *loss dispatch* is no-ERAF-only. Removing
    # its training flags changes no tensor or deployed no-ERAF computation.
    paired = result.get("paired_language_control")
    if paired is not None:
        for flag in ("enabled", "bidirectional_supervision", "deployment_matched_action_cache",
                     "correct_branch_action_ranking"):
            paired[flag] = False
    return result


def restore_policy_adapter(model, payload: Mapping[str, Any]) -> None:
    import torch
    model.configure_lora(compatible_lora_config(payload["lora_config"]))
    saved = payload["mot_trainable"]
    expected = model._lora_adapter_state_dict()
    if set(saved) != set(expected):
        raise ValueError("Policy LoRA tensor set changed during ERAF initialization.")
    if any(saved[k].shape != expected[k].shape for k in saved):
        raise ValueError("Policy LoRA geometry changed during ERAF initialization.")
    model.mot.load_state_dict(saved, strict=False)
    restored = model._lora_adapter_state_dict()
    if any(not torch.equal(restored[k].cpu(), saved[k].cpu()) for k in saved):
        raise ValueError("ERAF initialization did not preserve exact policy LoRA weights.")


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("protocol") != PROTOCOL:
        raise ValueError("Not an ERAF/FG repair checkpoint.")
    geometry = payload.get("geometry", {})
    expected = {"action_dim": 14, "proprio_dim": 14, "camera_count": 3,
                "camera_layout": "robotwin_mosaic", "action_horizon": 32,
                "replan_steps": 24, "inference_steps": 10}
    if geometry != expected or payload.get("guard_config") != eraf_guard_config():
        raise ValueError("ERAF/FG repair architecture or rollout contract changed.")
    if not payload.get("policy_guard") or not payload.get("mot_trainable"):
        raise ValueError("Repair checkpoint is missing ERAF or policy weights.")
    compatible_lora_config(payload["lora_config"])
    if payload.get("stage") not in {"bootstrap", "grounding", "interface", "joint"}:
        raise ValueError("Unknown repair stage.")
    if payload.get("fg_supervision") not in {"off", "local", "full"}:
        raise ValueError("Unknown FG ablation.")
    if int(payload.get("optimizer_steps", -1)) < 0:
        raise ValueError("Invalid stage optimizer count.")
    return dict(payload)


def load_repair_checkpoint(model, path: str | Path, *, payload=None) -> dict[str, Any]:
    import torch
    payload = validate_payload(payload if payload is not None else
                               torch.load(path, map_location="cpu", weights_only=False))
    validate_model_geometry(model)
    base = Path(payload["base_checkpoint"]).expanduser()
    if not base.is_absolute():
        base = Path(path).resolve().parent / base
    model.load_checkpoint(str(base))
    restore_policy_adapter(model, payload)
    model.policy_guard_modules.load_state_dict(payload["policy_guard"], strict=True)
    model.lora_base_checkpoint = str(base.resolve())
    model.policy_guard_base_checkpoint = str(base.resolve())
    return payload


def load_policy(checkpoint, manifest, *, device="cuda:0", seed=42, bootstrap=False):
    """Construct the exact deployment wrapper with the new architecture contract."""
    import torch
    from omegaconf import OmegaConf, open_dict
    from fastwam.utils.config_resolvers import register_default_resolvers
    from scripts.probe_robotwin_no_eraf import inference_bootstrap_configs
    from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy
    register_default_resolvers()
    cfg = OmegaConf.load(manifest["original_train_config"])
    model_cfg, processor_cfg = inference_bootstrap_configs(cfg)
    with open_dict(model_cfg):
        model_cfg.policy_guard = eraf_guard_config()
        model_cfg.lora.enabled = False
    parent = torch.load(checkpoint, map_location="cpu", weights_only=False) if bootstrap else None
    policy = WorldActionRobotWinPolicy(
        model_cfg=model_cfg, processor_cfg=processor_cfg,
        checkpoint_path=parent["base_checkpoint"] if bootstrap else str(checkpoint),
        dataset_stats_path=Path(manifest["stats_path"]), device=device,
        model_dtype=torch.bfloat16, action_horizon=32, replan_steps=24,
        num_inference_steps=10, sigma_shift=None, seed=seed, text_cfg_scale=1.,
        negative_prompt="", rand_device="cpu", tiled=False, timing_enabled=False,
        num_video_frames=9, task_name="eraf_fg_repair", task_config="demo_clean")
    if bootstrap:
        restore_policy_adapter(policy.model, parent)
    validate_model_geometry(policy.model)
    return policy


def validate_model_geometry(model) -> None:
    if (model.action_expert.action_dim != 14 or model.proprio_dim != 14
            or not model.policy_guard_enabled or model.policy_guard_version != 9
            or model.policy_guard_eraf_grounding_objective_version != 26):
        raise ValueError("Repair checkpoint requires the real 14-D RoboTwin ERAF model.")


def save_repair_checkpoint(model, path: str | Path, *, stage: str, steps: int,
                           parent: str | Path, fg_supervision: str,
                           provenance: Mapping[str, Any] | None = None,
                           record_hashes: bool = True) -> dict[str, Any]:
    import torch
    validate_model_geometry(model)
    payload = {
        "format": CHECKPOINT_FORMAT, "protocol": PROTOCOL, "stage": stage,
        "optimizer_steps": int(steps), "fg_supervision": fg_supervision,
        "base_checkpoint": model.lora_base_checkpoint,
        "parent_checkpoint": str(Path(parent).resolve()),
        "guard_config": eraf_guard_config(), "lora_config": dict(model.lora_config),
        "geometry": {"action_dim": 14, "proprio_dim": 14, "camera_count": 3,
                     "camera_layout": "robotwin_mosaic", "action_horizon": 32,
                     "replan_steps": 24, "inference_steps": 10},
        "mot_trainable": model._lora_adapter_state_dict(),
        "policy_guard": {k: v.detach().cpu().clone() for k, v in model.policy_guard_modules.state_dict().items()},
        "provenance": dict(provenance or {}),
    }
    if record_hashes:
        payload['parent_sha256'] = file_sha256(parent)
    validate_payload(payload)
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
    temporary.replace(target)
    return payload


def capture_frozen_inputs(policy, observation, instruction: str) -> dict[str, Any]:
    """Use the existing clean-context capture, then attach raw normalized state."""
    from experiments.robotwin.joint_adapter_repair import capture_inputs
    from experiments.robotwin.same_state_repair import move_cache
    enabled, state = policy.model.policy_guard_enabled, policy.policy_guard_state
    try:
        policy.model.policy_guard_enabled = False
        policy.policy_guard_state = None
        _, captured, _ = capture_inputs(policy.model, lambda: policy._infer_action_chunk(observation, instruction))
    finally:
        policy.model.policy_guard_enabled = enabled
        policy.policy_guard_state = state
    captured["proprio"] = policy._normalize_state(observation["joint_action"]["vector"]).detach().cpu()
    captured["policy_guard_state"] = move_cache(state, "cpu", clone=True) if state is not None else None
    return captured


def build_cache(model, captured: Mapping[str, Any], *, eraf: bool = True):
    pre = model.video_expert.pre_dit(**captured["video_inputs"])
    action = dict(captured["action_inputs"])
    length = int(action["video_seq_len"])
    if pre["tokens"].shape[1] != length:
        raise ValueError("Video token geometry changed.")
    prefill = model.mot.prefill_video_cache(
        video_tokens=pre["tokens"], video_freqs=pre["freqs"], video_t_mod=pre["t_mod"],
        video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
        video_attention_mask=action["attention_mask"][:length, :length], return_final_hidden=eraf)
    if not eraf:
        action["video_kv_cache"] = prefill
        return action, None, {}
    cache, final_hidden = prefill
    action["video_kv_cache"] = cache
    state_mask = action["state_only_context_mask"]
    if state_mask.shape[0] != 1:
        raise ValueError("Repair cache currently uses one state per GPU microbatch.")
    language_len = action["context"].shape[1] - int(state_mask.sum().item())
    proprio = captured.get("proprio")
    if proprio is None:
        raise ValueError("ERAF cache needs the original normalized 14-D proprio.")
    queries, _, metrics = model._encode_policy_guard_goal(
        final_video_hidden=final_hidden, current_visual_hidden=pre["tokens"],
        video_tokens_per_frame=int(pre["meta"]["tokens_per_frame"]),
        context=action["context"], context_mask=action["context_mask"],
        language_context_len=language_len, proprio=proprio,
        policy_guard_state=captured.get("policy_guard_state"))
    action["video_tokens_per_frame"] = int(pre["meta"]["tokens_per_frame"])
    return action, queries, metrics


def predict(model, captured, noisy, time, *, eraf: bool = True, checkpoint: bool = True):
    import torch
    from torch.utils.checkpoint import checkpoint as activation_checkpoint

    def forward(x, t):
        action, queries, _ = build_cache(model, captured, eraf=eraf)
        if not eraf:
            body = getattr(type(model)._predict_action_noise_with_cache, "__wrapped__", None)
            if body is None:
                raise ValueError("Expected the production no_grad action predictor.")
            return body(model, latents_action=x, timestep_action=t, **action)
        return model._forward_policy_guard_action_from_cache(
            action_tokens=x, timestep_action=t, context=action["context"],
            full_context_mask=action["context_mask"],
            state_only_context_mask=action["state_only_context_mask"],
            video_kv_cache=action["video_kv_cache"], video_seq_len=action["video_seq_len"],
            video_tokens_per_frame=action["video_tokens_per_frame"], routed_goal_queries=queries)

    with torch.enable_grad():
        return activation_checkpoint(forward, noisy, time, use_reentrant=False) if checkpoint else forward(noisy, time)


def masked_mse(prediction, target, valid):
    import torch
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Expected matching [B,T,D] action tensors.")
    mask = torch.as_tensor(valid, dtype=torch.bool, device=prediction.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape != prediction.shape[:2] or not bool(mask.any(dim=1).all()):
        raise ValueError("Every action example needs a nonempty [B,T] validity mask.")
    squared = (prediction.float() - target.float()).square().mean(dim=-1)
    return ((squared * mask).sum(dim=1) / mask.sum(dim=1)).mean()


ROUTE_PARTS = {"base_query_projection", "relation_attention", "query_delta_projection",
               "embedding_delta_projection"}
INTERFACE_PARTS = {"goal_graph", "goal_query_seeds", "eraf_action_grounding_bridge",
                   "eraf_action_context_injector"}


def trainable_parameters(model, stage: str, *, eraf=True, policy_scope='all'):
    """Freeze semantic prediction after grounding; train its action interfaces."""
    import torch
    if stage not in {"grounding", "interface", "joint"}:
        raise ValueError("Unknown optimization stage.")
    if policy_scope not in {'all', 'action'}:
        raise ValueError('Unknown policy parameter scope.')
    model.eval().requires_grad_(False)
    selected = {}
    if stage == "joint":
        for name, p in model.mot.named_parameters():
            if name.endswith((".lora_A", ".lora_B")):
                if policy_scope == 'action' and '.action.' not in '.' + name:
                    continue
                p.data = p.data.to(torch.float32)
                p.requires_grad_(True)
                selected["mot." + name] = p
    if eraf:
        for name, p in model.policy_guard_modules.named_parameters():
            parts = name.split(".")
            interface = (parts[0] in INTERFACE_PARTS or
                         (parts[0] == "entity_relation_affordance" and parts[1] in ROUTE_PARTS))
            semantic = parts[0] == "entity_relation_affordance" and not interface
            if (stage == "grounding" and semantic) or (stage != "grounding" and interface):
                p.requires_grad_(True)
                selected["guard." + name] = p
    if not selected:
        raise ValueError("The selected arm has no trainable parameters.")
    return selected


class MasterAdamW:
    """Retain small updates in FP32 while preserving deployed BF16 arithmetic.

    Updating BF16 weights directly at 1e-5 can discard repeated small updates.
    The master parameters accumulate these updates before each model copy.
    """
    def __init__(self, parameters, *, lr, weight_decay=0.):
        import torch
        self.live = list(parameters)
        self.master = [torch.nn.Parameter(p.detach().float().clone()) for p in self.live]
        self.optimizer = torch.optim.AdamW(self.master, lr=lr, weight_decay=weight_decay)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for p in self.live:
            p.grad = None

    def step(self, max_norm=1.):
        import torch
        for live, master in zip(self.live, self.master, strict=True):
            master.grad = live.grad.detach().float() if live.grad is not None else None
        norm = torch.nn.utils.clip_grad_norm_(self.master, max_norm, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for live, master in zip(self.live, self.master, strict=True):
                live.copy_(master)
        return float(norm)

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(),
                "master": [p.detach().cpu().clone() for p in self.master]}

    def load_state_dict(self, state):
        import torch
        values = state['master']
        if len(values) != len(self.master) or any(a.shape != b.shape for a, b in zip(values, self.master, strict=True)):
            raise ValueError('FP32 optimizer master geometry changed.')
        self.optimizer.load_state_dict(state['optimizer'])
        with torch.no_grad():
            for master, live, value in zip(self.master, self.live, values, strict=True):
                master.copy_(value.to(master))
                live.copy_(master)
