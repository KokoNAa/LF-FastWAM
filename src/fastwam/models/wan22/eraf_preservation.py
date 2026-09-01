"""Training-only V9.28 interface teacher; never copies an Action Expert."""

import copy
import hashlib
from collections.abc import Mapping

import torch
from torch import nn


INTERFACE_NAMES = (
    "eraf_action_token_compressor",
    "eraf_action_context_injector",
)
ROLE_SCOPE = "frozen_eraf_baseline_lora_and_gate_plus_compressor_injector"
ACTION_SCOPE = "eraf_action_token_compressor_plus_context_injector_only"
SCHEDULE = "injector_only_with_frozen_v928_teacher_and_gate"
GATE_CONTRACT = "frozen_v928_gate_no_optimization"
PRESERVATION_CONTRACT = (
    "same_noise_teacher_flow_on_noncorrective_teacher_no_worse_than_base_proxy"
)
SELECTIVE_FULL_GOAL_CONTRACT = (
    "same_noise_v928_teacher_no_worse_than_base_on_audited_full_goal_corrective_"
    "action_token_and_context_interface"
)


def tensor_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenInterfaceTeacher(nn.ModuleDict):
    def __init__(self, modules: nn.ModuleDict):
        super().__init__(
            {name: copy.deepcopy(modules[name]) for name in INTERFACE_NAMES}
        )
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Parent .train() calls must not activate the fixed teacher.
        return super().train(False)


def preservation_loss(
    *,
    student,
    teacher,
    target,
    base,
    action_is_pad,
    direct_valid,
    corrective,
    margin=0.0,
    candidate_mask=None,
):
    """Proxy eligibility is NOT a rollout-success label or a safety guarantee.

    By default, corrective rows explicitly target old failures and are not
    distilled.  A caller may supply an explicit ``candidate_mask`` for a
    separately audited subset (V9.31 uses only full-goal corrective rows).
    All-pad rows are excluded; the empty-mask result remains differentiable.
    """
    valid_steps = ~action_is_pad.bool()
    weights = valid_steps.to(student.dtype)
    count = weights.sum(dim=1).clamp_min(1)

    def error(prediction, reference):
        squared = (prediction.float() - reference.float()).square().mean(dim=-1)
        return (squared * weights).sum(dim=1) / count

    if candidate_mask is None:
        candidate_mask = ~corrective.bool()
    else:
        candidate_mask = torch.as_tensor(
            candidate_mask, device=student.device, dtype=torch.bool
        )
        if candidate_mask.shape != direct_valid.shape:
            raise ValueError(
                "Preservation candidate_mask must share the [B] validity shape."
            )
    with torch.no_grad():
        eligible = (
            direct_valid.bool()
            & candidate_mask
            & valid_steps.any(dim=1)
            & (error(teacher, target) <= error(base, target) + margin)
        )
    delta = error(student, teacher.detach())
    loss = (delta * eligible).sum() / eligible.sum().clamp_min(1)
    return loss, eligible


def interface_preservation_loss(*, student, teacher, eligible):
    """Return a differentiable per-sample interface MSE on audited rows only."""
    if student.shape != teacher.shape or student.ndim < 2:
        raise ValueError(
            "Student and teacher interface tensors must share shape [B,...]."
        )
    eligible = torch.as_tensor(eligible, device=student.device, dtype=torch.bool)
    if eligible.shape != (student.shape[0],):
        raise ValueError("Interface preservation eligibility must have shape [B].")
    per_sample = (
        student.float().sub(teacher.detach().float()).square().flatten(1).mean(dim=1)
    )
    return (per_sample * eligible).sum() / eligible.sum().clamp_min(1)


def validate_teacher_payload(payload):
    teacher = payload.get("eraf_preservation_teacher")
    if not isinstance(teacher, dict):
        raise ValueError("Preservation checkpoint is missing its fixed V9.28 teacher.")
    provenance = teacher.get("provenance", {})
    state = teacher.get("state_dict", {})
    checkpoint_sha = str(provenance.get("checkpoint_sha256", ""))
    expected_keys = {
        key
        for key in payload.get("policy_guard", {})
        if key.split(".", 1)[0] in INTERFACE_NAMES
    }
    if (
        provenance.get("objective") != 28
        or provenance.get("step") != 10000
        or not provenance.get("checkpoint")
        or len(checkpoint_sha) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_sha)
        or set(state) != expected_keys
        or payload.get("architecture_metadata", {}).get("eraf_preservation_source")
        != provenance
        or not state
        or tensor_digest(state) != provenance.get("teacher_sha256")
    ):
        raise ValueError("Invalid preservation teacher provenance or tensor checksum.")
    return state, dict(provenance)
