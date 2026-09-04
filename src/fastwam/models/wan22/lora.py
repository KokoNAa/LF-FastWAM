from __future__ import annotations

import fnmatch
import math
import types
from contextlib import contextmanager
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TARGET_MODULES = (
    "text_embedding.0",
    "text_embedding.2",
    "blocks.*.self_attn.q",
    "blocks.*.self_attn.k",
    "blocks.*.self_attn.v",
    "blocks.*.self_attn.o",
    "blocks.*.cross_attn.q",
    "blocks.*.cross_attn.k",
    "blocks.*.cross_attn.v",
    "blocks.*.cross_attn.o",
    "blocks.*.ffn.0",
    "blocks.*.ffn.2",
)

DEFAULT_EXTRA_TRAINABLE_PATTERNS = (
    "action_expert.latent_action_queries",
    "action_expert.action_encoder.*",
    "action_expert.head.*",
    "proprio_encoder.*",
)

DEFAULT_PAIRED_LANGUAGE_CONTROL = {
    "enabled": False,
    "bidirectional_supervision": False,
    # Existing LIBERO runs keep their published hybrid-cache objective. A
    # target can opt into deployment-matched Action ranking explicitly.
    "deployment_matched_action_cache": False,
    "world_language_weight": 0.10,
    "world_language_margin": 0.01,
    "native_action_weight": 1.0,
    "counterfactual_action_weight": 1.0,
    "action_language_weight": 1.0,
    "action_language_margin": 0.01,
    "regularization_weight": 1.0e-6,
}


def normalize_paired_language_control_config(
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize the strict no-ERAF paired-language LoRA control."""
    raw = dict(config or {})
    normalized = {
        "enabled": bool(raw.get("enabled", False)),
        "bidirectional_supervision": bool(
            raw.get("bidirectional_supervision", False)
        ),
        "deployment_matched_action_cache": bool(
            raw.get("deployment_matched_action_cache", False)
        ),
        **{
            name: float(raw.get(name, default))
            for name, default in DEFAULT_PAIRED_LANGUAGE_CONTROL.items()
            if name
            not in {
                "enabled",
                "bidirectional_supervision",
                "deployment_matched_action_cache",
            }
        },
    }
    non_negative = (
        "world_language_weight",
        "world_language_margin",
        "native_action_weight",
        "counterfactual_action_weight",
        "action_language_weight",
        "action_language_margin",
        "regularization_weight",
    )
    invalid = {
        name: normalized[name]
        for name in non_negative
        if normalized[name] < 0.0
    }
    if invalid:
        raise ValueError(
            "LoRA paired-language control values must be non-negative, "
            f"got {invalid}."
        )
    if normalized["bidirectional_supervision"] and not normalized["enabled"]:
        raise ValueError(
            "LoRA bidirectional supervision requires "
            "`paired_language_control.enabled=true`."
        )
    if (
        normalized["deployment_matched_action_cache"]
        and not normalized["enabled"]
    ):
        raise ValueError(
            "Deployment-matched Action ranking requires "
            "`paired_language_control.enabled=true`."
        )
    return normalized


def normalize_lora_config(config: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(config or {})
    enabled = bool(config.get("enabled", False))
    rank = int(config.get("rank", 16))
    alpha = float(config.get("alpha", 32.0))
    dropout = float(config.get("dropout", 0.05))
    experts = [str(value) for value in config.get("experts", ["video", "action"])]
    target_modules = [
        str(value)
        for value in config.get("target_modules", DEFAULT_TARGET_MODULES)
    ]
    extra_trainable_patterns = [
        str(value)
        for value in config.get(
            "extra_trainable_patterns", DEFAULT_EXTRA_TRAINABLE_PATTERNS
        )
    ]
    paired_language_control = normalize_paired_language_control_config(
        config.get("paired_language_control")
    )

    if rank <= 0:
        raise ValueError(f"LoRA `rank` must be positive, got {rank}.")
    if alpha <= 0:
        raise ValueError(f"LoRA `alpha` must be positive, got {alpha}.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"LoRA `dropout` must be in [0, 1), got {dropout}.")
    if not experts:
        raise ValueError("LoRA `experts` must not be empty.")
    unknown_experts = sorted(set(experts) - {"video", "action"})
    if unknown_experts:
        raise ValueError(
            f"Unsupported LoRA experts: {unknown_experts}; expected video/action."
        )
    if not target_modules:
        raise ValueError("LoRA `target_modules` must not be empty.")

    return {
        "enabled": enabled,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "experts": experts,
        "target_modules": target_modules,
        "extra_trainable_patterns": extra_trainable_patterns,
        "paired_language_control": paired_language_control,
    }


def matches_any_pattern(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def is_lora_parameter_name(name: str) -> bool:
    return name.endswith(".lora_A") or name.endswith(".lora_B")


def _lora_linear_forward(linear: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
    base = F.linear(inputs, linear.weight, linear.bias)
    adapter_inputs = F.dropout(
        inputs,
        p=float(linear.lora_dropout),
        training=linear.training,
    )
    # Keep adapter parameters in fp32 for optimizer stability. Explicit casts
    # also make inference work outside autocast when the frozen base is bf16.
    adapter_inputs = adapter_inputs.to(dtype=linear.lora_A.dtype)
    update = F.linear(adapter_inputs, linear.lora_A)
    update = F.linear(update, linear.lora_B)
    return base + update.to(dtype=base.dtype) * float(linear.lora_scaling)


def add_lora_to_linear(
    linear: nn.Linear,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> bool:
    if hasattr(linear, "lora_A") or hasattr(linear, "lora_B"):
        existing_rank = int(getattr(linear, "lora_rank", -1))
        if existing_rank != int(rank):
            raise ValueError(
                f"Linear already has LoRA rank={existing_rank}, requested rank={rank}."
            )
        return False

    adapter_device = linear.weight.device
    lora_a = nn.Parameter(
        torch.empty(
            rank,
            linear.in_features,
            device=adapter_device,
            dtype=torch.float32,
        )
    )
    lora_b = nn.Parameter(
        torch.zeros(
            linear.out_features,
            rank,
            device=adapter_device,
            dtype=torch.float32,
        )
    )
    nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
    linear.register_parameter("lora_A", lora_a)
    linear.register_parameter("lora_B", lora_b)
    linear.lora_rank = int(rank)
    linear.lora_alpha = float(alpha)
    linear.lora_scaling = float(alpha) / float(rank)
    linear.lora_dropout = float(dropout)
    # Patching an existing Linear keeps its original weight/bias state-dict
    # keys, so old FastWAM checkpoints remain directly loadable.
    linear.forward = types.MethodType(_lora_linear_forward, linear)
    return True


def inject_lora(
    module: nn.Module,
    *,
    target_modules: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    target_modules = tuple(target_modules)
    injected: list[str] = []
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        if not matches_any_pattern(name, target_modules):
            continue
        if add_lora_to_linear(
            child,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        ):
            injected.append(name)
    if not injected:
        already_injected = [
            name
            for name, child in module.named_modules()
            if isinstance(child, nn.Linear)
            and hasattr(child, "lora_A")
            and matches_any_pattern(name, target_modules)
        ]
        if not already_injected:
            raise ValueError(
                "LoRA target patterns matched no Linear modules: "
                f"{list(target_modules)}"
            )
    return injected


@contextmanager
def temporarily_disable_lora(module: nn.Module):
    """Evaluate an injected module through its frozen base weights only."""
    scaling: list[tuple[nn.Linear, float]] = []
    for child in module.modules():
        if isinstance(child, nn.Linear) and hasattr(child, "lora_scaling"):
            scaling.append((child, float(child.lora_scaling)))
            child.lora_scaling = 0.0
    try:
        yield
    finally:
        for child, value in scaling:
            child.lora_scaling = value
