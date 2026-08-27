#!/usr/bin/env python3
"""Validate a deployable PGC checkpoint and reconstruct Hydra overrides.

The evaluator must instantiate the exact ERAF architecture saved by training.
This tool keeps that contract in one testable place and adds the RoboTwin-only
checks that a LIBERO checkpoint cannot satisfy accidentally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


ERAF_CONFIG_MAPPING = {
    "eraf_hidden_dim": "hidden_dim",
    "eraf_num_heads": "num_heads",
    "eraf_max_clauses": "max_clauses",
    "eraf_camera_count": "camera_count",
    "eraf_visual_aspect_ratio": "visual_aspect_ratio",
    "eraf_camera_layout": "camera_layout",
    "eraf_temperature": "temperature",
    "eraf_grounding_aux_weight": "grounding_aux_weight",
    "eraf_completion_only_memory": "completion_only_memory",
    "eraf_action_joint_training": "action_joint_training",
    "eraf_fresh_joint_training": "fresh_joint_training",
    "eraf_pretrained_joint_training": "pretrained_joint_training",
    "eraf_bidirectional_supervision": "bidirectional_supervision",
    "eraf_context_injection_warmup_steps": "context_injection_warmup_steps",
    "eraf_context_injection_ramp_steps": "context_injection_ramp_steps",
    "eraf_action_grounding_hidden_dim": "action_grounding_hidden_dim",
    "eraf_action_grounding_num_heads": "action_grounding_num_heads",
    "eraf_action_grounding_learning_rate": "action_grounding_learning_rate",
    "eraf_action_geometry_hidden_dim": "action_geometry_hidden_dim",
    "eraf_action_geometry_learning_rate": "action_geometry_learning_rate",
    "eraf_action_geometry_residual_max_abs": "action_geometry_residual_max_abs",
    "eraf_action_phase_residual_imitation_weight": "action_phase_residual_imitation_weight",
    "eraf_action_phase_direction_weight": "action_phase_direction_weight",
    "eraf_action_phase_approach_weight": "action_phase_approach_weight",
    "eraf_action_phase_transport_weight": "action_phase_transport_weight",
    "eraf_action_phase_release_weight": "action_phase_release_weight",
    "eraf_action_phase_direction_min_norm": "action_phase_direction_min_norm",
    "eraf_action_servo_frame_weight": "action_servo_frame_weight",
    "eraf_action_waypoint_min_cosine": "action_waypoint_min_cosine",
    "eraf_action_waypoint_tangent_max_ratio": "action_waypoint_tangent_max_ratio",
    "eraf_action_expert_imitation_weight": "action_expert_imitation_weight",
    "eraf_action_expert_direction_weight": "action_expert_direction_weight",
    "eraf_action_expert_deployed_weight": "action_expert_deployed_weight",
    "eraf_action_expert_distillation_weight": "action_expert_distillation_weight",
    "eraf_action_expert_native_zero_weight": "action_expert_native_zero_weight",
    "eraf_action_clause_ranking_weight": "action_clause_ranking_weight",
    "eraf_action_clause_ranking_margin": "action_clause_ranking_margin",
    "eraf_action_clause_teacher_weight": "action_clause_teacher_weight",
    "eraf_action_clause_alignment_guard_weight": "action_clause_alignment_guard_weight",
    "eraf_action_eef_initial_scale": "action_eef_scale",
    "eraf_action_eef_initial_bias": "action_eef_bias",
    "eraf_action_causal_ranking_weight": "action_causal_ranking_weight",
    "eraf_action_causal_margin": "action_causal_margin",
    "eraf_expert_lora_world_language_weight": "expert_lora_world_language_weight",
    "eraf_expert_lora_world_language_margin": "expert_lora_world_language_margin",
    "eraf_expert_lora_native_action_weight": "expert_lora_native_action_weight",
    "eraf_expert_lora_counterfactual_action_weight": "expert_lora_counterfactual_action_weight",
    "eraf_expert_lora_regularization_weight": "expert_lora_regularization_weight",
    "eraf_attention_mask_weight": "attention_mask_weight",
    "eraf_role_swap_weight": "role_swap_weight",
    "eraf_role_overlap_weight": "role_overlap_weight",
    "eraf_role_swap_margin": "role_swap_margin",
    "eraf_role_assignment_weight": "role_assignment_weight",
    "eraf_role_assignment_temperature": "role_assignment_temperature",
    "eraf_role_assignment_hard_weight": "role_assignment_hard_weight",
    "eraf_structured_assignment_weight": "structured_assignment_weight",
    "eraf_structured_assignment_temperature": "structured_assignment_temperature",
    "eraf_structured_assignment_hard_weight": "structured_assignment_hard_weight",
    "eraf_multi_clause_consistency_weight": "multi_clause_consistency_weight",
    "eraf_clause_tuple_assignment_weight": "clause_tuple_assignment_weight",
    "eraf_clause_tuple_temperature": "clause_tuple_temperature",
    "eraf_clause_tuple_hard_weight": "clause_tuple_hard_weight",
    "eraf_clause_tuple_multi_consistency_weight": "clause_tuple_multi_consistency_weight",
    "eraf_clause_activation_balance_weight": "clause_activation_balance_weight",
    "eraf_clause_cardinality_weight": "clause_cardinality_weight",
    "eraf_clause_worst_slot_weight": "clause_worst_slot_weight",
    "eraf_clause_multi_group_weight": "clause_multi_group_weight",
    "eraf_clause_adapter_energy_weight": "clause_adapter_energy_weight",
    "eraf_view_fusion_weight": "view_fusion_weight",
    "eraf_view_fusion_energy_weight": "view_fusion_energy_weight",
    "eraf_clause_scheduler_weight": "clause_scheduler_weight",
    "eraf_clause_scheduler_energy_weight": "clause_scheduler_energy_weight",
    "eraf_phase_rebinding_energy_weight": "phase_rebinding_energy_weight",
    "eraf_phase_safe_memory_state_weight": "phase_safe_memory_state_weight",
    "eraf_phase_safe_memory_scheduler_weight": "phase_safe_memory_scheduler_weight",
    "eraf_phase_safe_memory_energy_weight": "phase_safe_memory_energy_weight",
    "eraf_role_adapter_hidden_dim": "role_adapter_hidden_dim",
    "eraf_structured_role_adapter_hidden_dim": "structured_role_adapter_hidden_dim",
    "eraf_balanced_role_adapter_hidden_dim": "balanced_role_adapter_hidden_dim",
    "eraf_clause_activation_adapter_hidden_dim": "clause_activation_adapter_hidden_dim",
    "eraf_clause_activation_residual_max_abs": "clause_activation_residual_max_abs",
    "eraf_view_fusion_adapter_hidden_dim": "view_fusion_adapter_hidden_dim",
    "eraf_view_fusion_residual_max_abs": "view_fusion_residual_max_abs",
    "eraf_clause_scheduler_hidden_dim": "clause_scheduler_hidden_dim",
    "eraf_clause_scheduler_residual_max_abs": "clause_scheduler_residual_max_abs",
    "eraf_closed_loop_rebinding_hidden_dim": "closed_loop_rebinding_hidden_dim",
    "eraf_closed_loop_query_residual_max_abs": "closed_loop_query_residual_max_abs",
    "eraf_closed_loop_state_residual_max_abs": "closed_loop_state_residual_max_abs",
    "eraf_phase_safe_memory_hidden_dim": "phase_safe_memory_hidden_dim",
    "eraf_phase_safe_memory_state_count": "phase_safe_memory_state_count",
    "eraf_phase_safe_memory_routing_residual_max_abs": "phase_safe_memory_routing_residual_max_abs",
    "eraf_role_attention_preservation_weight": "role_attention_preservation_weight",
    "eraf_role_position_preservation_weight": "role_position_preservation_weight",
    "eraf_role_anchor_preservation_weight": "role_anchor_preservation_weight",
    "eraf_role_relation_preservation_weight": "role_relation_preservation_weight",
    "eraf_role_adapter_energy_weight": "role_adapter_energy_weight",
}


def validate_payload(
    payload: Mapping[str, Any], *, target: str, inference_steps: int
) -> dict[str, Any]:
    metadata = payload.get("architecture_metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint architecture_metadata must be an object.")
    if metadata.get("architecture") != "pgc_fastwam":
        raise ValueError("Checkpoint is not a PGC-FastWAM checkpoint.")
    version = int(metadata.get("policy_guard_version", -1))
    objective = int(metadata.get("eraf_grounding_objective_version", -1))
    if version != 9 or objective != 26:
        raise ValueError(
            "RoboTwin integration requires the current PGC V9.26 contract; "
            f"got version={version}, objective={objective}."
        )
    if payload.get("format") != "fastwam_policy_guard_v9":
        raise ValueError("Checkpoint format/version does not match PGC V9.")
    fresh_joint = bool(metadata.get("eraf_fresh_joint_training", False))
    pretrained_joint = bool(
        metadata.get("eraf_pretrained_joint_training", False)
    )
    if fresh_joint and pretrained_joint:
        raise ValueError(
            "PGC V9.26 cannot be both fresh and pretrained ERAF joint."
        )
    expected_protection = (
        "pretrained_eraf_ramp_then_single_path_no_candidate_gate"
        if pretrained_joint
        else "fresh_eraf_warmup_then_single_path_no_candidate_gate"
        if fresh_joint
        else "single_eraf_path_no_candidate_gate"
    )
    required_metadata = {
        "eraf_single_path": True,
        "gate_mode": "eraf_only",
        "policy_protection": expected_protection,
        "eraf_post_action_residual_active": False,
        "counterfactual_action_interface": "single_eraf_conditioned_action_denoising_path",
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Checkpoint violates {key}: expected={expected!r}, "
                f"got={metadata.get(key)!r}."
            )
    if pretrained_joint:
        required_provenance = (
            "eraf_pretrained_source_checkpoint",
            "eraf_pretrained_source_sha256",
            "eraf_pretrained_source_objective",
            "eraf_pretrained_source_step",
            "eraf_pretrained_tensor_count",
        )
        missing = [
            name
            for name in required_provenance
            if metadata.get(name) in (None, "")
        ]
        if missing:
            raise ValueError(
                "Pretrained ERAF joint checkpoint lacks provenance: "
                f"{missing}."
            )
    rollout_steps = int(metadata.get("rollout_num_inference_steps", -1))
    if rollout_steps != int(inference_steps):
        raise ValueError(
            "Checkpoint/evaluation denoising-step mismatch: "
            f"checkpoint={rollout_steps}, evaluation={inference_steps}."
        )
    lora_state = payload.get("eraf_shared_expert_lora")
    lora_config = payload.get("eraf_shared_expert_lora_config")
    if not isinstance(lora_state, Mapping) or not lora_state:
        raise ValueError("PGC V9.26 checkpoint has no shared Expert LoRA tensors.")
    if not isinstance(lora_config, Mapping):
        raise ValueError("PGC V9.26 checkpoint has no shared Expert LoRA config.")
    if set(map(str, lora_config.get("experts", ()))) != {"video", "action"}:
        raise ValueError("PGC V9.26 must adapt both Video and Action Experts.")

    target = str(target).strip().lower()
    if target == "robotwin":
        robotwin_expected = {
            "action_output_dim": 14,
            "proprio_dim": 14,
            "eraf_camera_count": 3,
            "eraf_camera_layout": "robotwin_mosaic",
        }
        for key, expected in robotwin_expected.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"RoboTwin checkpoint requires {key}={expected!r}, "
                    f"got {metadata.get(key)!r}."
                )
        aspect = float(metadata.get("eraf_visual_aspect_ratio", -1.0))
        if abs(aspect - 5.0 / 6.0) > 1.0e-9:
            raise ValueError(
                "RoboTwin checkpoint requires ERAF visual_aspect_ratio=5/6, "
                f"got {aspect}."
            )
    elif target != "generic":
        raise ValueError(f"Unsupported checkpoint target: {target!r}.")
    return dict(metadata)


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def hydra_overrides(metadata: Mapping[str, Any]) -> list[str]:
    prefix = "model.policy_guard.entity_relation_grounding."
    overrides = [
        "model.action_dit_config.use_latent_action_queries=false",
        "model.langforce_mvp.enabled=false",
        "model.langforce_mvp.enable_prior=false",
        "model.langforce_mvp.enable_posterior_advantage=false",
        "model.transition_contract.enabled=false",
        "model.policy_guard.enabled=true",
        "model.policy_guard.version=9",
        "model.policy_guard.gate_mode=eraf_only",
        "model.lora.enabled=false",
        f"{prefix}training_stage={metadata['eraf_training_stage']}",
        f"{prefix}grounding_objective_version=26",
        f"{prefix}entity_only={_hydra_value(bool(metadata.get('eraf_entity_only', False)))}",
        f"{prefix}use_anchors={_hydra_value(bool(metadata.get('eraf_use_anchors', True)))}",
    ]
    for metadata_key, config_key in ERAF_CONFIG_MAPPING.items():
        value = metadata.get(metadata_key)
        if value is not None:
            overrides.append(f"{prefix}{config_key}={_hydra_value(value)}")
    return overrides


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint root must be a mapping.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--target", choices=("generic", "robotwin"), default="generic")
    parser.add_argument("--inference-steps", type=int, required=True)
    parser.add_argument("--format", choices=("json", "hydra"), default="json")
    args = parser.parse_args()
    if args.inference_steps <= 0:
        parser.error("--inference-steps must be positive")
    metadata = validate_payload(
        load_checkpoint(args.checkpoint.expanduser().resolve()),
        target=args.target,
        inference_steps=args.inference_steps,
    )
    if args.format == "hydra":
        print("\n".join(hydra_overrides(metadata)))
    else:
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
