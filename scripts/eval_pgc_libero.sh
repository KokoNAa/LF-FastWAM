#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:-4}"
NUM_TRIALS="${2:-5}"
CONDITION="${3:-correct}"
EVAL_SEED="${4:-42}"
NUM_INFERENCE_STEPS="${5:-10}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PGC_CHECKPOINT="${PGC_CHECKPOINT:?Set PGC_CHECKPOINT to a PGC-FastWAM checkpoint}"
STATS_PATH="${STATS_PATH:-${DIFFSYNTH_MODEL_BASE_PATH:-./checkpoints}/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
SUITES="${PGC_EVAL_SUITES:-[libero_spatial,libero_object,libero_goal,libero_10]}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(pwd)/evaluate_results/pgc_libero_${CONDITION}_seed${EVAL_SEED}_trials${NUM_TRIALS}}"
MANIFEST_PATH="${PGC_MANIFEST_PATH:-}"
GATE_MODE="${PGC_GATE_MODE:-guarded}"
GATE_THRESHOLD="${PGC_GATE_THRESHOLD:-0.20}"
MIN_COUNTERFACTUAL_SCORE="${PGC_MIN_COUNTERFACTUAL_SCORE:-0.60}"
MAX_POLICY_STEPS="${PGC_MAX_POLICY_STEPS:-}"
CLOSED_LOOP_CAPTURE_DIR="${PGC_CLOSED_LOOP_CAPTURE_DIR:-}"
CLOSED_LOOP_CAPTURE_STRIDE="${PGC_CLOSED_LOOP_CAPTURE_STRIDE_REPLANS:-1}"
CLOSED_LOOP_CAPTURE_MAX_STATES="${PGC_CLOSED_LOOP_CAPTURE_MAX_STATES_PER_EPISODE:-12}"
ERAF_CLOSED_LOOP_CAPTURE_DIR="${PGC_ERAF_CLOSED_LOOP_CAPTURE_DIR:-}"
ERAF_CLOSED_LOOP_CAPTURE_STRIDE="${PGC_ERAF_CLOSED_LOOP_CAPTURE_STRIDE_REPLANS:-1}"
ERAF_CLOSED_LOOP_CAPTURE_MAX_STATES="${PGC_ERAF_CLOSED_LOOP_CAPTURE_MAX_STATES_PER_EPISODE:-48}"
ERAF_CLOSED_LOOP_CAPTURE_STAGES="${PGC_ERAF_CLOSED_LOOP_CAPTURE_STAGES:-initial_search,holding,released_unfinished,next_clause_search}"
V9_ABLATION="${PGC_V9_ABLATION:-full}"
ERAF_SHADOW_AUDIT="${PGC_ERAF_SHADOW_AUDIT:-false}"
ERAF_SHADOW_SIDECAR_DIR="${PGC_ERAF_SHADOW_SIDECAR_DIR:-}"
ERAF_ORACLE="${PGC_ERAF_ORACLE:-false}"
ERAF_ORACLE_SIDECAR_DIR="${PGC_ERAF_ORACLE_SIDECAR_DIR:-}"
ERAF_ORACLE_PHASE_SERVO="${PGC_ERAF_ORACLE_PHASE_SERVO:-false}"
ERAF_ORACLE_SERVO_SCOPE="${PGC_ERAF_ORACLE_SERVO_SCOPE:-full}"
ERAF_ORACLE_SERVO_APPROACH_GAIN="${PGC_ERAF_ORACLE_SERVO_APPROACH_GAIN:-4.0}"
ERAF_ORACLE_SERVO_TRANSPORT_GAIN="${PGC_ERAF_ORACLE_SERVO_TRANSPORT_GAIN:-4.0}"
ERAF_ORACLE_SERVO_MAX_TRANSLATION="${PGC_ERAF_ORACLE_SERVO_MAX_TRANSLATION_ACTION:-0.20}"
ERAF_ORACLE_SERVO_REPLAN_STEPS="${PGC_ERAF_ORACLE_SERVO_REPLAN_STEPS:-5}"
ERAF_ORACLE_SERVO_APPROACH_HEIGHT="${PGC_ERAF_ORACLE_SERVO_APPROACH_HEIGHT_M:-0.08}"
ERAF_ORACLE_SERVO_TRANSPORT_HEIGHT="${PGC_ERAF_ORACLE_SERVO_TRANSPORT_HEIGHT_M:-0.10}"
ERAF_ORACLE_SERVO_GRASP_OFFSET="${PGC_ERAF_ORACLE_SERVO_GRASP_OFFSET_M:-0.01}"
ERAF_ORACLE_SERVO_RELEASE_HEIGHT="${PGC_ERAF_ORACLE_SERVO_RELEASE_HEIGHT_M:-0.04}"
ERAF_ORACLE_SERVO_HORIZONTAL_TOLERANCE="${PGC_ERAF_ORACLE_SERVO_HORIZONTAL_TOLERANCE_M:-0.035}"
ERAF_ORACLE_SERVO_GRASP_DISTANCE="${PGC_ERAF_ORACLE_SERVO_GRASP_DISTANCE_M:-0.035}"
ERAF_ORACLE_SERVO_RELEASE_DISTANCE="${PGC_ERAF_ORACLE_SERVO_RELEASE_DISTANCE_M:-0.05}"
ERAF_ORACLE_SERVO_INTERACTION_DISTANCE="${PGC_ERAF_ORACLE_SERVO_INTERACTION_DISTANCE_M:-0.045}"
ERAF_STATELESS_REPLAN_ABLATION="${PGC_ERAF_STATELESS_REPLAN_ABLATION:-false}"
ERAF_COMPLETION_ONLY_MEMORY_ABLATION="${PGC_ERAF_COMPLETION_ONLY_MEMORY_ABLATION:-false}"
ERAF_DIAGNOSTICS_DEFAULT=true
if [[ "${ERAF_SHADOW_AUDIT}" == "true" || "${ERAF_ORACLE}" == "true" ]]; then
  # A shadow audit scores every replan directly in JSON. Avoid writing several
  # thousand PNG/NPZ overlays unless the caller opts in explicitly.
  ERAF_DIAGNOSTICS_DEFAULT=false
fi
ERAF_DIAGNOSTICS="${PGC_ERAF_DIAGNOSTICS:-${ERAF_DIAGNOSTICS_DEFAULT}}"
ERAF_OVERLAY_DIR="${PGC_ERAF_OVERLAY_DIR:-${OUTPUT_ROOT}/eraf_overlays}"

for value_name in NUM_GPUS NUM_TRIALS NUM_INFERENCE_STEPS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got ${value}." >&2
    exit 1
  fi
done
if [[ -n "${MAX_POLICY_STEPS}" ]] && ! [[ "${MAX_POLICY_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PGC_MAX_POLICY_STEPS must be a positive integer when set." >&2
  exit 1
fi
case "${CONDITION}" in
  correct|null|shuffled|counterfactual) ;;
  *)
    echo "Condition must be correct, null, shuffled, or counterfactual." >&2
    exit 1
    ;;
esac
case "${GATE_MODE}" in
  guarded|base|counterfactual) ;;
  *)
    echo "PGC_GATE_MODE must be guarded, base, or counterfactual." >&2
    exit 1
    ;;
esac
case "${ERAF_DIAGNOSTICS}" in
  true|false) ;;
  *)
    echo "PGC_ERAF_DIAGNOSTICS must be true or false." >&2
    exit 1
    ;;
esac
case "${ERAF_SHADOW_AUDIT}" in
  true|false) ;;
  *)
    echo "PGC_ERAF_SHADOW_AUDIT must be true or false." >&2
    exit 1
    ;;
esac
case "${ERAF_ORACLE}" in
  true|false) ;;
  *)
    echo "PGC_ERAF_ORACLE must be true or false." >&2
    exit 1
    ;;
esac
case "${ERAF_ORACLE_PHASE_SERVO}" in
  true|false) ;;
  *)
    echo "PGC_ERAF_ORACLE_PHASE_SERVO must be true or false." >&2
    exit 1
    ;;
esac
if [[ "${ERAF_ORACLE_PHASE_SERVO}" == "true" && "${ERAF_ORACLE}" != "true" ]]; then
  echo "Oracle phase servo requires PGC_ERAF_ORACLE=true." >&2
  exit 1
fi
case "${ERAF_ORACLE_SERVO_SCOPE}" in
  full|transport_proposal_release|transport_oracle_release) ;;
  *)
    echo "PGC_ERAF_ORACLE_SERVO_SCOPE must be full, transport_proposal_release, or transport_oracle_release." >&2
    exit 1
    ;;
esac
if ! [[ "${ERAF_ORACLE_SERVO_REPLAN_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PGC_ERAF_ORACLE_SERVO_REPLAN_STEPS must be a positive integer." >&2
  exit 1
fi
if [[ "${ERAF_ORACLE}" == "true" && "${ERAF_SHADOW_AUDIT}" == "true" ]]; then
  echo "Oracle ERAF and passive ERAF shadow audit are mutually exclusive." >&2
  exit 1
fi
case "${ERAF_STATELESS_REPLAN_ABLATION}" in
  true|false) ;;
  *)
    echo "PGC_ERAF_STATELESS_REPLAN_ABLATION must be true or false." >&2
    exit 1
    ;;
esac
case "${ERAF_COMPLETION_ONLY_MEMORY_ABLATION}" in
  true|false) ;;
  *)
    echo "PGC_ERAF_COMPLETION_ONLY_MEMORY_ABLATION must be true or false." >&2
    exit 1
    ;;
esac
if [[ "${ERAF_STATELESS_REPLAN_ABLATION}" == "true" && "${ERAF_COMPLETION_ONLY_MEMORY_ABLATION}" == "true" ]]; then
  echo "Stateless and completion-only policy-state ablations are mutually exclusive." >&2
  exit 1
fi
if [[ ! -f "${PGC_CHECKPOINT}" ]]; then
  echo "PGC checkpoint not found: ${PGC_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${STATS_PATH}" ]]; then
  echo "Dataset stats not found: ${STATS_PATH}" >&2
  exit 1
fi
if [[ "${CONDITION}" == "shuffled" || "${CONDITION}" == "counterfactual" ]]; then
  if [[ -z "${MANIFEST_PATH}" || ! -f "${MANIFEST_PATH}" ]]; then
    echo "${CONDITION} evaluation requires PGC_MANIFEST_PATH to an audited manifest." >&2
    exit 1
  fi
  "${PYTHON_BIN}" scripts/validate_language_intervention_manifest.py "${MANIFEST_PATH}"
fi

PGC_CHECKPOINT_INFO="$("${PYTHON_BIN}" - \
  "${PGC_CHECKPOINT}" \
  "${NUM_INFERENCE_STEPS}" \
  "${V9_ABLATION}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
evaluation_inference_steps = int(sys.argv[2])
v9_ablation = sys.argv[3]
metadata = payload.get("architecture_metadata") or {}
if metadata.get("architecture") != "pgc_fastwam":
    raise SystemExit("Checkpoint is missing PGC architecture metadata")
version = int(metadata.get("policy_guard_version", -1))
if version not in {2, 3, 4, 5, 6, 7, 8, 9}:
    raise SystemExit(
        f"Only PGC versions 2 through 9 are supported, got {version}"
    )
if payload.get("format") != f"fastwam_policy_guard_v{version}":
    raise SystemExit("PGC checkpoint format/version mismatch")
objective = (
    int(metadata.get("eraf_grounding_objective_version", -1))
    if version == 9
    else 0
)
is_v926 = version == 9 and objective >= 26
fresh_eraf_joint = bool(metadata.get("eraf_fresh_joint_training", False))
expected_protection = (
    "fresh_eraf_warmup_then_single_path_no_candidate_gate"
    if fresh_eraf_joint
    else "single_eraf_path_no_candidate_gate"
    if is_v926
    else "single_immutable_base_plus_conservative_hard_gate"
    if version >= 3
    else "immutable_base_plus_conservative_hard_gate"
)
if metadata.get("policy_protection") != expected_protection:
    raise SystemExit("Checkpoint does not declare the protected hard-gate path")
expected_tuning = {
    2: "lora",
    3: "bounded_velocity_residual",
    4: "rollout_aligned_final_action_residual",
    5: "paired_language_prefix_aligned_action_residual",
    6: "visual_target_bottleneck_paired_action_residual",
    7: "object_token_mask_grounded_paired_action_residual",
    8: "closed_loop_replay_verified_target_acquisition_residual",
    9: "entity_relation_affordance_grounded_paired_action_residual",
}[version]
if is_v926:
    expected_tuning = (
        "fresh_eraf_native_counterfactual_bidirectional_world_action_joint_lora"
        if fresh_eraf_joint
        else "native_and_counterfactual_world_action_joint_lora"
    )
elif version == 9 and objective >= 25:
    expected_tuning = (
        "counterfactual_only_internal_action_expert_context_conditioning"
    )
if metadata.get("counterfactual_tuning") != expected_tuning:
    raise SystemExit(f"PGC v{version} tuning metadata is incompatible")
if version >= 3 and any(
    key in payload
    for key in (
        "counterfactual_action_adapter",
        "counterfactual_action_expert",
        "counterfactual_lora_config",
    )
):
    raise SystemExit("PGC v3+ must not contain an Action-Expert copy or LoRA")
if is_v926:
    if (
        not isinstance(payload.get("eraf_shared_expert_lora"), dict)
        or not payload["eraf_shared_expert_lora"]
        or not isinstance(payload.get("eraf_shared_expert_lora_config"), dict)
    ):
        raise SystemExit("PGC V9.26 checkpoint lacks shared Expert LoRA")
elif (
    payload.get("eraf_shared_expert_lora") is not None
    or payload.get("eraf_shared_expert_lora_config") is not None
):
    raise SystemExit("Pre-V9.26 checkpoint unexpectedly contains shared LoRA")
if version >= 4:
    rollout_steps = int(metadata.get("rollout_num_inference_steps", -1))
    if rollout_steps != evaluation_inference_steps:
        raise SystemExit(
            f"PGC v{version} requires rollout/evaluation step alignment: "
            f"checkpoint={rollout_steps}, evaluation={evaluation_inference_steps}"
        )
    if metadata.get("verifier_margin_space") != "raw_fp32_pairwise_advantage":
        raise SystemExit(
            f"PGC v{version} checkpoint lacks its FP32 raw-advantage contract"
        )
if version >= 5 and int(metadata.get("execution_prefix_steps", -1)) <= 0:
    raise SystemExit("PGC v5 checkpoint lacks its executed-prefix contract")
if version == 6 and (
    metadata.get("target_binding_bottleneck")
    != "visual_only_no_direct_language_residual"
):
    raise SystemExit("PGC v6 checkpoint lacks its visual-only target bottleneck")
if version in {6, 7} and (
    metadata.get("target_binding_visual_source")
    != "pre_dit_language_neutral_current_frame"
):
    raise SystemExit("PGC v6 checkpoint uses a language-leaking visual source")
if version == 6 and (
    metadata.get("target_prototype_bank_persisted") is not True
    or not isinstance(payload.get("target_prototype_bank"), dict)
):
    raise SystemExit("PGC v6 checkpoint lacks its persisted target prototypes")
if version == 7:
    if (
        metadata.get("target_binding_bottleneck")
        != "spatial_object_tokens_no_direct_language_residual"
    ):
        raise SystemExit("PGC v7 checkpoint lacks its spatial object-token bottleneck")
    if (
        metadata.get("target_mask_supervision")
        != "robosuite_element_current_frame_training_only"
    ):
        raise SystemExit("PGC v7 checkpoint lacks its explicit mask-supervision contract")
    if payload.get("target_prototype_bank") is not None:
        raise SystemExit("PGC v7 checkpoint unexpectedly contains V6 prototypes")
if version == 8 and (
    metadata.get("closed_loop_corrective_format")
    != "pgc_libero_closed_loop_corrective_v1"
    or metadata.get("acquisition_only") is not True
    or metadata.get("closed_loop_trainable_scope")
    != "action_chunk_proposal_only"
):
    raise SystemExit("PGC v8 checkpoint lacks its audited corrective contract")
if version == 9:
    completion_only_memory = bool(
        metadata.get("eraf_completion_only_memory", False)
    )
    expected_deployment_inputs = (
        "rgb_language_proprio_completed_clause_bitset"
        if completion_only_memory
        else (
            "rgb_language_proprio_previous_policy_state"
            if objective >= 14
            else "rgb_language_proprio"
        )
    )
    expected_ablation = {
        "full": (False, True),
        "entity-only": (True, False),
        "without-anchor": (False, False),
    }
    if v9_ablation not in expected_ablation:
        raise SystemExit(
            "PGC_V9_ABLATION must be full, entity-only, or without-anchor"
        )
    entity_only, use_anchors = expected_ablation[v9_ablation]
    if (
        metadata.get("warm_start_contract")
        not in {"exact_pgc_v5_sidecars", "released_base_fresh_eraf"}
        or metadata.get("grounding")
        != "predicate_entity_relation_affordance_field"
        or metadata.get("privileged_supervision") != "training_only"
        or metadata.get("deployment_inputs") != expected_deployment_inputs
        or bool(metadata.get("eraf_entity_only", False)) != entity_only
        or bool(metadata.get("eraf_use_anchors", True)) != use_anchors
    ):
        raise SystemExit("PGC v9 checkpoint lacks or mismatches its ERAF contract")
    if objective not in set(range(1, 27)):
        raise SystemExit(
            f"PGC v9 checkpoint has invalid grounding objective {objective}"
        )
    training_stage = str(metadata.get("eraf_training_stage", ""))
    if training_stage not in {"grounding", "action", "verifier"}:
        raise SystemExit(
            f"PGC v9 checkpoint has invalid ERAF stage {training_stage!r}"
        )
    if objective >= 9 and (
        metadata.get("eraf_clause_activation_contract")
        != "zero_init_cross_clause_active_logit_residual"
        or metadata.get("eraf_clause_gate")
        != "multi_clause_exact_at_least_80pct"
    ):
        raise SystemExit("PGC v9.8 checkpoint lacks clause calibration contract")
    if objective >= 10 and (
        metadata.get("eraf_view_fusion_contract")
        != "per_view_local_attention_visibility_gated_zero_init_residual"
        or metadata.get("eraf_clause_scheduler_contract")
        != "first_active_unfinished_predicate_zero_init_residual_route"
    ):
        raise SystemExit("PGC v9.9 checkpoint lacks view/scheduler contract")
    if objective >= 11 and (
        metadata.get("eraf_all_entity_role_contract")
        != "exclusive_evidence_same_state_all_entity_bipartite_assignment"
        or metadata.get("eraf_multi_clause_gate_contract")
        != "semantic_exact_with_exclusive_role_evidence"
    ):
        raise SystemExit("PGC v9.10 checkpoint lacks all-entity role contract")
    if objective >= 12 and (
        metadata.get("eraf_clause_tuple_contract")
        != "exclusive_same_state_subject_predicate_reference_assignment"
        or metadata.get("eraf_clause_tuple_curriculum_contract")
        != "v9_10_audit_native_hard_easy_plus_historical_strict_1_1_1_1"
    ):
        raise SystemExit("PGC v9.11 checkpoint lacks clause-tuple contract")
    if objective == 13 and (
        metadata.get("eraf_closed_loop_rebinding_contract")
        != "zero_init_second_pass_role_truth_phase_and_clause_route"
        or metadata.get("eraf_closed_loop_state_contract")
        != "immutable_base_correct_replan_exact_simulator_state"
        or metadata.get("eraf_closed_loop_curriculum_contract")
        != "offline_native_closed_loop_native_historical_strict_1_1_1_1"
    ):
        raise SystemExit("PGC v9.12 checkpoint lacks closed-loop rebinding contract")
    if objective >= 14 and (
        metadata.get("eraf_closed_loop_state_contract")
        != "immutable_base_correct_replan_exact_simulator_state"
        or metadata.get("eraf_closed_loop_curriculum_contract")
        != "offline_native_closed_loop_native_historical_strict_1_1_1_1"
        or metadata.get("eraf_phase_safe_memory_contract")
        != "explicit_cross_replan_pending_holding_retry_completed"
        or metadata.get("eraf_geometry_protection_contract")
        != (
            "jointly_trained_fresh_eraf_no_post_action_residual"
            if fresh_eraf_joint
            else "frozen_v9_11_no_query_token_anchor_or_heatmap_residual"
        )
        or metadata.get("eraf_release_transition_contract")
        != "release_true_advance_release_false_retry"
        or metadata.get("eraf_policy_state_contract")
        != (
            "monotonic_completed_bitset_no_pending_holding_retry_recurrence"
            if completion_only_memory
            else "explicit_caller_owned_reset_per_episode"
        )
        or metadata.get("eraf_phase_safe_memory_warm_start")
        != (
            "none_random_seeded_from_released_base"
            if fresh_eraf_joint
            else "exact_v9_11_geometry"
        )
    ):
        raise SystemExit("PGC v9.13 checkpoint lacks phase-safe memory contract")
    expected_action_joint_contract = (
        (
            "fresh_eraf_plus_shared_video_action_expert_lora_bidirectional_"
            "joint_training_with_delayed_internal_context_injection_single_path"
            if fresh_eraf_joint
            else "frozen_eraf_completion_memory_plus_shared_video_action_"
            "expert_lora_and_internal_context_injector_single_path"
            if objective >= 26
            else "frozen_eraf_and_shared_action_expert_plus_internal_context_"
            "injector_no_post_action_residual"
            if objective >= 25
            else "frozen_v921_expert_adapter_plus_isolated_clause_semantic_"
            "retention_residual"
            if objective >= 24
            else "frozen_v921_teacher_plus_alignment_preserving_negative_"
            "focused_final_action_clause_ranking"
            if objective >= 23
            else "frozen_v920_stack_plus_phase_specific_expert_adapter_with_"
            "balanced_final_action_clause_ranking"
            if objective >= 22
            else "frozen_v920_stack_plus_phase_specific_privileged_expert_"
            "prefix_residual_alignment"
            if objective >= 21
            else "frozen_v919_stack_plus_phase_compatible_local_waypoint_vector_field"
            if objective >= 20
            else "frozen_v918_stack_plus_hard_clause_phase_direction_preserving_servo"
            if objective >= 19
            else "frozen_eraf_v917_stack_plus_phase_balanced_direct_geometry_"
            "residual_imitation"
            if objective >= 18
            else "frozen_eraf_v916_bridge_and_proposal_plus_direct_eef_"
            "relative_geometry_action_adapter"
        )
        if objective >= 17
        else (
            "frozen_eraf_perception_proposal_and_legacy_bridge_plus_"
            "semantic_causal_action_grounding_bridge"
            if objective >= 16
            else (
                "frozen_eraf_perception_plus_phase_conditioned_geometry_bridge_"
                "legacy_bridge_and_proposal"
                if objective >= 15
                else "frozen_eraf_perception_plus_action_bridge_and_proposal"
            )
        )
    )
    expected_action_trainable_scope = (
        (
            "fresh_eraf_plus_shared_video_action_lora_plus_eraf_action_"
            "context_injector"
            if fresh_eraf_joint
            else "shared_video_action_lora_plus_eraf_action_context_injector"
            if objective >= 26
            else "eraf_action_context_injector_only"
            if objective >= 25
            else "clause_semantic_retention_residual_only"
            if objective >= 24
            else "phase_specific_privileged_expert_residual_adapter_only"
            if objective >= 21
            else "phase_compatible_local_waypoint_adapter_only"
            if objective >= 20
            else "hard_routed_phase_servo_only"
            if objective >= 19
            else "phase_conditioned_geometry_adapter_only_with_phase_balanced_"
            "residual_imitation"
            if objective >= 18
            else "phase_conditioned_relative_geometry_action_adapter_only"
        )
        if objective >= 17
        else (
            "semantic_causal_action_grounding_bridge_only"
            if objective >= 16
            else (
                "phase_conditioned_subject_reference_anchor_action_bridge_plus_"
                "legacy_bridge_and_action_chunk_proposal"
                if objective >= 15
                else "base_query_projection_relation_attention_query_embedding_"
                "delta_plus_action_chunk_proposal"
            )
        )
    )
    expected_role_trainable_scope = (
        (
            "fresh_eraf_plus_shared_video_action_lora_plus_eraf_action_"
            "context_injector"
            if fresh_eraf_joint
            else "shared_video_action_lora_plus_eraf_action_context_injector"
            if objective >= 26
            else "eraf_action_context_injector_only"
            if objective >= 25
            else "clause_semantic_retention_residual_only"
            if objective >= 24
            else "phase_specific_privileged_expert_residual_adapter_only"
            if objective >= 21
            else "phase_compatible_local_waypoint_adapter_only"
            if objective >= 20
            else "hard_routed_phase_servo_only"
            if objective >= 19
            else "phase_conditioned_geometry_adapter_only_with_phase_balanced_"
            "residual_imitation"
            if objective >= 18
            else "phase_conditioned_relative_geometry_action_adapter_only"
        )
        if objective >= 17
        else (
            "semantic_causal_action_grounding_bridge_only"
            if objective >= 16
            else "frozen_eraf_perception_action_bridge_plus_proposal"
        )
    )
    if objective >= 15 and (
        metadata.get("eraf_action_grounding_contract")
        != "separate_subject_reference_relation_grasp_goal_interaction_"
        "displacement_tokens_zero_init_v9_14_exact"
    ):
        raise SystemExit(
            "PGC v9.15 checkpoint lacks its explicit action-grounding contract"
        )
    if objective >= 18 and (
        metadata.get("eraf_action_phase_residual_contract")
        != "phase_balanced_bounded_expert_minus_frozen_v9_17_candidate_"
        "prefix_residual_imitation"
    ):
        raise SystemExit(
            "PGC v9.18 checkpoint lacks its phase-residual imitation contract"
        )
    if objective >= 19 and (
        metadata.get("eraf_action_phase_servo_contract")
        != "hard_single_clause_explicit_affine_eef_phase_specific_positive_"
        "cartesian_gain_with_legacy_suppression"
    ):
        raise SystemExit(
            "PGC v9.19 checkpoint lacks its hard-routed phase-servo contract"
        )
    if objective >= 20 and (
        metadata.get("eraf_action_waypoint_contract")
        != "hard_clause_phase_compatible_positive_progress_local_tangent_"
        "waypoint_with_privileged_training_only_compatibility_labels"
    ):
        raise SystemExit(
            "PGC v9.20 checkpoint lacks its phase-compatible waypoint contract"
        )
    if objective >= 21 and (
        metadata.get("eraf_action_expert_alignment_contract")
        != "training_only_privileged_phase_anchor_teacher_plus_deployed_"
        "full_action_prefix_residual_and_semantic_causal_ranking"
    ):
        raise SystemExit(
            "PGC expert-alignment checkpoint lacks its privileged prefix contract"
        )
    if objective >= 22 and (
        metadata.get("eraf_action_clause_ranking_contract")
        != (
            "frozen_v921_correct_route_identity_plus_isolated_wrong_clause_"
            "base_fallback_ranking"
            if objective >= 24
            else "frozen_v921_teacher_correct_output_preservation_plus_expert_"
            "nonregression_and_detached_correct_wrong_clause_ranking_"
            "balanced_over_approach_transport_release"
            if objective >= 23
            else "coherent_same_state_clause_swap_final_expert_prefix_mse_"
            "ranking_balanced_over_approach_transport_release"
        )
    ):
        raise SystemExit(
            "PGC clause-ranking checkpoint lacks its balanced final-action contract"
        )
    if objective == 23 and (
        metadata.get("eraf_action_clause_teacher_contract")
        != "training_only_frozen_exact_v921_expert_residual_adapter_excluded_"
        "from_rollout_and_optimizer"
    ):
        raise SystemExit(
            "PGC alignment-preserving clause checkpoint lacks its frozen teacher contract"
        )
    if objective >= 24 and (
        metadata.get("eraf_action_clause_residual_contract")
        != "frozen_v921_positive_action_plus_identity_initialized_clause_"
        "conditioned_base_fallback"
    ):
        raise SystemExit(
            "PGC isolated-clause checkpoint lacks its semantic residual contract"
        )
    if objective >= 25 and (
        metadata.get("eraf_action_context_injection_contract")
        != (
            "exact_no_injection_warmup_then_append_bounded_eraf_tokens_to_"
            "shared_action_expert_context_no_post_action_residual"
            if fresh_eraf_joint
            else "append_bounded_eraf_tokens_to_shared_action_expert_context_at_"
            "every_denoising_step_no_post_action_residual"
        )
        or metadata.get("eraf_post_action_residual_active") is not False
    ):
        raise SystemExit(
            "PGC internal Action-Expert checkpoint lacks its no-residual "
            "ERAF context-injection contract"
        )
    if objective >= 26 and (
        metadata.get("eraf_single_path") is not True
        or metadata.get("gate_mode") != "eraf_only"
        or metadata.get("verifier_deployment_role")
        != "diagnostic_only_no_action_selection"
        or metadata.get("world_model_supervision")
        != "future_video_flow_plus_paired_wrong_language_ranking"
        or set(
            (metadata.get("eraf_shared_expert_lora_config") or {}).get(
                "experts", []
            )
        )
        != {"video", "action"}
    ):
        raise SystemExit(
            "PGC V9.26 checkpoint lacks its single-path shared Expert-LoRA "
            "contract"
        )
    if completion_only_memory and (
        training_stage != "action"
        or metadata.get("eraf_action_joint_training") is not True
        or metadata.get("eraf_action_joint_contract")
        != expected_action_joint_contract
        or metadata.get("eraf_action_trainable_scope")
        != expected_action_trainable_scope
        or metadata.get("eraf_role_adapter_trainable_scope")
        != expected_role_trainable_scope
    ):
        raise SystemExit("PGC v9.14+ checkpoint lacks its joint-action contract")
else:
    objective = 0
    training_stage = "grounding"
print(f"{version}:{objective}:{training_stage}")
PY
)"
PGC_CHECKPOINT_VERSION="${PGC_CHECKPOINT_INFO%%:*}"
PGC_CHECKPOINT_REST="${PGC_CHECKPOINT_INFO#*:}"
PGC_V9_GROUNDING_OBJECTIVE_VERSION="${PGC_CHECKPOINT_REST%%:*}"
PGC_V9_TRAINING_STAGE="${PGC_CHECKPOINT_REST#*:}"
echo "Validated PGC v${PGC_CHECKPOINT_VERSION} checkpoint: ${PGC_CHECKPOINT}"
if [[ "${ERAF_SHADOW_AUDIT}" == "true" && "${PGC_CHECKPOINT_VERSION}" != "9" ]]; then
  echo "PGC ERAF shadow audit requires a PGC v9 checkpoint." >&2
  exit 1
fi
if [[ "${ERAF_ORACLE}" == "true" ]]; then
  if [[ "${PGC_CHECKPOINT_VERSION}" != "9" ]]; then
    echo "Oracle ERAF requires a PGC v9 checkpoint." >&2
    exit 1
  fi
  if [[ "${GATE_MODE}" != "counterfactual" ]]; then
    echo "Oracle ERAF requires PGC_GATE_MODE=counterfactual." >&2
    exit 1
  fi
  if [[ "${CONDITION}" != "correct" && "${CONDITION}" != "counterfactual" ]]; then
    echo "Oracle ERAF supports only correct or counterfactual." >&2
    exit 1
  fi
  if [[ -z "${ERAF_ORACLE_SIDECAR_DIR}" ]]; then
    echo "Set PGC_ERAF_ORACLE_SIDECAR_DIR for oracle workspace/mask metadata." >&2
    exit 1
  fi
  if [[ ! -f "${ERAF_ORACLE_SIDECAR_DIR%/}/index.json" && ! -f "${ERAF_ORACLE_SIDECAR_DIR}" ]]; then
    echo "Oracle ERAF sidecar index not found: ${ERAF_ORACLE_SIDECAR_DIR}" >&2
    exit 1
  fi
fi
if [[ "${ERAF_STATELESS_REPLAN_ABLATION}" == "true" || "${ERAF_COMPLETION_ONLY_MEMORY_ABLATION}" == "true" ]]; then
  if [[ "${PGC_CHECKPOINT_VERSION}" != "9" || "${PGC_V9_GROUNDING_OBJECTIVE_VERSION}" -lt 14 ]]; then
    echo "Policy-state ablations require a PGC V9.13+ phase-memory checkpoint." >&2
    exit 1
  fi
  if [[ "${ERAF_SHADOW_AUDIT}" != "true" || "${GATE_MODE}" != "base" ]]; then
    echo "Policy-state ablations require passive ERAF shadow audit with PGC_GATE_MODE=base." >&2
    exit 1
  fi
fi
PGC_CLOSED_LOOP_ENABLED=false
if [[ "${PGC_CHECKPOINT_VERSION}" == "8" ]]; then
  PGC_CLOSED_LOOP_ENABLED=true
fi
V9_ENTITY_ONLY=false
V9_USE_ANCHORS=true
case "${V9_ABLATION}" in
  full) ;;
  entity-only)
    V9_ENTITY_ONLY=true
    V9_USE_ANCHORS=false
    ;;
  without-anchor)
    V9_USE_ANCHORS=false
    ;;
  *)
    echo "PGC_V9_ABLATION must be full, entity-only, or without-anchor." >&2
    exit 1
    ;;
esac

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_LIST=""
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    [[ -z "${GPU_LIST}" ]] || GPU_LIST+=","
    GPU_LIST+="${gpu}"
  done
  export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
fi

EXTRA_OVERRIDES=(
  "task=libero_pgc_2cam224"
  "ckpt=${PGC_CHECKPOINT}"
  "seed=${EVAL_SEED}"
  "EVALUATION.dataset_stats_path=${STATS_PATH}"
  "EVALUATION.num_trials=${NUM_TRIALS}"
  "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}"
  "EVALUATION.instruction_condition=${CONDITION}"
  "EVALUATION.output_dir=${OUTPUT_ROOT}"
  "MULTIRUN.task_suite_names=${SUITES}"
  "MULTIRUN.num_gpus=${NUM_GPUS}"
  "MULTIRUN.max_tasks_per_gpu=1"
  "model.action_dit_config.use_latent_action_queries=false"
  "model.langforce_mvp.enabled=false"
  "model.langforce_mvp.enable_prior=false"
  "model.langforce_mvp.enable_posterior_advantage=false"
  "model.transition_contract.enabled=false"
  "model.policy_guard.enabled=true"
  "model.policy_guard.version=${PGC_CHECKPOINT_VERSION}"
  "model.policy_guard.closed_loop_corrective_enabled=${PGC_CLOSED_LOOP_ENABLED}"
  "model.policy_guard.gate_mode=${GATE_MODE}"
  "model.policy_guard.gate_threshold=${GATE_THRESHOLD}"
  "model.policy_guard.min_counterfactual_score=${MIN_COUNTERFACTUAL_SCORE}"
  # Keep construction adapter-free. The loader injects saved v2 independent
  # Action LoRA or V9.26 shared Video/Action LoRA before restoring tensors.
  "model.lora.enabled=false"
)
if [[ "${PGC_CHECKPOINT_VERSION}" == "9" ]]; then
  EXTRA_OVERRIDES+=(
    "model.policy_guard.entity_relation_grounding.training_stage=${PGC_V9_TRAINING_STAGE}"
    "model.policy_guard.entity_relation_grounding.grounding_objective_version=${PGC_V9_GROUNDING_OBJECTIVE_VERSION}"
    "model.policy_guard.entity_relation_grounding.entity_only=${V9_ENTITY_ONLY}"
    "model.policy_guard.entity_relation_grounding.use_anchors=${V9_USE_ANCHORS}"
    "EVALUATION.entity_relation_diagnostics=${ERAF_DIAGNOSTICS}"
    "EVALUATION.entity_relation_shadow_audit=${ERAF_SHADOW_AUDIT}"
    "EVALUATION.entity_relation_oracle=${ERAF_ORACLE}"
    "EVALUATION.entity_relation_stateless_replan_ablation=${ERAF_STATELESS_REPLAN_ABLATION}"
    "EVALUATION.entity_relation_completion_only_memory_ablation=${ERAF_COMPLETION_ONLY_MEMORY_ABLATION}"
  )
  # Reconstruct ERAF with the exact checkpoint architecture/loss contract.
  # These values are not merely training diagnostics: the strict V9 loader
  # validates them before restoring sidecar modules. Hard-coding defaults here
  # made a V9.12 checkpoint inherit V9.11's non-zero frozen-path weights and
  # fail identically in every evaluation worker during model construction.
  PGC_V9_METADATA_OVERRIDES="$("${PYTHON_BIN}" - "${PGC_CHECKPOINT}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
metadata = payload.get("architecture_metadata") or {}
mapping = {
    "warm_start_contract": "initialization_contract",
    "eraf_hidden_dim": "hidden_dim",
    "eraf_num_heads": "num_heads",
    "eraf_max_clauses": "max_clauses",
    "eraf_camera_count": "camera_count",
    "eraf_visual_aspect_ratio": "visual_aspect_ratio",
    "eraf_temperature": "temperature",
    "eraf_grounding_aux_weight": "grounding_aux_weight",
    "eraf_completion_only_memory": "completion_only_memory",
    "eraf_action_joint_training": "action_joint_training",
    "eraf_fresh_joint_training": "fresh_joint_training",
    "eraf_bidirectional_supervision": "bidirectional_supervision",
    "eraf_context_injection_warmup_steps": (
        "context_injection_warmup_steps"
    ),
    "eraf_context_injection_ramp_steps": "context_injection_ramp_steps",
    "eraf_action_grounding_hidden_dim": "action_grounding_hidden_dim",
    "eraf_action_grounding_num_heads": "action_grounding_num_heads",
    "eraf_action_grounding_learning_rate": "action_grounding_learning_rate",
    "eraf_action_geometry_hidden_dim": "action_geometry_hidden_dim",
    "eraf_action_geometry_learning_rate": "action_geometry_learning_rate",
    "eraf_action_geometry_residual_max_abs": "action_geometry_residual_max_abs",
    "eraf_action_phase_residual_imitation_weight": (
        "action_phase_residual_imitation_weight"
    ),
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
    "eraf_action_expert_distillation_weight": (
        "action_expert_distillation_weight"
    ),
    "eraf_action_expert_native_zero_weight": "action_expert_native_zero_weight",
    "eraf_action_clause_ranking_weight": "action_clause_ranking_weight",
    "eraf_action_clause_ranking_margin": "action_clause_ranking_margin",
    "eraf_action_clause_teacher_weight": "action_clause_teacher_weight",
    "eraf_action_clause_alignment_guard_weight": (
        "action_clause_alignment_guard_weight"
    ),
    "eraf_action_eef_initial_scale": "action_eef_scale",
    "eraf_action_eef_initial_bias": "action_eef_bias",
    "eraf_action_causal_ranking_weight": "action_causal_ranking_weight",
    "eraf_action_causal_margin": "action_causal_margin",
    "eraf_expert_lora_world_language_weight": (
        "expert_lora_world_language_weight"
    ),
    "eraf_expert_lora_world_language_margin": (
        "expert_lora_world_language_margin"
    ),
    "eraf_expert_lora_native_action_weight": (
        "expert_lora_native_action_weight"
    ),
    "eraf_expert_lora_counterfactual_action_weight": (
        "expert_lora_counterfactual_action_weight"
    ),
    "eraf_expert_lora_regularization_weight": (
        "expert_lora_regularization_weight"
    ),
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
    "eraf_clause_tuple_multi_consistency_weight": (
        "clause_tuple_multi_consistency_weight"
    ),
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
    "eraf_phase_safe_memory_scheduler_weight": (
        "phase_safe_memory_scheduler_weight"
    ),
    "eraf_phase_safe_memory_energy_weight": "phase_safe_memory_energy_weight",
    "eraf_role_adapter_hidden_dim": "role_adapter_hidden_dim",
    "eraf_structured_role_adapter_hidden_dim": (
        "structured_role_adapter_hidden_dim"
    ),
    "eraf_balanced_role_adapter_hidden_dim": "balanced_role_adapter_hidden_dim",
    "eraf_clause_activation_adapter_hidden_dim": (
        "clause_activation_adapter_hidden_dim"
    ),
    "eraf_clause_activation_residual_max_abs": (
        "clause_activation_residual_max_abs"
    ),
    "eraf_view_fusion_adapter_hidden_dim": "view_fusion_adapter_hidden_dim",
    "eraf_view_fusion_residual_max_abs": "view_fusion_residual_max_abs",
    "eraf_clause_scheduler_hidden_dim": "clause_scheduler_hidden_dim",
    "eraf_clause_scheduler_residual_max_abs": (
        "clause_scheduler_residual_max_abs"
    ),
    "eraf_closed_loop_rebinding_hidden_dim": "closed_loop_rebinding_hidden_dim",
    "eraf_closed_loop_query_residual_max_abs": (
        "closed_loop_query_residual_max_abs"
    ),
    "eraf_closed_loop_state_residual_max_abs": (
        "closed_loop_state_residual_max_abs"
    ),
    "eraf_phase_safe_memory_hidden_dim": "phase_safe_memory_hidden_dim",
    "eraf_phase_safe_memory_state_count": "phase_safe_memory_state_count",
    "eraf_phase_safe_memory_routing_residual_max_abs": (
        "phase_safe_memory_routing_residual_max_abs"
    ),
    "eraf_role_attention_preservation_weight": (
        "role_attention_preservation_weight"
    ),
    "eraf_role_position_preservation_weight": (
        "role_position_preservation_weight"
    ),
    "eraf_role_anchor_preservation_weight": "role_anchor_preservation_weight",
    "eraf_role_relation_preservation_weight": (
        "role_relation_preservation_weight"
    ),
    "eraf_role_adapter_energy_weight": "role_adapter_energy_weight",
}
prefix = "model.policy_guard.entity_relation_grounding."
for metadata_key, config_key in mapping.items():
    value = metadata.get(metadata_key)
    if value is None:
        continue
    if isinstance(value, bool):
        value = str(value).lower()
    print(f"{prefix}{config_key}={value}")
PY
)"
  while IFS= read -r override; do
    [[ -z "${override}" ]] || EXTRA_OVERRIDES+=("${override}")
  done <<< "${PGC_V9_METADATA_OVERRIDES}"
  if [[ "${ERAF_DIAGNOSTICS}" == "true" ]]; then
    EXTRA_OVERRIDES+=("EVALUATION.entity_relation_overlay_dir=${ERAF_OVERLAY_DIR}")
  fi
  if [[ "${ERAF_SHADOW_AUDIT}" == "true" ]]; then
    if [[ "${GATE_MODE}" != "base" ]]; then
      echo "PGC ERAF shadow audit requires PGC_GATE_MODE=base." >&2
      exit 1
    fi
    if [[ "${CONDITION}" != "correct" && "${CONDITION}" != "counterfactual" ]]; then
      echo "PGC ERAF shadow audit supports only correct or counterfactual." >&2
      exit 1
    fi
    if [[ -z "${ERAF_SHADOW_SIDECAR_DIR}" ]]; then
      echo "Set PGC_ERAF_SHADOW_SIDECAR_DIR for shadow workspace/mask metadata." >&2
      exit 1
    fi
    if [[ ! -f "${ERAF_SHADOW_SIDECAR_DIR%/}/index.json" && ! -f "${ERAF_SHADOW_SIDECAR_DIR}" ]]; then
      echo "ERAF shadow sidecar index not found: ${ERAF_SHADOW_SIDECAR_DIR}" >&2
      exit 1
    fi
    EXTRA_OVERRIDES+=(
      "EVALUATION.entity_relation_shadow_sidecar_dir=${ERAF_SHADOW_SIDECAR_DIR}"
    )
  fi
  if [[ "${ERAF_ORACLE}" == "true" ]]; then
    EXTRA_OVERRIDES+=(
      "EVALUATION.entity_relation_oracle_sidecar_dir=${ERAF_ORACLE_SIDECAR_DIR}"
    )
  fi
  if [[ "${ERAF_ORACLE_PHASE_SERVO}" == "true" ]]; then
    EXTRA_OVERRIDES+=(
      "+EVALUATION.entity_relation_oracle_phase_servo=true"
      "+EVALUATION.entity_relation_oracle_servo_scope=${ERAF_ORACLE_SERVO_SCOPE}"
      "+EVALUATION.entity_relation_oracle_servo_approach_gain=${ERAF_ORACLE_SERVO_APPROACH_GAIN}"
      "+EVALUATION.entity_relation_oracle_servo_transport_gain=${ERAF_ORACLE_SERVO_TRANSPORT_GAIN}"
      "+EVALUATION.entity_relation_oracle_servo_max_translation_action=${ERAF_ORACLE_SERVO_MAX_TRANSLATION}"
      "+EVALUATION.entity_relation_oracle_servo_approach_height_m=${ERAF_ORACLE_SERVO_APPROACH_HEIGHT}"
      "+EVALUATION.entity_relation_oracle_servo_transport_height_m=${ERAF_ORACLE_SERVO_TRANSPORT_HEIGHT}"
      "+EVALUATION.entity_relation_oracle_servo_grasp_offset_m=${ERAF_ORACLE_SERVO_GRASP_OFFSET}"
      "+EVALUATION.entity_relation_oracle_servo_release_height_m=${ERAF_ORACLE_SERVO_RELEASE_HEIGHT}"
      "+EVALUATION.entity_relation_oracle_servo_horizontal_tolerance_m=${ERAF_ORACLE_SERVO_HORIZONTAL_TOLERANCE}"
      "+EVALUATION.entity_relation_oracle_servo_grasp_distance_m=${ERAF_ORACLE_SERVO_GRASP_DISTANCE}"
      "+EVALUATION.entity_relation_oracle_servo_release_distance_m=${ERAF_ORACLE_SERVO_RELEASE_DISTANCE}"
      "+EVALUATION.entity_relation_oracle_servo_interaction_distance_m=${ERAF_ORACLE_SERVO_INTERACTION_DISTANCE}"
      "EVALUATION.replan_steps=${ERAF_ORACLE_SERVO_REPLAN_STEPS}"
    )
  fi
fi
if [[ -n "${MANIFEST_PATH}" ]]; then
  EXTRA_OVERRIDES+=("EVALUATION.language_intervention_manifest=${MANIFEST_PATH}")
fi
if [[ -n "${MAX_POLICY_STEPS}" ]]; then
  EXTRA_OVERRIDES+=("EVALUATION.max_steps=${MAX_POLICY_STEPS}")
fi
if [[ "${CONDITION}" == "counterfactual" ]]; then
  EXTRA_OVERRIDES+=("EVALUATION.counterfactual_diagnostics=true")
fi
if [[ -n "${CLOSED_LOOP_CAPTURE_DIR}" ]]; then
  if [[ "${CONDITION}" != "counterfactual" ]]; then
    echo "PGC closed-loop state capture requires condition=counterfactual." >&2
    exit 1
  fi
  if ! [[ "${CLOSED_LOOP_CAPTURE_STRIDE}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${CLOSED_LOOP_CAPTURE_MAX_STATES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PGC capture stride/max states must be positive integers." >&2
    exit 1
  fi
  EXTRA_OVERRIDES+=(
    "+EVALUATION.closed_loop_capture_dir=${CLOSED_LOOP_CAPTURE_DIR}"
    "+EVALUATION.closed_loop_capture_stride_replans=${CLOSED_LOOP_CAPTURE_STRIDE}"
    "+EVALUATION.closed_loop_capture_max_states_per_episode=${CLOSED_LOOP_CAPTURE_MAX_STATES}"
  )
fi
if [[ -n "${ERAF_CLOSED_LOOP_CAPTURE_DIR}" ]]; then
  if [[ "${PGC_CHECKPOINT_VERSION}" != "9" || "${ERAF_SHADOW_AUDIT}" != "true" ]]; then
    echo "ERAF phase capture requires a PGC v9 passive shadow audit." >&2
    exit 1
  fi
  if [[ "${CONDITION}" != "correct" || "${GATE_MODE}" != "base" ]]; then
    echo "ERAF phase capture requires condition=correct and PGC_GATE_MODE=base." >&2
    exit 1
  fi
  if ! [[ "${ERAF_CLOSED_LOOP_CAPTURE_STRIDE}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${ERAF_CLOSED_LOOP_CAPTURE_MAX_STATES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERAF phase capture stride/max states must be positive integers." >&2
    exit 1
  fi
  EXTRA_OVERRIDES+=(
    "+EVALUATION.entity_relation_closed_loop_capture_dir=${ERAF_CLOSED_LOOP_CAPTURE_DIR}"
    "+EVALUATION.entity_relation_closed_loop_capture_stride_replans=${ERAF_CLOSED_LOOP_CAPTURE_STRIDE}"
    "+EVALUATION.entity_relation_closed_loop_capture_max_states_per_episode=${ERAF_CLOSED_LOOP_CAPTURE_MAX_STATES}"
    "+EVALUATION.entity_relation_closed_loop_capture_stages='${ERAF_CLOSED_LOOP_CAPTURE_STAGES}'"
  )
fi

echo "[PGC-FastWAM] LIBERO ${CONDITION} evaluation"
echo "  checkpoint=${PGC_CHECKPOINT}"
echo "  suites=${SUITES} trials=${NUM_TRIALS}"
echo "  gate=${GATE_MODE} margin=${GATE_THRESHOLD} min_cf=${MIN_COUNTERFACTUAL_SCORE}"
echo "  max_policy_steps=${MAX_POLICY_STEPS:-suite_default}"
echo "  output=${OUTPUT_ROOT}"
echo "  closed_loop_capture=${CLOSED_LOOP_CAPTURE_DIR:-disabled}"
echo "  eraf_phase_capture=${ERAF_CLOSED_LOOP_CAPTURE_DIR:-disabled}"
if [[ "${PGC_CHECKPOINT_VERSION}" == "9" ]]; then
  echo "  v9_ablation=${V9_ABLATION} eraf_diagnostics=${ERAF_DIAGNOSTICS}"
  echo "  eraf_overlay_dir=${ERAF_OVERLAY_DIR}"
  echo "  eraf_shadow_audit=${ERAF_SHADOW_AUDIT}"
  echo "  eraf_shadow_sidecar=${ERAF_SHADOW_SIDECAR_DIR:-disabled}"
  echo "  eraf_oracle=${ERAF_ORACLE}"
  echo "  eraf_oracle_sidecar=${ERAF_ORACLE_SIDECAR_DIR:-disabled}"
  echo "  eraf_oracle_phase_servo=${ERAF_ORACLE_PHASE_SERVO}"
  echo "  eraf_oracle_servo_scope=${ERAF_ORACLE_SERVO_SCOPE}"
  if [[ "${ERAF_STATELESS_REPLAN_ABLATION}" == "true" ]]; then
    ERAF_POLICY_STATE_MODE=reset_each_replan
  elif [[ "${ERAF_COMPLETION_ONLY_MEMORY_ABLATION}" == "true" ]]; then
    ERAF_POLICY_STATE_MODE=completion_only
  else
    ERAF_POLICY_STATE_MODE=recurrent
  fi
  echo "  eraf_policy_state_mode=${ERAF_POLICY_STATE_MODE}"
fi

EXP_NAME="pgc-${CONDITION}" "${PYTHON_BIN}" \
  experiments/libero/run_libero_manager.py "${EXTRA_OVERRIDES[@]}"

echo "[PGC-FastWAM] evaluation complete: ${OUTPUT_ROOT}"
