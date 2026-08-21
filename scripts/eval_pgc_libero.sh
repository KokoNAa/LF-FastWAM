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
V9_ABLATION="${PGC_V9_ABLATION:-full}"
ERAF_SHADOW_AUDIT="${PGC_ERAF_SHADOW_AUDIT:-false}"
ERAF_SHADOW_SIDECAR_DIR="${PGC_ERAF_SHADOW_SIDECAR_DIR:-}"
ERAF_DIAGNOSTICS_DEFAULT=true
if [[ "${ERAF_SHADOW_AUDIT}" == "true" ]]; then
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
expected_protection = (
    "single_immutable_base_plus_conservative_hard_gate"
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
        metadata.get("warm_start_contract") != "exact_pgc_v5_sidecars"
        or metadata.get("grounding")
        != "predicate_entity_relation_affordance_field"
        or metadata.get("privileged_supervision") != "training_only"
        or metadata.get("deployment_inputs") != "rgb_language_proprio"
        or bool(metadata.get("eraf_entity_only", False)) != entity_only
        or bool(metadata.get("eraf_use_anchors", True)) != use_anchors
    ):
        raise SystemExit("PGC v9 checkpoint lacks or mismatches its ERAF contract")
    objective = int(metadata.get("eraf_grounding_objective_version", -1))
    if objective not in set(range(1, 13)):
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
  # Keep construction adapter-free. v2 loading injects its saved LoRA; v3+
  # strictly restore only their policy-guard sidecar tensors.
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
  )
  if [[ "${ERAF_DIAGNOSTICS}" == "true" ]]; then
    EXTRA_OVERRIDES+=("EVALUATION.entity_relation_overlay_dir=${ERAF_OVERLAY_DIR}")
  fi
  if (( PGC_V9_GROUNDING_OBJECTIVE_VERSION >= 9 )); then
    EXTRA_OVERRIDES+=(
      "model.policy_guard.entity_relation_grounding.role_assignment_weight=1.0"
      "model.policy_guard.entity_relation_grounding.role_assignment_hard_weight=0.5"
      "model.policy_guard.entity_relation_grounding.structured_assignment_weight=2.0"
      "model.policy_guard.entity_relation_grounding.structured_assignment_hard_weight=1.0"
      "model.policy_guard.entity_relation_grounding.multi_clause_consistency_weight=2.0"
      "model.policy_guard.entity_relation_grounding.role_attention_preservation_weight=5.0"
      "model.policy_guard.entity_relation_grounding.role_position_preservation_weight=2.0"
      "model.policy_guard.entity_relation_grounding.role_anchor_preservation_weight=10.0"
      "model.policy_guard.entity_relation_grounding.role_relation_preservation_weight=2.0"
      "model.policy_guard.entity_relation_grounding.role_adapter_energy_weight=0.01"
      "model.policy_guard.entity_relation_grounding.clause_activation_adapter_hidden_dim=256"
      "model.policy_guard.entity_relation_grounding.clause_activation_residual_max_abs=4.0"
      "model.policy_guard.entity_relation_grounding.clause_activation_balance_weight=1.0"
      "model.policy_guard.entity_relation_grounding.clause_cardinality_weight=1.0"
      "model.policy_guard.entity_relation_grounding.clause_worst_slot_weight=2.0"
      "model.policy_guard.entity_relation_grounding.clause_multi_group_weight=1.0"
      "model.policy_guard.entity_relation_grounding.clause_adapter_energy_weight=0.01"
    )
  fi
  if (( PGC_V9_GROUNDING_OBJECTIVE_VERSION >= 10 )); then
    EXTRA_OVERRIDES+=(
      "model.policy_guard.entity_relation_grounding.view_fusion_adapter_hidden_dim=256"
      "model.policy_guard.entity_relation_grounding.view_fusion_residual_max_abs=4.0"
      "model.policy_guard.entity_relation_grounding.view_fusion_weight=2.0"
      "model.policy_guard.entity_relation_grounding.view_fusion_energy_weight=0.01"
      "model.policy_guard.entity_relation_grounding.clause_scheduler_hidden_dim=256"
      "model.policy_guard.entity_relation_grounding.clause_scheduler_residual_max_abs=1.0"
      "model.policy_guard.entity_relation_grounding.clause_scheduler_weight=1.0"
      "model.policy_guard.entity_relation_grounding.clause_scheduler_energy_weight=0.01"
    )
  fi
  if (( PGC_V9_GROUNDING_OBJECTIVE_VERSION >= 12 )); then
    EXTRA_OVERRIDES+=(
      "model.policy_guard.entity_relation_grounding.clause_tuple_assignment_weight=4.0"
      "model.policy_guard.entity_relation_grounding.clause_tuple_temperature=0.10"
      "model.policy_guard.entity_relation_grounding.clause_tuple_hard_weight=1.0"
      "model.policy_guard.entity_relation_grounding.clause_tuple_multi_consistency_weight=2.0"
    )
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

echo "[PGC-FastWAM] LIBERO ${CONDITION} evaluation"
echo "  checkpoint=${PGC_CHECKPOINT}"
echo "  suites=${SUITES} trials=${NUM_TRIALS}"
echo "  gate=${GATE_MODE} margin=${GATE_THRESHOLD} min_cf=${MIN_COUNTERFACTUAL_SCORE}"
echo "  max_policy_steps=${MAX_POLICY_STEPS:-suite_default}"
echo "  output=${OUTPUT_ROOT}"
echo "  closed_loop_capture=${CLOSED_LOOP_CAPTURE_DIR:-disabled}"
if [[ "${PGC_CHECKPOINT_VERSION}" == "9" ]]; then
  echo "  v9_ablation=${V9_ABLATION} eraf_diagnostics=${ERAF_DIAGNOSTICS}"
  echo "  eraf_overlay_dir=${ERAF_OVERLAY_DIR}"
  echo "  eraf_shadow_audit=${ERAF_SHADOW_AUDIT}"
  echo "  eraf_shadow_sidecar=${ERAF_SHADOW_SIDECAR_DIR:-disabled}"
fi

EXP_NAME="pgc-${CONDITION}" "${PYTHON_BIN}" \
  experiments/libero/run_libero_manager.py "${EXTRA_OVERRIDES[@]}"

echo "[PGC-FastWAM] evaluation complete: ${OUTPUT_ROOT}"
