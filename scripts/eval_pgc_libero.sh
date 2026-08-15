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

for value_name in NUM_GPUS NUM_TRIALS NUM_INFERENCE_STEPS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got ${value}." >&2
    exit 1
  fi
done
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

"${PYTHON_BIN}" - "${PGC_CHECKPOINT}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
metadata = payload.get("architecture_metadata") or {}
if payload.get("format") != "fastwam_policy_guard_v1":
    raise SystemExit("Evaluation requires a fastwam_policy_guard_v1 checkpoint")
if metadata.get("architecture") != "pgc_fastwam":
    raise SystemExit("Checkpoint is missing PGC architecture metadata")
if int(metadata.get("policy_guard_version", -1)) != 1:
    raise SystemExit("Only PGC version 1 is supported")
if metadata.get("policy_protection") != "immutable_base_plus_conservative_hard_gate":
    raise SystemExit("Checkpoint does not declare the protected hard-gate path")
print(f"Validated PGC checkpoint: step={payload.get('step')} base={payload.get('base_checkpoint')}")
PY

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
  "model.policy_guard.version=1"
  "model.policy_guard.gate_mode=${GATE_MODE}"
  "model.policy_guard.gate_threshold=${GATE_THRESHOLD}"
  "model.policy_guard.min_counterfactual_score=${MIN_COUNTERFACTUAL_SCORE}"
  "model.lora.enabled=false"
)
if [[ -n "${MANIFEST_PATH}" ]]; then
  EXTRA_OVERRIDES+=("EVALUATION.language_intervention_manifest=${MANIFEST_PATH}")
fi
if [[ "${CONDITION}" == "counterfactual" ]]; then
  EXTRA_OVERRIDES+=("EVALUATION.counterfactual_diagnostics=true")
fi

echo "[PGC-FastWAM] LIBERO ${CONDITION} evaluation"
echo "  checkpoint=${PGC_CHECKPOINT}"
echo "  suites=${SUITES} trials=${NUM_TRIALS}"
echo "  gate=${GATE_MODE} margin=${GATE_THRESHOLD} min_cf=${MIN_COUNTERFACTUAL_SCORE}"
echo "  output=${OUTPUT_ROOT}"

EXP_NAME="pgc-${CONDITION}" "${PYTHON_BIN}" \
  experiments/libero/run_libero_manager.py "${EXTRA_OVERRIDES[@]}"

echo "[PGC-FastWAM] evaluation complete: ${OUTPUT_ROOT}"
