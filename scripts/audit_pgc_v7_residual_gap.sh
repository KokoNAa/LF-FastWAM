#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:?Usage: bash scripts/audit_pgc_v7_residual_gap.sh <v7_checkpoint> <rollout_results_dir> [num_samples] [output_json]}"
ROLLOUT_RESULTS="${2:?Missing the completed forced-CF rollout result directory}"
NUM_SAMPLES="${3:-20}"
OUTPUT_JSON="${4:-${ROLLOUT_RESULTS%/}/training_deployment_residual_audit.json}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "PGC v7 checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${ROLLOUT_RESULTS}" && ! -f "${ROLLOUT_RESULTS}" ]]; then
  echo "Rollout result path not found: ${ROLLOUT_RESULTS}" >&2
  exit 1
fi
if ! [[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "num_samples must be a positive integer, got ${NUM_SAMPLES}." >&2
  exit 1
fi

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --rollout-results "${ROLLOUT_RESULTS}"
  --num-samples "${NUM_SAMPLES}"
  --output "${OUTPUT_JSON}"
  --device "${PGC_AUDIT_DEVICE:-cuda:0}"
  --dtype "${PGC_AUDIT_DTYPE:-bf16}"
  --seed "${PGC_AUDIT_SEED:-42}"
)

if [[ -n "${PGC_AUDIT_TRAIN_CONFIG:-}" ]]; then
  ARGS+=(--training-config "${PGC_AUDIT_TRAIN_CONFIG}")
fi
if [[ -n "${PGC_AUDIT_NATIVE_DATASET:-}" ]]; then
  ARGS+=(--native-dataset-dir "${PGC_AUDIT_NATIVE_DATASET}")
fi
if [[ -n "${PGC_AUDIT_COUNTERFACTUAL_DATASET:-}" ]]; then
  ARGS+=(--counterfactual-dataset-dir "${PGC_AUDIT_COUNTERFACTUAL_DATASET}")
fi
if [[ -n "${PGC_AUDIT_TEXT_CACHE:-}" ]]; then
  ARGS+=(--text-cache-dir "${PGC_AUDIT_TEXT_CACHE}")
fi
if [[ -n "${PGC_AUDIT_DATASET_STATS:-}" ]]; then
  ARGS+=(--dataset-stats-path "${PGC_AUDIT_DATASET_STATS}")
fi

echo "[PGC audit] checkpoint=${CHECKPOINT}"
echo "[PGC audit] rollout_results=${ROLLOUT_RESULTS}"
echo "[PGC audit] demonstration_samples=${NUM_SAMPLES}"
echo "[PGC audit] output=${OUTPUT_JSON}"

exec "${PYTHON_BIN}" scripts/audit_pgc_residual_gap.py "${ARGS[@]}"
