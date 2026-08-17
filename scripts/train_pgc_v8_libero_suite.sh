#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: bash scripts/train_pgc_v8_libero_suite.sh <suite> <gpus> <base_checkpoint> <v5_checkpoint> <offline_counterfactual_dataset> <closed_loop_corrective_dataset> [seed] [max_steps]}"
NPROC_PER_NODE="${2:?Missing GPU count}"
BASE_CHECKPOINT="${3:?Missing released FastWAM base checkpoint}"
V5_CHECKPOINT="${4:?Missing validated suite-specific PGC V5 checkpoint}"
OFFLINE_DATASET="${5:?Missing original PGC counterfactual dataset}"
CLOSED_LOOP_DATASET="${6:?Missing replay-verified V8 corrective dataset}"
TRAIN_SEED="${7:-42}"
MAX_STEPS="${8:-4000}"

case "${SUITE}" in
  libero_spatial|libero_object|libero_goal|libero_10) ;;
  *)
    echo "Unsupported LIBERO suite: ${SUITE}." >&2
    exit 1
    ;;
esac
for path in "${BASE_CHECKPOINT}" "${V5_CHECKPOINT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Checkpoint not found: ${path}" >&2
    exit 1
  fi
done
for path in "${OFFLINE_DATASET}" "${CLOSED_LOOP_DATASET}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Dataset directory not found: ${path}" >&2
    exit 1
  fi
done

OFFLINE_DATASET="$(cd -- "${OFFLINE_DATASET}" && pwd -P)"
CLOSED_LOOP_DATASET="$(cd -- "${CLOSED_LOOP_DATASET}" && pwd -P)"
OFFLINE_LIST="$(mktemp "/tmp/pgc-v8-offline-${SUITE}.XXXXXX.txt")"
CLOSED_LOOP_LIST="$(mktemp "/tmp/pgc-v8-closed-${SUITE}.XXXXXX.txt")"
cleanup() {
  rm -f "${OFFLINE_LIST}" "${CLOSED_LOOP_LIST}"
}
trap cleanup EXIT
printf '%s\n' "${OFFLINE_DATASET}" >"${OFFLINE_LIST}"
printf '%s\n' "${CLOSED_LOOP_DATASET}" >"${CLOSED_LOOP_LIST}"

export PGC_VERSION=8
export PGC_TRAIN_SUITE="${SUITE}"
export PGC_INIT_CHECKPOINT="${V5_CHECKPOINT}"
export PGC_WARM_START_V5=true
export PGC_CLOSED_LOOP_CORRECTIVE_LIST="${CLOSED_LOOP_LIST}"
export PGC_CLOSED_LOOP_OVERSAMPLE_FACTOR="${PGC_CLOSED_LOOP_OVERSAMPLE_FACTOR:-4}"
export PGC_CLOSED_LOOP_CORRECTIVE_WEIGHT="${PGC_CLOSED_LOOP_CORRECTIVE_WEIGHT:-2.0}"
export PGC_OFFLINE_ACQUISITION_WEIGHT="${PGC_OFFLINE_ACQUISITION_WEIGHT:-1.0}"
export PGC_NATIVE_GUARD_WEIGHT="${PGC_NATIVE_GUARD_WEIGHT:-0.10}"
export PGC_CLOSED_LOOP_TRAIN_PROPOSAL_ONLY=true
export PGC_COUNTERFACTUAL_OVERSAMPLE_FACTOR=1
export PGC_BALANCE_NATIVE_COUNTERFACTUAL=true
export PGC_LEARNING_RATE="${PGC_LEARNING_RATE:-5.0e-6}"
export PGC_SAVE_EVERY="${PGC_SAVE_EVERY:-500}"
export PGC_SAVE_TRAINING_STATE="${PGC_SAVE_TRAINING_STATE:-false}"
export RUN_TAG="${RUN_TAG:-${SUITE}-pgc-v8-closed-loop-acquisition-${MAX_STEPS}-seed${TRAIN_SEED}-v1}"

echo "[PGC-FastWAM] V8 closed-loop target-acquisition training"
echo "  suite=${SUITE} v5_checkpoint=${V5_CHECKPOINT}"
echo "  offline_dataset=${OFFLINE_DATASET}"
echo "  corrective_dataset=${CLOSED_LOOP_DATASET}"
echo "  proposal_only=true lr=${PGC_LEARNING_RATE} steps=${MAX_STEPS}"

exec bash scripts/train_pgc_libero.sh \
  "${NPROC_PER_NODE}" \
  "${BASE_CHECKPOINT}" \
  "${OFFLINE_LIST}" \
  "${TRAIN_SEED}" \
  "${MAX_STEPS}"
