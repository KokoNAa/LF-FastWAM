#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: bash scripts/train_pgc_libero_suite.sh <suite> <gpus> <base_checkpoint> <counterfactual_dataset_dir> [seed] [max_steps]}"
NPROC_PER_NODE="${2:?Missing GPU count}"
BASE_CHECKPOINT="${3:?Missing released FastWAM base checkpoint}"
COUNTERFACTUAL_DATASET="${4:?Missing suite-specific PGC dataset directory}"
TRAIN_SEED="${5:-42}"
MAX_STEPS="${6:-4000}"

case "${SUITE}" in
  libero_spatial|libero_object|libero_goal|libero_10)
    ;;
  *)
    echo "suite must be libero_spatial, libero_object, libero_goal, or libero_10; got ${SUITE}." >&2
    exit 1
    ;;
esac

if [[ ! -d "${COUNTERFACTUAL_DATASET}" ]]; then
  echo "Counterfactual dataset directory not found: ${COUNTERFACTUAL_DATASET}" >&2
  exit 1
fi
COUNTERFACTUAL_DATASET="$(cd -- "${COUNTERFACTUAL_DATASET}" && pwd -P)"

LIST_FILE="$(mktemp "/tmp/pgc-${SUITE}.XXXXXX.txt")"
cleanup() {
  rm -f "${LIST_FILE}"
}
trap cleanup EXIT
printf '%s\n' "${COUNTERFACTUAL_DATASET}" >"${LIST_FILE}"

export PGC_TRAIN_SUITE="${SUITE}"
PGC_VERSION="${PGC_VERSION:-2}"
export PGC_VERSION
export RUN_TAG="${RUN_TAG:-${SUITE}-pgc-v${PGC_VERSION}-${MAX_STEPS}-seed${TRAIN_SEED}}"

bash scripts/train_pgc_libero.sh \
  "${NPROC_PER_NODE}" \
  "${BASE_CHECKPOINT}" \
  "${LIST_FILE}" \
  "${TRAIN_SEED}" \
  "${MAX_STEPS}"
