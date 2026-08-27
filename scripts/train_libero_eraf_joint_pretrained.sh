#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: bash scripts/train_libero_eraf_joint_pretrained.sh <suite> <gpus> <released_base> <pretrained_eraf_checkpoint> <historical_cf_dataset> <strict_cf_dataset> <native_sidecar> <historical_cf_sidecar> <strict_cf_sidecar> [seed]}"
NPROC_PER_NODE="${2:?Missing GPU count}"
BASE_CHECKPOINT="${3:?Missing released FastWAM checkpoint}"
PRETRAINED_ERAF_CHECKPOINT="${4:?Missing pretrained ERAF checkpoint}"
HISTORICAL_CF_DATASET="${5:?Missing historical counterfactual dataset}"
STRICT_CF_DATASET="${6:?Missing strict counterfactual dataset}"
NATIVE_SIDECAR="${7:?Missing native ERAF sidecar}"
HISTORICAL_CF_SIDECAR="${8:?Missing historical-CF ERAF sidecar}"
STRICT_CF_SIDECAR="${9:?Missing strict-CF ERAF sidecar}"
TRAIN_SEED="${10:-42}"

export ERAF_PRETRAINED_CHECKPOINT="${PRETRAINED_ERAF_CHECKPOINT}"
exec bash scripts/train_libero_eraf_joint_from_scratch.sh \
  "${SUITE}" \
  "${NPROC_PER_NODE}" \
  "${BASE_CHECKPOINT}" \
  "${HISTORICAL_CF_DATASET}" \
  "${STRICT_CF_DATASET}" \
  "${NATIVE_SIDECAR}" \
  "${HISTORICAL_CF_SIDECAR}" \
  "${STRICT_CF_SIDECAR}" \
  "${TRAIN_SEED}"
