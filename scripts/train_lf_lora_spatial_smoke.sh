#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_lf_lora_spatial_smoke.sh <gpus> <base_checkpoint> [steps] [seed]}"
BASE_CHECKPOINT="${2:?Missing base FastWAM checkpoint}"
TRAIN_STEPS="${3:-2}"
TRAIN_SEED="${4:-42}"

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "Base checkpoint not found: ${BASE_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot" ]]; then
  echo "LIBERO-Spatial dataset is missing under ./data." >&2
  exit 1
fi

RUN_ID="${RUN_ID:-lf-lora-spatial-smoke-$(date +%Y-%m-%d_%H-%M-%S)}" \
bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task=libero_spatial_lf_lora_2cam224 \
  "resume=${BASE_CHECKPOINT}" \
  "max_steps=${TRAIN_STEPS}" \
  "seed=${TRAIN_SEED}"
