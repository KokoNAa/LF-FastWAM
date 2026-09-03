#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${PROJECT_ROOT}"

: "${ROBOTWIN_BASE_CKPT:?Set ROBOTWIN_BASE_CKPT to robotwin_uncond_3cam_384.pt}"
: "${ROBOTWIN_STATS_PATH:?Set ROBOTWIN_STATS_PATH to the matching dataset_stats.json}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
ERAF_WORK_ROOT="${ERAF_WORK_ROOT:-/root/gpufree-data/pgc_robotwin_eraf_v1}"
GROUNDING_MANIFEST="${GROUNDING_MANIFEST:-${ERAF_WORK_ROOT}/formal/pgc_robotwin_eraf_grounding_manifest.json}"
TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-${PROJECT_ROOT}/data/text_embeds_cache/robotwin}"
ERAF_GROUNDING_SEED="${ERAF_GROUNDING_SEED:-42}"
ERAF_GROUNDING_STEPS="${ERAF_GROUNDING_STEPS:-1500}"
ERAF_GROUNDING_SAVE_EVERY="${ERAF_GROUNDING_SAVE_EVERY:-250}"
RUN_TAG="${RUN_TAG:-robotwin-v939-eraf-grounding-1500-4gpu-seed${ERAF_GROUNDING_SEED}}"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES=0,1,2,3 for the formal four-GPU run." >&2
  exit 1
fi

exec "${PYTHON_BIN}" scripts/train_pgc_robotwin_eraf_grounding.py \
  --gpus 4 \
  --base-checkpoint "${ROBOTWIN_BASE_CKPT}" \
  --grounding-manifest "${GROUNDING_MANIFEST}" \
  --stats-path "${ROBOTWIN_STATS_PATH}" \
  --cache-dir "${TEXT_CACHE_DIR}" \
  --seed "${ERAF_GROUNDING_SEED}" \
  --steps "${ERAF_GROUNDING_STEPS}" \
  --save-every "${ERAF_GROUNDING_SAVE_EVERY}" \
  --run-tag "${RUN_TAG}" \
  "${@}"
