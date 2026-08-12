#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_lf_lora_object_b0_m1.sh <gpus> <base_checkpoint> [seed]}"
BASE_CHECKPOINT="${2:?Missing base FastWAM checkpoint}"
TRAIN_SEED="${3:-42}"
RUN_TAG="${RUN_TAG:-lf-object-lora-1epoch-v1}"
TASK_NAME="libero_object_lf_lora_2cam224"
OBJECT_DATASET="data/libero_mujoco3.3.2/libero_object_no_noops_lerobot"
DEFAULT_STATS_PATH="$(dirname "${BASE_CHECKPOINT}")/libero_uncond_2cam224_dataset_stats.json"
STATS_PATH="${STATS_PATH:-${DEFAULT_STATS_PATH}}"

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU process count must be a positive integer, got: ${NPROC_PER_NODE}" >&2
  exit 1
fi
VISIBLE_GPU_COUNT="$(${PYTHON_BIN:-python} -c 'import torch; print(torch.cuda.device_count())')"
if (( VISIBLE_GPU_COUNT < NPROC_PER_NODE )); then
  echo "Requested ${NPROC_PER_NODE} GPU processes, but PyTorch sees only ${VISIBLE_GPU_COUNT} CUDA device(s)." >&2
  echo "Check nvidia-smi -L and CUDA_VISIBLE_DEVICES before starting B0/M1 training." >&2
  exit 1
fi

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "Base checkpoint not found: ${BASE_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${STATS_PATH}" ]]; then
  echo "Dataset stats not found: ${STATS_PATH}" >&2
  echo "Set STATS_PATH to libero_uncond_2cam224_dataset_stats.json." >&2
  exit 1
fi
if [[ ! -d "${OBJECT_DATASET}" ]]; then
  echo "LIBERO-Object dataset is missing: ${OBJECT_DATASET}" >&2
  echo "Download and extract libero_object_no_noops_lerobot.tar.gz first." >&2
  exit 1
fi
if [[ ! -f "${OBJECT_DATASET}/meta/tasks.jsonl" ]]; then
  echo "Invalid LIBERO-Object dataset; missing ${OBJECT_DATASET}/meta/tasks.jsonl" >&2
  exit 1
fi

CACHE_DIR="data/text_embeds_cache/libero"
MISSING_CACHE_COUNT="$(${PYTHON_BIN:-python} - "${OBJECT_DATASET}/meta/tasks.jsonl" "${CACHE_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

tasks_path = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
template = "A video recorded from a robot's point of view executing the following instruction: {task}"
missing = 0
with tasks_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        task = str(json.loads(line)["task"])
        prompt = template.format(task=task)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = cache_dir / f"{digest}.t5_len128.wan22ti2v5b.pt"
        missing += int(not path.is_file())
print(missing)
PY
)"
if (( MISSING_CACHE_COUNT > 0 )); then
  echo "Missing ${MISSING_CACHE_COUNT} Object text-embedding caches under ${CACHE_DIR}." >&2
  echo "Run scripts/precompute_text_embeds.py with task=${TASK_NAME} first." >&2
  exit 1
fi

echo "[LF-FastWAM] Object-only controlled comparison"
echo "  dataset=${OBJECT_DATASET}"
echo "  base=${BASE_CHECKPOINT}"
echo "  stats=${STATS_PATH}"
echo "  seed=${TRAIN_SEED}"
echo "  schedule=1 epoch (steps derived from dataset length)"

echo "[LF-FastWAM] Training B0: original action path + LoRA"
RUN_ID="lf-b0-${RUN_TAG}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  "task=${TASK_NAME}" \
  "resume=${BASE_CHECKPOINT}" \
  "data.train.pretrained_norm_stats=${STATS_PATH}" \
  "seed=${TRAIN_SEED}" \
  max_steps=null \
  num_epochs=1 \
  model.action_dit_config.use_latent_action_queries=false \
  model.langforce_mvp.enabled=false

echo "[LF-FastWAM] Training M1: latent queries + prior/advantage + LoRA"
RUN_ID="lf-m1-${RUN_TAG}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  "task=${TASK_NAME}" \
  "resume=${BASE_CHECKPOINT}" \
  "data.train.pretrained_norm_stats=${STATS_PATH}" \
  "seed=${TRAIN_SEED}" \
  max_steps=null \
  num_epochs=1 \
  model.action_dit_config.use_latent_action_queries=true \
  model.langforce_mvp.enabled=true \
  model.langforce_mvp.enable_prior=true \
  model.langforce_mvp.enable_posterior_advantage=true

echo "[LF-FastWAM] Object B0/M1 training complete."
