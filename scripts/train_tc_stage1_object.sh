#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_tc_stage1_object.sh <gpus> <m1_checkpoint> [seed] [max_steps]}"
M1_CHECKPOINT="${2:?Missing M1 LoRA checkpoint}"
TRAIN_SEED="${3:-42}"
MAX_STEPS="${4:-null}"
TC_VERSION="${TC_VERSION:-2}"
if [[ "${TC_VERSION}" != "2" && "${TC_VERSION}" != "3" ]]; then
  echo "TC_VERSION must be 2 or 3, got: ${TC_VERSION}" >&2
  exit 1
fi
if [[ "${TC_VERSION}" == "3" ]]; then
  DISTILLATION_DEFAULT=true
  FREEZE_POLICY_DEFAULT=true
else
  DISTILLATION_DEFAULT=false
  FREEZE_POLICY_DEFAULT=false
fi
TC_POLICY_DISTILLATION_ENABLED="${TC_POLICY_DISTILLATION_ENABLED:-${DISTILLATION_DEFAULT}}"
TC_POLICY_DISTILLATION_WEIGHT="${TC_POLICY_DISTILLATION_WEIGHT:-1.0}"
TC_FREEZE_M1_POLICY="${TC_FREEZE_M1_POLICY:-${FREEZE_POLICY_DEFAULT}}"
TC_SAVE_TRAINING_STATE="${TC_SAVE_TRAINING_STATE:-false}"
TC_RESUME_STATE="${TC_RESUME_STATE:-}"
RUN_TAG="${RUN_TAG:-v${TC_VERSION}-object-stage1-v1}"
TASK_NAME="libero_object_lf_lora_2cam224"
OBJECT_DATASET="data/libero_mujoco3.3.2/libero_object_no_noops_lerobot"
DEFAULT_STATS_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-./checkpoints}/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
STATS_PATH="${STATS_PATH:-${DEFAULT_STATS_PATH}}"

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU process count must be positive, got: ${NPROC_PER_NODE}" >&2
  exit 1
fi
if [[ "${MAX_STEPS}" != "null" ]] && ! [[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "max_steps must be 'null' or a positive integer, got: ${MAX_STEPS}" >&2
  exit 1
fi
if [[ ! -f "${M1_CHECKPOINT}" ]]; then
  echo "M1 checkpoint not found: ${M1_CHECKPOINT}" >&2
  exit 1
fi
if [[ -n "${TC_RESUME_STATE}" ]]; then
  if [[ ! -d "${TC_RESUME_STATE}" ]]; then
    echo "Training-state directory not found: ${TC_RESUME_STATE}" >&2
    exit 1
  fi
  if [[ ! -f "${TC_RESUME_STATE}/trainer_state.json" ]]; then
    echo "Training-state directory is incomplete: ${TC_RESUME_STATE}" >&2
    exit 1
  fi
  RESUME_TARGET="${TC_RESUME_STATE}"
else
  RESUME_TARGET="${M1_CHECKPOINT}"
fi
if [[ ! -f "${STATS_PATH}" ]]; then
  echo "Dataset stats not found: ${STATS_PATH}" >&2
  exit 1
fi
if [[ ! -f "${OBJECT_DATASET}/meta/tasks.jsonl" ]]; then
  echo "LIBERO-Object dataset is missing or incomplete: ${OBJECT_DATASET}" >&2
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
        digest = hashlib.sha256(template.format(task=task).encode("utf-8")).hexdigest()
        missing += int(not (cache_dir / f"{digest}.t5_len128.wan22ti2v5b.pt").is_file())
print(missing)
PY
)"
if (( MISSING_CACHE_COUNT > 0 )); then
  echo "Missing ${MISSING_CACHE_COUNT} Object text-embedding caches under ${CACHE_DIR}." >&2
  echo "Precompute Object text embeddings before TC-C training." >&2
  exit 1
fi

VISIBLE_GPU_COUNT="$(${PYTHON_BIN:-python} -c 'import torch; print(torch.cuda.device_count())')"
if (( VISIBLE_GPU_COUNT < NPROC_PER_NODE )); then
  echo "Requested ${NPROC_PER_NODE} GPUs, PyTorch sees ${VISIBLE_GPU_COUNT}." >&2
  exit 1
fi

echo "[TC-FastWAM] Stage 1 TC-C v${TC_VERSION} Object training"
echo "  initialization=M1 LoRA checkpoint (policy recovery)"
echo "  checkpoint=${M1_CHECKPOINT}"
echo "  dataset=${OBJECT_DATASET}"
echo "  seed=${TRAIN_SEED}"
echo "  max_steps=${MAX_STEPS} (null means one complete epoch)"
echo "  policy_distillation=${TC_POLICY_DISTILLATION_ENABLED} weight=${TC_POLICY_DISTILLATION_WEIGHT}"
echo "  freeze_m1_policy=${TC_FREEZE_M1_POLICY}"
echo "  save_training_state=${TC_SAVE_TRAINING_STATE}"
echo "  resume=${RESUME_TARGET}"
echo "  method=final-video-hidden Router + policy recovery + LF InfoNCE + optional frozen-M1 distillation"

CHECKPOINT_KIND="$(${PYTHON_BIN:-python} - "${M1_CHECKPOINT}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
fmt = payload.get("format")
meta = payload.get("architecture_metadata") or {}
if fmt != "fastwam_lora_adapter_v1":
    raise SystemExit(f"TC-C requires an M1 LoRA adapter, got format={fmt!r}")
if meta.get("architecture") == "tc_fastwam":
    raise SystemExit("TC-C must initialize from M1, not a previous TC adapter")
mot_keys = payload.get("mot_trainable") or {}
if not any(key.endswith("latent_action_queries") for key in mot_keys):
    raise SystemExit("Checkpoint does not contain trained M1 latent queries")
print("m1_lora_adapter")
PY
)"
echo "  checkpoint_kind=${CHECKPOINT_KIND}"

RUN_ID="tc-c-${RUN_TAG}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  "task=${TASK_NAME}" \
  "resume=${RESUME_TARGET}" \
  "data.train.pretrained_norm_stats=${STATS_PATH}" \
  "seed=${TRAIN_SEED}" \
  "max_steps=${MAX_STEPS}" \
  num_epochs=1 \
  "save_training_state=${TC_SAVE_TRAINING_STATE}" \
  model.action_dit_config.use_latent_action_queries=true \
  model.langforce_mvp.enabled=false \
  model.langforce_mvp.enable_prior=false \
  model.langforce_mvp.enable_posterior_advantage=false \
  model.transition_contract.enabled=true \
  "model.transition_contract.version=${TC_VERSION}" \
  "model.transition_contract.policy_distillation_enabled=${TC_POLICY_DISTILLATION_ENABLED}" \
  "model.transition_contract.policy_distillation_weight=${TC_POLICY_DISTILLATION_WEIGHT}" \
  "model.transition_contract.freeze_m1_policy=${TC_FREEZE_M1_POLICY}"

echo "[TC-FastWAM] Stage 1 training complete."
