#!/usr/bin/env bash
set -euo pipefail

FASTWAM_CHECKPOINT="${1:?Usage: bash scripts/eval_fastwam_libero_object_language_ood.sh <released_fastwam_checkpoint> <ood_manifest> [gpus] [trials] [seed]}"
OOD_MANIFEST="${2:?Missing audited LIBERO-Object language-OOD manifest}"
NUM_GPUS="${3:-4}"
NUM_TRIALS="${4:-2}"
EVAL_SEED="${5:-42}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
STATS_PATH="${STATS_PATH:-$(dirname "${FASTWAM_CHECKPOINT}")/libero_uncond_2cam224_dataset_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(pwd)/evaluate_results/fastwam_libero_object_language_ood_seed${EVAL_SEED}_trials${NUM_TRIALS}}"
OOD_CONDITIONS="${OOD_CONDITIONS:-correct paraphrase_near paraphrase_sequence paraphrase_goal null shuffled}"

for value_name in NUM_GPUS NUM_TRIALS EVAL_SEED; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value_name}" != "EVAL_SEED" && "${value}" == "0" ]]; then
    echo "${value_name} has an invalid value: ${value}." >&2
    exit 1
  fi
done
if [[ ! -f "${FASTWAM_CHECKPOINT}" ]]; then
  echo "Released FastWAM checkpoint not found: ${FASTWAM_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${STATS_PATH}" ]]; then
  echo "Released FastWAM dataset stats not found: ${STATS_PATH}" >&2
  exit 1
fi
if [[ ! -f "${OOD_MANIFEST}" ]]; then
  echo "Language-OOD manifest not found: ${OOD_MANIFEST}" >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required by the LIBERO evaluation manager." >&2
  exit 1
fi

"${PYTHON_BIN}" - "${FASTWAM_CHECKPOINT}" <<'PY'
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1]).expanduser().resolve()
payload = torch.load(path, map_location="cpu", weights_only=False)
fmt = payload.get("format")
if fmt == "fastwam_lora_adapter_v1" or str(fmt).startswith("fastwam_policy_guard_v"):
    raise SystemExit(f"Language-OOD baseline requires released native FastWAM, got {fmt!r}.")
if "mot" not in payload and "dit" not in payload:
    raise SystemExit("Released native FastWAM checkpoint must contain `mot` or `dit` weights.")
if payload.get("transition_contract") is not None or payload.get("policy_guard") is not None:
    raise SystemExit("Native language-OOD checkpoint unexpectedly contains project add-on tensors.")
print(f"Validated released native FastWAM checkpoint: {path}")
PY

"${PYTHON_BIN}" scripts/validate_libero_language_ood_manifest.py "${OOD_MANIFEST}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_LIST=""
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    [[ -z "${GPU_LIST}" ]] || GPU_LIST+=","
    GPU_LIST+="${gpu}"
  done
  export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
else
  VISIBLE_GPU_COUNT="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
  if (( VISIBLE_GPU_COUNT != NUM_GPUS )); then
    echo "NUM_GPUS=${NUM_GPUS}, but CUDA_VISIBLE_DEVICES exposes ${VISIBLE_GPU_COUNT}." >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_ROOT}"

run_condition() {
  local label="$1"
  local condition="$2"
  local variant="$3"
  local condition_output="${OUTPUT_ROOT}/${label}"
  local completed=0
  if [[ -d "${condition_output}/libero_object" ]]; then
    completed="$(find "${condition_output}/libero_object" -maxdepth 1 -type f -name 'gpu*_task*_results.json' | wc -l | tr -d ' ')"
  fi
  if [[ "${completed}" == "10" ]]; then
    echo "[FastWAM language OOD] skip completed condition ${label}: ${condition_output}"
    return
  fi

  local overrides=(
    task=libero_uncond_2cam224_1e-4
    "ckpt=${FASTWAM_CHECKPOINT}"
    "seed=${EVAL_SEED}"
    "EVALUATION.dataset_stats_path=${STATS_PATH}"
    "EVALUATION.num_trials=${NUM_TRIALS}"
    EVALUATION.num_inference_steps=10
    EVALUATION.max_steps=400
    EVALUATION.replan_steps=10
    EVALUATION.rand_device=cpu
    EVALUATION.use_action_ensembler=false
    "EVALUATION.instruction_condition=${condition}"
    "EVALUATION.output_dir=${condition_output}"
    'MULTIRUN.task_suite_names=[libero_object]'
    "MULTIRUN.num_gpus=${NUM_GPUS}"
    MULTIRUN.max_tasks_per_gpu=1
    model.action_dit_config.use_latent_action_queries=true
    model.langforce_mvp.enabled=false
    model.langforce_mvp.enable_prior=false
    model.langforce_mvp.enable_posterior_advantage=false
    model.transition_contract.enabled=false
    model.policy_guard.enabled=false
    model.lora.enabled=false
    model.lora.paired_language_control.enabled=false
  )
  if [[ "${condition}" == "paraphrase" ]]; then
    overrides+=(
      "EVALUATION.language_intervention_manifest=${OOD_MANIFEST}"
      "EVALUATION.language_ood_variant=${variant}"
    )
  elif [[ "${condition}" == "shuffled" ]]; then
    overrides+=("EVALUATION.language_intervention_manifest=${OOD_MANIFEST}")
  fi

  echo "[FastWAM language OOD] condition=${label} output=${condition_output}"
  LIBERO_TMUX_SESSION_NAME="fastwam_object_ood_${EVAL_SEED}_${label}" \
  EXP_NAME="fastwam-native-object-language-ood-${label}" \
    "${PYTHON_BIN}" experiments/libero/run_libero_manager.py "${overrides[@]}"
}

for label in ${OOD_CONDITIONS}; do
  case "${label}" in
    correct) run_condition correct correct null ;;
    paraphrase_near) run_condition paraphrase_near paraphrase near ;;
    paraphrase_sequence) run_condition paraphrase_sequence paraphrase sequence ;;
    paraphrase_goal) run_condition paraphrase_goal paraphrase goal ;;
    null) run_condition null null null ;;
    shuffled) run_condition shuffled shuffled null ;;
    *)
      echo "Unsupported OOD_CONDITIONS entry: ${label}" >&2
      exit 1
      ;;
  esac
done

"${PYTHON_BIN}" scripts/summarize_libero_language_ood.py \
  --run-root "${OUTPUT_ROOT}" \
  --output-prefix "${OUTPUT_ROOT}/language_ood_summary"

echo "[FastWAM language OOD] complete: ${OUTPUT_ROOT}"
