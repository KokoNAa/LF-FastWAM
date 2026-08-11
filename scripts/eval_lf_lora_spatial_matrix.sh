#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:-4}"
NUM_TRIALS="${2:-5}"
NUM_INFERENCE_STEPS="${3:-10}"
EVAL_SEED="${4:-42}"

if (( NUM_GPUS < 1 )); then
  echo "NUM_GPUS must be at least 1, got ${NUM_GPUS}." >&2
  exit 1
fi
if (( NUM_TRIALS < 1 )); then
  echo "NUM_TRIALS must be at least 1, got ${NUM_TRIALS}." >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT:-$(pwd)}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
cd "${RUN_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
export PYTHON_BIN

STATS_PATH="${STATS_PATH:-${RUN_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
B0_CKPT="${B0_CKPT:-${RUN_ROOT}/runs/libero_spatial_lf_lora_2cam224/lf-b0-lf-spatial-lora-200-v1/checkpoints/weights/step_000200.pt}"
B1_CKPT="${B1_CKPT:-${RUN_ROOT}/runs/libero_spatial_lf_lora_2cam224/lf-b1-lf-spatial-lora-200-v1/checkpoints/weights/step_000200.pt}"
M1_CKPT="${M1_CKPT:-${RUN_ROOT}/runs/libero_spatial_lf_lora_2cam224/lf-m1-lf-spatial-lora-200-v1/checkpoints/weights/step_000200.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/evaluate_results/lf_spatial_200step_seed${EVAL_SEED}_trials${NUM_TRIALS}}"
EVAL_CONDITIONS="${EVAL_CONDITIONS:-correct null}"
read -r -a CONDITION_LIST <<<"${EVAL_CONDITIONS}"

if (( ${#CONDITION_LIST[@]} == 0 )); then
  echo "EVAL_CONDITIONS must contain at least one condition." >&2
  exit 1
fi
for condition in "${CONDITION_LIST[@]}"; do
  if [[ "${condition}" != "correct" && "${condition}" != "null" ]]; then
    echo "Unsupported Spatial matrix condition: ${condition}. Expected correct and/or null." >&2
    exit 1
  fi
done

for required_file in "${STATS_PATH}" "${B0_CKPT}" "${B1_CKPT}" "${M1_CKPT}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required evaluation file not found: ${required_file}" >&2
    exit 1
  fi
done
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required by the LIBERO parallel evaluation manager." >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_LIST=""
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    if [[ -n "${GPU_LIST}" ]]; then
      GPU_LIST+=","
    fi
    GPU_LIST+="${gpu}"
  done
  export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
else
  VISIBLE_GPU_COUNT="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
  if (( VISIBLE_GPU_COUNT != NUM_GPUS )); then
    echo "NUM_GPUS=${NUM_GPUS}, but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} exposes ${VISIBLE_GPU_COUNT} GPUs." >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_ROOT}"

run_condition() {
  local model_label="$1"
  local condition="$2"
  local checkpoint="$3"
  shift 3
  local condition_output="${OUTPUT_ROOT}/${model_label}/${condition}"

  echo "[LF-FastWAM] Evaluating ${model_label} / ${condition}"
  echo "  checkpoint=${checkpoint}"
  echo "  output=${condition_output}"

  EXP_NAME="lf-${model_label}-${condition}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" experiments/libero/run_libero_manager.py \
    task=libero_spatial_lf_lora_2cam224 \
    "ckpt=${checkpoint}" \
    "seed=${EVAL_SEED}" \
    "EVALUATION.dataset_stats_path=${STATS_PATH}" \
    "EVALUATION.num_trials=${NUM_TRIALS}" \
    "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}" \
    "EVALUATION.instruction_condition=${condition}" \
    "EVALUATION.output_dir=${condition_output}" \
    'MULTIRUN.task_suite_names=[libero_spatial]' \
    "MULTIRUN.num_gpus=${NUM_GPUS}" \
    MULTIRUN.max_tasks_per_gpu=1 \
    "$@"
}

B0_OVERRIDES=(
  model.action_dit_config.use_latent_action_queries=false
  model.langforce_mvp.enabled=false
)
B1_OVERRIDES=(
  model.action_dit_config.use_latent_action_queries=true
  model.langforce_mvp.enabled=true
  model.langforce_mvp.enable_prior=false
  model.langforce_mvp.enable_posterior_advantage=false
)
M1_OVERRIDES=(
  model.action_dit_config.use_latent_action_queries=true
  model.langforce_mvp.enabled=true
  model.langforce_mvp.enable_prior=true
  model.langforce_mvp.enable_posterior_advantage=true
)

for condition in "${CONDITION_LIST[@]}"; do
  run_condition B0 "${condition}" "${B0_CKPT}" "${B0_OVERRIDES[@]}"
done
for condition in "${CONDITION_LIST[@]}"; do
  run_condition B1 "${condition}" "${B1_CKPT}" "${B1_OVERRIDES[@]}"
done
for condition in "${CONDITION_LIST[@]}"; do
  run_condition M1 "${condition}" "${M1_CKPT}" "${M1_OVERRIDES[@]}"
done

"${PYTHON_BIN}" scripts/summarize_language_interventions.py \
  --run "B0=${OUTPUT_ROOT}/B0" \
  --run "B1=${OUTPUT_ROOT}/B1" \
  --run "M1=${OUTPUT_ROOT}/M1" \
  --output-prefix "${OUTPUT_ROOT}/lf_mvp_summary"

echo "[LF-FastWAM] Spatial matrix complete (conditions: ${EVAL_CONDITIONS}): ${OUTPUT_ROOT}"
echo "[LF-FastWAM] Spatial tasks share the same object and receptacle; do not report shuffled DTL from this suite."
