#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:-4}"
NUM_TRIALS="${2:-1}"
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
B0_CKPT="${B0_CKPT:-${RUN_ROOT}/runs/libero_spatial_lf_lora_2cam224/lf-b0-lf-spatial-lora-1epoch-v1/checkpoints/weights/step_003328.pt}"
B1_CKPT="${B1_CKPT:-${RUN_ROOT}/runs/libero_spatial_lf_lora_2cam224/lf-b1-lf-spatial-lora-1epoch-v1/checkpoints/weights/step_003328.pt}"
M1_CKPT="${M1_CKPT:-${RUN_ROOT}/runs/libero_spatial_lf_lora_2cam224/lf-m1-lf-spatial-lora-1epoch-v1/checkpoints/weights/step_003328.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/evaluate_results/lf_object_dtl_cis_1epoch_seed${EVAL_SEED}_trials${NUM_TRIALS}}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_ROOT}/libero_object_dtl_cis.jsonl}"
EVAL_CONDITIONS="${EVAL_CONDITIONS:-correct shuffled counterfactual}"
EVAL_MODELS="${EVAL_MODELS:-B0 B1 M1}"
read -r -a CONDITION_LIST <<<"${EVAL_CONDITIONS}"
read -r -a MODEL_LIST <<<"${EVAL_MODELS}"

if (( ${#CONDITION_LIST[@]} == 0 )); then
  echo "EVAL_CONDITIONS must contain at least one condition." >&2
  exit 1
fi
for condition in "${CONDITION_LIST[@]}"; do
  if [[ "${condition}" != "correct" && "${condition}" != "shuffled" && "${condition}" != "counterfactual" ]]; then
    echo "Unsupported DTL/CIS condition: ${condition}." >&2
    exit 1
  fi
done
if (( ${#MODEL_LIST[@]} == 0 )); then
  echo "EVAL_MODELS must contain at least one of B0, B1, M1." >&2
  exit 1
fi
for model_label in "${MODEL_LIST[@]}"; do
  if [[ "${model_label}" != "B0" && "${model_label}" != "B1" && "${model_label}" != "M1" ]]; then
    echo "Unsupported model label: ${model_label}." >&2
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
"${PYTHON_BIN}" scripts/prepare_libero_object_interventions.py \
  --suite libero_object \
  --output "${MANIFEST_PATH}"
"${PYTHON_BIN}" scripts/validate_language_intervention_manifest.py \
  "${MANIFEST_PATH}"

run_condition() {
  local model_label="$1"
  local condition="$2"
  local checkpoint="$3"
  shift 3
  local condition_output="${OUTPUT_ROOT}/${model_label}/${condition}"

  echo "[LF-FastWAM] Evaluating ${model_label} / ${condition} on LIBERO-Object"
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
    "EVALUATION.language_intervention_manifest=${MANIFEST_PATH}" \
    "EVALUATION.output_dir=${condition_output}" \
    'MULTIRUN.task_suite_names=[libero_object]' \
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

SUMMARY_ARGS=()
for model_label in "${MODEL_LIST[@]}"; do
  case "${model_label}" in
    B0)
      checkpoint="${B0_CKPT}"
      model_overrides=("${B0_OVERRIDES[@]}")
      ;;
    B1)
      checkpoint="${B1_CKPT}"
      model_overrides=("${B1_OVERRIDES[@]}")
      ;;
    M1)
      checkpoint="${M1_CKPT}"
      model_overrides=("${M1_OVERRIDES[@]}")
      ;;
  esac
  for condition in "${CONDITION_LIST[@]}"; do
    run_condition \
      "${model_label}" \
      "${condition}" \
      "${checkpoint}" \
      "${model_overrides[@]}"
  done
  SUMMARY_ARGS+=(--run "${model_label}=${OUTPUT_ROOT}/${model_label}")
done

"${PYTHON_BIN}" scripts/summarize_language_interventions.py \
  "${SUMMARY_ARGS[@]}" \
  --output-prefix "${OUTPUT_ROOT}/lf_dtl_cis_summary"

echo "[LF-FastWAM] DTL/CIS matrix complete: ${OUTPUT_ROOT}"
echo "[LF-FastWAM] DTL is shuffled/source-goal success; CIS is counterfactual/alternate-goal success."
