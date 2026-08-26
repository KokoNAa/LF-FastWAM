#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:-8}"
NUM_EPISODES="${2:-5}"
NUM_INFERENCE_STEPS="${3:-10}"
EVAL_SEED="${4:-42}"
ROBOTWIN_CKPT="${5:-${ROBOTWIN_CKPT:-}}"
STATS_PATH="${6:-${STATS_PATH:-}}"
if (( $# >= 6 )); then
  shift 6
else
  shift "$#"
fi

for value_name in NUM_GPUS NUM_EPISODES NUM_INFERENCE_STEPS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got ${value}." >&2
    exit 1
  fi
done
if ! [[ "${EVAL_SEED}" =~ ^[0-9]+$ ]]; then
  echo "EVAL_SEED must be a non-negative integer, got ${EVAL_SEED}." >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT:-$(pwd)}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
cd "${RUN_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"

if [[ -z "${ROBOTWIN_CKPT}" ]]; then
  echo "Pass checkpoint as argument 5 or set ROBOTWIN_CKPT." >&2
  exit 1
fi
if [[ -z "${STATS_PATH}" ]]; then
  echo "Pass dataset stats as argument 6 or set STATS_PATH." >&2
  exit 1
fi
if [[ ! -f "${ROBOTWIN_CKPT}" ]]; then
  echo "Checkpoint not found: ${ROBOTWIN_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${STATS_PATH}" ]]; then
  echo "Dataset stats not found: ${STATS_PATH}" >&2
  exit 1
fi
ROBOTWIN_CKPT="$(cd "$(dirname "${ROBOTWIN_CKPT}")" && pwd)/$(basename "${ROBOTWIN_CKPT}")"
STATS_PATH="$(cd "$(dirname "${STATS_PATH}")" && pwd)/$(basename "${STATS_PATH}")"

MANIFEST_PATH="${MANIFEST_PATH:-${RUN_ROOT}/configs/eval/robotwin_cis_spatial.json}"
CIS_CONDITIONS="${CIS_CONDITIONS:-correct,shuffled,counterfactual}"
CIS_TASK_CONFIGS="${CIS_TASK_CONFIGS:-demo_clean,demo_randomized}"
CIS_TASKS="${CIS_TASKS:-}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
FASTWAM_EVAL_MODE="${FASTWAM_EVAL_MODE:-B0}"
ROBOTWIN_TASK_CONFIG="${ROBOTWIN_TASK_CONFIG:-robotwin_uncond_3cam_384_1e-4}"
CKPT_STEM="$(basename "${ROBOTWIN_CKPT}")"
CKPT_STEM="${CKPT_STEM%.*}"
RUN_TAG="${RUN_TAG:-robotwin_cis_${FASTWAM_EVAL_MODE}_${CKPT_STEM}_seed${EVAL_SEED}_episodes${NUM_EPISODES}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/evaluate_results/robotwin_cis_runs/${RUN_TAG}}"

for csv_name in CIS_CONDITIONS CIS_TASK_CONFIGS; do
  csv_value="${!csv_name}"
  if ! [[ "${csv_value}" =~ ^[a-z0-9_]+(,[a-z0-9_]+)*$ ]]; then
    echo "${csv_name} must be a comma-separated identifier list, got ${csv_value}." >&2
    exit 1
  fi
done
if [[ -n "${CIS_TASKS}" ]] && ! [[ "${CIS_TASKS}" =~ ^[a-z0-9_]+(,[a-z0-9_]+)*$ ]]; then
  echo "CIS_TASKS must be a comma-separated identifier list, got ${CIS_TASKS}." >&2
  exit 1
fi
if ! [[ "${MAX_TASKS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TASKS_PER_GPU must be a positive integer." >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/validate_robotwin_cis_manifest.py \
  "${MANIFEST_PATH}" \
  --robotwin-root "${RUN_ROOT}/third_party/RoboTwin"

MODEL_OVERRIDES=()
case "${FASTWAM_EVAL_MODE}" in
  B0)
    MODEL_OVERRIDES+=(
      model.action_dit_config.use_latent_action_queries=false
      model.langforce_mvp.enabled=false
      model.transition_contract.enabled=false
      model.policy_guard.enabled=false
    )
    ;;
  M1)
    MODEL_OVERRIDES+=(
      model.action_dit_config.use_latent_action_queries=true
      model.langforce_mvp.enabled=true
      model.langforce_mvp.enable_prior=true
      model.langforce_mvp.enable_posterior_advantage=true
      model.transition_contract.enabled=false
      model.policy_guard.enabled=false
    )
    ;;
  PGC)
    ROBOTWIN_TASK_CONFIG=robotwin_pgc_3cam_384
    PGC_OVERRIDES="$("${PYTHON_BIN}" scripts/inspect_pgc_checkpoint.py \
      "${ROBOTWIN_CKPT}" \
      --target robotwin \
      --inference-steps "${NUM_INFERENCE_STEPS}" \
      --format hydra)"
    while IFS= read -r override; do
      [[ -z "${override}" ]] || MODEL_OVERRIDES+=("${override}")
    done <<< "${PGC_OVERRIDES}"
    ;;
  CUSTOM)
    ;;
  *)
    echo "FASTWAM_EVAL_MODE must be B0, M1, PGC, or CUSTOM." >&2
    exit 1
    ;;
esac

TASK_OVERRIDES=()
if [[ -n "${CIS_TASKS}" ]]; then
  TASK_OVERRIDES+=("EVALUATION.cis_tasks=[${CIS_TASKS}]")
fi

echo "[RoboTwin CIS] checkpoint=${ROBOTWIN_CKPT}"
echo "[RoboTwin CIS] output selector=${OUTPUT_ROOT}"
echo "[RoboTwin CIS] conditions=${CIS_CONDITIONS} task_configs=${CIS_TASK_CONFIGS}"

"${PYTHON_BIN}" experiments/robotwin/run_robotwin_cis_manager.py \
  "task=${ROBOTWIN_TASK_CONFIG}" \
  "ckpt=${ROBOTWIN_CKPT}" \
  "seed=${EVAL_SEED}" \
  "EVALUATION.dataset_stats_path=${STATS_PATH}" \
  "EVALUATION.language_intervention_manifest=${MANIFEST_PATH}" \
  "EVALUATION.cis_conditions=[${CIS_CONDITIONS}]" \
  "EVALUATION.cis_task_configs=[${CIS_TASK_CONFIGS}]" \
  "EVALUATION.eval_num_episodes=${NUM_EPISODES}" \
  "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}" \
  "EVALUATION.instruction_type=${INSTRUCTION_TYPE}" \
  "EVALUATION.output_dir=${OUTPUT_ROOT}" \
  EVALUATION.resume_completed=true \
  "MULTIRUN.num_gpus=${NUM_GPUS}" \
  "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}" \
  "${TASK_OVERRIDES[@]}" \
  "${MODEL_OVERRIDES[@]}" \
  "$@"

echo "[RoboTwin CIS] matrix complete; see the manager's run_root above."
