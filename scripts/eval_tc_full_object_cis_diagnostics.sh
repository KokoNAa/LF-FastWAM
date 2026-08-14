#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:-4}"
NUM_TRIALS="${2:-5}"
NUM_INFERENCE_STEPS="${3:-10}"
EVAL_SEED="${4:-42}"
MAX_POLICY_STEPS="${5:-600}"

RUN_ROOT="${RUN_ROOT:-$(pwd)}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
cd "${RUN_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
TC_CHECKPOINT="${TC_CHECKPOINT:?Set TC_CHECKPOINT to a TC-Full v4/v5/v6 adapter}"
STATS_PATH="${STATS_PATH:-${DIFFSYNTH_MODEL_BASE_PATH:-${RUN_ROOT}/checkpoints}/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/evaluate_results/tc_full_object_cis_diagnostics_seed${EVAL_SEED}_trials${NUM_TRIALS}_h${MAX_POLICY_STEPS}}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_ROOT}/libero_object_dtl_cis.jsonl}"

for value_name in NUM_GPUS NUM_TRIALS NUM_INFERENCE_STEPS MAX_POLICY_STEPS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got ${value}." >&2
    exit 1
  fi
done
if [[ ! -f "${TC_CHECKPOINT}" ]]; then
  echo "TC-Full checkpoint not found: ${TC_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${STATS_PATH}" ]]; then
  echo "Dataset stats not found: ${STATS_PATH}" >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required by the LIBERO parallel evaluation manager." >&2
  exit 1
fi

TC_VERSION="$("${PYTHON_BIN}" - "${TC_CHECKPOINT}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
metadata = payload.get("architecture_metadata") or {}
if payload.get("format") != "fastwam_lora_adapter_v1":
    raise SystemExit("CIS diagnostics require a FastWAM LoRA adapter")
if int(payload.get("step", -1)) <= 0:
    raise SystemExit("Checkpoint has no positive training step")
version = int(metadata.get("transition_contract_version", -1))
if version not in {4, 5, 6}:
    raise SystemExit("CIS diagnostics require TC-Full v4/v5/v6")
if not metadata.get("use_action_effect") or not metadata.get("use_cf_ranking"):
    raise SystemExit("Checkpoint does not contain the complete TC-Full method")
if version >= 5 and not metadata.get("use_cf_action_positive"):
    raise SystemExit(f"TC-Full v{version} checkpoint has no action-positive supervision")
if version == 6 and not metadata.get("use_state_conditioned_grounding"):
    raise SystemExit("TC-Full v6 checkpoint has no current-state target grounding")
if not metadata.get("freeze_m1_policy"):
    raise SystemExit("Checkpoint does not protect the M1 policy")
print(version)
PY
)"
if [[ "${TC_VERSION}" == "5" ]]; then
  USE_CF_ACTION_POSITIVE=true
  USE_STATE_CONDITIONED_GROUNDING=false
elif [[ "${TC_VERSION}" == "6" ]]; then
  USE_CF_ACTION_POSITIVE=true
  USE_STATE_CONDITIONED_GROUNDING=true
else
  USE_CF_ACTION_POSITIVE=false
  USE_STATE_CONDITIONED_GROUNDING=false
fi
echo "Validated TC-Full v${TC_VERSION} checkpoint: ${TC_CHECKPOINT}"

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
"${PYTHON_BIN}" scripts/prepare_libero_object_interventions.py \
  --suite libero_object \
  --output "${MANIFEST_PATH}"
"${PYTHON_BIN}" scripts/validate_language_intervention_manifest.py \
  "${MANIFEST_PATH}"

echo "[TC-FastWAM] Running $((10 * NUM_TRIALS))-episode behavior-level CIS diagnostics"
echo "  checkpoint=${TC_CHECKPOINT}"
echo "  trials=${NUM_TRIALS} horizon=${MAX_POLICY_STEPS}"
echo "  output=${OUTPUT_ROOT}"

EXP_NAME="tc-v${TC_VERSION}-cis-diagnostics" \
"${PYTHON_BIN}" experiments/libero/run_libero_manager.py \
  task=libero_object_lf_lora_2cam224 \
  "ckpt=${TC_CHECKPOINT}" \
  "seed=${EVAL_SEED}" \
  "EVALUATION.dataset_stats_path=${STATS_PATH}" \
  "EVALUATION.num_trials=${NUM_TRIALS}" \
  "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}" \
  "EVALUATION.max_steps=${MAX_POLICY_STEPS}" \
  EVALUATION.instruction_condition=counterfactual \
  "EVALUATION.language_intervention_manifest=${MANIFEST_PATH}" \
  EVALUATION.counterfactual_diagnostics=true \
  "EVALUATION.output_dir=${OUTPUT_ROOT}" \
  'MULTIRUN.task_suite_names=[libero_object]' \
  "MULTIRUN.num_gpus=${NUM_GPUS}" \
  MULTIRUN.max_tasks_per_gpu=1 \
  model.action_dit_config.use_latent_action_queries=true \
  model.langforce_mvp.enabled=false \
  model.langforce_mvp.enable_prior=false \
  model.langforce_mvp.enable_posterior_advantage=false \
  model.transition_contract.enabled=true \
  "model.transition_contract.version=${TC_VERSION}" \
  model.transition_contract.use_action_effect=true \
  model.transition_contract.use_counterfactual_ranking=true \
  "model.transition_contract.use_counterfactual_action_positive=${USE_CF_ACTION_POSITIVE}" \
  "model.transition_contract.use_state_conditioned_grounding=${USE_STATE_CONDITIONED_GROUNDING}" \
  model.transition_contract.policy_distillation_enabled=true \
  model.transition_contract.policy_distillation_weight=1.0 \
  model.transition_contract.freeze_m1_policy=true

"${PYTHON_BIN}" scripts/summarize_counterfactual_behaviors.py \
  "${OUTPUT_ROOT}" \
  --expected-episodes "$((10 * NUM_TRIALS))" \
  --output-prefix "${OUTPUT_ROOT}/counterfactual_behavior_summary"

echo "[TC-FastWAM] CIS behavior diagnostics complete: ${OUTPUT_ROOT}"
