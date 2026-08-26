#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: bash scripts/eval_libero_lora_only_ablation.sh <suite> <checkpoint> [gpus] [trials] [seed] [inference_steps]}"
LORA_ONLY_CHECKPOINT="${2:?Missing LoRA-only checkpoint}"
NUM_GPUS="${3:-4}"
NUM_TRIALS="${4:-5}"
EVAL_SEED="${5:-42}"
NUM_INFERENCE_STEPS="${6:-10}"

case "${SUITE}" in
  libero_spatial|libero_object|libero_goal|libero_10) ;;
  *)
    echo "Unsupported LIBERO suite: ${SUITE}." >&2
    exit 1
    ;;
esac
for value_name in NUM_GPUS NUM_TRIALS EVAL_SEED NUM_INFERENCE_STEPS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value_name}" != "EVAL_SEED" && "${value}" == "0" ]]; then
    echo "${value_name} has an invalid value: ${value}." >&2
    exit 1
  fi
done

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
EXPECTED_STEP="${LORA_ONLY_EXPECTED_STEP:-10000}"
CONDITION="${LORA_ONLY_EVAL_CONDITION:-correct}"
MANIFEST_PATH="${LORA_ONLY_MANIFEST_PATH:-}"
STATS_PATH="${STATS_PATH:-${DIFFSYNTH_MODEL_BASE_PATH:-./checkpoints}/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(pwd)/evaluate_results/${SUITE}_lora_only_no_eraf_${CONDITION}_seed${EVAL_SEED}_trials${NUM_TRIALS}}"

case "${CONDITION}" in
  correct|null|shuffled|counterfactual) ;;
  *)
    echo "LORA_ONLY_EVAL_CONDITION must be correct, null, shuffled, or counterfactual." >&2
    exit 1
    ;;
esac
if [[ ! -f "${LORA_ONLY_CHECKPOINT}" ]]; then
  echo "LoRA-only checkpoint not found: ${LORA_ONLY_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${STATS_PATH}" ]]; then
  echo "Dataset stats not found: ${STATS_PATH}" >&2
  exit 1
fi
if [[ "${CONDITION}" == "shuffled" || "${CONDITION}" == "counterfactual" ]]; then
  if [[ -z "${MANIFEST_PATH}" || ! -f "${MANIFEST_PATH}" ]]; then
    echo "${CONDITION} evaluation requires LORA_ONLY_MANIFEST_PATH." >&2
    exit 1
  fi
  "${PYTHON_BIN}" scripts/validate_language_intervention_manifest.py "${MANIFEST_PATH}"
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required by the LIBERO evaluation manager." >&2
  exit 1
fi

"${PYTHON_BIN}" - "${LORA_ONLY_CHECKPOINT}" "${EXPECTED_STEP}" <<'PY'
import math
import sys

import torch

checkpoint, expected_step = sys.argv[1], int(sys.argv[2])
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
if payload.get("format") != "fastwam_lora_adapter_v1":
    raise SystemExit("Expected a generic FastWAM LoRA adapter checkpoint.")
if int(payload.get("step", -1)) != expected_step:
    raise SystemExit(
        f"Expected formal LoRA-only step {expected_step}, got {payload.get('step')}."
    )
if payload.get("transition_contract") is not None:
    raise SystemExit("LoRA-only checkpoint unexpectedly contains Transition Contract tensors.")
config = payload.get("lora_config")
if not isinstance(config, dict):
    raise SystemExit("LoRA-only checkpoint has no LoRA configuration.")
control = config.get("paired_language_control")
if not isinstance(control, dict) or control.get("enabled") is not True:
    raise SystemExit("Checkpoint is not the strict paired-language no-ERAF control.")
if set(config.get("experts", [])) != {"video", "action"}:
    raise SystemExit("Checkpoint does not contain shared Video+Action LoRA.")
if config.get("extra_trainable_patterns"):
    raise SystemExit("Checkpoint contains non-LoRA trainable tensors.")
expected = {"rank": 16, "alpha": 16.0, "dropout": 0.05}
for name, value in expected.items():
    actual = float(config.get(name, -1))
    if not math.isclose(actual, value, rel_tol=0.0, abs_tol=1.0e-12):
        raise SystemExit(f"LoRA {name} mismatch: expected={value} got={actual}")
PY

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

OVERRIDES=(
  task=libero_lora_only_2cam224
  "ckpt=${LORA_ONLY_CHECKPOINT}"
  "seed=${EVAL_SEED}"
  "EVALUATION.dataset_stats_path=${STATS_PATH}"
  "EVALUATION.num_trials=${NUM_TRIALS}"
  "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}"
  "EVALUATION.instruction_condition=${CONDITION}"
  "EVALUATION.output_dir=${OUTPUT_ROOT}"
  "MULTIRUN.task_suite_names=[${SUITE}]"
  "MULTIRUN.num_gpus=${NUM_GPUS}"
  MULTIRUN.max_tasks_per_gpu=1
  model.action_dit_config.use_latent_action_queries=false
  model.langforce_mvp.enabled=false
  model.langforce_mvp.enable_prior=false
  model.langforce_mvp.enable_posterior_advantage=false
  model.transition_contract.enabled=false
  model.policy_guard.enabled=false
  model.lora.enabled=false
  model.lora.paired_language_control.enabled=false
)
if [[ -n "${MANIFEST_PATH}" ]]; then
  OVERRIDES+=("EVALUATION.language_intervention_manifest=${MANIFEST_PATH}")
fi

echo "[FastWAM] LIBERO LoRA-only / no-ERAF evaluation"
echo "  suite=${SUITE} checkpoint=${LORA_ONLY_CHECKPOINT}"
echo "  condition=${CONDITION} trials=${NUM_TRIALS} seed=${EVAL_SEED}"
echo "  output=${OUTPUT_ROOT}"

EXP_NAME="lora-only-${CONDITION}" "${PYTHON_BIN}" \
  experiments/libero/run_libero_manager.py "${OVERRIDES[@]}"

echo "[FastWAM] evaluation complete: ${OUTPUT_ROOT}"
