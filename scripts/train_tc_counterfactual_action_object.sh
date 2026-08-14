#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_tc_counterfactual_action_object.sh <gpus> <tc_checkpoint> [seed] [max_steps]}"
INITIAL_CHECKPOINT="${2:?Missing protected TC-Full checkpoint}"
TRAIN_SEED="${3:-42}"
MAX_STEPS="${4:-4000}"
TC_CONTRACT_VERSION="${TC_CONTRACT_VERSION:-5}"
TC_USE_STATE_CONDITIONED_GROUNDING="${TC_USE_STATE_CONDITIONED_GROUNDING:-false}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_NAME="libero_object_lf_lora_2cam224"
OBJECT_DATASET="data/libero_mujoco3.3.2/libero_object_no_noops_lerobot"
DEFAULT_STATS_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-./checkpoints}/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
STATS_PATH="${STATS_PATH:-${DEFAULT_STATS_PATH}}"
MANIFEST_PATH="${TC_FULL_MANIFEST_PATH:-configs/eval/libero_object_tc_full_train.jsonl}"
NEGATIVE_PROBABILITY="${TC_FULL_NEGATIVE_PROBABILITY:-1.0}"
CONTRACT_WEIGHT="${TC_FULL_CONTRACT_WEIGHT:-0.05}"
ACTION_FUTURE_WEIGHT="${TC_FULL_ACTION_FUTURE_WEIGHT:-1.0}"
COUNTERFACTUAL_WEIGHT="${TC_FULL_COUNTERFACTUAL_WEIGHT:-0.05}"
COUNTERFACTUAL_MARGIN="${TC_FULL_COUNTERFACTUAL_MARGIN:-0.2}"
CAP_WEIGHT="${TC_CAP_WEIGHT:-0.10}"
CAP_QUERY_WEIGHT="${TC_CAP_QUERY_WEIGHT:-1.0}"
CAP_ACTION_EFFECT_WEIGHT="${TC_CAP_ACTION_EFFECT_WEIGHT:-1.0}"
CAP_SEPARATION_WEIGHT="${TC_CAP_SEPARATION_WEIGHT:-0.05}"
CAP_SEPARATION_MARGIN="${TC_CAP_SEPARATION_MARGIN:-0.05}"
CAP_PROTOTYPE_SLOTS="${TC_CAP_PROTOTYPE_SLOTS:-64}"
CAP_PROTOTYPE_MOMENTUM="${TC_CAP_PROTOTYPE_MOMENTUM:-0.95}"
STATE_GROUNDING_WEIGHT="${TC_STATE_GROUNDING_WEIGHT:-0.10}"
STATE_GROUNDING_CORRECT_WEIGHT="${TC_STATE_GROUNDING_CORRECT_WEIGHT:-1.0}"
STATE_GROUNDING_COUNTERFACTUAL_WEIGHT="${TC_STATE_GROUNDING_COUNTERFACTUAL_WEIGHT:-1.0}"
STATE_GROUNDING_SEPARATION_WEIGHT="${TC_STATE_GROUNDING_SEPARATION_WEIGHT:-0.25}"
STATE_GROUNDING_OVERLAP_MARGIN="${TC_STATE_GROUNDING_OVERLAP_MARGIN:-0.25}"
STATE_GROUNDING_ROUTER_BIAS="${TC_STATE_GROUNDING_ROUTER_BIAS:-2.0}"
STATE_GROUNDING_TEACHER_TOPK="${TC_STATE_GROUNDING_TEACHER_TOPK:-0.15}"
STATE_GROUNDING_TEACHER_TEMPERATURE="${TC_STATE_GROUNDING_TEACHER_TEMPERATURE:-0.25}"
STATE_GROUNDING_HIDDEN_DIM="${TC_STATE_GROUNDING_HIDDEN_DIM:-256}"
STATE_GROUNDING_TEMPERATURE="${TC_STATE_GROUNDING_TEMPERATURE:-0.07}"
STATE_GROUNDING_PROTOTYPE_SLOTS="${TC_STATE_GROUNDING_PROTOTYPE_SLOTS:-64}"
STATE_GROUNDING_PROTOTYPE_MOMENTUM="${TC_STATE_GROUNDING_PROTOTYPE_MOMENTUM:-0.95}"
STATE_GROUNDING_PROTOTYPE_TEMPERATURE="${TC_STATE_GROUNDING_PROTOTYPE_TEMPERATURE:-0.07}"
STATE_GROUNDING_PROTOTYPE_TOPK="${TC_STATE_GROUNDING_PROTOTYPE_TOPK:-0.10}"
POLICY_DISTILLATION_WEIGHT="${TC_POLICY_DISTILLATION_WEIGHT:-1.0}"
ACTION_EFFECT_HIDDEN_DIM="${TC_ACTION_EFFECT_HIDDEN_DIM:-512}"
ACTION_EFFECT_NUM_HEADS="${TC_ACTION_EFFECT_NUM_HEADS:-8}"
ACTION_EFFECT_NUM_LAYERS="${TC_ACTION_EFFECT_NUM_LAYERS:-2}"
TC_SAVE_TRAINING_STATE="${TC_SAVE_TRAINING_STATE:-false}"
TC_RESUME_STATE="${TC_RESUME_STATE:-}"
RUN_TAG="${RUN_TAG:-v${TC_CONTRACT_VERSION}-cf-action-positive-object-${MAX_STEPS}-seed${TRAIN_SEED}-v1}"
RUN_PREFIX="${TC_RUN_PREFIX:-tc-cap}"

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU process count must be positive, got: ${NPROC_PER_NODE}" >&2
  exit 1
fi
if ! [[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "max_steps must be a positive integer, got: ${MAX_STEPS}" >&2
  exit 1
fi
if [[ "${TC_CONTRACT_VERSION}" != "5" && "${TC_CONTRACT_VERSION}" != "6" ]]; then
  echo "TC_CONTRACT_VERSION must be 5 or 6, got: ${TC_CONTRACT_VERSION}" >&2
  exit 1
fi
if [[ "${TC_CONTRACT_VERSION}" == "6" && "${TC_USE_STATE_CONDITIONED_GROUNDING}" != "true" ]]; then
  echo "TC v6 requires TC_USE_STATE_CONDITIONED_GROUNDING=true." >&2
  exit 1
fi
if [[ ! -f "${INITIAL_CHECKPOINT}" ]]; then
  echo "TC initialization checkpoint not found: ${INITIAL_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${STATS_PATH}" ]]; then
  echo "Dataset stats not found: ${STATS_PATH}" >&2
  exit 1
fi
if [[ ! -f "${OBJECT_DATASET}/meta/tasks.jsonl" ]]; then
  echo "LIBERO-Object dataset is missing or incomplete: ${OBJECT_DATASET}" >&2
  exit 1
fi
if [[ -n "${TC_RESUME_STATE}" ]]; then
  if [[ ! -d "${TC_RESUME_STATE}" || ! -f "${TC_RESUME_STATE}/trainer_state.json" ]]; then
    echo "Training-state directory is incomplete: ${TC_RESUME_STATE}" >&2
    exit 1
  fi
  RESUME_TARGET="${TC_RESUME_STATE}"
else
  RESUME_TARGET="${INITIAL_CHECKPOINT}"
fi

mkdir -p "$(dirname "${MANIFEST_PATH}")"
"${PYTHON_BIN}" scripts/prepare_libero_object_interventions.py \
  --suite libero_object \
  --output "${MANIFEST_PATH}"
"${PYTHON_BIN}" scripts/validate_language_intervention_manifest.py \
  "${MANIFEST_PATH}"

CACHE_DIR="data/text_embeds_cache/libero"
MISSING_CACHE_COUNT="$("${PYTHON_BIN}" - "${OBJECT_DATASET}/meta/tasks.jsonl" "${MANIFEST_PATH}" "${CACHE_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

tasks_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
cache_dir = Path(sys.argv[3])
template = "A video recorded from a robot's point of view executing the following instruction: {task}"
tasks = set()
with tasks_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            tasks.add(str(json.loads(line)["task"]))
with manifest_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        record = json.loads(line)
        tasks.add(str(record["correct_instruction"]))
        tasks.add(str(record["counterfactual_instruction"]))
missing = 0
for task in sorted(tasks):
    prompt = template.format(task=task)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    missing += int(
        not (cache_dir / f"{digest}.t5_len128.wan22ti2v5b.pt").is_file()
    )
print(missing)
PY
)"
if (( MISSING_CACHE_COUNT > 0 )); then
  echo "Missing ${MISSING_CACHE_COUNT} text caches under ${CACHE_DIR}." >&2
  echo "Run scripts/precompute_text_embeds.py for the Object task first." >&2
  exit 1
fi

CHECKPOINT_KIND="$("${PYTHON_BIN}" - "${INITIAL_CHECKPOINT}" "${TC_CONTRACT_VERSION}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
target_version = int(sys.argv[2])
metadata = payload.get("architecture_metadata") or {}
if payload.get("format") != "fastwam_lora_adapter_v1":
    raise SystemExit(f"TC v{target_version} requires a FastWAM LoRA adapter checkpoint")
if metadata.get("architecture") != "tc_fastwam":
    raise SystemExit(f"TC v{target_version} must initialize from a protected TC checkpoint")
version = int(metadata.get("transition_contract_version", -1))
allowed = {5: {4, 5}, 6: {5, 6}}[target_version]
if version not in allowed:
    expected = "/".join(f"v{item}" for item in sorted(allowed))
    raise SystemExit(
        f"TC v{target_version} requires {expected} initialization, got v{version}"
    )
if not metadata.get("use_action_effect") or not metadata.get("use_cf_ranking"):
    raise SystemExit("Initialization checkpoint is not TC-Full")
if not metadata.get("freeze_m1_policy"):
    raise SystemExit("Initialization checkpoint does not protect the M1 policy")
if not metadata.get("policy_distillation_enabled"):
    raise SystemExit("Initialization checkpoint has no M1 distillation")
if not payload.get("transition_contract"):
    raise SystemExit("Initialization checkpoint has no Transition Contract tensors")
if version >= 5 and not metadata.get("use_cf_action_positive"):
    raise SystemExit(f"TC v{version} checkpoint is missing action-positive metadata")
if version == 6 and not metadata.get("use_state_conditioned_grounding"):
    raise SystemExit("TC v6 checkpoint is missing state-grounding metadata")
print(f"tc_v{version}_protected_adapter")
PY
)"

VISIBLE_GPU_COUNT="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if (( VISIBLE_GPU_COUNT < NPROC_PER_NODE )); then
  echo "Requested ${NPROC_PER_NODE} GPUs, PyTorch sees ${VISIBLE_GPU_COUNT}." >&2
  exit 1
fi

echo "[TC-FastWAM] TC-Full v${TC_CONTRACT_VERSION} counterfactual training"
echo "  checkpoint=${INITIAL_CHECKPOINT}"
echo "  checkpoint_kind=${CHECKPOINT_KIND}"
echo "  dataset=${OBJECT_DATASET}"
echo "  manifest=${MANIFEST_PATH}"
echo "  counterfactual_probability=${NEGATIVE_PROBABILITY}"
echo "  seed=${TRAIN_SEED} max_steps=${MAX_STEPS}"
echo "  contract=${CONTRACT_WEIGHT} AF=${ACTION_FUTURE_WEIGHT} CF=${COUNTERFACTUAL_WEIGHT}"
echo "  CAP=${CAP_WEIGHT} query=${CAP_QUERY_WEIGHT} action_effect=${CAP_ACTION_EFFECT_WEIGHT}"
echo "  separation=${CAP_SEPARATION_WEIGHT} margin=${CAP_SEPARATION_MARGIN}"
echo "  prototype_slots=${CAP_PROTOTYPE_SLOTS} momentum=${CAP_PROTOTYPE_MOMENTUM}"
echo "  state_grounding=${TC_USE_STATE_CONDITIONED_GROUNDING} weight=${STATE_GROUNDING_WEIGHT} router_bias=${STATE_GROUNDING_ROUTER_BIAS}"
echo "  state_teacher=clean_latent_change topk=${STATE_GROUNDING_TEACHER_TOPK} temperature=${STATE_GROUNDING_TEACHER_TEMPERATURE}"
echo "  policy_distillation=${POLICY_DISTILLATION_WEIGHT}"
echo "  save_training_state=${TC_SAVE_TRAINING_STATE} resume=${RESUME_TARGET}"
echo "  method=V4 contracts + V5 action positives + optional V6 current-state target grounding"

RUN_ID="${RUN_PREFIX}-${RUN_TAG}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  "task=${TASK_NAME}" \
  "resume=${RESUME_TARGET}" \
  "data.train.pretrained_norm_stats=${STATS_PATH}" \
  "data.train.counterfactual_manifest_path=${MANIFEST_PATH}" \
  "data.train.counterfactual_negative_probability=${NEGATIVE_PROBABILITY}" \
  "seed=${TRAIN_SEED}" \
  "max_steps=${MAX_STEPS}" \
  num_epochs=1 \
  save_every=500 \
  "save_training_state=${TC_SAVE_TRAINING_STATE}" \
  model.action_dit_config.use_latent_action_queries=true \
  model.langforce_mvp.enabled=false \
  model.langforce_mvp.enable_prior=false \
  model.langforce_mvp.enable_posterior_advantage=false \
  model.transition_contract.enabled=true \
  "model.transition_contract.version=${TC_CONTRACT_VERSION}" \
  model.transition_contract.use_action_effect=true \
  model.transition_contract.use_counterfactual_ranking=true \
  model.transition_contract.use_counterfactual_action_positive=true \
  "model.transition_contract.use_state_conditioned_grounding=${TC_USE_STATE_CONDITIONED_GROUNDING}" \
  "model.transition_contract.contract_weight=${CONTRACT_WEIGHT}" \
  "model.transition_contract.action_future_weight=${ACTION_FUTURE_WEIGHT}" \
  "model.transition_contract.counterfactual_weight=${COUNTERFACTUAL_WEIGHT}" \
  "model.transition_contract.counterfactual_margin=${COUNTERFACTUAL_MARGIN}" \
  "model.transition_contract.counterfactual_action_positive_weight=${CAP_WEIGHT}" \
  "model.transition_contract.counterfactual_action_query_weight=${CAP_QUERY_WEIGHT}" \
  "model.transition_contract.counterfactual_action_effect_weight=${CAP_ACTION_EFFECT_WEIGHT}" \
  "model.transition_contract.counterfactual_action_separation_weight=${CAP_SEPARATION_WEIGHT}" \
  "model.transition_contract.counterfactual_action_separation_margin=${CAP_SEPARATION_MARGIN}" \
  "model.transition_contract.counterfactual_action_prototype_slots=${CAP_PROTOTYPE_SLOTS}" \
  "model.transition_contract.counterfactual_action_prototype_momentum=${CAP_PROTOTYPE_MOMENTUM}" \
  "model.transition_contract.state_grounding_weight=${STATE_GROUNDING_WEIGHT}" \
  "model.transition_contract.state_grounding_correct_weight=${STATE_GROUNDING_CORRECT_WEIGHT}" \
  "model.transition_contract.state_grounding_counterfactual_weight=${STATE_GROUNDING_COUNTERFACTUAL_WEIGHT}" \
  "model.transition_contract.state_grounding_separation_weight=${STATE_GROUNDING_SEPARATION_WEIGHT}" \
  "model.transition_contract.state_grounding_overlap_margin=${STATE_GROUNDING_OVERLAP_MARGIN}" \
  "model.transition_contract.state_grounding_router_bias=${STATE_GROUNDING_ROUTER_BIAS}" \
  "model.transition_contract.state_grounding_teacher_topk=${STATE_GROUNDING_TEACHER_TOPK}" \
  "model.transition_contract.state_grounding_teacher_temperature=${STATE_GROUNDING_TEACHER_TEMPERATURE}" \
  "model.transition_contract.state_grounding_hidden_dim=${STATE_GROUNDING_HIDDEN_DIM}" \
  "model.transition_contract.state_grounding_temperature=${STATE_GROUNDING_TEMPERATURE}" \
  "model.transition_contract.state_grounding_prototype_slots=${STATE_GROUNDING_PROTOTYPE_SLOTS}" \
  "model.transition_contract.state_grounding_prototype_momentum=${STATE_GROUNDING_PROTOTYPE_MOMENTUM}" \
  "model.transition_contract.state_grounding_prototype_temperature=${STATE_GROUNDING_PROTOTYPE_TEMPERATURE}" \
  "model.transition_contract.state_grounding_prototype_topk=${STATE_GROUNDING_PROTOTYPE_TOPK}" \
  "model.transition_contract.action_effect_hidden_dim=${ACTION_EFFECT_HIDDEN_DIM}" \
  "model.transition_contract.action_effect_num_heads=${ACTION_EFFECT_NUM_HEADS}" \
  "model.transition_contract.action_effect_num_layers=${ACTION_EFFECT_NUM_LAYERS}" \
  model.transition_contract.policy_distillation_enabled=true \
  "model.transition_contract.policy_distillation_weight=${POLICY_DISTILLATION_WEIGHT}" \
  model.transition_contract.freeze_m1_policy=true

echo "[TC-FastWAM] TC-Full v${TC_CONTRACT_VERSION} training complete."
