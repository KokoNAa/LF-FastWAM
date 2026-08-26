#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: bash scripts/train_libero_lora_only_ablation.sh <suite> <gpus> <released_base> <historical_cf_dataset> <strict_cf_dataset> <native_sidecar> <historical_cf_sidecar> <strict_cf_sidecar> [seed]}"
NPROC_PER_NODE="${2:?Missing GPU count}"
BASE_CHECKPOINT="${3:?Missing released FastWAM checkpoint}"
HISTORICAL_CF_DATASET="${4:?Missing historical counterfactual dataset}"
STRICT_CF_DATASET="${5:?Missing strict counterfactual dataset}"
NATIVE_SIDECAR="${6:?Missing native ERAF sidecar}"
HISTORICAL_CF_SIDECAR="${7:?Missing historical-CF ERAF sidecar}"
STRICT_CF_SIDECAR="${8:?Missing strict-CF ERAF sidecar}"
TRAIN_SEED="${9:-42}"

case "${SUITE}" in
  libero_spatial|libero_object|libero_goal|libero_10) ;;
  *)
    echo "Unsupported LIBERO suite: ${SUITE}." >&2
    exit 1
    ;;
esac
if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU process count must be a positive integer." >&2
  exit 1
fi
if ! [[ "${TRAIN_SEED}" =~ ^[0-9]+$ ]]; then
  echo "Seed must be a non-negative integer." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/root/gpufree-data/fastwam/FastWAM/data/libero_mujoco3.3.2}"
NATIVE_DATASET="${LIBERO_DATA_ROOT}/${SUITE}_no_noops_lerobot"
CLOSED_LOOP_DATASET="${PGC_V9_CLOSED_LOOP_GROUNDING_DATASET:-}"
CLOSED_LOOP_SIDECAR="${PGC_V9_CLOSED_LOOP_GROUNDING_SIDECAR:-}"
STATS_PATH="${STATS_PATH:-${DIFFSYNTH_MODEL_BASE_PATH:-./checkpoints}/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
CACHE_DIR="${TEXT_CACHE_DIR:-data/text_embeds_cache/libero}"
MAX_STEPS="${LORA_ONLY_MAX_STEPS:-10000}"
LEARNING_RATE="${LORA_ONLY_LEARNING_RATE:-5.0e-6}"
GRADIENT_ACCUMULATION_STEPS="${LORA_ONLY_GRADIENT_ACCUMULATION_STEPS:-4}"
SAVE_EVERY="${LORA_ONLY_SAVE_EVERY:-250}"

for value_name in MAX_STEPS GRADIENT_ACCUMULATION_STEPS SAVE_EVERY; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got ${value}." >&2
    exit 1
  fi
done
if [[ -z "${CLOSED_LOOP_DATASET}" || -z "${CLOSED_LOOP_SIDECAR}" ]]; then
  echo "Set PGC_V9_CLOSED_LOOP_GROUNDING_DATASET and PGC_V9_CLOSED_LOOP_GROUNDING_SIDECAR." >&2
  exit 1
fi

for file_path in "${BASE_CHECKPOINT}" "${STATS_PATH}"; do
  if [[ ! -f "${file_path}" ]]; then
    echo "Required file not found: ${file_path}" >&2
    exit 1
  fi
done
for dataset_path in \
  "${NATIVE_DATASET}" \
  "${CLOSED_LOOP_DATASET}" \
  "${HISTORICAL_CF_DATASET}" \
  "${STRICT_CF_DATASET}"; do
  if [[ ! -f "${dataset_path}/meta/tasks.jsonl" ]]; then
    echo "Invalid or missing dataset: ${dataset_path}" >&2
    exit 1
  fi
done
for sidecar_path in \
  "${NATIVE_SIDECAR}" \
  "${CLOSED_LOOP_SIDECAR}" \
  "${HISTORICAL_CF_SIDECAR}" \
  "${STRICT_CF_SIDECAR}"; do
  if [[ ! -f "${sidecar_path}/index.json" ]]; then
    echo "Invalid or missing sidecar: ${sidecar_path}" >&2
    exit 1
  fi
done
if [[ ! -d "${CACHE_DIR}" ]]; then
  echo "Text embedding cache directory not found: ${CACHE_DIR}" >&2
  exit 1
fi

VISIBLE_GPU_COUNT="$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count())')"
if (( VISIBLE_GPU_COUNT < NPROC_PER_NODE )); then
  echo "Requested ${NPROC_PER_NODE} processes, but PyTorch sees ${VISIBLE_GPU_COUNT} CUDA devices." >&2
  exit 1
fi

resolve_dir() {
  (cd -- "$1" && pwd -P)
}
resolve_file() {
  local parent
  parent="$(cd -- "$(dirname -- "$1")" && pwd -P)"
  printf '%s/%s\n' "${parent}" "$(basename -- "$1")"
}

BASE_CHECKPOINT="$(resolve_file "${BASE_CHECKPOINT}")"
STATS_PATH="$(resolve_file "${STATS_PATH}")"
NATIVE_DATASET="$(resolve_dir "${NATIVE_DATASET}")"
CLOSED_LOOP_DATASET="$(resolve_dir "${CLOSED_LOOP_DATASET}")"
HISTORICAL_CF_DATASET="$(resolve_dir "${HISTORICAL_CF_DATASET}")"
STRICT_CF_DATASET="$(resolve_dir "${STRICT_CF_DATASET}")"
NATIVE_SIDECAR="$(resolve_dir "${NATIVE_SIDECAR}")"
CLOSED_LOOP_SIDECAR="$(resolve_dir "${CLOSED_LOOP_SIDECAR}")"
HISTORICAL_CF_SIDECAR="$(resolve_dir "${HISTORICAL_CF_SIDECAR}")"
STRICT_CF_SIDECAR="$(resolve_dir "${STRICT_CF_SIDECAR}")"
CACHE_DIR="$(resolve_dir "${CACHE_DIR}")"

"${PYTHON_BIN}" - \
  "${BASE_CHECKPOINT}" \
  "${NATIVE_DATASET}" "${CLOSED_LOOP_DATASET}" \
  "${HISTORICAL_CF_DATASET}" "${STRICT_CF_DATASET}" \
  "${NATIVE_SIDECAR}" "${CLOSED_LOOP_SIDECAR}" \
  "${HISTORICAL_CF_SIDECAR}" "${STRICT_CF_SIDECAR}" <<'PY'
import json
import math
import pathlib
import sys

import torch

base_checkpoint = pathlib.Path(sys.argv[1]).resolve()
datasets = [pathlib.Path(value).resolve() for value in sys.argv[2:6]]
sidecars = [pathlib.Path(value).resolve() for value in sys.argv[6:10]]

payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
if str(payload.get("format", "")).startswith("fastwam_policy_guard_"):
    raise SystemExit("LoRA-only must start from the released Base, not a PGC checkpoint.")
if not isinstance(payload.get("mot"), dict) and not isinstance(payload.get("dit"), dict):
    raise SystemExit("Released Base checkpoint must contain `mot` or `dit` weights.")

expected_contracts = [
    ("native", "fastwam_gripper_open_1_close_0", "fastwam_to_libero_env"),
    ("native", "fastwam_gripper_open_1_close_0", "fastwam_to_libero_env"),
    ("counterfactual", "libero_env_gripper_open_minus1_close_plus1", "identity"),
    ("counterfactual", "libero_env_gripper_open_minus1_close_plus1", "identity"),
]
workspace_contracts = []
for index, (dataset, sidecar, expected) in enumerate(
    zip(datasets, sidecars, expected_contracts, strict=True)
):
    sidecar_index = json.loads((sidecar / "index.json").read_text(encoding="utf-8"))
    if sidecar_index.get("format") != "pgc_libero_entity_relation_v1":
        raise SystemExit(f"Invalid sidecar format: {sidecar / 'index.json'}")
    if pathlib.Path(sidecar_index["dataset"]).resolve() != dataset:
        raise SystemExit(f"Sidecar does not bind dataset {dataset}: {sidecar}")
    actual = (
        sidecar_index.get("dataset_kind"),
        sidecar_index.get("dataset_action_convention"),
        sidecar_index.get("simulator_replay_action_transform"),
    )
    if actual != expected:
        raise SystemExit(
            f"Sidecar action contract mismatch for {sidecar}: expected={expected} got={actual}"
        )
    lower = tuple(float(value) for value in sidecar_index["workspace_min"])
    upper = tuple(float(value) for value in sidecar_index["workspace_max"])
    if len(lower) != 3 or len(upper) != 3:
        raise SystemExit(f"Invalid workspace bounds in {sidecar / 'index.json'}")
    workspace_contracts.append((sidecar, lower, upper))
    if index == 1 and sidecar_index.get("state_distribution") != "immutable_base_closed_loop_replan":
        raise SystemExit("Closed-loop native sidecar has the wrong state-distribution contract.")

_, reference_min, reference_max = workspace_contracts[0]
for sidecar, lower, upper in workspace_contracts[1:]:
    aligned = all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-6)
        for left, right in zip(lower + upper, reference_min + reference_max)
    )
    if not aligned:
        raise SystemExit(f"Workspace contract mismatch: {sidecar}")
PY

json_array() {
  "${PYTHON_BIN}" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"
}
NATIVE_JSON="$(json_array "${NATIVE_DATASET}" "${CLOSED_LOOP_DATASET}")"
CF_JSON="$(json_array "${HISTORICAL_CF_DATASET}" "${STRICT_CF_DATASET}")"
SIDECAR_JSON="$(json_array "${NATIVE_SIDECAR}" "${CLOSED_LOOP_SIDECAR}" "${HISTORICAL_CF_SIDECAR}" "${STRICT_CF_SIDECAR}")"

RUN_TAG="${RUN_TAG:-${SUITE}-lora-only-no-eraf-10k-seed${TRAIN_SEED}-v1}"
echo "[FastWAM] strict LIBERO LoRA-only / no-ERAF ablation"
echo "  suite=${SUITE} steps=${MAX_STEPS} seed=${TRAIN_SEED}"
echo "  base=${BASE_CHECKPOINT}"
echo "  mixture=offline_native:closed_loop_native:historical_cf:strict_cf 1:1:1:1"
echo "  lora=video+action rank16 alpha16 dropout0.05; extra_trainables=none"
echo "  objectives=world_flow+world_language_rank+native_action+counterfactual_action+action_language_rank+lora_reg"
echo "  removed=ERAF+context_injector+completion_memory+ERAF_preservation+policy_guard"
echo "  optimizer=AdamW lr=${LEARNING_RATE} cosine weight_decay=1e-2 grad_accum=${GRADIENT_ACCUMULATION_STEPS}"

RUN_ID="lora-only-${RUN_TAG}" exec bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task=libero_lora_only_2cam224 \
  "resume=${BASE_CHECKPOINT}" \
  "data.train.dataset_dirs=${NATIVE_JSON}" \
  "data.train.pgc_counterfactual_dataset_dirs=${CF_JSON}" \
  data.train.pgc_closed_loop_corrective_dataset_dirs=[] \
  data.train.pgc_counterfactual_oversample_factor=1 \
  data.train.pgc_balance_native_counterfactual=true \
  data.train.pgc_entity_relation_supervision_required=true \
  "data.train.pgc_entity_relation_sidecar_dirs=${SIDECAR_JSON}" \
  data.train.pgc_v9_balanced_sampling=true \
  data.train.pgc_v9_structured_role_sampling=false \
  data.train.pgc_v9_hard_role_curriculum=false \
  data.train.pgc_v9_hard_role_index_path=null \
  data.train.pgc_v9_closed_loop_rebinding=false \
  data.train.pgc_v9_phase_safe_memory=true \
  data.train.pgc_v9_closed_loop_native_dataset_count=1 \
  "++data.train.pretrained_norm_stats=${STATS_PATH}" \
  "data.train.text_embedding_cache_dir=${CACHE_DIR}" \
  "seed=${TRAIN_SEED}" \
  "max_steps=${MAX_STEPS}" \
  num_epochs=1 \
  "learning_rate=${LEARNING_RATE}" \
  "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
  "save_every=${SAVE_EVERY}" \
  save_training_state=false \
  model.action_dit_config.use_latent_action_queries=false \
  model.langforce_mvp.enabled=false \
  model.langforce_mvp.enable_prior=false \
  model.langforce_mvp.enable_posterior_advantage=false \
  model.transition_contract.enabled=false \
  model.policy_guard.enabled=false \
  model.lora.enabled=true \
  model.lora.rank=16 \
  model.lora.alpha=16 \
  model.lora.dropout=0.05 \
  'model.lora.experts=[video,action]' \
  'model.lora.extra_trainable_patterns=[]' \
  model.lora.paired_language_control.enabled=true \
  model.lora.paired_language_control.world_language_weight=0.10 \
  model.lora.paired_language_control.world_language_margin=0.01 \
  model.lora.paired_language_control.native_action_weight=1.0 \
  model.lora.paired_language_control.counterfactual_action_weight=1.0 \
  model.lora.paired_language_control.action_language_weight=1.0 \
  model.lora.paired_language_control.action_language_margin=0.01 \
  model.lora.paired_language_control.regularization_weight=1.0e-6
