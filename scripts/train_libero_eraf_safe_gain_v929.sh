#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: bash scripts/train_libero_eraf_safe_gain_v929.sh <suite> <gpus> <v928_checkpoint> <historical_cf_dataset> <strict_cf_dataset> <closed_loop_cf_dataset> <native_sidecar> <closed_loop_native_dataset> <closed_loop_native_sidecar> <historical_cf_sidecar> <strict_cf_sidecar> <closed_loop_cf_sidecar> [seed]}"
NPROC_PER_NODE="${2:?Missing GPU count}"
V928_CHECKPOINT="${3:?Missing completed V9.28 checkpoint}"
HISTORICAL_CF_DATASET="${4:?Missing historical counterfactual dataset}"
STRICT_CF_DATASET="${5:?Missing strict counterfactual dataset}"
CLOSED_LOOP_CF_DATASET="${6:?Missing replay-verified closed-loop counterfactual dataset}"
NATIVE_SIDECAR="${7:?Missing native ERAF sidecar}"
CLOSED_LOOP_NATIVE_DATASET="${8:?Missing closed-loop native memory dataset}"
CLOSED_LOOP_NATIVE_SIDECAR="${9:?Missing closed-loop native ERAF sidecar}"
HISTORICAL_CF_SIDECAR="${10:?Missing historical-CF ERAF sidecar}"
STRICT_CF_SIDECAR="${11:?Missing strict-CF ERAF sidecar}"
CLOSED_LOOP_CF_SIDECAR="${12:?Missing closed-loop-CF ERAF sidecar}"
TRAIN_SEED="${13:-42}"

case "${SUITE}" in
  libero_spatial|libero_object|libero_goal|libero_10) ;;
  *) echo "Unsupported LIBERO suite: ${SUITE}." >&2; exit 1 ;;
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
STATS_PATH="${STATS_PATH:-${DIFFSYNTH_MODEL_BASE_PATH:-./checkpoints}/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
CACHE_DIR="${TEXT_CACHE_DIR:-data/text_embeds_cache/libero}"
MAX_STEPS="${ERAF_SAFE_GAIN_MAX_STEPS:-10000}"
SIDE_MODULE_LEARNING_RATE="${ERAF_SAFE_GAIN_LEARNING_RATE:-2.0e-5}"
GRADIENT_ACCUMULATION_STEPS="${ERAF_SAFE_GAIN_GRADIENT_ACCUMULATION_STEPS:-4}"
SAVE_EVERY="${ERAF_SAFE_GAIN_SAVE_EVERY:-250}"
VARIANT="${ERAF_SAFE_GAIN_VARIANT:-v929}"
VARIANT_OVERRIDES=()
case "${VARIANT}" in
  v929) ;;
  v930)
    if ! [[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || (( MAX_STEPS > 500 )); then
      echo "V9.30 is a short 1..500-step probe; evaluate before extending training." >&2
      exit 1
    fi
    VARIANT_OVERRIDES+=(
      "model.policy_guard.entity_relation_grounding.safe_gain_injector_training_steps=${MAX_STEPS}"
      "model.policy_guard.entity_relation_grounding.preservation_weight=${ERAF_PRESERVATION_WEIGHT:-1.0}"
    )
    ;;
  *) echo "Unsupported safe-gain variant: ${VARIANT}" >&2; exit 1 ;;
esac

if [[ "${VARIANT}" == v929 && "${MAX_STEPS}" != "10000" ]]; then
  echo "PGC V9.29 has a fixed 7000+3000=10000-step schedule." >&2
  exit 1
fi
for file_path in "${V928_CHECKPOINT}" "${STATS_PATH}"; do
  [[ -f "${file_path}" ]] || { echo "Required file not found: ${file_path}" >&2; exit 1; }
done
for dataset_path in \
  "${NATIVE_DATASET}" \
  "${CLOSED_LOOP_NATIVE_DATASET}" \
  "${HISTORICAL_CF_DATASET}" \
  "${STRICT_CF_DATASET}" \
  "${CLOSED_LOOP_CF_DATASET}"; do
  [[ -f "${dataset_path}/meta/tasks.jsonl" ]] || {
    echo "Invalid or missing dataset: ${dataset_path}" >&2
    exit 1
  }
done
[[ -f "${CLOSED_LOOP_CF_DATASET}/meta/pgc_v8_closed_loop/index.json" ]] || {
  echo "Closed-loop CF dataset is not replay-verified PGC V8 data: ${CLOSED_LOOP_CF_DATASET}" >&2
  exit 1
}
for sidecar_path in \
  "${NATIVE_SIDECAR}" \
  "${CLOSED_LOOP_NATIVE_SIDECAR}" \
  "${HISTORICAL_CF_SIDECAR}" \
  "${STRICT_CF_SIDECAR}" \
  "${CLOSED_LOOP_CF_SIDECAR}"; do
  [[ -f "${sidecar_path}/index.json" ]] || {
    echo "Invalid or missing sidecar: ${sidecar_path}" >&2
    exit 1
  }
done
[[ -d "${CACHE_DIR}" ]] || {
  echo "Text embedding cache directory not found: ${CACHE_DIR}" >&2
  exit 1
}

resolve_dir() { (cd -- "$1" && pwd -P); }
resolve_file() {
  local parent
  parent="$(cd -- "$(dirname -- "$1")" && pwd -P)"
  printf '%s/%s\n' "${parent}" "$(basename -- "$1")"
}

V928_CHECKPOINT="$(resolve_file "${V928_CHECKPOINT}")"
STATS_PATH="$(resolve_file "${STATS_PATH}")"
NATIVE_DATASET="$(resolve_dir "${NATIVE_DATASET}")"
CLOSED_LOOP_NATIVE_DATASET="$(resolve_dir "${CLOSED_LOOP_NATIVE_DATASET}")"
HISTORICAL_CF_DATASET="$(resolve_dir "${HISTORICAL_CF_DATASET}")"
STRICT_CF_DATASET="$(resolve_dir "${STRICT_CF_DATASET}")"
CLOSED_LOOP_CF_DATASET="$(resolve_dir "${CLOSED_LOOP_CF_DATASET}")"
NATIVE_SIDECAR="$(resolve_dir "${NATIVE_SIDECAR}")"
CLOSED_LOOP_NATIVE_SIDECAR="$(resolve_dir "${CLOSED_LOOP_NATIVE_SIDECAR}")"
HISTORICAL_CF_SIDECAR="$(resolve_dir "${HISTORICAL_CF_SIDECAR}")"
STRICT_CF_SIDECAR="$(resolve_dir "${STRICT_CF_SIDECAR}")"
CLOSED_LOOP_CF_SIDECAR="$(resolve_dir "${CLOSED_LOOP_CF_SIDECAR}")"
CACHE_DIR="$(resolve_dir "${CACHE_DIR}")"

"${PYTHON_BIN}" - \
  "${V928_CHECKPOINT}" \
  "${NATIVE_DATASET}" "${CLOSED_LOOP_NATIVE_DATASET}" \
  "${HISTORICAL_CF_DATASET}" "${STRICT_CF_DATASET}" "${CLOSED_LOOP_CF_DATASET}" \
  "${NATIVE_SIDECAR}" "${CLOSED_LOOP_NATIVE_SIDECAR}" \
  "${HISTORICAL_CF_SIDECAR}" "${STRICT_CF_SIDECAR}" "${CLOSED_LOOP_CF_SIDECAR}" <<'PY'
import json
import math
import pathlib
import sys

import torch

checkpoint = pathlib.Path(sys.argv[1]).resolve()
datasets = [pathlib.Path(value).resolve() for value in sys.argv[2:7]]
sidecars = [pathlib.Path(value).resolve() for value in sys.argv[7:12]]

payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
metadata = payload.get("architecture_metadata") or {}
if payload.get("format") != "fastwam_policy_guard_v9":
    raise SystemExit("V9.29 must warm-start a PGC V9 policy checkpoint.")
if int(metadata.get("eraf_grounding_objective_version", -1)) != 28:
    raise SystemExit("V9.29 requires the completed objective-28 safe-gain checkpoint.")
if not (
    metadata.get("eraf_safe_gain_training") is True
    and metadata.get("eraf_single_path") is True
    and metadata.get("eraf_post_action_residual_active") is False
    and metadata.get("eraf_safe_gain_gate_supervision_contract")
    == "correct_advantage_bce_plus_wrong_language_rejection_bce_plus_positive_pair_logit_margin"
):
    raise SystemExit("V9.28 checkpoint lacks the corrected single-path safe-gain contract.")
if int(payload.get("step", metadata.get("step", -1))) < 10000:
    raise SystemExit("V9.29 requires the completed V9.28 step-10000 checkpoint.")

expected = [
    ("native", "fastwam_gripper_open_1_close_0", "fastwam_to_libero_env"),
    ("native", "fastwam_gripper_open_1_close_0", "fastwam_to_libero_env"),
    ("counterfactual", "libero_env_gripper_open_minus1_close_plus1", "identity"),
    ("counterfactual", "libero_env_gripper_open_minus1_close_plus1", "identity"),
    ("counterfactual", "libero_env_gripper_open_minus1_close_plus1", "identity"),
]
workspace = []
for index_value, (dataset, sidecar, contract) in enumerate(
    zip(datasets, sidecars, expected, strict=True)
):
    index = json.loads((sidecar / "index.json").read_text(encoding="utf-8"))
    if pathlib.Path(index["dataset"]).resolve() != dataset:
        raise SystemExit(f"Sidecar does not bind dataset {dataset}: {sidecar}")
    actual = (
        index.get("dataset_kind"),
        index.get("dataset_action_convention"),
        index.get("simulator_replay_action_transform"),
    )
    if actual != contract:
        raise SystemExit(f"Sidecar action contract mismatch: {sidecar}")
    if index_value == 1 and index.get("state_distribution") != (
        "immutable_base_closed_loop_replan"
    ):
        raise SystemExit("Closed-loop native sidecar lacks its state distribution contract.")
    workspace.append(
        (
            tuple(float(value) for value in index["workspace_min"]),
            tuple(float(value) for value in index["workspace_max"]),
        )
    )
reference = workspace[0]
for current in workspace[1:]:
    if not all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-6)
        for left, right in zip(current[0] + current[1], reference[0] + reference[1])
    ):
        raise SystemExit("Workspace contract mismatch in V9.29 sidecars.")
PY

VISIBLE_GPU_COUNT="$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count())')"
if (( VISIBLE_GPU_COUNT < NPROC_PER_NODE )); then
  echo "Requested ${NPROC_PER_NODE} processes, but PyTorch sees ${VISIBLE_GPU_COUNT} CUDA devices." >&2
  exit 1
fi

json_array() {
  "${PYTHON_BIN}" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"
}
NATIVE_JSON="$(json_array "${NATIVE_DATASET}" "${CLOSED_LOOP_NATIVE_DATASET}")"
CF_JSON="$(json_array "${HISTORICAL_CF_DATASET}" "${STRICT_CF_DATASET}")"
CORRECTIVE_JSON="$(json_array "${CLOSED_LOOP_CF_DATASET}")"
SIDECAR_JSON="$(json_array \
  "${NATIVE_SIDECAR}" "${CLOSED_LOOP_NATIVE_SIDECAR}" \
  "${HISTORICAL_CF_SIDECAR}" "${STRICT_CF_SIDECAR}" \
  "${CLOSED_LOOP_CF_SIDECAR}")"

if [[ "${VARIANT}" == v929 ]]; then
  RUN_TAG="${RUN_TAG:-${SUITE}-eraf-safe-gain-rollout-repair-10k-seed${TRAIN_SEED}-v929}"
else
  RUN_TAG="${RUN_TAG:-${SUITE}-eraf-safe-gain-${VARIANT}-${MAX_STEPS}-seed${TRAIN_SEED}}"
fi
echo "[FastWAM] LIBERO ${VARIANT} ERAF safe-gain training"
echo "  warm_start=${V928_CHECKPOINT}"
echo "  productive_mix=offline-native:historical-CF:strict-CF:closed-loop-CF=1:1:1:1"
if [[ "${VARIANT}" == v930 ]]; then
  echo "  schedule=injector-only ${MAX_STEPS}; fixed V9.28 teacher; gate excluded from optimizer"
  echo "  preservation=teacher-no-worse-than-base proxy on noncorrective rows; not rollout safety"
else
  echo "  schedule=injector-multinoise[0,7000)+detached-gate[7000,10000)"
fi
echo "  frozen=no-ERAF-Video/Action-LoRA+GoalGraph+complete-ERAF"
echo "  deployment=one pre-action gate then one Action Expert denoising path"

RUN_ID="eraf-safe-gain-${VARIANT}-${RUN_TAG}" exec bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  "task=libero_eraf_safe_gain_${VARIANT}_2cam224" \
  "resume=${V928_CHECKPOINT}" \
  "data.train.dataset_dirs=${NATIVE_JSON}" \
  "data.train.pgc_counterfactual_dataset_dirs=${CF_JSON}" \
  "data.train.pgc_closed_loop_corrective_dataset_dirs=${CORRECTIVE_JSON}" \
  data.train.pgc_counterfactual_oversample_factor=1 \
  data.train.pgc_closed_loop_corrective_oversample_factor=1 \
  data.train.pgc_balance_native_counterfactual=true \
  data.train.pgc_entity_relation_supervision_required=true \
  data.train.pgc_bidirectional_language_supervision_required=true \
  "data.train.pgc_entity_relation_sidecar_dirs=${SIDECAR_JSON}" \
  data.train.pgc_v9_balanced_sampling=true \
  data.train.pgc_v9_structured_role_sampling=false \
  data.train.pgc_v9_hard_role_curriculum=false \
  data.train.pgc_v9_hard_role_index_path=null \
  data.train.pgc_v9_closed_loop_rebinding=false \
  data.train.pgc_v9_phase_safe_memory=true \
  data.train.pgc_v9_closed_loop_native_dataset_count=1 \
  data.train.pgc_v9_safe_gain_counterfactual_replay=true \
  "++data.train.pretrained_norm_stats=${STATS_PATH}" \
  "data.train.text_embedding_cache_dir=${CACHE_DIR}" \
  "model.policy_guard.entity_relation_grounding.action_geometry_learning_rate=${SIDE_MODULE_LEARNING_RATE}" \
  "seed=${TRAIN_SEED}" \
  "max_steps=${MAX_STEPS}" \
  num_epochs=1 \
  "learning_rate=${SIDE_MODULE_LEARNING_RATE}" \
  "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
  "save_every=${SAVE_EVERY}" \
  save_training_state=false \
  "${VARIANT_OVERRIDES[@]}"
