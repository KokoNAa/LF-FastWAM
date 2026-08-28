#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: bash scripts/train_libero_eraf_safe_gain.sh <suite> <gpus> <no_eraf_lora_checkpoint> <pretrained_eraf_checkpoint> <historical_cf_dataset> <strict_cf_dataset> <native_sidecar> <historical_cf_sidecar> <strict_cf_sidecar> [seed]}"
NPROC_PER_NODE="${2:?Missing GPU count}"
BASELINE_CHECKPOINT="${3:?Missing no-ERAF LoRA checkpoint}"
PRETRAINED_ERAF_CHECKPOINT="${4:?Missing pretrained ERAF checkpoint}"
HISTORICAL_CF_DATASET="${5:?Missing historical counterfactual dataset}"
STRICT_CF_DATASET="${6:?Missing strict counterfactual dataset}"
NATIVE_SIDECAR="${7:?Missing native ERAF sidecar}"
HISTORICAL_CF_SIDECAR="${8:?Missing historical-CF ERAF sidecar}"
STRICT_CF_SIDECAR="${9:?Missing strict-CF ERAF sidecar}"
TRAIN_SEED="${10:-42}"

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
MAX_STEPS="${ERAF_SAFE_GAIN_MAX_STEPS:-10000}"
SIDE_MODULE_LEARNING_RATE="${ERAF_SAFE_GAIN_LEARNING_RATE:-2.0e-5}"
GRADIENT_ACCUMULATION_STEPS="${ERAF_SAFE_GAIN_GRADIENT_ACCUMULATION_STEPS:-4}"
SAVE_EVERY="${ERAF_SAFE_GAIN_SAVE_EVERY:-250}"

if [[ -z "${CLOSED_LOOP_DATASET}" || -z "${CLOSED_LOOP_SIDECAR}" ]]; then
  echo "Set PGC_V9_CLOSED_LOOP_GROUNDING_DATASET and PGC_V9_CLOSED_LOOP_GROUNDING_SIDECAR." >&2
  exit 1
fi
for file_path in "${BASELINE_CHECKPOINT}" "${PRETRAINED_ERAF_CHECKPOINT}" "${STATS_PATH}"; do
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

resolve_dir() {
  (cd -- "$1" && pwd -P)
}
resolve_file() {
  local parent
  parent="$(cd -- "$(dirname -- "$1")" && pwd -P)"
  printf '%s/%s\n' "${parent}" "$(basename -- "$1")"
}

BASELINE_CHECKPOINT="$(resolve_file "${BASELINE_CHECKPOINT}")"
PRETRAINED_ERAF_CHECKPOINT="$(resolve_file "${PRETRAINED_ERAF_CHECKPOINT}")"
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
  "${BASELINE_CHECKPOINT}" "${PRETRAINED_ERAF_CHECKPOINT}" \
  "${NATIVE_DATASET}" "${CLOSED_LOOP_DATASET}" \
  "${HISTORICAL_CF_DATASET}" "${STRICT_CF_DATASET}" \
  "${NATIVE_SIDECAR}" "${CLOSED_LOOP_SIDECAR}" \
  "${HISTORICAL_CF_SIDECAR}" "${STRICT_CF_SIDECAR}" <<'PY'
import json
import math
import pathlib
import sys

import torch

baseline = pathlib.Path(sys.argv[1]).resolve()
eraf = pathlib.Path(sys.argv[2]).resolve()
datasets = [pathlib.Path(value).resolve() for value in sys.argv[3:7]]
sidecars = [pathlib.Path(value).resolve() for value in sys.argv[7:11]]

baseline_payload = torch.load(baseline, map_location="cpu", weights_only=False)
if baseline_payload.get("format") != "fastwam_lora_adapter_v1":
    raise SystemExit("Safe-gain baseline must be a no-ERAF LoRA adapter checkpoint.")
lora = baseline_payload.get("lora_config") or {}
paired = lora.get("paired_language_control") or {}
if not paired.get("enabled") or not paired.get("bidirectional_supervision"):
    raise SystemExit("Safe-gain baseline is not the formal bidirectional no-ERAF control.")
if set(lora.get("experts") or []) != {"video", "action"}:
    raise SystemExit("Safe-gain baseline must contain shared Video/Action LoRA.")

eraf_payload = torch.load(eraf, map_location="cpu", weights_only=False)
metadata = eraf_payload.get("architecture_metadata") or {}
objective = int(metadata.get("eraf_grounding_objective_version", -1))
if eraf_payload.get("format") != "fastwam_policy_guard_v9" or not 14 <= objective <= 26:
    raise SystemExit("ERAF source must be an admitted pre-joint V9.13+ checkpoint.")
if bool(metadata.get("eraf_fresh_joint_training", False)) or bool(
    metadata.get("eraf_pretrained_joint_training", False)
):
    raise SystemExit("ERAF source must precede the joint experiments.")
guard = eraf_payload.get("policy_guard") or {}
for prefix in ("goal_query_seeds.", "goal_graph.", "entity_relation_affordance."):
    if not any(str(name).startswith(prefix) for name in guard):
        raise SystemExit(f"ERAF source is missing the required {prefix} tensors.")
if eraf_payload.get("eraf_shared_expert_lora"):
    raise SystemExit("ERAF source must not contain shared Expert LoRA.")

expected = [
    ("native", "fastwam_gripper_open_1_close_0", "fastwam_to_libero_env"),
    ("native", "fastwam_gripper_open_1_close_0", "fastwam_to_libero_env"),
    ("counterfactual", "libero_env_gripper_open_minus1_close_plus1", "identity"),
    ("counterfactual", "libero_env_gripper_open_minus1_close_plus1", "identity"),
]
workspace = []
for dataset, sidecar, contract in zip(datasets, sidecars, expected, strict=True):
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
        raise SystemExit("Workspace contract mismatch in safe-gain sidecars.")
PY

VISIBLE_GPU_COUNT="$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count())')"
if (( VISIBLE_GPU_COUNT < NPROC_PER_NODE )); then
  echo "Requested ${NPROC_PER_NODE} processes, but PyTorch sees ${VISIBLE_GPU_COUNT} CUDA devices." >&2
  exit 1
fi

json_array() {
  "${PYTHON_BIN}" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"
}
NATIVE_JSON="$(json_array "${NATIVE_DATASET}" "${CLOSED_LOOP_DATASET}")"
CF_JSON="$(json_array "${HISTORICAL_CF_DATASET}" "${STRICT_CF_DATASET}")"
SIDECAR_JSON="$(json_array "${NATIVE_SIDECAR}" "${CLOSED_LOOP_SIDECAR}" "${HISTORICAL_CF_SIDECAR}" "${STRICT_CF_SIDECAR}")"

RUN_TAG="${RUN_TAG:-${SUITE}-eraf-safe-gain-bidirectional-10k-seed${TRAIN_SEED}-v1}"
echo "[FastWAM] LIBERO V9.27 ERAF safe-gain training"
echo "  baseline=${BASELINE_CHECKPOINT}"
echo "  eraf_bundle=${PRETRAINED_ERAF_CHECKPOINT}"
echo "  trainable=4-token-compressor+context-injector+gain-gate"
echo "  frozen=no-ERAF-Video/Action-LoRA+GoalGraph+ERAF"
echo "  deployment=pre-action gate then one Action Expert denoising path"

RUN_ID="eraf-safe-gain-${RUN_TAG}" exec bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task=libero_eraf_safe_gain_2cam224 \
  "resume=${BASELINE_CHECKPOINT}" \
  "data.train.dataset_dirs=${NATIVE_JSON}" \
  "data.train.pgc_counterfactual_dataset_dirs=${CF_JSON}" \
  data.train.pgc_closed_loop_corrective_dataset_dirs=[] \
  data.train.pgc_counterfactual_oversample_factor=1 \
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
  "++data.train.pretrained_norm_stats=${STATS_PATH}" \
  "data.train.text_embedding_cache_dir=${CACHE_DIR}" \
  "model.policy_guard.entity_relation_grounding.pretrained_checkpoint=${PRETRAINED_ERAF_CHECKPOINT}" \
  "model.policy_guard.entity_relation_grounding.action_geometry_learning_rate=${SIDE_MODULE_LEARNING_RATE}" \
  "seed=${TRAIN_SEED}" \
  "max_steps=${MAX_STEPS}" \
  num_epochs=1 \
  "learning_rate=${SIDE_MODULE_LEARNING_RATE}" \
  "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
  "save_every=${SAVE_EVERY}" \
  save_training_state=false
