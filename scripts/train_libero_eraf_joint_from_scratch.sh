#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: bash scripts/train_libero_eraf_joint_from_scratch.sh <suite> <gpus> <released_base> <historical_cf_dataset> <strict_cf_dataset> <native_sidecar> <historical_cf_sidecar> <strict_cf_sidecar> [seed]}"
NPROC_PER_NODE="${2:?Missing GPU count}"
BASE_CHECKPOINT="${3:?Missing released FastWAM checkpoint}"
HISTORICAL_CF_DATASET="${4:?Missing historical counterfactual dataset}"
STRICT_CF_DATASET="${5:?Missing strict counterfactual dataset}"
NATIVE_SIDECAR="${6:?Missing native ERAF sidecar}"
HISTORICAL_CF_SIDECAR="${7:?Missing historical-CF ERAF sidecar}"
STRICT_CF_SIDECAR="${8:?Missing strict-CF ERAF sidecar}"
TRAIN_SEED="${9:-42}"
PRETRAINED_ERAF_CHECKPOINT="${ERAF_PRETRAINED_CHECKPOINT:-}"
if [[ -n "${PRETRAINED_ERAF_CHECKPOINT}" ]]; then
  ERAF_INITIALIZATION_CONTRACT="released_base_pretrained_eraf"
  ERAF_FRESH_JOINT="false"
  ERAF_PRETRAINED_JOINT="true"
  DEFAULT_INJECTION_WARMUP_STEPS=0
  DEFAULT_ERAF_LEARNING_RATE="2.0e-5"
  ERAF_JOINT_KIND="pretrained"
else
  ERAF_INITIALIZATION_CONTRACT="released_base_fresh_eraf"
  ERAF_FRESH_JOINT="true"
  ERAF_PRETRAINED_JOINT="false"
  DEFAULT_INJECTION_WARMUP_STEPS=1500
  DEFAULT_ERAF_LEARNING_RATE="1.0e-4"
  ERAF_JOINT_KIND="fresh"
fi

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
MAX_STEPS="${ERAF_JOINT_MAX_STEPS:-15000}"
LEARNING_RATE="${ERAF_JOINT_LORA_LEARNING_RATE:-5.0e-6}"
ERAF_LEARNING_RATE="${ERAF_JOINT_ERAF_LEARNING_RATE:-${DEFAULT_ERAF_LEARNING_RATE}}"
INJECTOR_LEARNING_RATE="${ERAF_JOINT_INJECTOR_LEARNING_RATE:-2.0e-5}"
INJECTION_WARMUP_STEPS="${ERAF_JOINT_INJECTION_WARMUP_STEPS:-${DEFAULT_INJECTION_WARMUP_STEPS}}"
INJECTION_RAMP_STEPS="${ERAF_JOINT_INJECTION_RAMP_STEPS:-1000}"
GRADIENT_ACCUMULATION_STEPS="${ERAF_JOINT_GRADIENT_ACCUMULATION_STEPS:-4}"
SAVE_EVERY="${ERAF_JOINT_SAVE_EVERY:-250}"

for value_name in MAX_STEPS INJECTION_WARMUP_STEPS INJECTION_RAMP_STEPS GRADIENT_ACCUMULATION_STEPS SAVE_EVERY; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "${value_name} must be a non-negative integer, got ${value}." >&2
    exit 1
  fi
done
if (( MAX_STEPS == 0 || GRADIENT_ACCUMULATION_STEPS == 0 || SAVE_EVERY == 0 )); then
  echo "MAX_STEPS, GRADIENT_ACCUMULATION_STEPS, and SAVE_EVERY must be positive." >&2
  exit 1
fi
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
if [[ -n "${PRETRAINED_ERAF_CHECKPOINT}" && ! -f "${PRETRAINED_ERAF_CHECKPOINT}" ]]; then
  echo "Pretrained ERAF checkpoint not found: ${PRETRAINED_ERAF_CHECKPOINT}" >&2
  exit 1
fi
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
if [[ -n "${PRETRAINED_ERAF_CHECKPOINT}" ]]; then
  PRETRAINED_ERAF_CHECKPOINT="$(resolve_file "${PRETRAINED_ERAF_CHECKPOINT}")"
fi
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

# Keep the exact same released-base, action-convention, sidecar binding, and
# workspace checks as the formal no-ERAF control.  An ERAF joint run is invalid
# if any one of the four pools is substituted or expressed in another frame.
"${PYTHON_BIN}" - \
  "${BASE_CHECKPOINT}" \
  "${NATIVE_DATASET}" "${CLOSED_LOOP_DATASET}" \
  "${HISTORICAL_CF_DATASET}" "${STRICT_CF_DATASET}" \
  "${NATIVE_SIDECAR}" "${CLOSED_LOOP_SIDECAR}" \
  "${HISTORICAL_CF_SIDECAR}" "${STRICT_CF_SIDECAR}" \
  "${PRETRAINED_ERAF_CHECKPOINT}" <<'PY'
import json
import math
import pathlib
import sys

import torch

base_checkpoint = pathlib.Path(sys.argv[1]).resolve()
datasets = [pathlib.Path(value).resolve() for value in sys.argv[2:6]]
sidecars = [pathlib.Path(value).resolve() for value in sys.argv[6:10]]
pretrained_eraf = (
    pathlib.Path(sys.argv[10]).resolve() if len(sys.argv) > 10 and sys.argv[10] else None
)

payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
if str(payload.get("format", "")).startswith("fastwam_policy_guard_"):
    raise SystemExit(
        "ERAF joint training must start from the released Base, not a PGC checkpoint."
    )
if not isinstance(payload.get("mot"), dict) and not isinstance(payload.get("dit"), dict):
    raise SystemExit("Released Base checkpoint must contain `mot` or `dit` weights.")

if pretrained_eraf is not None:
    eraf_payload = torch.load(
        pretrained_eraf, map_location="cpu", weights_only=False
    )
    eraf_metadata = eraf_payload.get("architecture_metadata") or {}
    objective = int(eraf_metadata.get("eraf_grounding_objective_version", -1))
    source_step = int(eraf_payload.get("step", -1))
    if eraf_payload.get("format") != "fastwam_policy_guard_v9":
        raise SystemExit("Pretrained ERAF source must be a PGC V9 checkpoint.")
    if not 14 <= objective <= 26:
        raise SystemExit(
            f"Pretrained ERAF source must be admitted V9.13+ objective 14..26; got {objective}."
        )
    if source_step < 7250:
        raise SystemExit(
            f"Pretrained ERAF source must be completed V9.13+ step >=7250; got {source_step}."
        )
    if eraf_metadata.get("eraf_phase_safe_memory_contract") != (
        "explicit_cross_replan_pending_holding_retry_completed"
    ):
        raise SystemExit("Pretrained ERAF source lacks the phase-safe-memory contract.")
    if eraf_metadata.get("eraf_geometry_protection_contract") != (
        "frozen_v9_11_no_query_token_anchor_or_heatmap_residual"
    ):
        raise SystemExit("Pretrained ERAF source is not the frozen admitted geometry path.")
    if bool(eraf_metadata.get("eraf_fresh_joint_training", False)) or bool(
        eraf_metadata.get("eraf_pretrained_joint_training", False)
    ):
        raise SystemExit(
            "Pretrained ERAF source must precede the new end-to-end joint experiments."
        )
    guard = eraf_payload.get("policy_guard") or {}
    eraf_keys = [
        str(name)
        for name in guard
        if str(name).startswith("entity_relation_affordance.")
    ]
    if not eraf_keys:
        raise SystemExit("Pretrained ERAF source contains no ERAF tensors.")

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

RUN_TAG="${RUN_TAG:-${SUITE}-${ERAF_JOINT_KIND}-eraf-joint-bidirectional-15k-seed${TRAIN_SEED}-v1}"
echo "[FastWAM] LIBERO ${ERAF_JOINT_KIND}-ERAF + shared Expert LoRA joint training"
echo "  suite=${SUITE} steps=${MAX_STEPS} seed=${TRAIN_SEED}"
echo "  base=${BASE_CHECKPOINT}"
echo "  mixture=offline_native:closed_loop_native:historical_cf:strict_cf 1:1:1:1"
echo "  lora=video+action rank16 alpha16 dropout0.05; extra_trainables=none"
echo "  eraf=${ERAF_JOINT_KIND} trainable lr=${ERAF_LEARNING_RATE}; injector=fresh lr=${INJECTOR_LEARNING_RATE}"
if [[ -n "${PRETRAINED_ERAF_CHECKPOINT}" ]]; then
  echo "  eraf_pretrained_checkpoint=${PRETRAINED_ERAF_CHECKPOINT} (ERAF tensors only)"
fi
echo "  injection=exact-off through step ${INJECTION_WARMUP_STEPS}, ramp ${INJECTION_RAMP_STEPS} steps"
echo "  objectives=source+target world/action positives + bidirectional same-row language ranks + ERAF labels + lora_reg"

RUN_ID="eraf-joint-${RUN_TAG}" exec bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task=libero_eraf_joint_2cam224 \
  "resume=${BASE_CHECKPOINT}" \
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
  "seed=${TRAIN_SEED}" \
  "max_steps=${MAX_STEPS}" \
  num_epochs=1 \
  "learning_rate=${LEARNING_RATE}" \
  "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
  "save_every=${SAVE_EVERY}" \
  save_training_state=false \
  model.policy_guard.enabled=true \
  model.policy_guard.version=9 \
  model.policy_guard.entity_relation_grounding.training_stage=action \
  "model.policy_guard.entity_relation_grounding.initialization_contract=${ERAF_INITIALIZATION_CONTRACT}" \
  model.policy_guard.entity_relation_grounding.grounding_objective_version=26 \
  model.policy_guard.entity_relation_grounding.completion_only_memory=true \
  model.policy_guard.entity_relation_grounding.action_joint_training=true \
  "model.policy_guard.entity_relation_grounding.fresh_joint_training=${ERAF_FRESH_JOINT}" \
  "model.policy_guard.entity_relation_grounding.pretrained_joint_training=${ERAF_PRETRAINED_JOINT}" \
  "model.policy_guard.entity_relation_grounding.pretrained_checkpoint=${PRETRAINED_ERAF_CHECKPOINT:-null}" \
  model.policy_guard.entity_relation_grounding.bidirectional_supervision=true \
  "model.policy_guard.entity_relation_grounding.context_injection_warmup_steps=${INJECTION_WARMUP_STEPS}" \
  "model.policy_guard.entity_relation_grounding.context_injection_ramp_steps=${INJECTION_RAMP_STEPS}" \
  "model.policy_guard.entity_relation_grounding.learning_rate=${ERAF_LEARNING_RATE}" \
  "model.policy_guard.entity_relation_grounding.action_geometry_learning_rate=${INJECTOR_LEARNING_RATE}" \
  model.action_dit_config.use_latent_action_queries=false \
  model.langforce_mvp.enabled=false \
  model.transition_contract.enabled=false \
  model.lora.enabled=true \
  model.lora.rank=16 \
  model.lora.alpha=16 \
  model.lora.dropout=0.05 \
  'model.lora.experts=[video,action]' \
  'model.lora.extra_trainable_patterns=[]' \
  model.lora.paired_language_control.enabled=false
