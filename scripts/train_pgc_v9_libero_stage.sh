#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: bash scripts/train_pgc_v9_libero_stage.sh <suite> <grounding|action|verifier> <gpus> <base_checkpoint> <init_checkpoint> <original_cf_dataset> <strict_cf_dataset> <native_sidecar> <original_cf_sidecar> <strict_cf_sidecar> [seed] [full|entity-only|without-anchor]}"
STAGE="${2:?Missing V9 training stage}"
NPROC_PER_NODE="${3:?Missing GPU count}"
BASE_CHECKPOINT="${4:?Missing released FastWAM checkpoint}"
INIT_CHECKPOINT="${5:?Missing exact V5/V9 initialization checkpoint}"
ORIGINAL_CF_DATASET="${6:?Missing historical counterfactual dataset}"
STRICT_CF_DATASET="${7:?Missing strict-conflict counterfactual dataset}"
NATIVE_SIDECAR="${8:?Missing native ERAF sidecar}"
ORIGINAL_CF_SIDECAR="${9:?Missing historical-CF ERAF sidecar}"
STRICT_CF_SIDECAR="${10:?Missing strict-CF ERAF sidecar}"
TRAIN_SEED="${11:-42}"
ABLATION="${12:-full}"

case "${SUITE}" in
  libero_spatial|libero_object|libero_goal|libero_10) ;;
  *)
    echo "Unsupported LIBERO suite: ${SUITE}." >&2
    exit 1
    ;;
esac
case "${STAGE}" in
  grounding)
    START_STEP="null"
    STAGE_START_STEP=0
    DEFAULT_STAGE_STEPS=1500
    LEARNING_RATE="1.0e-4"
    ;;
  action)
    START_STEP=1500
    STAGE_START_STEP=1500
    DEFAULT_STAGE_STEPS=4000
    LEARNING_RATE="1.0e-4"
    ;;
  verifier)
    START_STEP=5500
    STAGE_START_STEP=5500
    DEFAULT_STAGE_STEPS=1000
    LEARNING_RATE="1.0e-4"
    ;;
  *)
    echo "Stage must be grounding, action, or verifier; got ${STAGE}." >&2
    exit 1
    ;;
esac
STAGE_STEPS="${PGC_V9_STAGE_STEPS:-${DEFAULT_STAGE_STEPS}}"
if ! [[ "${STAGE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PGC_V9_STAGE_STEPS must be a positive integer." >&2
  exit 1
fi
GRADIENT_ACCUMULATION_STEPS="${PGC_V9_GRADIENT_ACCUMULATION_STEPS:-4}"
if ! [[ "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PGC_V9_GRADIENT_ACCUMULATION_STEPS must be a positive integer." >&2
  exit 1
fi
GROUNDING_OBJECTIVE_VERSION="${PGC_V9_GROUNDING_OBJECTIVE_VERSION:-2}"
ATTENTION_MASK_WEIGHT="${PGC_V9_ATTENTION_MASK_WEIGHT:-2.0}"
ROLE_SWAP_WEIGHT="${PGC_V9_ROLE_SWAP_WEIGHT:-2.0}"
ROLE_OVERLAP_WEIGHT="${PGC_V9_ROLE_OVERLAP_WEIGHT:-1.0}"
ROLE_SWAP_MARGIN="${PGC_V9_ROLE_SWAP_MARGIN:-0.20}"
if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "2" ]]; then
  echo "Formal V9.1 training requires PGC_V9_GROUNDING_OBJECTIVE_VERSION=2." >&2
  exit 1
fi
MAX_STEPS=$((STAGE_START_STEP + STAGE_STEPS))
case "${ABLATION}" in
  full)
    ENTITY_ONLY=false
    USE_ANCHORS=true
    RELATION_WEIGHT=1.0
    ANCHOR_WEIGHT=1.0
    POSITION_WEIGHT=0.5
    PHASE_WEIGHT=1.0
    WRONG_RELATION_WEIGHT=0.5
    ;;
  entity-only)
    ENTITY_ONLY=true
    USE_ANCHORS=false
    RELATION_WEIGHT=0.0
    ANCHOR_WEIGHT=0.0
    POSITION_WEIGHT=0.0
    PHASE_WEIGHT=0.0
    WRONG_RELATION_WEIGHT=0.0
    ;;
  without-anchor)
    ENTITY_ONLY=false
    USE_ANCHORS=false
    RELATION_WEIGHT=1.0
    ANCHOR_WEIGHT=0.0
    POSITION_WEIGHT=0.0
    PHASE_WEIGHT=1.0
    WRONG_RELATION_WEIGHT=0.5
    ;;
  *)
    echo "Ablation must be full, entity-only, or without-anchor." >&2
    exit 1
    ;;
esac

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU count must be a positive integer." >&2
  exit 1
fi
if ! [[ "${TRAIN_SEED}" =~ ^[0-9]+$ ]]; then
  echo "Seed must be a non-negative integer." >&2
  exit 1
fi
for checkpoint in "${BASE_CHECKPOINT}" "${INIT_CHECKPOINT}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found: ${checkpoint}" >&2
    exit 1
  fi
done
for directory in \
  "${ORIGINAL_CF_DATASET}" "${STRICT_CF_DATASET}" \
  "${NATIVE_SIDECAR}" "${ORIGINAL_CF_SIDECAR}" "${STRICT_CF_SIDECAR}"; do
  if [[ ! -d "${directory}" ]]; then
    echo "Dataset/sidecar directory not found: ${directory}" >&2
    exit 1
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/root/gpufree-data/fastwam/FastWAM/data/libero_mujoco3.3.2}"
NATIVE_DATASET="${LIBERO_DATA_ROOT}/${SUITE}_no_noops_lerobot"
STATS_PATH="${STATS_PATH:-${DIFFSYNTH_MODEL_BASE_PATH:-./checkpoints}/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
CACHE_DIR="${TEXT_CACHE_DIR:-data/text_embeds_cache/libero}"
if [[ ! -d "${NATIVE_DATASET}" ]]; then
  echo "Native suite dataset not found: ${NATIVE_DATASET}" >&2
  exit 1
fi
if [[ ! -f "${STATS_PATH}" ]]; then
  echo "Dataset normalization stats not found: ${STATS_PATH}" >&2
  exit 1
fi

NATIVE_DATASET="$(cd -- "${NATIVE_DATASET}" && pwd -P)"
ORIGINAL_CF_DATASET="$(cd -- "${ORIGINAL_CF_DATASET}" && pwd -P)"
STRICT_CF_DATASET="$(cd -- "${STRICT_CF_DATASET}" && pwd -P)"
NATIVE_SIDECAR="$(cd -- "${NATIVE_SIDECAR}" && pwd -P)"
ORIGINAL_CF_SIDECAR="$(cd -- "${ORIGINAL_CF_SIDECAR}" && pwd -P)"
STRICT_CF_SIDECAR="$(cd -- "${STRICT_CF_SIDECAR}" && pwd -P)"

"${PYTHON_BIN}" - \
  "${STAGE}" "${START_STEP}" "${BASE_CHECKPOINT}" "${INIT_CHECKPOINT}" \
  "${NATIVE_DATASET}" "${ORIGINAL_CF_DATASET}" "${STRICT_CF_DATASET}" \
  "${NATIVE_SIDECAR}" "${ORIGINAL_CF_SIDECAR}" "${STRICT_CF_SIDECAR}" <<'PY'
import json
import pathlib
import sys
import torch

stage, expected_step, base_checkpoint, checkpoint, *paths = sys.argv[1:]
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
fmt = str(payload.get("format", ""))
version = int((payload.get("architecture_metadata") or {}).get("policy_guard_version", 0))
if stage == "grounding":
    if fmt != "fastwam_policy_guard_v5" or version != 5:
        raise SystemExit(
            "V9 grounding must warm-start from an exact suite-specific PGC V5 checkpoint."
        )
else:
    if fmt != "fastwam_policy_guard_v9" or version != 9:
        raise SystemExit(f"V9 {stage} must resume from a V9 checkpoint.")
    grounding_objective_version = int(
        (payload.get("architecture_metadata") or {}).get(
            "eraf_grounding_objective_version", 1
        )
    )
    if grounding_objective_version != 2:
        raise SystemExit(
            f"V9.1 {stage} requires a completed gate-aligned grounding "
            f"checkpoint (objective version 2); got {grounding_objective_version}."
        )
    if int(payload.get("step", -1)) != int(expected_step):
        raise SystemExit(
            f"V9 {stage} requires checkpoint step {expected_step}; got {payload.get('step')}."
        )
    expected_input_stage = {"action": "grounding", "verifier": "action"}[stage]
    saved_stage = str(
        (payload.get("architecture_metadata") or {}).get(
            "eraf_training_stage", ""
        )
    )
    if saved_stage != expected_input_stage:
        raise SystemExit(
            f"V9 {stage} must resume the completed {expected_input_stage} "
            f"stage; checkpoint declares {saved_stage!r}."
        )
saved_base = payload.get("base_checkpoint")
if not saved_base:
    raise SystemExit(f"Initialization checkpoint has no protected base: {checkpoint}")
saved_base_path = pathlib.Path(str(saved_base)).expanduser()
if not saved_base_path.is_absolute():
    saved_base_path = pathlib.Path(checkpoint).resolve().parent / saved_base_path
if saved_base_path.resolve() != pathlib.Path(base_checkpoint).resolve():
    raise SystemExit(
        "Protected Base mismatch: initialization checkpoint references "
        f"{saved_base_path.resolve()}, but the command supplied "
        f"{pathlib.Path(base_checkpoint).resolve()}."
    )
expected_action_contracts = (
    (
        "native",
        "fastwam_gripper_open_1_close_0",
        "fastwam_to_libero_env",
    ),
    (
        "counterfactual",
        "libero_env_gripper_open_minus1_close_plus1",
        "identity",
    ),
    (
        "counterfactual",
        "libero_env_gripper_open_minus1_close_plus1",
        "identity",
    ),
)
for dataset, sidecar, expected_contract in zip(
    paths[:3], paths[3:], expected_action_contracts, strict=True
):
    index_path = pathlib.Path(sidecar) / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("format") != "pgc_libero_entity_relation_v1":
        raise SystemExit(f"Invalid ERAF sidecar: {index_path}")
    if pathlib.Path(index["dataset"]).resolve() != pathlib.Path(dataset).resolve():
        raise SystemExit(
            f"Sidecar order mismatch: {index['dataset']} does not audit {dataset}."
        )
    actual_contract = (
        index.get("dataset_kind"),
        index.get("dataset_action_convention"),
        index.get("simulator_replay_action_transform"),
    )
    if actual_contract != expected_contract:
        raise SystemExit(
            f"Sidecar action contract mismatch at {index_path}: "
            f"expected={expected_contract} got={actual_contract}."
        )
PY

json_array() {
  "${PYTHON_BIN}" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"
}
NATIVE_JSON="$(json_array "${NATIVE_DATASET}")"
CF_JSON="$(json_array "${ORIGINAL_CF_DATASET}" "${STRICT_CF_DATASET}")"
SIDECAR_JSON="$(json_array "${NATIVE_SIDECAR}" "${ORIGINAL_CF_SIDECAR}" "${STRICT_CF_SIDECAR}")"

RUN_TAG="${RUN_TAG:-${SUITE}-pgc-v9-eraf-${ABLATION}-${STAGE}-seed${TRAIN_SEED}-v1}"
echo "[PGC-FastWAM] V9 ERAF ${STAGE} training"
echo "  suite=${SUITE} cumulative_steps=${MAX_STEPS} start_step=${START_STEP} stage_steps=${STAGE_STEPS}"
echo "  ablation=${ABLATION} entity_only=${ENTITY_ONLY} use_anchors=${USE_ANCHORS}"
echo "  effective_batch=$((NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS)) (${NPROC_PER_NODE} GPUs x batch1 x grad_accum${GRADIENT_ACCUMULATION_STEPS})"
echo "  grounding_objective=v${GROUNDING_OBJECTIVE_VERSION} attention_mask=${ATTENTION_MASK_WEIGHT} role_swap=${ROLE_SWAP_WEIGHT} role_overlap=${ROLE_OVERLAP_WEIGHT} margin=${ROLE_SWAP_MARGIN}"
echo "  mixture=native:CF 1:1; CF=historical:strict 1:1; strict=relation-balanced"
echo "  init=${INIT_CHECKPOINT}"
echo "  native=${NATIVE_DATASET}"
echo "  historical_cf=${ORIGINAL_CF_DATASET}"
echo "  strict_cf=${STRICT_CF_DATASET}"

RUN_ID="pgc-${RUN_TAG}" exec bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task=libero_pgc_2cam224 \
  "resume=${INIT_CHECKPOINT}" \
  "weight_only_start_step=${START_STEP}" \
  "data.train.dataset_dirs=${NATIVE_JSON}" \
  "data.train.pgc_counterfactual_dataset_dirs=${CF_JSON}" \
  data.train.pgc_closed_loop_corrective_dataset_dirs=[] \
  data.train.pgc_counterfactual_oversample_factor=1 \
  data.train.pgc_balance_native_counterfactual=true \
  data.train.pgc_entity_relation_supervision_required=true \
  "data.train.pgc_entity_relation_sidecar_dirs=${SIDECAR_JSON}" \
  data.train.pgc_v9_balanced_sampling=true \
  "++data.train.pretrained_norm_stats=${STATS_PATH}" \
  "data.train.text_embedding_cache_dir=${CACHE_DIR}" \
  "seed=${TRAIN_SEED}" \
  "max_steps=${MAX_STEPS}" \
  num_epochs=1 \
  "learning_rate=${LEARNING_RATE}" \
  "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
  save_every=500 \
  save_training_state=false \
  model.action_dit_config.use_latent_action_queries=false \
  model.langforce_mvp.enabled=false \
  model.langforce_mvp.enable_prior=false \
  model.langforce_mvp.enable_posterior_advantage=false \
  model.transition_contract.enabled=false \
  model.policy_guard.enabled=true \
  model.policy_guard.version=9 \
  "model.policy_guard.entity_relation_grounding.training_stage=${STAGE}" \
  "model.policy_guard.entity_relation_grounding.grounding_objective_version=${GROUNDING_OBJECTIVE_VERSION}" \
  "model.policy_guard.entity_relation_grounding.entity_only=${ENTITY_ONLY}" \
  "model.policy_guard.entity_relation_grounding.use_anchors=${USE_ANCHORS}" \
  model.policy_guard.entity_relation_grounding.learning_rate=2.0e-5 \
  model.policy_guard.entity_relation_grounding.grounding_aux_weight=0.25 \
  model.policy_guard.entity_relation_grounding.mask_weight=1.0 \
  "model.policy_guard.entity_relation_grounding.attention_mask_weight=${ATTENTION_MASK_WEIGHT}" \
  model.policy_guard.entity_relation_grounding.entity_weight=1.0 \
  "model.policy_guard.entity_relation_grounding.relation_weight=${RELATION_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.anchor_weight=${ANCHOR_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.position_weight=${POSITION_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.role_swap_weight=${ROLE_SWAP_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.role_overlap_weight=${ROLE_OVERLAP_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.role_swap_margin=${ROLE_SWAP_MARGIN}" \
  "model.policy_guard.entity_relation_grounding.phase_weight=${PHASE_WEIGHT}" \
  model.policy_guard.counterfactual_action_weight=1.0 \
  model.policy_guard.execution_prefix_steps=10 \
  model.policy_guard.suffix_loss_weight=0.1 \
  model.policy_guard.same_state_source_zero_weight=1.0 \
  model.policy_guard.native_guard_weight=0.1 \
  model.policy_guard.residual_regularization_weight=0.01 \
  model.policy_guard.residual_smoothness_weight=0.01 \
  model.policy_guard.verifier_wrong_entity_weight=0.5 \
  "model.policy_guard.verifier_wrong_relation_weight=${WRONG_RELATION_WEIGHT}" \
  model.policy_guard.require_direct_counterfactual_actions=true \
  model.lora.enabled=false
