#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: bash scripts/train_pgc_v9_libero_stage.sh <suite> <grounding|grounding-role|grounding-role-adapter|grounding-structured-role|grounding-balanced-role|grounding-hard-role|grounding-exclusive-role|grounding-clause-calibration|grounding-view-scheduler|grounding-all-entity-role|grounding-clause-tuple|grounding-phase-rebinding|grounding-phase-memory|action-completion-only|action-geometry-causal|action-semantic-causal|action-direct-geometry|action-phase-residual|action-phase-servo|action|verifier> <gpus> <base_checkpoint> <init_checkpoint> <original_cf_dataset> <strict_cf_dataset> <native_sidecar> <original_cf_sidecar> <strict_cf_sidecar> [seed] [full|entity-only|without-anchor]}"
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
REQUESTED_GROUNDING_OBJECTIVE_VERSION="${PGC_V9_GROUNDING_OBJECTIVE_VERSION:-}"

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
    CONFIG_STAGE="grounding"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=2
    SAVE_EVERY=500
    ;;
  grounding-role)
    START_STEP=1500
    STAGE_START_STEP=1500
    DEFAULT_STAGE_STEPS=1000
    LEARNING_RATE="2.0e-5"
    CONFIG_STAGE="grounding"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=3
    SAVE_EVERY=500
    ;;
  grounding-role-adapter)
    START_STEP=1500
    STAGE_START_STEP=1500
    DEFAULT_STAGE_STEPS=1000
    LEARNING_RATE="5.0e-5"
    CONFIG_STAGE="grounding"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=4
    SAVE_EVERY=250
    ;;
  grounding-structured-role)
    START_STEP=2500
    STAGE_START_STEP=2500
    DEFAULT_STAGE_STEPS=1000
    LEARNING_RATE="2.0e-5"
    CONFIG_STAGE="grounding"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=5
    SAVE_EVERY=250
    ;;
  grounding-balanced-role)
    START_STEP=2500
    STAGE_START_STEP=2500
    DEFAULT_STAGE_STEPS=1000
    LEARNING_RATE="2.0e-5"
    CONFIG_STAGE="grounding"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=6
    SAVE_EVERY=250
    ;;
  grounding-hard-role)
    START_STEP=2500
    STAGE_START_STEP=2500
    DEFAULT_STAGE_STEPS=500
    LEARNING_RATE="2.0e-5"
    CONFIG_STAGE="grounding"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=7
    SAVE_EVERY=250
    ;;
  grounding-exclusive-role)
    START_STEP=3000
    STAGE_START_STEP=3000
    DEFAULT_STAGE_STEPS=250
    LEARNING_RATE="1.0e-5"
    CONFIG_STAGE="grounding"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=8
    SAVE_EVERY=250
    ;;
  grounding-clause-calibration)
    START_STEP=3250
    STAGE_START_STEP=3250
    DEFAULT_STAGE_STEPS=500
    LEARNING_RATE="1.0e-5"
    CONFIG_STAGE="grounding"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=9
    SAVE_EVERY=250
    ;;
  grounding-view-scheduler)
    START_STEP=3750
    STAGE_START_STEP=3750
    DEFAULT_STAGE_STEPS=1000
    LEARNING_RATE="2.0e-5"
    CONFIG_STAGE="grounding"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=10
    SAVE_EVERY=250
    ;;
  grounding-all-entity-role)
    START_STEP=4750
    STAGE_START_STEP=4750
    DEFAULT_STAGE_STEPS=1000
    LEARNING_RATE="1.0e-5"
    CONFIG_STAGE="grounding"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=11
    SAVE_EVERY=250
    ;;
  grounding-clause-tuple)
    START_STEP=5750
    STAGE_START_STEP=5750
    DEFAULT_STAGE_STEPS=500
    LEARNING_RATE="5.0e-6"
    CONFIG_STAGE="grounding"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=12
    SAVE_EVERY=250
    ;;
  grounding-phase-rebinding)
    START_STEP=6250
    STAGE_START_STEP=6250
    DEFAULT_STAGE_STEPS=1000
    LEARNING_RATE="1.0e-5"
    CONFIG_STAGE="grounding"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=13
    SAVE_EVERY=250
    ;;
  grounding-phase-memory)
    START_STEP=6250
    STAGE_START_STEP=6250
    DEFAULT_STAGE_STEPS=1000
    LEARNING_RATE="1.0e-5"
    CONFIG_STAGE="grounding"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=14
    SAVE_EVERY=250
    ;;
  action-completion-only)
    START_STEP=7250
    STAGE_START_STEP=7250
    DEFAULT_STAGE_STEPS=4000
    LEARNING_RATE="1.0e-4"
    CONFIG_STAGE="action"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=14
    SAVE_EVERY=500
    ;;
  action-geometry-causal)
    START_STEP=11250
    STAGE_START_STEP=11250
    DEFAULT_STAGE_STEPS=2000
    LEARNING_RATE="5.0e-5"
    CONFIG_STAGE="action"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=15
    SAVE_EVERY=250
    ;;
  action-semantic-causal)
    START_STEP=13250
    STAGE_START_STEP=13250
    DEFAULT_STAGE_STEPS=500
    LEARNING_RATE="2.0e-5"
    CONFIG_STAGE="action"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=16
    SAVE_EVERY=250
    ;;
  action-direct-geometry)
    START_STEP=13750
    STAGE_START_STEP=13750
    DEFAULT_STAGE_STEPS=500
    LEARNING_RATE="2.0e-5"
    CONFIG_STAGE="action"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=17
    SAVE_EVERY=250
    ;;
  action-phase-residual)
    START_STEP=14250
    STAGE_START_STEP=14250
    DEFAULT_STAGE_STEPS=1000
    LEARNING_RATE="2.0e-5"
    CONFIG_STAGE="action"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=18
    SAVE_EVERY=250
    ;;
  action-phase-servo)
    START_STEP=15250
    STAGE_START_STEP=15250
    DEFAULT_STAGE_STEPS=1000
    LEARNING_RATE="2.0e-5"
    CONFIG_STAGE="action"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=19
    SAVE_EVERY=250
    ;;
  action)
    if [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "14" || "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "13" ]]; then
      START_STEP=7250
      STAGE_START_STEP=7250
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "12" ]]; then
      START_STEP=6250
      STAGE_START_STEP=6250
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "11" ]]; then
      START_STEP=5750
      STAGE_START_STEP=5750
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "10" ]]; then
      START_STEP=4750
      STAGE_START_STEP=4750
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "9" ]]; then
      START_STEP=3750
      STAGE_START_STEP=3750
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "8" ]]; then
      START_STEP=3250
      STAGE_START_STEP=3250
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "7" ]]; then
      START_STEP=3000
      STAGE_START_STEP=3000
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "5" || "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "6" ]]; then
      START_STEP=3500
      STAGE_START_STEP=3500
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "3" || "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "4" ]]; then
      START_STEP=2500
      STAGE_START_STEP=2500
    else
      START_STEP=1500
      STAGE_START_STEP=1500
    fi
    DEFAULT_STAGE_STEPS=4000
    LEARNING_RATE="1.0e-4"
    CONFIG_STAGE="action"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=2
    SAVE_EVERY=500
    ;;
  verifier)
    if [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "14" || "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "13" ]]; then
      START_STEP=11250
      STAGE_START_STEP=11250
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "12" ]]; then
      START_STEP=10250
      STAGE_START_STEP=10250
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "11" ]]; then
      START_STEP=9750
      STAGE_START_STEP=9750
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "10" ]]; then
      START_STEP=8750
      STAGE_START_STEP=8750
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "9" ]]; then
      START_STEP=7750
      STAGE_START_STEP=7750
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "8" ]]; then
      START_STEP=7250
      STAGE_START_STEP=7250
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "7" ]]; then
      START_STEP=7000
      STAGE_START_STEP=7000
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "5" || "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "6" ]]; then
      START_STEP=7500
      STAGE_START_STEP=7500
    elif [[ "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "3" || "${REQUESTED_GROUNDING_OBJECTIVE_VERSION}" == "4" ]]; then
      START_STEP=6500
      STAGE_START_STEP=6500
    else
      START_STEP=5500
      STAGE_START_STEP=5500
    fi
    DEFAULT_STAGE_STEPS=1000
    LEARNING_RATE="1.0e-4"
    CONFIG_STAGE="verifier"
    DEFAULT_GROUNDING_OBJECTIVE_VERSION=2
    SAVE_EVERY=500
    ;;
  *)
    echo "Stage must be grounding, grounding-role, grounding-role-adapter, grounding-structured-role, grounding-balanced-role, grounding-hard-role, grounding-exclusive-role, grounding-clause-calibration, grounding-view-scheduler, grounding-all-entity-role, grounding-clause-tuple, grounding-phase-rebinding, grounding-phase-memory, action-completion-only, action-geometry-causal, action-semantic-causal, action-direct-geometry, action-phase-residual, action-phase-servo, action, or verifier; got ${STAGE}." >&2
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
GROUNDING_OBJECTIVE_VERSION="${REQUESTED_GROUNDING_OBJECTIVE_VERSION:-${DEFAULT_GROUNDING_OBJECTIVE_VERSION}}"
ATTENTION_MASK_WEIGHT="${PGC_V9_ATTENTION_MASK_WEIGHT:-2.0}"
ROLE_SWAP_WEIGHT="${PGC_V9_ROLE_SWAP_WEIGHT:-2.0}"
ROLE_OVERLAP_WEIGHT="${PGC_V9_ROLE_OVERLAP_WEIGHT:-1.0}"
ROLE_SWAP_MARGIN="${PGC_V9_ROLE_SWAP_MARGIN:-0.20}"
ROLE_ASSIGNMENT_TEMPERATURE="${PGC_V9_ROLE_ASSIGNMENT_TEMPERATURE:-0.10}"
if [[ "${GROUNDING_OBJECTIVE_VERSION}" == "4" || "${GROUNDING_OBJECTIVE_VERSION}" == "5" || "${GROUNDING_OBJECTIVE_VERSION}" == "6" || "${GROUNDING_OBJECTIVE_VERSION}" == "7" || "${GROUNDING_OBJECTIVE_VERSION}" == "8" || "${GROUNDING_OBJECTIVE_VERSION}" == "9" || "${GROUNDING_OBJECTIVE_VERSION}" == "10" || "${GROUNDING_OBJECTIVE_VERSION}" == "11" || "${GROUNDING_OBJECTIVE_VERSION}" == "12" ]]; then
  DEFAULT_ROLE_ASSIGNMENT_WEIGHT=1.0
  DEFAULT_ROLE_ASSIGNMENT_HARD_WEIGHT=0.5
elif [[ "${GROUNDING_OBJECTIVE_VERSION}" == "3" ]]; then
  DEFAULT_ROLE_ASSIGNMENT_WEIGHT=4.0
  DEFAULT_ROLE_ASSIGNMENT_HARD_WEIGHT=2.0
else
  DEFAULT_ROLE_ASSIGNMENT_WEIGHT=0.0
  DEFAULT_ROLE_ASSIGNMENT_HARD_WEIGHT=0.0
fi
ROLE_ASSIGNMENT_WEIGHT="${PGC_V9_ROLE_ASSIGNMENT_WEIGHT:-${DEFAULT_ROLE_ASSIGNMENT_WEIGHT}}"
ROLE_ASSIGNMENT_HARD_WEIGHT="${PGC_V9_ROLE_ASSIGNMENT_HARD_WEIGHT:-${DEFAULT_ROLE_ASSIGNMENT_HARD_WEIGHT}}"
if [[ "${GROUNDING_OBJECTIVE_VERSION}" == "6" || "${GROUNDING_OBJECTIVE_VERSION}" == "7" || "${GROUNDING_OBJECTIVE_VERSION}" == "8" || "${GROUNDING_OBJECTIVE_VERSION}" == "9" || "${GROUNDING_OBJECTIVE_VERSION}" == "10" || "${GROUNDING_OBJECTIVE_VERSION}" == "11" || "${GROUNDING_OBJECTIVE_VERSION}" == "12" ]]; then
  DEFAULT_ROLE_ATTENTION_PRESERVATION_WEIGHT=5.0
  DEFAULT_ROLE_POSITION_PRESERVATION_WEIGHT=2.0
  DEFAULT_ROLE_ANCHOR_PRESERVATION_WEIGHT=10.0
  DEFAULT_ROLE_RELATION_PRESERVATION_WEIGHT=2.0
  DEFAULT_ROLE_ADAPTER_ENERGY_WEIGHT=0.01
elif [[ "${GROUNDING_OBJECTIVE_VERSION}" == "4" || "${GROUNDING_OBJECTIVE_VERSION}" == "5" ]]; then
  DEFAULT_ROLE_ATTENTION_PRESERVATION_WEIGHT=1.0
  DEFAULT_ROLE_POSITION_PRESERVATION_WEIGHT=0.5
  DEFAULT_ROLE_ANCHOR_PRESERVATION_WEIGHT=1.0
  DEFAULT_ROLE_RELATION_PRESERVATION_WEIGHT=0.5
  DEFAULT_ROLE_ADAPTER_ENERGY_WEIGHT=0.01
else
  DEFAULT_ROLE_ATTENTION_PRESERVATION_WEIGHT=0.0
  DEFAULT_ROLE_POSITION_PRESERVATION_WEIGHT=0.0
  DEFAULT_ROLE_ANCHOR_PRESERVATION_WEIGHT=0.0
  DEFAULT_ROLE_RELATION_PRESERVATION_WEIGHT=0.0
  DEFAULT_ROLE_ADAPTER_ENERGY_WEIGHT=0.0
fi
ROLE_ATTENTION_PRESERVATION_WEIGHT="${PGC_V9_ROLE_ATTENTION_PRESERVATION_WEIGHT:-${DEFAULT_ROLE_ATTENTION_PRESERVATION_WEIGHT}}"
ROLE_POSITION_PRESERVATION_WEIGHT="${PGC_V9_ROLE_POSITION_PRESERVATION_WEIGHT:-${DEFAULT_ROLE_POSITION_PRESERVATION_WEIGHT}}"
ROLE_ANCHOR_PRESERVATION_WEIGHT="${PGC_V9_ROLE_ANCHOR_PRESERVATION_WEIGHT:-${DEFAULT_ROLE_ANCHOR_PRESERVATION_WEIGHT}}"
ROLE_RELATION_PRESERVATION_WEIGHT="${PGC_V9_ROLE_RELATION_PRESERVATION_WEIGHT:-${DEFAULT_ROLE_RELATION_PRESERVATION_WEIGHT}}"
ROLE_ADAPTER_ENERGY_WEIGHT="${PGC_V9_ROLE_ADAPTER_ENERGY_WEIGHT:-${DEFAULT_ROLE_ADAPTER_ENERGY_WEIGHT}}"
if [[ "${GROUNDING_OBJECTIVE_VERSION}" == "6" || "${GROUNDING_OBJECTIVE_VERSION}" == "7" || "${GROUNDING_OBJECTIVE_VERSION}" == "8" || "${GROUNDING_OBJECTIVE_VERSION}" == "9" || "${GROUNDING_OBJECTIVE_VERSION}" == "10" || "${GROUNDING_OBJECTIVE_VERSION}" == "11" || "${GROUNDING_OBJECTIVE_VERSION}" == "12" ]]; then
  DEFAULT_STRUCTURED_ASSIGNMENT_WEIGHT=2.0
  # V9.5 interprets this as hard-group:easy-group mass.  1.0 is exact 1:1.
  DEFAULT_STRUCTURED_ASSIGNMENT_HARD_WEIGHT=1.0
  DEFAULT_MULTI_CLAUSE_CONSISTENCY_WEIGHT=2.0
  STRUCTURED_ROLE_SAMPLING=true
elif [[ "${GROUNDING_OBJECTIVE_VERSION}" == "5" ]]; then
  DEFAULT_STRUCTURED_ASSIGNMENT_WEIGHT=2.0
  DEFAULT_STRUCTURED_ASSIGNMENT_HARD_WEIGHT=2.0
  DEFAULT_MULTI_CLAUSE_CONSISTENCY_WEIGHT=1.0
  STRUCTURED_ROLE_SAMPLING=true
else
  DEFAULT_STRUCTURED_ASSIGNMENT_WEIGHT=0.0
  DEFAULT_STRUCTURED_ASSIGNMENT_HARD_WEIGHT=0.0
  DEFAULT_MULTI_CLAUSE_CONSISTENCY_WEIGHT=0.0
  STRUCTURED_ROLE_SAMPLING=false
fi
STRUCTURED_ASSIGNMENT_WEIGHT="${PGC_V9_STRUCTURED_ASSIGNMENT_WEIGHT:-${DEFAULT_STRUCTURED_ASSIGNMENT_WEIGHT}}"
STRUCTURED_ASSIGNMENT_TEMPERATURE="${PGC_V9_STRUCTURED_ASSIGNMENT_TEMPERATURE:-0.10}"
STRUCTURED_ASSIGNMENT_HARD_WEIGHT="${PGC_V9_STRUCTURED_ASSIGNMENT_HARD_WEIGHT:-${DEFAULT_STRUCTURED_ASSIGNMENT_HARD_WEIGHT}}"
MULTI_CLAUSE_CONSISTENCY_WEIGHT="${PGC_V9_MULTI_CLAUSE_CONSISTENCY_WEIGHT:-${DEFAULT_MULTI_CLAUSE_CONSISTENCY_WEIGHT}}"
if [[ "${GROUNDING_OBJECTIVE_VERSION}" == "12" ]]; then
  DEFAULT_CLAUSE_TUPLE_ASSIGNMENT_WEIGHT=4.0
  DEFAULT_CLAUSE_TUPLE_HARD_WEIGHT=1.0
  DEFAULT_CLAUSE_TUPLE_MULTI_CONSISTENCY_WEIGHT=2.0
else
  DEFAULT_CLAUSE_TUPLE_ASSIGNMENT_WEIGHT=0.0
  DEFAULT_CLAUSE_TUPLE_HARD_WEIGHT=0.0
  DEFAULT_CLAUSE_TUPLE_MULTI_CONSISTENCY_WEIGHT=0.0
fi
CLAUSE_TUPLE_ASSIGNMENT_WEIGHT="${PGC_V9_CLAUSE_TUPLE_ASSIGNMENT_WEIGHT:-${DEFAULT_CLAUSE_TUPLE_ASSIGNMENT_WEIGHT}}"
CLAUSE_TUPLE_TEMPERATURE="${PGC_V9_CLAUSE_TUPLE_TEMPERATURE:-0.10}"
CLAUSE_TUPLE_HARD_WEIGHT="${PGC_V9_CLAUSE_TUPLE_HARD_WEIGHT:-${DEFAULT_CLAUSE_TUPLE_HARD_WEIGHT}}"
CLAUSE_TUPLE_MULTI_CONSISTENCY_WEIGHT="${PGC_V9_CLAUSE_TUPLE_MULTI_CONSISTENCY_WEIGHT:-${DEFAULT_CLAUSE_TUPLE_MULTI_CONSISTENCY_WEIGHT}}"
if [[ "${GROUNDING_OBJECTIVE_VERSION}" == "9" || "${GROUNDING_OBJECTIVE_VERSION}" == "10" || "${GROUNDING_OBJECTIVE_VERSION}" == "11" || "${GROUNDING_OBJECTIVE_VERSION}" == "12" ]]; then
  DEFAULT_CLAUSE_ACTIVATION_BALANCE_WEIGHT=1.0
  DEFAULT_CLAUSE_CARDINALITY_WEIGHT=1.0
  DEFAULT_CLAUSE_WORST_SLOT_WEIGHT=2.0
  DEFAULT_CLAUSE_ADAPTER_ENERGY_WEIGHT=0.01
else
  DEFAULT_CLAUSE_ACTIVATION_BALANCE_WEIGHT=0.0
  DEFAULT_CLAUSE_CARDINALITY_WEIGHT=0.0
  DEFAULT_CLAUSE_WORST_SLOT_WEIGHT=0.0
  DEFAULT_CLAUSE_ADAPTER_ENERGY_WEIGHT=0.0
fi
CLAUSE_ACTIVATION_BALANCE_WEIGHT="${PGC_V9_CLAUSE_ACTIVATION_BALANCE_WEIGHT:-${DEFAULT_CLAUSE_ACTIVATION_BALANCE_WEIGHT}}"
CLAUSE_CARDINALITY_WEIGHT="${PGC_V9_CLAUSE_CARDINALITY_WEIGHT:-${DEFAULT_CLAUSE_CARDINALITY_WEIGHT}}"
CLAUSE_WORST_SLOT_WEIGHT="${PGC_V9_CLAUSE_WORST_SLOT_WEIGHT:-${DEFAULT_CLAUSE_WORST_SLOT_WEIGHT}}"
CLAUSE_MULTI_GROUP_WEIGHT="${PGC_V9_CLAUSE_MULTI_GROUP_WEIGHT:-1.0}"
CLAUSE_ADAPTER_ENERGY_WEIGHT="${PGC_V9_CLAUSE_ADAPTER_ENERGY_WEIGHT:-${DEFAULT_CLAUSE_ADAPTER_ENERGY_WEIGHT}}"
CLAUSE_ACTIVATION_RESIDUAL_MAX_ABS="${PGC_V9_CLAUSE_ACTIVATION_RESIDUAL_MAX_ABS:-4.0}"
if [[ "${GROUNDING_OBJECTIVE_VERSION}" == "10" || "${GROUNDING_OBJECTIVE_VERSION}" == "11" || "${GROUNDING_OBJECTIVE_VERSION}" == "12" ]]; then
  DEFAULT_VIEW_FUSION_WEIGHT=2.0
  DEFAULT_VIEW_FUSION_ENERGY_WEIGHT=0.01
  DEFAULT_CLAUSE_SCHEDULER_WEIGHT=1.0
  DEFAULT_CLAUSE_SCHEDULER_ENERGY_WEIGHT=0.01
else
  DEFAULT_VIEW_FUSION_WEIGHT=0.0
  DEFAULT_VIEW_FUSION_ENERGY_WEIGHT=0.0
  DEFAULT_CLAUSE_SCHEDULER_WEIGHT=0.0
  DEFAULT_CLAUSE_SCHEDULER_ENERGY_WEIGHT=0.0
fi
VIEW_FUSION_WEIGHT="${PGC_V9_VIEW_FUSION_WEIGHT:-${DEFAULT_VIEW_FUSION_WEIGHT}}"
VIEW_FUSION_ENERGY_WEIGHT="${PGC_V9_VIEW_FUSION_ENERGY_WEIGHT:-${DEFAULT_VIEW_FUSION_ENERGY_WEIGHT}}"
VIEW_FUSION_RESIDUAL_MAX_ABS="${PGC_V9_VIEW_FUSION_RESIDUAL_MAX_ABS:-4.0}"
CLAUSE_SCHEDULER_WEIGHT="${PGC_V9_CLAUSE_SCHEDULER_WEIGHT:-${DEFAULT_CLAUSE_SCHEDULER_WEIGHT}}"
CLAUSE_SCHEDULER_ENERGY_WEIGHT="${PGC_V9_CLAUSE_SCHEDULER_ENERGY_WEIGHT:-${DEFAULT_CLAUSE_SCHEDULER_ENERGY_WEIGHT}}"
CLAUSE_SCHEDULER_RESIDUAL_MAX_ABS="${PGC_V9_CLAUSE_SCHEDULER_RESIDUAL_MAX_ABS:-1.0}"
CLOSED_LOOP_REBINDING_HIDDEN_DIM="${PGC_V9_CLOSED_LOOP_REBINDING_HIDDEN_DIM:-256}"
CLOSED_LOOP_QUERY_RESIDUAL_MAX_ABS="${PGC_V9_CLOSED_LOOP_QUERY_RESIDUAL_MAX_ABS:-1.0}"
CLOSED_LOOP_STATE_RESIDUAL_MAX_ABS="${PGC_V9_CLOSED_LOOP_STATE_RESIDUAL_MAX_ABS:-2.0}"
PHASE_REBINDING_ENERGY_WEIGHT="${PGC_V9_PHASE_REBINDING_ENERGY_WEIGHT:-0.01}"
PHASE_SAFE_MEMORY_HIDDEN_DIM="${PGC_V9_PHASE_SAFE_MEMORY_HIDDEN_DIM:-256}"
PHASE_SAFE_MEMORY_STATE_COUNT="${PGC_V9_PHASE_SAFE_MEMORY_STATE_COUNT:-4}"
PHASE_SAFE_MEMORY_ROUTING_RESIDUAL_MAX_ABS="${PGC_V9_PHASE_SAFE_MEMORY_ROUTING_RESIDUAL_MAX_ABS:-1.0}"
if [[ "${GROUNDING_OBJECTIVE_VERSION}" == "14" ]]; then
  DEFAULT_PHASE_SAFE_MEMORY_STATE_WEIGHT=1.0
  DEFAULT_PHASE_SAFE_MEMORY_SCHEDULER_WEIGHT=1.0
  DEFAULT_PHASE_SAFE_MEMORY_ENERGY_WEIGHT=0.01
else
  DEFAULT_PHASE_SAFE_MEMORY_STATE_WEIGHT=0.0
  DEFAULT_PHASE_SAFE_MEMORY_SCHEDULER_WEIGHT=0.0
  DEFAULT_PHASE_SAFE_MEMORY_ENERGY_WEIGHT=0.0
fi
PHASE_SAFE_MEMORY_STATE_WEIGHT="${PGC_V9_PHASE_SAFE_MEMORY_STATE_WEIGHT:-${DEFAULT_PHASE_SAFE_MEMORY_STATE_WEIGHT}}"
PHASE_SAFE_MEMORY_SCHEDULER_WEIGHT="${PGC_V9_PHASE_SAFE_MEMORY_SCHEDULER_WEIGHT:-${DEFAULT_PHASE_SAFE_MEMORY_SCHEDULER_WEIGHT}}"
PHASE_SAFE_MEMORY_ENERGY_WEIGHT="${PGC_V9_PHASE_SAFE_MEMORY_ENERGY_WEIGHT:-${DEFAULT_PHASE_SAFE_MEMORY_ENERGY_WEIGHT}}"
if [[ "${GROUNDING_OBJECTIVE_VERSION}" == "13" ]]; then
  # Only the new second-pass adapter is trainable. Losses on frozen-only
  # outputs are disabled; role/tuple, state, phase, anchor, and scheduler
  # supervision all backpropagate through the rebinding path.
  ROLE_ASSIGNMENT_WEIGHT="${PGC_V9_ROLE_ASSIGNMENT_WEIGHT:-1.0}"
  ROLE_ASSIGNMENT_HARD_WEIGHT="${PGC_V9_ROLE_ASSIGNMENT_HARD_WEIGHT:-0.5}"
  STRUCTURED_ASSIGNMENT_WEIGHT="${PGC_V9_STRUCTURED_ASSIGNMENT_WEIGHT:-2.0}"
  STRUCTURED_ASSIGNMENT_HARD_WEIGHT="${PGC_V9_STRUCTURED_ASSIGNMENT_HARD_WEIGHT:-1.0}"
  MULTI_CLAUSE_CONSISTENCY_WEIGHT="${PGC_V9_MULTI_CLAUSE_CONSISTENCY_WEIGHT:-2.0}"
  CLAUSE_TUPLE_ASSIGNMENT_WEIGHT="${PGC_V9_CLAUSE_TUPLE_ASSIGNMENT_WEIGHT:-4.0}"
  CLAUSE_TUPLE_HARD_WEIGHT="${PGC_V9_CLAUSE_TUPLE_HARD_WEIGHT:-1.0}"
  CLAUSE_TUPLE_MULTI_CONSISTENCY_WEIGHT="${PGC_V9_CLAUSE_TUPLE_MULTI_CONSISTENCY_WEIGHT:-2.0}"
  CLAUSE_ACTIVATION_BALANCE_WEIGHT="${PGC_V9_CLAUSE_ACTIVATION_BALANCE_WEIGHT:-0.0}"
  CLAUSE_CARDINALITY_WEIGHT="${PGC_V9_CLAUSE_CARDINALITY_WEIGHT:-0.0}"
  CLAUSE_WORST_SLOT_WEIGHT="${PGC_V9_CLAUSE_WORST_SLOT_WEIGHT:-0.0}"
  CLAUSE_ADAPTER_ENERGY_WEIGHT="${PGC_V9_CLAUSE_ADAPTER_ENERGY_WEIGHT:-0.0}"
  VIEW_FUSION_WEIGHT="${PGC_V9_VIEW_FUSION_WEIGHT:-0.0}"
  VIEW_FUSION_ENERGY_WEIGHT="${PGC_V9_VIEW_FUSION_ENERGY_WEIGHT:-0.0}"
  CLAUSE_SCHEDULER_WEIGHT="${PGC_V9_CLAUSE_SCHEDULER_WEIGHT:-1.0}"
  CLAUSE_SCHEDULER_ENERGY_WEIGHT="${PGC_V9_CLAUSE_SCHEDULER_ENERGY_WEIGHT:-0.0}"
  ROLE_ATTENTION_PRESERVATION_WEIGHT="${PGC_V9_ROLE_ATTENTION_PRESERVATION_WEIGHT:-0.0}"
  ROLE_POSITION_PRESERVATION_WEIGHT="${PGC_V9_ROLE_POSITION_PRESERVATION_WEIGHT:-0.0}"
  ROLE_ANCHOR_PRESERVATION_WEIGHT="${PGC_V9_ROLE_ANCHOR_PRESERVATION_WEIGHT:-0.0}"
  ROLE_RELATION_PRESERVATION_WEIGHT="${PGC_V9_ROLE_RELATION_PRESERVATION_WEIGHT:-0.0}"
  ROLE_ADAPTER_ENERGY_WEIGHT="${PGC_V9_ROLE_ADAPTER_ENERGY_WEIGHT:-0.0}"
  STRUCTURED_ROLE_SAMPLING=true
fi
case "${STAGE}" in
  grounding)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "2" ]]; then
      echo "Formal V9.1 grounding requires objective version 2." >&2
      exit 1
    fi
    ;;
  action-direct-geometry)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "17" ]]; then
      echo "Formal V9.17 direct geometry-action calibration requires objective version 17." >&2
      exit 1
    fi
    ;;
  action-phase-residual)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "18" ]]; then
      echo "Formal V9.18 phase-residual imitation requires objective version 18." >&2
      exit 1
    fi
    ;;
  action-phase-servo)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "19" ]]; then
      echo "Formal V9.19 hard-routed phase servo requires objective version 19." >&2
      exit 1
    fi
    ;;
  grounding-role)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "3" ]]; then
      echo "Formal V9.2 role repair requires objective version 3." >&2
      exit 1
    fi
    ;;
  grounding-role-adapter)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "4" ]]; then
      echo "Formal V9.3 role-adapter repair requires objective version 4." >&2
      exit 1
    fi
    ;;
  grounding-structured-role)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "5" ]]; then
      echo "Formal V9.4 structured role repair requires objective version 5." >&2
      exit 1
    fi
    ;;
  grounding-balanced-role)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "6" ]]; then
      echo "Formal V9.5 balanced role binding requires objective version 6." >&2
      exit 1
    fi
    ;;
  grounding-hard-role)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "7" ]]; then
      echo "Formal V9.6 hard-role curriculum requires objective version 7." >&2
      exit 1
    fi
    ;;
  grounding-exclusive-role)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "8" ]]; then
      echo "Formal V9.7 exclusive-role calibration requires objective version 8." >&2
      exit 1
    fi
    ;;
  grounding-clause-calibration)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "9" ]]; then
      echo "Formal V9.8 clause calibration requires objective version 9." >&2
      exit 1
    fi
    ;;
  grounding-view-scheduler)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "10" ]]; then
      echo "Formal V9.9 view fusion and clause scheduling requires objective version 10." >&2
      exit 1
    fi
    ;;
  grounding-all-entity-role)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "11" ]]; then
      echo "Formal V9.10 exclusive all-entity role binding requires objective version 11." >&2
      exit 1
    fi
    ;;
  grounding-clause-tuple)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "12" ]]; then
      echo "Formal V9.11 clause-tuple binding requires objective version 12." >&2
      exit 1
    fi
    ;;
  grounding-phase-rebinding)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "13" ]]; then
      echo "Formal V9.12 closed-loop phase rebinding requires objective version 13." >&2
      exit 1
    fi
    ;;
  grounding-phase-memory)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "14" ]]; then
      echo "Formal V9.13 phase-safe clause memory requires objective version 14." >&2
      exit 1
    fi
    ;;
  action-completion-only)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "14" ]]; then
      echo "Formal V9.14 completion-only ERAF--Proposal joint training requires objective version 14." >&2
      exit 1
    fi
    ;;
  action-geometry-causal)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "15" ]]; then
      echo "Formal V9.15 geometry-causal action training requires objective version 15." >&2
      exit 1
    fi
    ;;
  action-semantic-causal)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "16" ]]; then
      echo "Formal V9.16 semantic-causal action calibration requires objective version 16." >&2
      exit 1
    fi
    ;;
  action|verifier)
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" == "14" ]]; then
      echo "Objective-v14 action must use the audited action-completion-only V9.14 stage." >&2
      exit 1
    fi
    if [[ "${GROUNDING_OBJECTIVE_VERSION}" != "2" && "${GROUNDING_OBJECTIVE_VERSION}" != "3" && "${GROUNDING_OBJECTIVE_VERSION}" != "4" && "${GROUNDING_OBJECTIVE_VERSION}" != "5" && "${GROUNDING_OBJECTIVE_VERSION}" != "6" && "${GROUNDING_OBJECTIVE_VERSION}" != "7" && "${GROUNDING_OBJECTIVE_VERSION}" != "8" && "${GROUNDING_OBJECTIVE_VERSION}" != "9" && "${GROUNDING_OBJECTIVE_VERSION}" != "10" && "${GROUNDING_OBJECTIVE_VERSION}" != "11" && "${GROUNDING_OBJECTIVE_VERSION}" != "12" && "${GROUNDING_OBJECTIVE_VERSION}" != "13" ]]; then
      echo "V9 action/verifier requires grounding objective version 2 through 13." >&2
      exit 1
    fi
    ;;
esac
HARD_ROLE_CURRICULUM=false
HARD_ROLE_INDEX_PATH="${PGC_V9_HARD_ROLE_INDEX_PATH:-}"
if [[ "${STAGE}" == "grounding-hard-role" || "${STAGE}" == "grounding-exclusive-role" || "${STAGE}" == "grounding-clause-tuple" ]]; then
  HARD_ROLE_CURRICULUM=true
  if [[ -z "${HARD_ROLE_INDEX_PATH}" || ! -f "${HARD_ROLE_INDEX_PATH}" ]]; then
    echo "V9.6/V9.7/V9.11 requires PGC_V9_HARD_ROLE_INDEX_PATH from its audited teacher." >&2
    exit 1
  fi
  HARD_ROLE_INDEX_PATH="$(cd -- "$(dirname -- "${HARD_ROLE_INDEX_PATH}")" && pwd -P)/$(basename -- "${HARD_ROLE_INDEX_PATH}")"
fi
CLOSED_LOOP_GROUNDING_DATASET="${PGC_V9_CLOSED_LOOP_GROUNDING_DATASET:-}"
CLOSED_LOOP_GROUNDING_SIDECAR="${PGC_V9_CLOSED_LOOP_GROUNDING_SIDECAR:-}"
if [[ "${STAGE}" == "grounding-phase-rebinding" || "${STAGE}" == "grounding-phase-memory" || "${STAGE}" == "action-completion-only" || "${STAGE}" == "action-geometry-causal" || "${STAGE}" == "action-semantic-causal" || "${STAGE}" == "action-direct-geometry" || "${STAGE}" == "action-phase-residual" || "${STAGE}" == "action-phase-servo" ]]; then
  if [[ -z "${CLOSED_LOOP_GROUNDING_DATASET}" || ! -d "${CLOSED_LOOP_GROUNDING_DATASET}" ]]; then
    echo "V9.12 through V9.19 requires PGC_V9_CLOSED_LOOP_GROUNDING_DATASET." >&2
    exit 1
  fi
  if [[ -z "${CLOSED_LOOP_GROUNDING_SIDECAR}" || ! -d "${CLOSED_LOOP_GROUNDING_SIDECAR}" ]]; then
    echo "V9.12 through V9.19 requires PGC_V9_CLOSED_LOOP_GROUNDING_SIDECAR." >&2
    exit 1
  fi
fi
MAX_STEPS=$((STAGE_START_STEP + STAGE_STEPS))
MASK_WEIGHT=1.0
ENTITY_WEIGHT=1.0
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
if [[ "${GROUNDING_OBJECTIVE_VERSION}" == "14" || "${GROUNDING_OBJECTIVE_VERSION}" == "15" || "${GROUNDING_OBJECTIVE_VERSION}" == "16" || "${GROUNDING_OBJECTIVE_VERSION}" == "17" || "${GROUNDING_OBJECTIVE_VERSION}" == "18" || "${GROUNDING_OBJECTIVE_VERSION}" == "19" ]]; then
  # V9.13+ freeze the validated V9.11 perception/grounding losses. V9.15-19
  # inherit V9.14's exact zero-loss contract while calibrating action paths.
  MASK_WEIGHT=0.0
  ATTENTION_MASK_WEIGHT=0.0
  ENTITY_WEIGHT=0.0
  RELATION_WEIGHT=0.0
  ANCHOR_WEIGHT=0.0
  POSITION_WEIGHT=0.0
  PHASE_WEIGHT=0.0
  ROLE_SWAP_WEIGHT=0.0
  ROLE_OVERLAP_WEIGHT=0.0
  ROLE_ASSIGNMENT_WEIGHT=0.0
  ROLE_ASSIGNMENT_HARD_WEIGHT=0.0
  STRUCTURED_ASSIGNMENT_WEIGHT=0.0
  STRUCTURED_ASSIGNMENT_HARD_WEIGHT=0.0
  MULTI_CLAUSE_CONSISTENCY_WEIGHT=0.0
  CLAUSE_TUPLE_ASSIGNMENT_WEIGHT=0.0
  CLAUSE_TUPLE_HARD_WEIGHT=0.0
  CLAUSE_TUPLE_MULTI_CONSISTENCY_WEIGHT=0.0
  CLAUSE_ACTIVATION_BALANCE_WEIGHT=0.0
  CLAUSE_CARDINALITY_WEIGHT=0.0
  CLAUSE_WORST_SLOT_WEIGHT=0.0
  CLAUSE_ADAPTER_ENERGY_WEIGHT=0.0
  VIEW_FUSION_WEIGHT=0.0
  VIEW_FUSION_ENERGY_WEIGHT=0.0
  CLAUSE_SCHEDULER_WEIGHT=0.0
  CLAUSE_SCHEDULER_ENERGY_WEIGHT=0.0
  PHASE_REBINDING_ENERGY_WEIGHT=0.0
  ROLE_ATTENTION_PRESERVATION_WEIGHT=0.0
  ROLE_POSITION_PRESERVATION_WEIGHT=0.0
  ROLE_ANCHOR_PRESERVATION_WEIGHT=0.0
  ROLE_RELATION_PRESERVATION_WEIGHT=0.0
  ROLE_ADAPTER_ENERGY_WEIGHT=0.0
  STRUCTURED_ROLE_SAMPLING=false
fi

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

ACTION_EEF_SCALE="[1.0,1.0,1.0]"
ACTION_EEF_BIAS="[0.0,0.0,0.0]"
if [[ "${GROUNDING_OBJECTIVE_VERSION}" == "19" ]]; then
  EEF_AFFINE="$(${PYTHON_BIN} - "${STATS_PATH}" "${NATIVE_SIDECAR}/index.json" <<'PY'
import json
import sys

stats = json.load(open(sys.argv[1], "r", encoding="utf-8"))
index = json.load(open(sys.argv[2], "r", encoding="utf-8"))
state_stats = stats["state"]
field = state_stats.get("default")
if field is None:
    if len(state_stats) != 1:
        raise SystemExit(
            "V9.19 cannot identify the canonical proprio state statistics."
        )
    field = next(iter(state_stats.values()))
state_min = [float(value) for value in field["global_min"][:3]]
state_max = [float(value) for value in field["global_max"][:3]]
workspace_min = [float(value) for value in index["workspace_min"]]
workspace_max = [float(value) for value in index["workspace_max"]]
scale = []
bias = []
for lower, upper, workspace_lower, workspace_upper in zip(
    state_min, state_max, workspace_min, workspace_max
):
    workspace_range = workspace_upper - workspace_lower
    if upper <= lower or workspace_range <= 0:
        raise SystemExit("V9.19 received invalid state/workspace bounds.")
    scale.append((upper - lower) / workspace_range)
    bias.append(
        (upper + lower - 2.0 * workspace_lower) / workspace_range - 1.0
    )
print(json.dumps(scale, separators=(",", ":")))
print(json.dumps(bias, separators=(",", ":")))
PY
)"
  ACTION_EEF_SCALE="$(printf '%s\n' "${EEF_AFFINE}" | sed -n '1p')"
  ACTION_EEF_BIAS="$(printf '%s\n' "${EEF_AFFINE}" | sed -n '2p')"
fi

NATIVE_DATASET="$(cd -- "${NATIVE_DATASET}" && pwd -P)"
ORIGINAL_CF_DATASET="$(cd -- "${ORIGINAL_CF_DATASET}" && pwd -P)"
STRICT_CF_DATASET="$(cd -- "${STRICT_CF_DATASET}" && pwd -P)"
NATIVE_SIDECAR="$(cd -- "${NATIVE_SIDECAR}" && pwd -P)"
ORIGINAL_CF_SIDECAR="$(cd -- "${ORIGINAL_CF_SIDECAR}" && pwd -P)"
STRICT_CF_SIDECAR="$(cd -- "${STRICT_CF_SIDECAR}" && pwd -P)"
if [[ "${STAGE}" == "grounding-phase-rebinding" || "${STAGE}" == "grounding-phase-memory" || "${STAGE}" == "action-completion-only" || "${STAGE}" == "action-geometry-causal" || "${STAGE}" == "action-semantic-causal" || "${STAGE}" == "action-direct-geometry" || "${STAGE}" == "action-phase-residual" || "${STAGE}" == "action-phase-servo" ]]; then
  CLOSED_LOOP_GROUNDING_DATASET="$(cd -- "${CLOSED_LOOP_GROUNDING_DATASET}" && pwd -P)"
  CLOSED_LOOP_GROUNDING_SIDECAR="$(cd -- "${CLOSED_LOOP_GROUNDING_SIDECAR}" && pwd -P)"
fi

"${PYTHON_BIN}" - \
  "${STAGE}" "${START_STEP}" "${GROUNDING_OBJECTIVE_VERSION}" \
  "${BASE_CHECKPOINT}" "${INIT_CHECKPOINT}" \
  "${NATIVE_DATASET}" "${ORIGINAL_CF_DATASET}" "${STRICT_CF_DATASET}" \
  "${NATIVE_SIDECAR}" "${ORIGINAL_CF_SIDECAR}" "${STRICT_CF_SIDECAR}" \
  "${CLOSED_LOOP_GROUNDING_DATASET:-null}" \
  "${CLOSED_LOOP_GROUNDING_SIDECAR:-null}" <<'PY'
import json
import math
import pathlib
import sys
import torch

(
    stage,
    expected_step,
    requested_objective,
    base_checkpoint,
    checkpoint,
    native_dataset,
    original_dataset,
    strict_dataset,
    native_sidecar,
    original_sidecar,
    strict_sidecar,
    closed_loop_dataset,
    closed_loop_sidecar,
) = sys.argv[1:]
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
fmt = str(payload.get("format", ""))
version = int((payload.get("architecture_metadata") or {}).get("policy_guard_version", 0))
if stage == "grounding":
    if fmt != "fastwam_policy_guard_v5" or version != 5:
        raise SystemExit(
            "V9 grounding must warm-start from an exact suite-specific PGC V5 checkpoint."
        )
elif stage in {
    "grounding-role",
    "grounding-role-adapter",
    "grounding-structured-role",
    "grounding-balanced-role",
    "grounding-hard-role",
    "grounding-exclusive-role",
    "grounding-clause-calibration",
    "grounding-view-scheduler",
    "grounding-all-entity-role",
    "grounding-clause-tuple",
    "grounding-phase-rebinding",
    "grounding-phase-memory",
}:
    metadata = payload.get("architecture_metadata") or {}
    if fmt != "fastwam_policy_guard_v9" or version != 9:
        raise SystemExit("V9 role repair must resume from a V9 checkpoint.")
    expected_saved_objective = {
        "grounding-role": 2,
        "grounding-role-adapter": 2,
        "grounding-structured-role": 4,
        "grounding-balanced-role": 4,
        "grounding-hard-role": 4,
        "grounding-exclusive-role": 7,
        "grounding-clause-calibration": 8,
        "grounding-view-scheduler": 9,
        "grounding-all-entity-role": 10,
        "grounding-clause-tuple": 11,
        "grounding-phase-rebinding": 12,
        "grounding-phase-memory": 12,
    }[stage]
    if (
        int(metadata.get("eraf_grounding_objective_version", -1))
        != expected_saved_objective
    ):
        raise SystemExit(
            "V9 role repair requires the completed objective-v"
            f"{expected_saved_objective} grounding checkpoint."
        )
    if int(payload.get("step", -1)) != int(expected_step):
        raise SystemExit(
            f"V9 role repair requires checkpoint step {expected_step}; "
            f"got {payload.get('step')}."
        )
    if metadata.get("eraf_training_stage") != "grounding":
        raise SystemExit("V9 role repair requires a grounding-stage checkpoint.")
    expected_objective = {
        "grounding-role": 3,
        "grounding-role-adapter": 4,
        "grounding-structured-role": 5,
        "grounding-balanced-role": 6,
        "grounding-hard-role": 7,
        "grounding-exclusive-role": 8,
        "grounding-clause-calibration": 9,
        "grounding-view-scheduler": 10,
        "grounding-all-entity-role": 11,
        "grounding-clause-tuple": 12,
        "grounding-phase-rebinding": 13,
        "grounding-phase-memory": 14,
    }[stage]
    if int(requested_objective) != expected_objective:
        raise SystemExit(
            f"{stage} must configure objective version {expected_objective}."
        )
    if stage == "grounding-structured-role" and (
        metadata.get("eraf_role_adapter_trainable_scope")
        != "role_assignment_adapter_only"
    ):
        raise SystemExit(
            "V9.4 must warm-start from the completed V9.3 role-adapter-only "
            "checkpoint."
        )
    if stage == "grounding-balanced-role" and (
        metadata.get("eraf_role_adapter_trainable_scope")
        != "role_assignment_adapter_only"
    ):
        raise SystemExit(
            "V9.5 must warm-start from the completed clean V9.3 "
            "role-adapter-only checkpoint, not V9.4."
        )
    if stage == "grounding-hard-role" and (
        metadata.get("eraf_role_adapter_trainable_scope")
        != "role_assignment_adapter_only"
    ):
        raise SystemExit(
            "V9.6 must warm-start from the completed clean V9.3 "
            "role-adapter-only checkpoint, not V9.4/V9.5."
        )
    if stage == "grounding-exclusive-role" and (
        metadata.get("eraf_role_adapter_trainable_scope")
        != "global_hard_curriculum_balanced_visual_role_binding_adapter_only"
    ):
        raise SystemExit(
            "V9.7 must warm-start from the completed V9.6 hard-role "
            "checkpoint at step 3000."
        )
    if stage == "grounding-clause-calibration" and (
        metadata.get("eraf_role_adapter_trainable_scope")
        != "exclusive_evidence_global_hard_curriculum_"
        "balanced_visual_role_binding_adapter_only"
    ):
        raise SystemExit(
            "V9.8 must warm-start from the completed V9.7 exclusive-role "
            "checkpoint at step 3250."
        )
    if stage == "grounding-view-scheduler" and (
        metadata.get("eraf_role_adapter_trainable_scope")
        != "clause_activation_calibration_adapter_only"
    ):
        raise SystemExit(
            "V9.9 must warm-start from the completed V9.8 clause-calibration "
            "checkpoint at step 3750."
        )
    if stage == "grounding-all-entity-role" and (
        metadata.get("eraf_role_adapter_trainable_scope")
        != "clause_activation_plus_balanced_role_plus_visibility_gated_"
        "view_fusion_plus_unfinished_clause_scheduler"
    ):
        raise SystemExit(
            "V9.10 must warm-start from the completed V9.9 view/scheduler "
            "checkpoint at step 4750."
        )
    if stage == "grounding-clause-tuple" and (
        metadata.get("eraf_role_adapter_trainable_scope")
        != "exclusive_all_entity_balanced_visual_role_binding_adapter_only"
    ):
        raise SystemExit(
            "V9.11 must warm-start from the completed V9.10 all-entity "
            "checkpoint at step 5750."
        )
    if stage == "grounding-phase-rebinding" and (
        metadata.get("eraf_role_adapter_trainable_scope")
        != "audited_hard_clause_tuple_balanced_visual_role_binding_adapter_only"
    ):
        raise SystemExit(
            "V9.12 must warm-start from the completed V9.11 clause-tuple "
            "checkpoint at step 6250."
        )
    if stage == "grounding-phase-memory" and (
        metadata.get("eraf_role_adapter_trainable_scope")
        != "audited_hard_clause_tuple_balanced_visual_role_binding_adapter_only"
    ):
        raise SystemExit(
            "V9.13 must warm-start directly from the completed V9.11 "
            "clause-tuple checkpoint at step 6250, not V9.12."
        )
else:
    if fmt != "fastwam_policy_guard_v9" or version != 9:
        raise SystemExit(f"V9 {stage} must resume from a V9 checkpoint.")
    grounding_objective_version = int(
        (payload.get("architecture_metadata") or {}).get(
            "eraf_grounding_objective_version", 1
        )
    )
    objective_upgrade = (
        (
            stage == "action-geometry-causal"
            and grounding_objective_version == 14
            and int(requested_objective) == 15
        )
        or (
            stage == "action-semantic-causal"
            and grounding_objective_version == 15
            and int(requested_objective) == 16
        )
        or (
            stage == "action-direct-geometry"
            and grounding_objective_version == 16
            and int(requested_objective) == 17
        )
        or (
            stage == "action-phase-residual"
            and grounding_objective_version == 17
            and int(requested_objective) == 18
        )
        or (
            stage == "action-phase-servo"
            and grounding_objective_version == 18
            and int(requested_objective) == 19
        )
    )
    if grounding_objective_version != int(requested_objective) and not objective_upgrade:
        raise SystemExit(
            f"V9 {stage} grounding objective mismatch: checkpoint="
            f"{grounding_objective_version}, requested={requested_objective}."
        )
    if int(payload.get("step", -1)) != int(expected_step):
        raise SystemExit(
            f"V9 {stage} requires checkpoint step {expected_step}; got {payload.get('step')}."
        )
    expected_input_stage = {
        "action": "grounding",
        "action-completion-only": "grounding",
        "action-geometry-causal": "action",
        "action-semantic-causal": "action",
        "action-direct-geometry": "action",
        "action-phase-residual": "action",
        "action-phase-servo": "action",
        "verifier": "action",
    }[stage]
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
    if stage == "action-completion-only":
        metadata = payload.get("architecture_metadata") or {}
        if (
            metadata.get("eraf_role_adapter_trainable_scope")
            != "phase_safe_temporal_clause_memory_only"
            or metadata.get("eraf_phase_safe_memory_contract")
            != "explicit_cross_replan_pending_holding_retry_completed"
            or metadata.get("eraf_policy_state_contract")
            != "explicit_caller_owned_reset_per_episode"
        ):
            raise SystemExit(
                "V9.14 must warm-start from the admitted V9.13 phase-memory "
                "checkpoint at step 7250."
            )
    if stage == "action-geometry-causal":
        metadata = payload.get("architecture_metadata") or {}
        if (
            grounding_objective_version != 14
            or metadata.get("eraf_training_stage") != "action"
            or not bool(metadata.get("eraf_action_joint_training", False))
            or metadata.get("eraf_action_joint_contract")
            != "frozen_eraf_perception_plus_action_bridge_and_proposal"
            or metadata.get("eraf_action_trainable_scope")
            != "base_query_projection_relation_attention_query_embedding_"
            "delta_plus_action_chunk_proposal"
            or metadata.get("eraf_role_adapter_trainable_scope")
            != "frozen_eraf_perception_action_bridge_plus_proposal"
            or metadata.get("eraf_policy_state_contract")
            != "monotonic_completed_bitset_no_pending_holding_retry_recurrence"
        ):
            raise SystemExit(
                "V9.15 must warm-start from the completed V9.14 "
                "completion-only ERAF--Proposal checkpoint at step 11250."
            )
    if stage == "action-semantic-causal":
        metadata = payload.get("architecture_metadata") or {}
        if (
            grounding_objective_version != 15
            or metadata.get("eraf_training_stage") != "action"
            or not bool(metadata.get("eraf_action_joint_training", False))
            or metadata.get("eraf_action_joint_contract")
            != "frozen_eraf_perception_plus_phase_conditioned_geometry_"
            "bridge_legacy_bridge_and_proposal"
            or metadata.get("eraf_action_trainable_scope")
            != "phase_conditioned_subject_reference_anchor_action_bridge_"
            "plus_legacy_bridge_and_action_chunk_proposal"
            or metadata.get("eraf_role_adapter_trainable_scope")
            != "frozen_eraf_perception_action_bridge_plus_proposal"
            or metadata.get("eraf_policy_state_contract")
            != "monotonic_completed_bitset_no_pending_holding_retry_recurrence"
        ):
            raise SystemExit(
                "V9.16 must warm-start from the completed V9.15 "
                "geometry-causal action checkpoint at step 13250."
            )
    if stage == "action-direct-geometry":
        metadata = payload.get("architecture_metadata") or {}
        if (
            grounding_objective_version != 16
            or metadata.get("eraf_training_stage") != "action"
            or not bool(metadata.get("eraf_action_joint_training", False))
            or metadata.get("eraf_action_joint_contract")
            != "frozen_eraf_perception_proposal_and_legacy_bridge_plus_"
            "semantic_causal_action_grounding_bridge"
            or metadata.get("eraf_action_trainable_scope")
            != "semantic_causal_action_grounding_bridge_only"
            or metadata.get("eraf_role_adapter_trainable_scope")
            != "semantic_causal_action_grounding_bridge_only"
            or metadata.get("eraf_policy_state_contract")
            != "monotonic_completed_bitset_no_pending_holding_retry_recurrence"
        ):
            raise SystemExit(
                "V9.17 must warm-start from the completed V9.16 semantic-causal "
                "action checkpoint at step 13750."
            )
    if stage == "action-phase-residual":
        metadata = payload.get("architecture_metadata") or {}
        if (
            grounding_objective_version != 17
            or metadata.get("eraf_training_stage") != "action"
            or not bool(metadata.get("eraf_action_joint_training", False))
            or metadata.get("eraf_action_joint_contract")
            != "frozen_eraf_v916_bridge_and_proposal_plus_direct_"
            "eef_relative_geometry_action_adapter"
            or metadata.get("eraf_action_trainable_scope")
            != "phase_conditioned_relative_geometry_action_adapter_only"
            or metadata.get("eraf_role_adapter_trainable_scope")
            != "phase_conditioned_relative_geometry_action_adapter_only"
            or metadata.get("eraf_policy_state_contract")
            != "monotonic_completed_bitset_no_pending_holding_retry_recurrence"
        ):
            raise SystemExit(
                "V9.18 must warm-start from the completed V9.17 direct "
                "geometry-action checkpoint at step 14250."
            )
    if stage == "action-phase-servo":
        metadata = payload.get("architecture_metadata") or {}
        if (
            grounding_objective_version != 18
            or metadata.get("eraf_training_stage") != "action"
            or not bool(metadata.get("eraf_action_joint_training", False))
            or metadata.get("eraf_action_joint_contract")
            != "frozen_eraf_v917_stack_plus_phase_balanced_direct_"
            "geometry_residual_imitation"
            or metadata.get("eraf_action_trainable_scope")
            != "phase_conditioned_geometry_adapter_only_with_phase_"
            "balanced_residual_imitation"
            or metadata.get("eraf_role_adapter_trainable_scope")
            != "phase_conditioned_geometry_adapter_only_with_phase_"
            "balanced_residual_imitation"
            or metadata.get("eraf_action_phase_residual_contract")
            != "phase_balanced_bounded_expert_minus_frozen_v9_17_candidate_"
            "prefix_residual_imitation"
            or metadata.get("eraf_policy_state_contract")
            != "monotonic_completed_bitset_no_pending_holding_retry_recurrence"
        ):
            raise SystemExit(
                "V9.19 must warm-start from the completed V9.18 phase-residual "
                "checkpoint at step 15250."
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
datasets = [native_dataset]
sidecars = [native_sidecar]
expected_action_contracts = [
    ("native", "fastwam_gripper_open_1_close_0", "fastwam_to_libero_env")
]
if stage in {
    "grounding-phase-rebinding",
    "grounding-phase-memory",
    "action-completion-only",
    "action-geometry-causal",
    "action-semantic-causal",
    "action-direct-geometry",
    "action-phase-residual",
    "action-phase-servo",
}:
    if closed_loop_dataset == "null" or closed_loop_sidecar == "null":
        raise SystemExit(
            "V9.12/V9.13/V9.14 closed-loop dataset/sidecar is missing."
        )
    datasets.append(closed_loop_dataset)
    sidecars.append(closed_loop_sidecar)
    expected_action_contracts.append(
        ("native", "fastwam_gripper_open_1_close_0", "fastwam_to_libero_env")
    )
datasets.extend((original_dataset, strict_dataset))
sidecars.extend((original_sidecar, strict_sidecar))
expected_action_contracts.extend(
    (
        ("counterfactual", "libero_env_gripper_open_minus1_close_plus1", "identity"),
        ("counterfactual", "libero_env_gripper_open_minus1_close_plus1", "identity"),
    )
)
workspace_contracts = []
for dataset, sidecar, expected_contract in zip(
    datasets, sidecars, expected_action_contracts, strict=True
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
    try:
        workspace_min = tuple(float(value) for value in index["workspace_min"])
        workspace_max = tuple(float(value) for value in index["workspace_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"Invalid ERAF workspace contract at {index_path}: {exc}"
        ) from exc
    if len(workspace_min) != 3 or len(workspace_max) != 3:
        raise SystemExit(
            f"Invalid ERAF workspace dimensions at {index_path}: "
            f"min={workspace_min} max={workspace_max}."
        )
    workspace_contracts.append(
        (str(pathlib.Path(sidecar).resolve()), workspace_min, workspace_max)
    )
    if dataset == closed_loop_dataset and (
        index.get("state_distribution")
        != "immutable_base_closed_loop_replan"
    ):
        raise SystemExit(
            "V9.12/V9.13 closed-loop sidecar has the wrong state-distribution contract."
        )
reference_sidecar, reference_min, reference_max = workspace_contracts[0]
workspace_mismatches = [
    (sidecar, lower, upper)
    for sidecar, lower, upper in workspace_contracts[1:]
    if not all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-6)
        for left, right in zip(lower + upper, reference_min + reference_max)
    )
]
if workspace_mismatches:
    details = "\n".join(
        f"  {sidecar}: min={lower} max={upper}"
        for sidecar, lower, upper in workspace_contracts
    )
    raise SystemExit(
        "ERAF workspace mismatch detected before distributed launch.\n"
        f"Reference: {reference_sidecar}\n{details}\n"
        "Select an already canonical sidecar or create an out-of-place copy "
        "with scripts/migrate_pgc_eraf_workspace.py."
    )
PY

json_array() {
  "${PYTHON_BIN}" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"
}
if [[ "${STAGE}" == "grounding-phase-rebinding" || "${STAGE}" == "grounding-phase-memory" || "${STAGE}" == "action-completion-only" || "${STAGE}" == "action-geometry-causal" || "${STAGE}" == "action-semantic-causal" || "${STAGE}" == "action-direct-geometry" || "${STAGE}" == "action-phase-residual" || "${STAGE}" == "action-phase-servo" ]]; then
  NATIVE_JSON="$(json_array "${NATIVE_DATASET}" "${CLOSED_LOOP_GROUNDING_DATASET}")"
  SIDECAR_JSON="$(json_array "${NATIVE_SIDECAR}" "${CLOSED_LOOP_GROUNDING_SIDECAR}" "${ORIGINAL_CF_SIDECAR}" "${STRICT_CF_SIDECAR}")"
  CLOSED_LOOP_NATIVE_DATASET_COUNT=1
else
  NATIVE_JSON="$(json_array "${NATIVE_DATASET}")"
  SIDECAR_JSON="$(json_array "${NATIVE_SIDECAR}" "${ORIGINAL_CF_SIDECAR}" "${STRICT_CF_SIDECAR}")"
  CLOSED_LOOP_NATIVE_DATASET_COUNT=0
fi
if [[ "${STAGE}" == "grounding-phase-rebinding" ]]; then
  CLOSED_LOOP_REBINDING=true
else
  CLOSED_LOOP_REBINDING=false
fi
if [[ "${STAGE}" == "grounding-phase-memory" || "${STAGE}" == "action-completion-only" || "${STAGE}" == "action-geometry-causal" || "${STAGE}" == "action-semantic-causal" || "${STAGE}" == "action-direct-geometry" || "${STAGE}" == "action-phase-residual" || "${STAGE}" == "action-phase-servo" ]]; then
  PHASE_SAFE_MEMORY=true
else
  PHASE_SAFE_MEMORY=false
fi
if [[ "${STAGE}" == "action-completion-only" || "${STAGE}" == "action-geometry-causal" || "${STAGE}" == "action-semantic-causal" || "${STAGE}" == "action-direct-geometry" || "${STAGE}" == "action-phase-residual" || "${STAGE}" == "action-phase-servo" ]]; then
  COMPLETION_ONLY_MEMORY=true
  ACTION_JOINT_TRAINING=true
  GROUNDING_AUX_WEIGHT=0.0
else
  COMPLETION_ONLY_MEMORY=false
  ACTION_JOINT_TRAINING=false
  GROUNDING_AUX_WEIGHT=0.25
fi
if [[ "${STAGE}" == "action-semantic-causal" ]]; then
  ACTION_GROUNDING_LEARNING_RATE="${PGC_V9_ACTION_GROUNDING_LEARNING_RATE:-2.0e-5}"
  ACTION_CAUSAL_RANKING_WEIGHT="${PGC_V9_ACTION_CAUSAL_RANKING_WEIGHT:-2.0}"
else
  ACTION_GROUNDING_LEARNING_RATE="${PGC_V9_ACTION_GROUNDING_LEARNING_RATE:-1.0e-4}"
  ACTION_CAUSAL_RANKING_WEIGHT="${PGC_V9_ACTION_CAUSAL_RANKING_WEIGHT:-1.0}"
fi
ACTION_CAUSAL_MARGIN="${PGC_V9_ACTION_CAUSAL_MARGIN:-0.01}"
ACTION_GEOMETRY_LEARNING_RATE="${PGC_V9_ACTION_GEOMETRY_LEARNING_RATE:-2.0e-5}"
ACTION_GEOMETRY_RESIDUAL_MAX_ABS="${PGC_V9_ACTION_GEOMETRY_RESIDUAL_MAX_ABS:-0.25}"
ACTION_PHASE_RESIDUAL_IMITATION_WEIGHT="${PGC_V9_ACTION_PHASE_RESIDUAL_IMITATION_WEIGHT:-2.0}"
ACTION_PHASE_DIRECTION_WEIGHT="${PGC_V9_ACTION_PHASE_DIRECTION_WEIGHT:-0.5}"
ACTION_PHASE_APPROACH_WEIGHT="${PGC_V9_ACTION_PHASE_APPROACH_WEIGHT:-1.0}"
ACTION_PHASE_TRANSPORT_WEIGHT="${PGC_V9_ACTION_PHASE_TRANSPORT_WEIGHT:-2.0}"
ACTION_PHASE_RELEASE_WEIGHT="${PGC_V9_ACTION_PHASE_RELEASE_WEIGHT:-3.0}"
ACTION_PHASE_DIRECTION_MIN_NORM="${PGC_V9_ACTION_PHASE_DIRECTION_MIN_NORM:-1.0e-3}"
ACTION_SERVO_FRAME_WEIGHT="${PGC_V9_ACTION_SERVO_FRAME_WEIGHT:-0.01}"
CF_JSON="$(json_array "${ORIGINAL_CF_DATASET}" "${STRICT_CF_DATASET}")"

RUN_TAG="${RUN_TAG:-${SUITE}-pgc-v9-eraf-${ABLATION}-${STAGE}-seed${TRAIN_SEED}-v1}"
echo "[PGC-FastWAM] V9 ERAF ${STAGE} training"
echo "  suite=${SUITE} cumulative_steps=${MAX_STEPS} start_step=${START_STEP} stage_steps=${STAGE_STEPS}"
echo "  ablation=${ABLATION} entity_only=${ENTITY_ONLY} use_anchors=${USE_ANCHORS}"
echo "  effective_batch=$((NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS)) (${NPROC_PER_NODE} GPUs x batch1 x grad_accum${GRADIENT_ACCUMULATION_STEPS})"
echo "  grounding_objective=v${GROUNDING_OBJECTIVE_VERSION} attention_mask=${ATTENTION_MASK_WEIGHT} role_swap=${ROLE_SWAP_WEIGHT} role_overlap=${ROLE_OVERLAP_WEIGHT} margin=${ROLE_SWAP_MARGIN}"
echo "  role_assignment=${ROLE_ASSIGNMENT_WEIGHT} temperature=${ROLE_ASSIGNMENT_TEMPERATURE} hard_weight=${ROLE_ASSIGNMENT_HARD_WEIGHT}"
echo "  structured_assignment=${STRUCTURED_ASSIGNMENT_WEIGHT} temperature=${STRUCTURED_ASSIGNMENT_TEMPERATURE} hard_weight=${STRUCTURED_ASSIGNMENT_HARD_WEIGHT} multi_clause=${MULTI_CLAUSE_CONSISTENCY_WEIGHT}"
echo "  clause_tuple=assignment:${CLAUSE_TUPLE_ASSIGNMENT_WEIGHT} temperature:${CLAUSE_TUPLE_TEMPERATURE} hard_weight:${CLAUSE_TUPLE_HARD_WEIGHT} multi_consistency:${CLAUSE_TUPLE_MULTI_CONSISTENCY_WEIGHT}"
echo "  clause_calibration=active:${CLAUSE_ACTIVATION_BALANCE_WEIGHT} cardinality:${CLAUSE_CARDINALITY_WEIGHT} worst_slot:${CLAUSE_WORST_SLOT_WEIGHT} multi_group:${CLAUSE_MULTI_GROUP_WEIGHT} energy:${CLAUSE_ADAPTER_ENERGY_WEIGHT} max_abs:${CLAUSE_ACTIVATION_RESIDUAL_MAX_ABS}"
echo "  view_fusion=weight:${VIEW_FUSION_WEIGHT} energy:${VIEW_FUSION_ENERGY_WEIGHT} max_abs:${VIEW_FUSION_RESIDUAL_MAX_ABS}"
echo "  clause_scheduler=weight:${CLAUSE_SCHEDULER_WEIGHT} energy:${CLAUSE_SCHEDULER_ENERGY_WEIGHT} max_abs:${CLAUSE_SCHEDULER_RESIDUAL_MAX_ABS}"
echo "  phase_rebinding=enabled:${CLOSED_LOOP_REBINDING} hidden:${CLOSED_LOOP_REBINDING_HIDDEN_DIM} query_max_abs:${CLOSED_LOOP_QUERY_RESIDUAL_MAX_ABS} state_max_abs:${CLOSED_LOOP_STATE_RESIDUAL_MAX_ABS} energy:${PHASE_REBINDING_ENERGY_WEIGHT}"
echo "  phase_safe_memory=enabled:${PHASE_SAFE_MEMORY} hidden:${PHASE_SAFE_MEMORY_HIDDEN_DIM} states:${PHASE_SAFE_MEMORY_STATE_COUNT} routing_max_abs:${PHASE_SAFE_MEMORY_ROUTING_RESIDUAL_MAX_ABS} state_weight:${PHASE_SAFE_MEMORY_STATE_WEIGHT} scheduler_weight:${PHASE_SAFE_MEMORY_SCHEDULER_WEIGHT} energy:${PHASE_SAFE_MEMORY_ENERGY_WEIGHT}"
echo "  v9.14_action_joint=enabled:${ACTION_JOINT_TRAINING} completion_only_memory:${COMPLETION_ONLY_MEMORY} eraf_lr:2.0e-5 proposal_lr:${LEARNING_RATE} grounding_aux:${GROUNDING_AUX_WEIGHT}"
echo "  v9.15_action_grounding=lr:${ACTION_GROUNDING_LEARNING_RATE} causal_weight:${ACTION_CAUSAL_RANKING_WEIGHT} causal_margin:${ACTION_CAUSAL_MARGIN}"
echo "  v9.18_phase_residual=imitation:${ACTION_PHASE_RESIDUAL_IMITATION_WEIGHT} direction:${ACTION_PHASE_DIRECTION_WEIGHT} phase_weights:${ACTION_PHASE_APPROACH_WEIGHT},${ACTION_PHASE_TRANSPORT_WEIGHT},${ACTION_PHASE_RELEASE_WEIGHT} direction_min_norm:${ACTION_PHASE_DIRECTION_MIN_NORM}"
echo "  v9.19_phase_servo=hard_single_clause:true frame_weight:${ACTION_SERVO_FRAME_WEIGHT}"
echo "  v9.19_eef_affine=scale:${ACTION_EEF_SCALE} bias:${ACTION_EEF_BIAS}"
echo "  role_preservation=attention:${ROLE_ATTENTION_PRESERVATION_WEIGHT} position:${ROLE_POSITION_PRESERVATION_WEIGHT} anchor:${ROLE_ANCHOR_PRESERVATION_WEIGHT} relation:${ROLE_RELATION_PRESERVATION_WEIGHT} energy:${ROLE_ADAPTER_ENERGY_WEIGHT}"
if [[ "${CLOSED_LOOP_REBINDING}" == "true" || "${PHASE_SAFE_MEMORY}" == "true" ]]; then
  echo "  mixture=offline_native:closed_loop_native:historical_cf:strict_cf 1:1:1:1; closed_loop_phase_balanced=true"
  echo "  closed_loop_native=${CLOSED_LOOP_GROUNDING_DATASET}"
  echo "  closed_loop_sidecar=${CLOSED_LOOP_GROUNDING_SIDECAR}"
else
  echo "  mixture=native:CF 1:1; CF=historical:strict 1:1; structured_task_balance=${STRUCTURED_ROLE_SAMPLING}"
fi
echo "  hard_role_curriculum=${HARD_ROLE_CURRICULUM} hard_index=${HARD_ROLE_INDEX_PATH:-none}"
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
  "data.train.pgc_v9_structured_role_sampling=${STRUCTURED_ROLE_SAMPLING}" \
  "data.train.pgc_v9_hard_role_curriculum=${HARD_ROLE_CURRICULUM}" \
  "data.train.pgc_v9_hard_role_index_path=${HARD_ROLE_INDEX_PATH:-null}" \
  "data.train.pgc_v9_closed_loop_rebinding=${CLOSED_LOOP_REBINDING}" \
  "data.train.pgc_v9_phase_safe_memory=${PHASE_SAFE_MEMORY}" \
  "data.train.pgc_v9_closed_loop_native_dataset_count=${CLOSED_LOOP_NATIVE_DATASET_COUNT}" \
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
  model.policy_guard.enabled=true \
  model.policy_guard.version=9 \
  "model.policy_guard.entity_relation_grounding.training_stage=${CONFIG_STAGE}" \
  "model.policy_guard.entity_relation_grounding.grounding_objective_version=${GROUNDING_OBJECTIVE_VERSION}" \
  "model.policy_guard.entity_relation_grounding.entity_only=${ENTITY_ONLY}" \
  "model.policy_guard.entity_relation_grounding.use_anchors=${USE_ANCHORS}" \
  model.policy_guard.entity_relation_grounding.learning_rate=2.0e-5 \
  "model.policy_guard.entity_relation_grounding.grounding_aux_weight=${GROUNDING_AUX_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.completion_only_memory=${COMPLETION_ONLY_MEMORY}" \
  "model.policy_guard.entity_relation_grounding.action_joint_training=${ACTION_JOINT_TRAINING}" \
  "model.policy_guard.entity_relation_grounding.action_grounding_learning_rate=${ACTION_GROUNDING_LEARNING_RATE}" \
  "model.policy_guard.entity_relation_grounding.action_causal_ranking_weight=${ACTION_CAUSAL_RANKING_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.action_causal_margin=${ACTION_CAUSAL_MARGIN}" \
  "model.policy_guard.entity_relation_grounding.action_geometry_learning_rate=${ACTION_GEOMETRY_LEARNING_RATE}" \
  "model.policy_guard.entity_relation_grounding.action_geometry_residual_max_abs=${ACTION_GEOMETRY_RESIDUAL_MAX_ABS}" \
  "model.policy_guard.entity_relation_grounding.action_phase_residual_imitation_weight=${ACTION_PHASE_RESIDUAL_IMITATION_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.action_phase_direction_weight=${ACTION_PHASE_DIRECTION_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.action_phase_approach_weight=${ACTION_PHASE_APPROACH_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.action_phase_transport_weight=${ACTION_PHASE_TRANSPORT_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.action_phase_release_weight=${ACTION_PHASE_RELEASE_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.action_phase_direction_min_norm=${ACTION_PHASE_DIRECTION_MIN_NORM}" \
  "model.policy_guard.entity_relation_grounding.action_servo_frame_weight=${ACTION_SERVO_FRAME_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.action_eef_scale=${ACTION_EEF_SCALE}" \
  "model.policy_guard.entity_relation_grounding.action_eef_bias=${ACTION_EEF_BIAS}" \
  "model.policy_guard.entity_relation_grounding.mask_weight=${MASK_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.attention_mask_weight=${ATTENTION_MASK_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.entity_weight=${ENTITY_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.relation_weight=${RELATION_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.anchor_weight=${ANCHOR_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.position_weight=${POSITION_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.role_swap_weight=${ROLE_SWAP_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.role_overlap_weight=${ROLE_OVERLAP_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.role_swap_margin=${ROLE_SWAP_MARGIN}" \
  "model.policy_guard.entity_relation_grounding.role_assignment_weight=${ROLE_ASSIGNMENT_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.role_assignment_temperature=${ROLE_ASSIGNMENT_TEMPERATURE}" \
  "model.policy_guard.entity_relation_grounding.role_assignment_hard_weight=${ROLE_ASSIGNMENT_HARD_WEIGHT}" \
  model.policy_guard.entity_relation_grounding.role_adapter_hidden_dim=256 \
  "model.policy_guard.entity_relation_grounding.structured_assignment_weight=${STRUCTURED_ASSIGNMENT_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.structured_assignment_temperature=${STRUCTURED_ASSIGNMENT_TEMPERATURE}" \
  "model.policy_guard.entity_relation_grounding.structured_assignment_hard_weight=${STRUCTURED_ASSIGNMENT_HARD_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.multi_clause_consistency_weight=${MULTI_CLAUSE_CONSISTENCY_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.clause_tuple_assignment_weight=${CLAUSE_TUPLE_ASSIGNMENT_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.clause_tuple_temperature=${CLAUSE_TUPLE_TEMPERATURE}" \
  "model.policy_guard.entity_relation_grounding.clause_tuple_hard_weight=${CLAUSE_TUPLE_HARD_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.clause_tuple_multi_consistency_weight=${CLAUSE_TUPLE_MULTI_CONSISTENCY_WEIGHT}" \
  model.policy_guard.entity_relation_grounding.structured_role_adapter_hidden_dim=256 \
  model.policy_guard.entity_relation_grounding.balanced_role_adapter_hidden_dim=256 \
  model.policy_guard.entity_relation_grounding.clause_activation_adapter_hidden_dim=256 \
  "model.policy_guard.entity_relation_grounding.clause_activation_residual_max_abs=${CLAUSE_ACTIVATION_RESIDUAL_MAX_ABS}" \
  "model.policy_guard.entity_relation_grounding.clause_activation_balance_weight=${CLAUSE_ACTIVATION_BALANCE_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.clause_cardinality_weight=${CLAUSE_CARDINALITY_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.clause_worst_slot_weight=${CLAUSE_WORST_SLOT_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.clause_multi_group_weight=${CLAUSE_MULTI_GROUP_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.clause_adapter_energy_weight=${CLAUSE_ADAPTER_ENERGY_WEIGHT}" \
  model.policy_guard.entity_relation_grounding.view_fusion_adapter_hidden_dim=256 \
  "model.policy_guard.entity_relation_grounding.view_fusion_residual_max_abs=${VIEW_FUSION_RESIDUAL_MAX_ABS}" \
  "model.policy_guard.entity_relation_grounding.view_fusion_weight=${VIEW_FUSION_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.view_fusion_energy_weight=${VIEW_FUSION_ENERGY_WEIGHT}" \
  model.policy_guard.entity_relation_grounding.clause_scheduler_hidden_dim=256 \
  "model.policy_guard.entity_relation_grounding.clause_scheduler_residual_max_abs=${CLAUSE_SCHEDULER_RESIDUAL_MAX_ABS}" \
  "model.policy_guard.entity_relation_grounding.clause_scheduler_weight=${CLAUSE_SCHEDULER_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.clause_scheduler_energy_weight=${CLAUSE_SCHEDULER_ENERGY_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.closed_loop_rebinding_hidden_dim=${CLOSED_LOOP_REBINDING_HIDDEN_DIM}" \
  "model.policy_guard.entity_relation_grounding.closed_loop_query_residual_max_abs=${CLOSED_LOOP_QUERY_RESIDUAL_MAX_ABS}" \
  "model.policy_guard.entity_relation_grounding.closed_loop_state_residual_max_abs=${CLOSED_LOOP_STATE_RESIDUAL_MAX_ABS}" \
  "model.policy_guard.entity_relation_grounding.phase_rebinding_energy_weight=${PHASE_REBINDING_ENERGY_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.phase_safe_memory_hidden_dim=${PHASE_SAFE_MEMORY_HIDDEN_DIM}" \
  "model.policy_guard.entity_relation_grounding.phase_safe_memory_state_count=${PHASE_SAFE_MEMORY_STATE_COUNT}" \
  "model.policy_guard.entity_relation_grounding.phase_safe_memory_routing_residual_max_abs=${PHASE_SAFE_MEMORY_ROUTING_RESIDUAL_MAX_ABS}" \
  "model.policy_guard.entity_relation_grounding.phase_safe_memory_state_weight=${PHASE_SAFE_MEMORY_STATE_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.phase_safe_memory_scheduler_weight=${PHASE_SAFE_MEMORY_SCHEDULER_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.phase_safe_memory_energy_weight=${PHASE_SAFE_MEMORY_ENERGY_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.role_attention_preservation_weight=${ROLE_ATTENTION_PRESERVATION_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.role_position_preservation_weight=${ROLE_POSITION_PRESERVATION_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.role_anchor_preservation_weight=${ROLE_ANCHOR_PRESERVATION_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.role_relation_preservation_weight=${ROLE_RELATION_PRESERVATION_WEIGHT}" \
  "model.policy_guard.entity_relation_grounding.role_adapter_energy_weight=${ROLE_ADAPTER_ENERGY_WEIGHT}" \
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
