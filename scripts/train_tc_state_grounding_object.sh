#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_tc_state_grounding_object.sh <gpus> <tc_v5_or_v6_checkpoint> [seed] [max_steps]}"
INITIAL_CHECKPOINT="${2:?Missing protected TC v5/v6 checkpoint}"
TRAIN_SEED="${3:-42}"
MAX_STEPS="${4:-4000}"

export TC_CONTRACT_VERSION=6
export TC_USE_STATE_CONDITIONED_GROUNDING=true
export TC_RUN_PREFIX="${TC_RUN_PREFIX:-tc-ground}"
export RUN_TAG="${RUN_TAG:-v6-state-grounding-object-${MAX_STEPS}-seed${TRAIN_SEED}-v1}"

echo "[TC-FastWAM] Enabling v6 current-state target-object grounding."
exec bash scripts/train_tc_counterfactual_action_object.sh \
  "${NPROC_PER_NODE}" \
  "${INITIAL_CHECKPOINT}" \
  "${TRAIN_SEED}" \
  "${MAX_STEPS}"
