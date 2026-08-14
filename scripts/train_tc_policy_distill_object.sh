#!/usr/bin/env bash
set -euo pipefail

# TC-C v3 keeps the loaded M1 policy immutable and uses its exact joint-MoT
# posterior output as a per-sample action-flow teacher. Router/projection/
# outcome modules remain trainable under action, distillation, and LF losses.
export TC_VERSION=3
export TC_POLICY_DISTILLATION_ENABLED=true
export TC_POLICY_DISTILLATION_WEIGHT="${TC_POLICY_DISTILLATION_WEIGHT:-1.0}"
export TC_FREEZE_M1_POLICY=true
# Optimizer state is small enough with a Router-only optimizer and makes long
# server jobs resumable after shutdowns.
export TC_SAVE_TRAINING_STATE="${TC_SAVE_TRAINING_STATE:-true}"
export RUN_TAG="${RUN_TAG:-v3-policy-distill-object-v1}"

exec bash scripts/train_tc_stage1_object.sh "$@"
