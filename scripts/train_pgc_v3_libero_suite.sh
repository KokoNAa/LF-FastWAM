#!/usr/bin/env bash
set -euo pipefail

# PGC v3 keeps one immutable released Base Action Expert and trains only the
# Goal Graph, bounded post-DiT velocity residual, and delayed Verifier.
export PGC_VERSION=3
export PGC_LEARNING_RATE="${PGC_LEARNING_RATE:-1.0e-4}"
export PGC_RESIDUAL_REGULARIZATION_WEIGHT="${PGC_RESIDUAL_REGULARIZATION_WEIGHT:-0.01}"
export PGC_RESIDUAL_SMOOTHNESS_WEIGHT="${PGC_RESIDUAL_SMOOTHNESS_WEIGHT:-0.01}"
export PGC_VELOCITY_RESIDUAL_MAX_ABS="${PGC_VELOCITY_RESIDUAL_MAX_ABS:-1.0}"
export PGC_VERIFIER_START_STEP="${PGC_VERIFIER_START_STEP:-1000}"
export PGC_VERIFIER_RAMP_STEPS="${PGC_VERIFIER_RAMP_STEPS:-500}"

exec bash scripts/train_pgc_libero_suite.sh "$@"
