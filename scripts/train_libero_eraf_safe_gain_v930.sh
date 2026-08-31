#!/usr/bin/env bash
set -euo pipefail

# Same arguments and validated data bindings as V9.29, but a new short run.
# Do not source this script into an interactive login shell.
export ERAF_SAFE_GAIN_VARIANT=v930
export ERAF_SAFE_GAIN_MAX_STEPS="${ERAF_SAFE_GAIN_MAX_STEPS:-250}"
export ERAF_SAFE_GAIN_LEARNING_RATE="${ERAF_SAFE_GAIN_LEARNING_RATE:-2.0e-6}"
export ERAF_SAFE_GAIN_SAVE_EVERY="${ERAF_SAFE_GAIN_SAVE_EVERY:-50}"
exec bash scripts/train_libero_eraf_safe_gain_v929.sh "$@"
