#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_pgc_v5_completion_object.sh <gpus> <base_checkpoint> <v5_checkpoint> <counterfactual_datasets.txt> [seed] [additional_steps]}"
BASE_CHECKPOINT="${2:?Missing released FastWAM base checkpoint}"
V5_CHECKPOINT="${3:?Missing PGC V5 checkpoint}"
COUNTERFACTUAL_LIST="${4:?Missing direct-counterfactual dataset list}"
TRAIN_SEED="${5:-42}"
ADDITIONAL_STEPS="${6:-1500}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! [[ "${ADDITIONAL_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "additional_steps must be a positive integer." >&2
  exit 1
fi

START_STEP="$("${PYTHON_BIN}" - "${V5_CHECKPOINT}" "${BASE_CHECKPOINT}" <<'PY'
import sys
from pathlib import Path

import torch

checkpoint_path = Path(sys.argv[1]).expanduser().resolve()
base_path = Path(sys.argv[2]).expanduser().resolve()
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
metadata = payload.get("architecture_metadata") or {}
if payload.get("format") != "fastwam_policy_guard_v5":
    raise SystemExit("V5-completion must initialize from fastwam_policy_guard_v5")
if int(metadata.get("policy_guard_version", -1)) != 5:
    raise SystemExit("V5-completion checkpoint metadata is not version 5")
if metadata.get("counterfactual_tuning") != (
    "paired_language_prefix_aligned_action_residual"
):
    raise SystemExit("V5-completion checkpoint lacks the paired V5 contract")
recorded_base = Path(str(payload.get("base_checkpoint", ""))).expanduser()
if not recorded_base.is_absolute():
    recorded_base = checkpoint_path.parent / recorded_base
if recorded_base.resolve() != base_path:
    raise SystemExit(
        f"Protected base mismatch: checkpoint={recorded_base.resolve()} "
        f"requested={base_path}"
    )
step = int(payload.get("step", -1))
if step <= 0:
    raise SystemExit(f"V5 checkpoint has invalid step: {step}")
print(step)
PY
)"
MAX_STEPS=$((START_STEP + ADDITIONAL_STEPS))

export PGC_VERSION=5
export PGC_TRAIN_SUITE=libero_object
export PGC_INIT_CHECKPOINT="${V5_CHECKPOINT}"
export PGC_CONTINUE_FROM_STEP="${START_STEP}"
export PGC_COMPLETION_PHASE_ENABLED=true
export PGC_COMPLETION_TRAIN_PROPOSAL_ONLY=true
export PGC_COMPLETION_TRANSPORT_WEIGHT="${PGC_COMPLETION_TRANSPORT_WEIGHT:-2.0}"
export PGC_COMPLETION_RELEASE_WEIGHT="${PGC_COMPLETION_RELEASE_WEIGHT:-3.0}"
export PGC_LEARNING_RATE="${PGC_LEARNING_RATE:-5.0e-6}"
export RUN_TAG="${RUN_TAG:-object-pgc-v5-completion-${START_STEP}-to-${MAX_STEPS}-seed${TRAIN_SEED}-v1}"

echo "[PGC-FastWAM] V5 completion continuation"
echo "  checkpoint=${V5_CHECKPOINT}"
echo "  steps=${START_STEP}->${MAX_STEPS}"
echo "  proposal_only=true transport_weight=${PGC_COMPLETION_TRANSPORT_WEIGHT} release_weight=${PGC_COMPLETION_RELEASE_WEIGHT}"

exec bash scripts/train_pgc_libero.sh \
  "${NPROC_PER_NODE}" \
  "${BASE_CHECKPOINT}" \
  "${COUNTERFACTUAL_LIST}" \
  "${TRAIN_SEED}" \
  "${MAX_STEPS}"
