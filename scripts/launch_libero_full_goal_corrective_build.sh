#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ROOT="${FULL_GOAL_ARTIFACT_ROOT:-/root/gpufree-data}"
CAPTURE_STATE_FILE="${1:-${ARTIFACT_ROOT}/v938-full-goal-capture-latest.env}"
OLD_CAPTURE_ROOT="${FULL_GOAL_OLD_CAPTURE_ROOT:-${ARTIFACT_ROOT}/pgc_libero_data_v1/v9/libero_10_seed42/v929_base_counterfactual_captures_20260830-115457}"
OLD_FULL_GOAL_DATASET="${FULL_GOAL_OLD_DATASET:-${ARTIFACT_ROOT}/pgc_libero_data_v1/v9/libero_10_seed42/libero_10_v938_full_goal_corrective_lerobot_20260902-104440}"
BEHAVIOR_SUMMARY="${FULL_GOAL_BEHAVIOR_SUMMARY:-${REPO_ROOT}/evaluate_results/v937_step000025_forced_eraf_counterfactual_seed42_trials5_20260902-000240/counterfactual_behavior_summary.json}"
MANIFEST="${FULL_GOAL_MANIFEST:-${ARTIFACT_ROOT}/pgc_libero_data_v1/manifests/libero_10_pgc.jsonl}"
MAX_CAPTURES="${FULL_GOAL_MAX_CAPTURES_PER_PAIR:-120}"
MAX_CANDIDATES="${FULL_GOAL_MAX_CANDIDATES_PER_CAPTURE:-20}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"

if [[ ! -f "$CAPTURE_STATE_FILE" ]]; then
  echo "Capture state file not found: $CAPTURE_STATE_FILE" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CAPTURE_STATE_FILE"
: "${CAP43:?CAP43 is missing from capture state file}"
: "${CAP44:?CAP44 is missing from capture state file}"

for path in \
  "$OLD_CAPTURE_ROOT" \
  "$OLD_FULL_GOAL_DATASET" \
  "$CAP43" \
  "$CAP44"; do
  if [[ ! -d "$path" ]]; then
    echo "Required directory not found: $path" >&2
    exit 1
  fi
done
for path in "$MANIFEST" "$BEHAVIOR_SUMMARY"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
done

PROVENANCE="${OLD_FULL_GOAL_DATASET}/meta/pgc_provenance.json"
if [[ ! -f "$PROVENANCE" ]]; then
  echo "Old full-goal provenance not found: $PROVENANCE" >&2
  exit 1
fi

REFERENCE_DATASET="$("$PYTHON_BIN" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["reference_dataset"])' \
  "$PROVENANCE")"
CAMERA_RESOLUTION="$("$PYTHON_BIN" -c \
  'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("camera_resolution", 512)))' \
  "$PROVENANCE")"
FPS="$("$PYTHON_BIN" -c \
  'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("fps", 20)))' \
  "$PROVENANCE")"

if [[ ! -d "$REFERENCE_DATASET" ]]; then
  echo "Reference dataset not found: $REFERENCE_DATASET" >&2
  exit 1
fi

if pgrep -af "build_pgc_v8_corrective_data.py.*v938_full_goal_corrective_complete" \
  >/tmp/v938_full_goal_builder_processes.txt; then
  echo "Refusing to launch a duplicate full-goal builder:" >&2
  cat /tmp/v938_full_goal_builder_processes.txt >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT="${ARTIFACT_ROOT}/pgc_libero_data_v1/v9/libero_10_seed42/libero_10_v938_full_goal_corrective_complete_${STAMP}"
LOG="${ARTIFACT_ROOT}/libero10-v938-full-goal-corrective-build-${STAMP}.log"
BUILD_STATE_FILE="${ARTIFACT_ROOT}/v938-full-goal-build-latest.env"
COVERAGE="${OUTPUT}/meta/v938_full_goal_coverage.json"

nohup env \
  BUILD_REPO="$REPO_ROOT" \
  BUILD_PYTHON="$PYTHON_BIN" \
  BUILD_MANIFEST="$MANIFEST" \
  BUILD_OLD_CAPTURES="$OLD_CAPTURE_ROOT" \
  BUILD_CAP43="$CAP43" \
  BUILD_CAP44="$CAP44" \
  BUILD_REFERENCE="$REFERENCE_DATASET" \
  BUILD_OUTPUT="$OUTPUT" \
  BUILD_BEHAVIOR="$BEHAVIOR_SUMMARY" \
  BUILD_COVERAGE="$COVERAGE" \
  BUILD_CAMERA_RESOLUTION="$CAMERA_RESOLUTION" \
  BUILD_FPS="$FPS" \
  BUILD_MAX_CAPTURES="$MAX_CAPTURES" \
  BUILD_MAX_CANDIDATES="$MAX_CANDIDATES" \
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  bash -c '
    set -euo pipefail
    cd "$BUILD_REPO"
    "$BUILD_PYTHON" -u scripts/build_pgc_v8_corrective_data.py \
      --manifest "$BUILD_MANIFEST" \
      --captures "$BUILD_OLD_CAPTURES" \
      --captures "$BUILD_CAP43" \
      --captures "$BUILD_CAP44" \
      --reference-dataset "$BUILD_REFERENCE" \
      --output "$BUILD_OUTPUT" \
      --suite libero_10 \
      --episodes-per-pair 5 \
      --max-captures-per-pair "$BUILD_MAX_CAPTURES" \
      --max-candidates-per-capture "$BUILD_MAX_CANDIDATES" \
      --reference-index-stride 1 \
      --post-lift-steps 8 \
      --verification-policy full_goal \
      --min-actions 12 \
      --lift-threshold-m 0.04 \
      --state-atol 1e-7 \
      --camera-resolution "$BUILD_CAMERA_RESOLUTION" \
      --fps "$BUILD_FPS" \
      --video-codec h264 \
      --seed 42 \
      --allow-partial
    "$BUILD_PYTHON" scripts/audit_pgc_corrective_coverage.py \
      --manifest "$BUILD_MANIFEST" \
      --dataset "$BUILD_OUTPUT" \
      --suite libero_10 \
      --behavior-summary "$BUILD_BEHAVIOR" \
      --minimum-source-directed-failures 1 \
      --minimum-full-goal-per-pair 5 \
      --output "$BUILD_COVERAGE"
    echo "[ALL_DONE] full-goal corrective dataset and coverage audit complete"
    echo "OUTPUT=$BUILD_OUTPUT"
    echo "COVERAGE=$BUILD_COVERAGE"
  ' >"$LOG" 2>&1 </dev/null &
PID=$!

printf 'OUTPUT=%q\nLOG=%q\nPID=%q\nCOVERAGE=%q\nREFERENCE_DATASET=%q\nCAP43=%q\nCAP44=%q\n' \
  "$OUTPUT" "$LOG" "$PID" "$COVERAGE" "$REFERENCE_DATASET" "$CAP43" "$CAP44" \
  >"$BUILD_STATE_FILE"

sleep 8
if ! kill -0 "$PID" 2>/dev/null; then
  echo "Full-goal builder exited during startup: $LOG" >&2
  tail -160 "$LOG" >&2 || true
  exit 1
fi

echo "PASS: full-goal corrective builder is running"
echo "PID=$PID"
echo "LOG=$LOG"
echo "OUTPUT=$OUTPUT"
echo "STATE=$BUILD_STATE_FILE"
echo "Monitor: source '$BUILD_STATE_FILE'; tail -F \"\$LOG\""
