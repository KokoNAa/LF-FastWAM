#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 CHECKPOINT" >&2
  exit 2
fi

CHECKPOINT="$1"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ROOT="${FULL_GOAL_ARTIFACT_ROOT:-/root/gpufree-data}"
DATA_ROOT="${LIBERO_DATA_ROOT:-${ARTIFACT_ROOT}/fastwam/FastWAM/data/libero_mujoco3.3.2}"
MODEL_ROOT="${DIFFSYNTH_MODEL_BASE_PATH:-${ARTIFACT_ROOT}/fastwam/FastWAM/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
MANIFEST="${FULL_GOAL_MANIFEST:-${ARTIFACT_ROOT}/pgc_libero_data_v1/manifests/libero_10_pgc.jsonl}"
STATS="${FULL_GOAL_STATS:-${MODEL_ROOT}/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
PRIMARY_TASK_IDS="${FULL_GOAL_PRIMARY_TASK_IDS:-[6,7]}"
FOCUS_TASK_IDS="${FULL_GOAL_FOCUS_TASK_IDS:-[7]}"
PRIMARY_GPUS="${FULL_GOAL_PRIMARY_GPUS:-0,1}"
FOCUS_GPUS="${FULL_GOAL_FOCUS_GPUS:-2}"
PRIMARY_SEED="${FULL_GOAL_PRIMARY_SEED:-43}"
FOCUS_SEED="${FULL_GOAL_FOCUS_SEED:-44}"
NUM_TRIALS="${FULL_GOAL_NUM_TRIALS:-10}"
MAX_CAPTURE_STATES="${FULL_GOAL_MAX_CAPTURE_STATES:-64}"
STAMP="$(date +%Y%m%d-%H%M%S)"
STATE_FILE="${FULL_GOAL_STATE_FILE:-${ARTIFACT_ROOT}/v938-full-goal-capture-latest.env}"
PRIMARY_SESSION="${FULL_GOAL_PRIMARY_SESSION_NAME:-v938_fg_primary_${STAMP}_$$}"
FOCUS_SESSION="${FULL_GOAL_FOCUS_SESSION_NAME:-v938_fg_focus_${STAMP}_$$}"

for path in "$CHECKPOINT" "$MANIFEST" "$STATS"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
done

validate_visible_gpus() {
  local label="$1"
  local gpu_list="$2"
  local expected
  local observed
  expected="$(awk -F, '{print NF}' <<<"$gpu_list")"
  observed="$(
    env CUDA_VISIBLE_DEVICES="$gpu_list" "$PYTHON_BIN" -c \
      'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)'
  )"
  if [[ "$observed" != "$expected" ]]; then
    echo "$label GPU preflight failed: requested=$gpu_list expected=$expected visible=$observed" >&2
    exit 1
  fi
  echo "[preflight] $label GPUs=$gpu_list visible=$observed"
}

validate_visible_gpus primary "$PRIMARY_GPUS"
validate_visible_gpus focus "$FOCUS_GPUS"

if pgrep -af "v938_full_goal_capture_task" >/tmp/v938_capture_processes.txt; then
  echo "Refusing to launch duplicate V9.38 capture jobs:" >&2
  cat /tmp/v938_capture_processes.txt >&2
  exit 1
fi

CAP_PRIMARY="${ARTIFACT_ROOT}/pgc_libero_data_v1/v9/libero_10_seed42/v938_full_goal_captures_seed${PRIMARY_SEED}_${STAMP}"
CAP_FOCUS="${ARTIFACT_ROOT}/pgc_libero_data_v1/v9/libero_10_seed42/v938_full_goal_captures_seed${FOCUS_SEED}_${STAMP}"
EVAL_PRIMARY="${REPO_ROOT}/evaluate_results/v938_full_goal_capture_task67_seed${PRIMARY_SEED}_trials${NUM_TRIALS}_${STAMP}"
EVAL_FOCUS="${REPO_ROOT}/evaluate_results/v938_full_goal_capture_task7_seed${FOCUS_SEED}_trials${NUM_TRIALS}_${STAMP}"
LOG_PRIMARY="${ARTIFACT_ROOT}/libero10-v938-full-goal-capture-seed${PRIMARY_SEED}-${STAMP}.log"
LOG_FOCUS="${ARTIFACT_ROOT}/libero10-v938-full-goal-capture-seed${FOCUS_SEED}-${STAMP}.log"

cd "$REPO_ROOT"

echo "[launch] GPUs ${PRIMARY_GPUS}: tasks=${PRIMARY_TASK_IDS} seed=${PRIMARY_SEED}"
nohup env \
  CUDA_VISIBLE_DEVICES="$PRIMARY_GPUS" \
  LIBERO_TMUX_SESSION_NAME="$PRIMARY_SESSION" \
  PYTHON_BIN="$PYTHON_BIN" \
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  DIFFSYNTH_MODEL_BASE_PATH="$MODEL_ROOT" \
  LIBERO_DATA_ROOT="$DATA_ROOT" \
  TEXT_CACHE_DIR="${REPO_ROOT}/data/text_embeds_cache/libero" \
  PGC_CHECKPOINT="$CHECKPOINT" \
  PGC_EVAL_SUITES='[libero_10]' \
  PGC_EVAL_TASK_IDS="$PRIMARY_TASK_IDS" \
  PGC_MANIFEST_PATH="$MANIFEST" \
  PGC_GATE_MODE=counterfactual \
  PGC_V9_ABLATION=full \
  PGC_ERAF_DIAGNOSTICS=false \
  PGC_CLOSED_LOOP_CAPTURE_DIR="$CAP_PRIMARY" \
  PGC_CLOSED_LOOP_CAPTURE_STAGE_POLICY=all_replans \
  PGC_CLOSED_LOOP_CAPTURE_STRIDE_REPLANS=1 \
  PGC_CLOSED_LOOP_CAPTURE_MAX_STATES_PER_EPISODE="$MAX_CAPTURE_STATES" \
  PGC_CLOSED_LOOP_CAPTURE_NAMESPACE="v938fg_s${PRIMARY_SEED}_${STAMP}" \
  STATS_PATH="$STATS" \
  OUTPUT_ROOT="$EVAL_PRIMARY" \
  bash scripts/eval_pgc_libero.sh 2 "$NUM_TRIALS" counterfactual "$PRIMARY_SEED" 10 \
  >"$LOG_PRIMARY" 2>&1 </dev/null &
PID_PRIMARY=$!

echo "[launch] GPUs ${FOCUS_GPUS}: tasks=${FOCUS_TASK_IDS} seed=${FOCUS_SEED}"
nohup env \
  CUDA_VISIBLE_DEVICES="$FOCUS_GPUS" \
  LIBERO_TMUX_SESSION_NAME="$FOCUS_SESSION" \
  PYTHON_BIN="$PYTHON_BIN" \
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  DIFFSYNTH_MODEL_BASE_PATH="$MODEL_ROOT" \
  LIBERO_DATA_ROOT="$DATA_ROOT" \
  TEXT_CACHE_DIR="${REPO_ROOT}/data/text_embeds_cache/libero" \
  PGC_CHECKPOINT="$CHECKPOINT" \
  PGC_EVAL_SUITES='[libero_10]' \
  PGC_EVAL_TASK_IDS="$FOCUS_TASK_IDS" \
  PGC_MANIFEST_PATH="$MANIFEST" \
  PGC_GATE_MODE=counterfactual \
  PGC_V9_ABLATION=full \
  PGC_ERAF_DIAGNOSTICS=false \
  PGC_CLOSED_LOOP_CAPTURE_DIR="$CAP_FOCUS" \
  PGC_CLOSED_LOOP_CAPTURE_STAGE_POLICY=all_replans \
  PGC_CLOSED_LOOP_CAPTURE_STRIDE_REPLANS=1 \
  PGC_CLOSED_LOOP_CAPTURE_MAX_STATES_PER_EPISODE="$MAX_CAPTURE_STATES" \
  PGC_CLOSED_LOOP_CAPTURE_NAMESPACE="v938fg_s${FOCUS_SEED}_${STAMP}" \
  STATS_PATH="$STATS" \
  OUTPUT_ROOT="$EVAL_FOCUS" \
  bash scripts/eval_pgc_libero.sh 1 "$NUM_TRIALS" counterfactual "$FOCUS_SEED" 10 \
  >"$LOG_FOCUS" 2>&1 </dev/null &
PID_FOCUS=$!

mkdir -p "$(dirname -- "$STATE_FILE")"
printf 'REPO=%q\nCAP43=%q\nCAP44=%q\nEVAL43=%q\nEVAL44=%q\nLOG43=%q\nLOG44=%q\nPID43=%q\nPID44=%q\nSESSION43=%q\nSESSION44=%q\n' \
  "$REPO_ROOT" "$CAP_PRIMARY" "$CAP_FOCUS" "$EVAL_PRIMARY" "$EVAL_FOCUS" \
  "$LOG_PRIMARY" "$LOG_FOCUS" "$PID_PRIMARY" "$PID_FOCUS" \
  "$PRIMARY_SESSION" "$FOCUS_SESSION" >"$STATE_FILE"

sleep 8
failed=0
for item in "$PID_PRIMARY:$LOG_PRIMARY" "$PID_FOCUS:$LOG_FOCUS"; do
  pid="${item%%:*}"
  log="${item#*:}"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Capture process $pid exited during startup: $log" >&2
    tail -120 "$log" >&2 || true
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo "PASS: three-GPU full-goal capture is running"
echo "STATE=$STATE_FILE"
echo "GPU ${PRIMARY_GPUS}: PID=$PID_PRIMARY LOG=$LOG_PRIMARY"
echo "GPU ${FOCUS_GPUS}: PID=$PID_FOCUS LOG=$LOG_FOCUS"
echo "TMUX primary=$PRIMARY_SESSION focus=$FOCUS_SESSION"
echo "Monitor: source '$STATE_FILE'; tail -F \"\$LOG43\" \"\$LOG44\""
