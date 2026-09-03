#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CLOSED_LOOP_STAGE=${CLOSED_LOOP_STAGE:-smoke}
CLOSED_LOOP_WORK_ROOT=${CLOSED_LOOP_WORK_ROOT:-/root/gpufree-data/pgc_robotwin_no_eraf_v1}
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/bin/python}
ROBOTWIN_CKPT=${ROBOTWIN_CKPT:-}
STATS_PATH=${STATS_PATH:-}
MANIFEST_PATH=${MANIFEST_PATH:-$PROJECT_ROOT/configs/eval/robotwin_cis_v939_four_tasks.json}

case "$CLOSED_LOOP_STAGE" in
  smoke|formal) ;;
  *) echo "CLOSED_LOOP_STAGE must be smoke or formal" >&2; exit 2 ;;
esac
for path_name in ROBOTWIN_CKPT STATS_PATH MANIFEST_PATH; do
  path_value=${!path_name}
  if [[ -z $path_value || ! -f $path_value ]]; then
    echo "$path_name must name an existing file, got: $path_value" >&2
    exit 2
  fi
done

STAGE_ROOT=$CLOSED_LOOP_WORK_ROOT/$CLOSED_LOOP_STAGE/closed_loop_native
CAPTURE_ROOT=$STAGE_ROOT/captures
DATASET_ROOT=$STAGE_ROOT/lerobot
SIDECAR_ROOT=$STAGE_ROOT/sidecar
LOG_ROOT=$STAGE_ROOT/logs
mkdir -p "$CAPTURE_ROOT" "$LOG_ROOT"

run_config() {
  local task_config=$1
  local episodes=$2
  local eval_seed=$3
  local run_tag="robotwin_v939_closed_loop_native_${CLOSED_LOOP_STAGE}_${task_config}_seed${eval_seed}"
  echo "[robotwin-closed-loop-native] config=$task_config episodes=$episodes seed=$eval_seed"
  PGC_ROBOTWIN_CLOSED_LOOP_CAPTURE_DIR="$CAPTURE_ROOT" \
  PGC_ROBOTWIN_CLOSED_LOOP_CAPTURE_STRIDE_REPLANS=1 \
  PGC_ROBOTWIN_CLOSED_LOOP_CAPTURE_MAX_STATES_PER_EPISODE=12 \
  MANIFEST_PATH="$MANIFEST_PATH" \
  CIS_TASKS=place_a2b_left,place_a2b_right,stack_blocks_two,blocks_ranking_rgb,place_burger_fries \
  CIS_TASK_CONFIGS="$task_config" \
  CIS_CONDITIONS=correct \
  FASTWAM_EVAL_MODE=B0 \
  MAX_TASKS_PER_GPU=1 \
  RUN_TAG="$run_tag" \
  OUTPUT_ROOT="$PROJECT_ROOT/evaluate_results/robotwin_closed_loop_native/$run_tag" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/eval_robotwin_cis.sh \
    4 "$episodes" 10 "$eval_seed" "$ROBOTWIN_CKPT" "$STATS_PATH" \
    EVALUATION.skip_get_obs_within_replan=false \
    2>&1 | tee "$LOG_ROOT/eval_${task_config}.log"
}

cd "$PROJECT_ROOT"
if [[ -e $DATASET_ROOT || -e $SIDECAR_ROOT ]]; then
  echo "Refusing to overwrite prepared closed-loop-native output under $STAGE_ROOT" >&2
  exit 2
fi

if [[ $CLOSED_LOOP_STAGE == smoke ]]; then
  run_config demo_clean 1 54
  EXPECTED_CONFIGS=demo_clean
  MAX_PER_GROUP=12
else
  run_config demo_clean 3 55
  run_config demo_randomized 2 65
  EXPECTED_CONFIGS=demo_clean,demo_randomized
  MAX_PER_GROUP=25
fi

"$PYTHON_BIN" scripts/build_pgc_robotwin_closed_loop_native.py \
  --captures "$CAPTURE_ROOT" \
  --output "$DATASET_ROOT" \
  --sidecar-output "$SIDECAR_ROOT" \
  --expected-task-configs "$EXPECTED_CONFIGS" \
  --max-per-task-config-stage "$MAX_PER_GROUP" \
  --seed 42 2>&1 | tee "$LOG_ROOT/prepare.log"

echo "[robotwin-closed-loop-native] complete stage=$CLOSED_LOOP_STAGE full_goal=false"
