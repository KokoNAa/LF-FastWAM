#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ERAF_WORK_ROOT=${ERAF_WORK_ROOT:-/root/gpufree-data/pgc_robotwin_eraf_v1}
PYTHON_BIN=${PYTHON_BIN:-python}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-$PROJECT_ROOT/third_party/RoboTwin}

ACTIVE_PIDS=()

terminate_children() {
  local pid
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

on_error() {
  local exit_code=$?
  terminate_children
  echo "[robotwin-eraf-formal] failed exit_code=$exit_code" >&2
  exit "$exit_code"
}

trap on_error ERR
trap 'terminate_children; exit 130' INT TERM

collect_split() {
  local task_config=$1
  local episodes=$2
  local base_seed=$3
  local raw_root=$ERAF_WORK_ROOT/formal/$task_config/raw
  local log_root=$ERAF_WORK_ROOT/logs/formal_$task_config

  if [[ -e $raw_root ]]; then
    echo "Refusing to overwrite existing raw root: $raw_root" >&2
    return 2
  fi
  mkdir -p "$log_root"
  echo "[robotwin-eraf-formal] collecting config=$task_config episodes=$episodes"

  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/collect_pgc_robotwin_pairs.py \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --output-root "$raw_root" \
    --task-config "$task_config" \
    --episodes "$episodes" \
    --start-seed "$base_seed" \
    --source-tasks place_a2b_left \
    >"$log_root/gpu0_place_left.log" 2>&1 &
  local p0=$!

  CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" scripts/collect_pgc_robotwin_pairs.py \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --output-root "$raw_root" \
    --task-config "$task_config" \
    --episodes "$episodes" \
    --start-seed "$((base_seed + 1000000))" \
    --source-tasks place_a2b_right \
    >"$log_root/gpu1_place_right.log" 2>&1 &
  local p1=$!

  CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" scripts/collect_pgc_robotwin_pairs.py \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --output-root "$raw_root" \
    --task-config "$task_config" \
    --episodes "$episodes" \
    --start-seed "$((base_seed + 2000000))" \
    --source-tasks stack_blocks_two \
    >"$log_root/gpu2_stack.log" 2>&1 &
  local p2=$!

  CUDA_VISIBLE_DEVICES=3 "$PYTHON_BIN" scripts/collect_pgc_robotwin_pairs.py \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --output-root "$raw_root" \
    --task-config "$task_config" \
    --episodes "$episodes" \
    --start-seed "$((base_seed + 3000000))" \
    --source-tasks blocks_ranking_rgb place_burger_fries \
    >"$log_root/gpu3_ranking_burger.log" 2>&1 &
  local p3=$!

  ACTIVE_PIDS=("$p0" "$p1" "$p2" "$p3")
  local status=0
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    wait "$pid" || status=1
  done
  ACTIVE_PIDS=()
  if [[ $status -ne 0 ]]; then
    echo "Collection failed for $task_config; inspect $log_root" >&2
    return 1
  fi

  echo "[robotwin-eraf-formal] preparing config=$task_config"
  "$PYTHON_BIN" scripts/prepare_pgc_robotwin_eraf_data.py \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --work-root "$ERAF_WORK_ROOT" \
    --stage formal \
    --task-config "$task_config" \
    --episodes "$episodes" \
    --start-seed "$base_seed" \
    --skip-collection \
    2>&1 | tee "$log_root/prepare.log"
}

cd "$PROJECT_ROOT"
mkdir -p "$ERAF_WORK_ROOT/logs"

collect_split demo_clean 3 4400000
collect_split demo_randomized 2 9400000

echo "[robotwin-eraf-formal] complete trajectories=50 full_goal_verified=false"
