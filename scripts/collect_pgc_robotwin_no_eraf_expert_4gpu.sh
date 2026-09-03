#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NO_ERAF_WORK_ROOT=${NO_ERAF_WORK_ROOT:-/root/gpufree-data/pgc_robotwin_no_eraf_v1}
NO_ERAF_EXPERT_STAGE=${NO_ERAF_EXPERT_STAGE:-formal}
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/bin/python}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-$PROJECT_ROOT/third_party/RoboTwin}

case "$NO_ERAF_EXPERT_STAGE" in
  smoke|formal) ;;
  *)
    echo "NO_ERAF_EXPERT_STAGE must be smoke or formal" >&2
    exit 2
    ;;
esac

STAGE_ROOT=$NO_ERAF_WORK_ROOT/$NO_ERAF_EXPERT_STAGE

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
  echo "[robotwin-no-eraf-expert] failed exit_code=$exit_code" >&2
  exit "$exit_code"
}

trap on_error ERR
trap 'terminate_children; exit 130' INT TERM

collect_split() {
  local profile=$1
  local collector_profile=$2
  local task_config=$3
  local episodes=$4
  local base_seed=$5
  local raw_root=$STAGE_ROOT/expert/$profile/$task_config/raw
  local log_root=$STAGE_ROOT/logs/expert_${profile}_${task_config}

  if [[ -e $raw_root ]]; then
    echo "Refusing to overwrite existing raw root: $raw_root" >&2
    return 2
  fi
  mkdir -p "$log_root"
  echo "[robotwin-no-eraf-expert] profile=$profile config=$task_config episodes=$episodes"

  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/collect_pgc_robotwin_pairs.py \
    --robotwin-root "$ROBOTWIN_ROOT" --output-root "$raw_root" \
    --task-config "$task_config" --episodes "$episodes" \
    --start-seed "$base_seed" --collection-profile "$collector_profile" \
    --source-tasks place_a2b_left >"$log_root/gpu0_place_left.log" 2>&1 &
  local p0=$!

  CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" scripts/collect_pgc_robotwin_pairs.py \
    --robotwin-root "$ROBOTWIN_ROOT" --output-root "$raw_root" \
    --task-config "$task_config" --episodes "$episodes" \
    --start-seed "$((base_seed + 1000000))" --collection-profile "$collector_profile" \
    --source-tasks place_a2b_right >"$log_root/gpu1_place_right.log" 2>&1 &
  local p1=$!

  CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" scripts/collect_pgc_robotwin_pairs.py \
    --robotwin-root "$ROBOTWIN_ROOT" --output-root "$raw_root" \
    --task-config "$task_config" --episodes "$episodes" \
    --start-seed "$((base_seed + 2000000))" --collection-profile "$collector_profile" \
    --source-tasks stack_blocks_two >"$log_root/gpu2_stack.log" 2>&1 &
  local p2=$!

  CUDA_VISIBLE_DEVICES=3 "$PYTHON_BIN" scripts/collect_pgc_robotwin_pairs.py \
    --robotwin-root "$ROBOTWIN_ROOT" --output-root "$raw_root" \
    --task-config "$task_config" --episodes "$episodes" \
    --start-seed "$((base_seed + 3000000))" --collection-profile "$collector_profile" \
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
    echo "Collection failed; inspect $log_root" >&2
    return 1
  fi

  "$PYTHON_BIN" scripts/prepare_pgc_robotwin_no_eraf_expert_data.py \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --work-root "$STAGE_ROOT" \
    --profile "$profile" \
    --task-config "$task_config" \
    --episodes "$episodes" \
    --start-seed "$base_seed" \
    --skip-collection 2>&1 | tee "$log_root/prepare.log"
}

cd "$PROJECT_ROOT"
mkdir -p "$STAGE_ROOT/logs"

if [[ $NO_ERAF_EXPERT_STAGE == smoke ]]; then
  collect_split historical no_eraf_historical demo_clean 1 14000000
  collect_split strict no_eraf_strict demo_clean 1 34000000
else
  # Match the prior 3:2 clean/randomized ratio while providing 50 expert pairs
  # per directed task for the broad offline-native/historical pools.
  collect_split historical no_eraf_historical demo_clean 30 14000000
  collect_split historical no_eraf_historical demo_randomized 20 24000000

  # Strict-CF follows the LIBERO five-demonstration-per-direction scale and uses
  # disjoint scene seeds from the historical pool.
  collect_split strict no_eraf_strict demo_clean 3 34000000
  collect_split strict no_eraf_strict demo_randomized 2 44000000
fi

echo "[robotwin-no-eraf-expert] complete stage=$NO_ERAF_EXPERT_STAGE full_goal=false"
