#!/usr/bin/env bash
set -euo pipefail

ROBOTWIN_CKPT="${1:?Usage: bash scripts/validate_robotwin_cis_server.sh <checkpoint> <dataset_stats> [manifest]}"
STATS_PATH="${2:?Missing dataset stats path}"
MANIFEST_PATH="${3:-configs/eval/robotwin_cis_spatial.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-third_party/RoboTwin}"

for required_file in \
  "${ROBOTWIN_CKPT}" \
  "${STATS_PATH}" \
  "${MANIFEST_PATH}" \
  "${ROBOTWIN_ROOT}/task_config/_eval_step_limit.yml" \
  "${ROBOTWIN_ROOT}/task_config/_camera_config.yml" \
  "${ROBOTWIN_ROOT}/task_config/_embodiment_config.yml" \
  "${ROBOTWIN_ROOT}/task_config/demo_clean.yml" \
  "${ROBOTWIN_ROOT}/task_config/demo_randomized.yml"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done
for required_dir in \
  "${ROBOTWIN_ROOT}/assets/background_texture" \
  "${ROBOTWIN_ROOT}/assets/objects" \
  "${ROBOTWIN_ROOT}/assets/embodiments" \
  "${ROBOTWIN_ROOT}/assets/embodiments/aloha-agilex" \
  "${ROBOTWIN_ROOT}/policy"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Required directory not found: ${required_dir}" >&2
    exit 1
  fi
done
for required_command in nvidia-smi ffmpeg; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "Required command not found: ${required_command}" >&2
    exit 1
  fi
done

nvidia-smi -L
"${PYTHON_BIN}" -c 'import cv2, gymnasium, hydra, mplib, numpy, omegaconf, open3d, sapien, torch, transforms3d, trimesh; assert torch.cuda.is_available(); print("torch={} cuda_devices={} sapien={}".format(torch.__version__, torch.cuda.device_count(), getattr(sapien, "__version__", "unknown")))'
(
  cd "${ROBOTWIN_ROOT}"
  "${PYTHON_BIN}" -c 'from envs.robot.planner import CuroboPlanner, MplibPlanner; print("RoboTwin planners import passed")'
)
"${PYTHON_BIN}" scripts/validate_robotwin_cis_manifest.py \
  "${MANIFEST_PATH}" \
  --robotwin-root "${ROBOTWIN_ROOT}"
"${PYTHON_BIN}" -m unittest \
  tests.test_robotwin_language_interventions \
  tests.test_robotwin_cis_results -v
"${PYTHON_BIN}" -m compileall -q \
  experiments/robotwin \
  scripts/summarize_robotwin_cis.py \
  scripts/validate_robotwin_cis_manifest.py \
  third_party/RoboTwin/script/eval_policy.py

echo "RoboTwin CIS server preflight passed."
