#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-$(pwd)}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${RUN_ROOT}/third_party/RoboTwin}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_ROOT}/checkpoints/fastwam_release}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"
KEEP_ASSET_ARCHIVES="${KEEP_ASSET_ARCHIVES:-0}"
UPSTREAM_COMMIT="bf44be51cf5717a5595ce59447f2cf5263d2aa95"
UPSTREAM_RAW="https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/${UPSTREAM_COMMIT}"

export CHECKPOINT_ROOT HF_ENDPOINT ROBOTWIN_ROOT

for required_command in curl "${PYTHON_BIN}"; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "Required command not found: ${required_command}" >&2
    exit 1
  fi
done

if ! "${PYTHON_BIN}" -c 'import huggingface_hub; assert huggingface_hub.__version__ == "0.29.2"' >/dev/null 2>&1; then
  echo "FastWAM requires huggingface_hub==0.29.2." >&2
  echo "Run: pip install --no-deps --force-reinstall 'huggingface_hub==0.29.2'" >&2
  exit 1
fi

mkdir -p "${CHECKPOINT_ROOT}" "${ROBOTWIN_ROOT}/assets" "${ROBOTWIN_ROOT}/task_config"

echo "[setup] Hugging Face endpoint: ${HF_ENDPOINT}"
df -h "${RUN_ROOT}"

"${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import hf_hub_download

target = os.environ["CHECKPOINT_ROOT"]
for filename in (
    "robotwin_uncond_3cam_384.pt",
    "robotwin_uncond_3cam_384_dataset_stats.json",
):
    path = hf_hub_download(
        repo_id="yuanty/fastwam",
        filename=filename,
        local_dir=target,
        resume_download=True,
    )
    print(f"[setup] checkpoint artifact: {path}")
PY

for config_name in \
  _eval_step_limit.yml \
  _camera_config.yml \
  _embodiment_config.yml \
  demo_clean.yml \
  demo_randomized.yml; do
  config_path="${ROBOTWIN_ROOT}/task_config/${config_name}"
  if [[ ! -s "${config_path}" ]]; then
    echo "[setup] downloading task_config/${config_name}"
    config_tmp="${config_path}.download"
    curl --fail --location --retry 5 --retry-delay 3 --connect-timeout 20 \
      "${UPSTREAM_RAW}/task_config/${config_name}" \
      --output "${config_tmp}"
    mv "${config_tmp}" "${config_path}"
  fi
done

for archive_name in background_texture.zip embodiments.zip objects.zip; do
  stem="${archive_name%.zip}"
  archive_path="${ROBOTWIN_ROOT}/assets/${archive_name}"
  marker="${ROBOTWIN_ROOT}/assets/.${stem}.extract_complete"
  if [[ ! -d "${ROBOTWIN_ROOT}/assets/${stem}" ]]; then
    rm -f "${marker}"
  fi
  if [[ ! -f "${marker}" ]]; then
    ASSET_ARCHIVE_NAME="${archive_name}"
    export ASSET_ARCHIVE_NAME
    "${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    repo_type="dataset",
    allow_patterns=[os.environ["ASSET_ARCHIVE_NAME"]],
    local_dir=os.path.join(os.environ["ROBOTWIN_ROOT"], "assets"),
    resume_download=True,
)
PY
    echo "[setup] extracting ${archive_name}"
    ASSET_ARCHIVE_PATH="${archive_path}"
    export ASSET_ARCHIVE_PATH
    "${PYTHON_BIN}" - <<'PY'
import os
import zipfile

archive = os.environ["ASSET_ARCHIVE_PATH"]
destination = os.path.join(os.environ["ROBOTWIN_ROOT"], "assets")
with zipfile.ZipFile(archive) as bundle:
    bundle.extractall(destination)
PY
    touch "${marker}"
  fi
  if [[ "${KEEP_ASSET_ARCHIVES}" == "0" ]]; then
    rm -f "${archive_path}"
  fi
done

(
  cd "${ROBOTWIN_ROOT}"
  "${PYTHON_BIN}" script/update_embodiment_config_path.py
)

echo "[setup] RoboTwin CIS artifacts are ready."
echo "[setup] checkpoint=${CHECKPOINT_ROOT}/robotwin_uncond_3cam_384.pt"
echo "[setup] stats=${CHECKPOINT_ROOT}/robotwin_uncond_3cam_384_dataset_stats.json"
