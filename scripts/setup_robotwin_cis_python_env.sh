#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-$(pwd)}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${RUN_ROOT}/third_party/RoboTwin}"
CUROBO_ROOT="${CUROBO_ROOT:-${ROBOTWIN_ROOT}/envs/curobo}"
CUROBO_TAG="${CUROBO_TAG:-v0.7.8}"
MAX_JOBS="${MAX_JOBS:-4}"

for required_command in git "${PYTHON_BIN}"; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "Required command not found: ${required_command}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" -m pip install --upgrade-strategy only-if-needed \
  "sapien==3.0.0b1" \
  "mplib==0.2.1" \
  "transforms3d==0.4.2" \
  "gymnasium==0.29.1" \
  "trimesh==4.4.3" \
  "open3d==0.18.0" \
  "ninja" \
  "warp-lang==1.12.0"

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

import mplib
import sapien


def replace_in_file(path: Path, replacements: tuple[tuple[str, str], ...]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    source = path.read_text(encoding="utf-8")
    updated = source
    for old, new in replacements:
        updated = updated.replace(old, new)
    if updated != source:
        path.write_text(updated, encoding="utf-8")
        print(f"[python-env] patched {path}")
    else:
        print(f"[python-env] patch already applied or not needed: {path}")


sapien_root = Path(sapien.__file__).resolve().parent
replace_in_file(
    sapien_root / "wrapper" / "urdf_loader.py",
    (
        (
            'with open(urdf_file, "r") as f:',
            'with open(urdf_file, "r", encoding="utf-8") as f:',
        ),
        (
            'with open(srdf_file, "r") as f:',
            'with open(srdf_file, "r", encoding="utf-8") as f:',
        ),
        (
            'srdf_file = urdf_file[:-4] + "srdf"',
            'srdf_file = urdf_file[:-4] + ".srdf"',
        ),
    ),
)

mplib_root = Path(mplib.__file__).resolve().parent
replace_in_file(
    mplib_root / "planner.py",
    ((" or collide or not within_joint_limit:", " or not within_joint_limit:"),),
)
PY

if ! command -v nvcc >/dev/null 2>&1; then
  for cuda_candidate in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12; do
    if [[ -x "${cuda_candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${cuda_candidate}"
      export PATH="${CUDA_HOME}/bin:${PATH}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      break
    fi
  done
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA compiler nvcc is required to build CuRobo ${CUROBO_TAG}." >&2
  echo "Use a CUDA devel container/toolkit matching the installed CUDA 12 PyTorch." >&2
  exit 1
fi

if [[ -e "${CUROBO_ROOT}" && ! -d "${CUROBO_ROOT}/.git" ]]; then
  echo "CuRobo target exists but is not a git checkout: ${CUROBO_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${CUROBO_ROOT}/.git" ]]; then
  git clone --branch "${CUROBO_TAG}" --depth 1 \
    https://github.com/NVlabs/curobo.git \
    "${CUROBO_ROOT}"
fi

installed_curobo_tag="$(git -C "${CUROBO_ROOT}" describe --tags --exact-match 2>/dev/null || true)"
if [[ "${installed_curobo_tag}" != "${CUROBO_TAG}" ]]; then
  echo "Expected CuRobo ${CUROBO_TAG}, found ${installed_curobo_tag:-an untagged checkout}." >&2
  exit 1
fi

TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$("${PYTHON_BIN}" -c 'import torch; major, minor = torch.cuda.get_device_capability(0); print(f"{major}.{minor}")')}"
export MAX_JOBS TORCH_CUDA_ARCH_LIST
echo "[python-env] building CuRobo ${CUROBO_TAG} with nvcc=$(command -v nvcc) arch=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS}"
nvcc --version
"${PYTHON_BIN}" -m pip install -e "${CUROBO_ROOT}" --no-build-isolation

# CuRobo's transitive dependencies currently select newer releases than the
# FastWAM/Requests environment supports. Restore the shared compatibility pins
# after the editable CuRobo install; neither package affects the CUDA build.
"${PYTHON_BIN}" -m pip install --no-deps \
  "packaging==25.0" \
  "chardet==5.2.0"

"${PYTHON_BIN}" - <<'PY'
import curobo
import gymnasium
import mplib
import open3d
import requests
import sapien
import transforms3d
import trimesh

print(
    "[python-env] imports passed:",
    f"curobo={getattr(curobo, '__version__', 'unknown')}",
    f"sapien={getattr(sapien, '__version__', 'unknown')}",
    f"mplib={getattr(mplib, '__version__', 'unknown')}",
)
PY
