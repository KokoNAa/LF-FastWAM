#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip install --upgrade-strategy only-if-needed \
  "sapien==3.0.0b1" \
  "mplib==0.2.1" \
  "transforms3d==0.4.2" \
  "gymnasium==0.29.1" \
  "trimesh==4.4.3" \
  "open3d==0.18.0"

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

"${PYTHON_BIN}" - <<'PY'
import gymnasium
import mplib
import open3d
import sapien
import transforms3d
import trimesh

print(
    "[python-env] imports passed:",
    f"sapien={getattr(sapien, '__version__', 'unknown')}",
    f"mplib={getattr(mplib, '__version__', 'unknown')}",
)
PY
