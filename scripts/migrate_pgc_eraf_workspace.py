#!/usr/bin/env python3
"""Migrate ERAF normalized 3D labels into a shared workspace contract.

The migration is intentionally out-of-place. Simulator state/action audit
digests remain unchanged; only normalized position and anchor arrays are
rewritten, and every sidecar file digest is recomputed in the copied index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from fastwam.datasets.pgc_libero import (  # noqa: E402
    PGC_ENTITY_RELATION_FORMAT,
    PGC_ENTITY_RELATION_INDEX,
    PGC_ENTITY_RELATION_WORKSPACE_MAX,
    PGC_ENTITY_RELATION_WORKSPACE_MIN,
    atomic_write_json,
    load_pgc_entity_relation_index,
)


_POSITION_ARRAY_VALIDITY = {
    "subject_positions": "subject_position_valid",
    "reference_positions": "reference_position_valid",
    "grasp_anchors": "grasp_anchor_valid",
    "goal_anchors": "goal_anchor_valid",
    "interaction_anchors": "interaction_anchor_valid",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate a PGC ERAF sidecar to canonical workspace bounds."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workspace-min",
        type=float,
        nargs=3,
        default=PGC_ENTITY_RELATION_WORKSPACE_MIN,
    )
    parser.add_argument(
        "--workspace-max",
        type=float,
        nargs=3,
        default=PGC_ENTITY_RELATION_WORKSPACE_MAX,
    )
    args = parser.parse_args()
    if args.input.expanduser().resolve() == args.output.expanduser().resolve():
        parser.error("--output must differ from --input; in-place migration is forbidden")
    lower = np.asarray(args.workspace_min, dtype=np.float32)
    upper = np.asarray(args.workspace_max, dtype=np.float32)
    if np.any(upper <= lower):
        parser.error("workspace max must exceed min on every axis")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remap_normalized(
    value: np.ndarray,
    *,
    old_min: np.ndarray,
    old_max: np.ndarray,
    new_min: np.ndarray,
    new_max: np.ndarray,
) -> np.ndarray:
    source = np.asarray(value)
    world = (source.astype(np.float64) + 1.0) * 0.5
    world = world * (old_max.astype(np.float64) - old_min.astype(np.float64))
    world = world + old_min.astype(np.float64)
    migrated = 2.0 * (world - new_min.astype(np.float64))
    migrated = migrated / (new_max.astype(np.float64) - new_min.astype(np.float64))
    migrated = np.clip(migrated - 1.0, -1.0, 1.0)
    return migrated.astype(source.dtype, copy=False)


def _migrate_episode(
    path: Path,
    *,
    old_min: np.ndarray,
    old_max: np.ndarray,
    new_min: np.ndarray,
    new_max: np.ndarray,
) -> int:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    migrated_count = 0
    for role in ("target", "source"):
        for array_suffix, validity_suffix in _POSITION_ARRAY_VALIDITY.items():
            array_name = f"{role}_{array_suffix}"
            validity_name = f"{role}_{validity_suffix}"
            if array_name not in arrays or validity_name not in arrays:
                raise ValueError(
                    f"ERAF episode {path} lacks {array_name!r} or {validity_name!r}."
                )
            values = arrays[array_name]
            valid = np.asarray(arrays[validity_name], dtype=bool)
            if values.shape[:-1] != valid.shape or values.shape[-1] != 3:
                raise ValueError(
                    f"ERAF position/validity shape mismatch for {array_name}: "
                    f"{values.shape} versus {valid.shape}."
                )
            remapped = _remap_normalized(
                values,
                old_min=old_min,
                old_max=old_max,
                new_min=new_min,
                new_max=new_max,
            )
            values[valid] = remapped[valid]
            arrays[array_name] = values
            migrated_count += int(valid.sum())
    temporary = path.with_name(f".{path.name}.workspace-migration.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return migrated_count


def migrate(args: argparse.Namespace) -> dict[str, Any]:
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_index_path = source / PGC_ENTITY_RELATION_INDEX
    if not source_index_path.is_file():
        raise FileNotFoundError(f"Missing ERAF index: {source_index_path}")
    payload = json.loads(source_index_path.read_text(encoding="utf-8"))
    if payload.get("format") != PGC_ENTITY_RELATION_FORMAT:
        raise ValueError(f"Unsupported ERAF format: {payload.get('format')!r}")
    old_min = np.asarray(payload.get("workspace_min"), dtype=np.float32)
    old_max = np.asarray(payload.get("workspace_max"), dtype=np.float32)
    new_min = np.asarray(args.workspace_min, dtype=np.float32)
    new_max = np.asarray(args.workspace_max, dtype=np.float32)
    if old_min.shape != (3,) or old_max.shape != (3,) or np.any(old_max <= old_min):
        raise ValueError("Input ERAF workspace bounds are invalid.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite migration output: {output}")
    temporary_root = output.with_name(f".{output.name}.workspace-migration.tmp")
    if temporary_root.exists():
        raise FileExistsError(f"Stale migration temporary directory: {temporary_root}")
    shutil.copytree(source, temporary_root)
    total_labels = 0
    try:
        episodes = payload.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("Input ERAF index contains no episode records.")
        for record in episodes:
            relpath = Path(str(record.get("file", "")))
            if relpath.is_absolute() or ".." in relpath.parts:
                raise ValueError(f"Unsafe ERAF episode path: {relpath}")
            episode_path = temporary_root / relpath
            total_labels += _migrate_episode(
                episode_path,
                old_min=old_min,
                old_max=old_max,
                new_min=new_min,
                new_max=new_max,
            )
            record["sha256"] = _sha256(episode_path)
        payload["workspace_min"] = new_min.tolist()
        payload["workspace_max"] = new_max.tolist()
        payload["workspace_migration"] = {
            "format": "pgc_eraf_workspace_migration_v1",
            "source": str(source),
            "source_workspace_min": old_min.tolist(),
            "source_workspace_max": old_max.tolist(),
            "normalized_label_count": total_labels,
            "state_action_audits_preserved": True,
        }
        atomic_write_json(temporary_root / PGC_ENTITY_RELATION_INDEX, payload)
        load_pgc_entity_relation_index(temporary_root)
        temporary_root.replace(output)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return {
        "format": "pgc_eraf_workspace_migration_v1",
        "input": str(source),
        "output": str(output),
        "episode_count": len(payload["episodes"]),
        "normalized_label_count": total_labels,
        "workspace_min": new_min.tolist(),
        "workspace_max": new_max.tolist(),
        "validated": True,
    }


def main() -> None:
    print(json.dumps(migrate(_parse_args()), indent=2))


if __name__ == "__main__":
    main()
