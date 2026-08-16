"""Pure data-contract helpers for collecting PGC LIBERO demonstrations.

The simulator-facing entry point lives in ``scripts/build_pgc_libero_data.py``.
This module intentionally has no torch, LIBERO, or robosuite imports so that
manifest/demo validation can run on a CPU workstation before a server job is
started.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


PGC_DATA_FORMAT = "pgc_counterfactual_actions_v1"
PGC_ACTION_SUPERVISION = "executed_counterfactual_success_trajectory"
LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)
PGC_STATE_TRANSFER_MODES = ("flat_exact", "named_joint_remap")


@dataclass(frozen=True)
class LiberoDemo:
    """One action demonstration lazily copied out of a LIBERO HDF5 file."""

    group_name: str
    initial_state: np.ndarray
    actions: np.ndarray


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            records.append(record)
    return records


def load_pgc_episode_language_pairs(
    dataset_root: str | Path,
) -> dict[int, dict[str, Any]]:
    """Map each audited PGC episode to its same-state language pair.

    PGC v5 reuses the already collected successful counterfactual trajectory
    twice: once with its recorded counterfactual instruction (positive action
    supervision), and once with the source instruction on the *identical*
    visual state (strict zero-residual supervision).  The collection contract
    deliberately stores the language pair at dataset level and the pair ID at
    episode level, so the training dataset can recover this mapping without
    duplicating videos or actions.
    """
    dataset_root = Path(dataset_root).expanduser()
    provenance_path = dataset_root / "meta/pgc_provenance.json"
    episodes_path = dataset_root / "meta/pgc_episodes.jsonl"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing PGC provenance: {provenance_path}")
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing PGC episode audit: {episodes_path}")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("format") != PGC_DATA_FORMAT:
        raise ValueError(
            f"Unsupported PGC data format at {provenance_path}: "
            f"{provenance.get('format')!r}."
        )
    if provenance.get("state_aligned") is not True:
        raise ValueError(
            f"PGC paired-language training requires state_aligned=true: "
            f"{provenance_path}."
        )

    pairs_by_id: dict[str, dict[str, Any]] = {}
    for pair in provenance.get("pairs") or []:
        pair_id = str(pair.get("pair_id", "")).strip()
        source_instruction = str(pair.get("source_instruction", "")).strip()
        counterfactual_instruction = str(
            pair.get("counterfactual_instruction", "")
        ).strip()
        if not pair_id or not source_instruction or not counterfactual_instruction:
            raise ValueError(
                f"PGC provenance contains an incomplete language pair: {pair!r}."
            )
        if source_instruction.casefold() == counterfactual_instruction.casefold():
            raise ValueError(
                f"PGC pair {pair_id!r} must change the instruction for v5."
            )
        if pair_id in pairs_by_id:
            raise ValueError(f"Duplicate PGC provenance pair ID: {pair_id!r}.")
        pairs_by_id[pair_id] = {
            "pair_id": pair_id,
            "source_instruction": source_instruction,
            "counterfactual_instruction": counterfactual_instruction,
            "source_suite": str(pair.get("source_suite", "")),
            "source_task_id": int(pair.get("source_task_id", -1)),
        }
    if not pairs_by_id:
        raise ValueError(f"PGC provenance has no language pairs: {provenance_path}.")

    result: dict[int, dict[str, Any]] = {}
    for audit in read_jsonl(episodes_path):
        try:
            episode_index = int(audit["episode_index"])
            pair_id = str(audit["pair_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid PGC episode audit in {episodes_path}: {audit!r}."
            ) from exc
        if episode_index < 0:
            raise ValueError("PGC episode indices must be non-negative.")
        if episode_index in result:
            raise ValueError(
                f"Duplicate PGC episode audit index {episode_index} in "
                f"{episodes_path}."
            )
        try:
            pair = pairs_by_id[pair_id]
        except KeyError as exc:
            raise ValueError(
                f"PGC episode {episode_index} references unknown pair "
                f"{pair_id!r}."
            ) from exc
        result[episode_index] = dict(pair)

    expected_count = int(provenance.get("successful_episode_count", len(result)))
    if expected_count != len(result):
        raise ValueError(
            "PGC successful episode count does not match its audit table: "
            f"provenance={expected_count}, audits={len(result)}."
        )
    if not result:
        raise ValueError(f"PGC episode audit is empty: {episodes_path}.")
    return result


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def canonical_state_array(state: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a stable little-endian float64 representation of simulator state."""
    array = np.asarray(state)
    if array.ndim == 0:
        raise ValueError("Simulator state must have at least one dimension.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Simulator state must be numeric, got {array.dtype}.")
    array = np.ascontiguousarray(array.astype("<f8", copy=False))
    if not np.isfinite(array).all():
        raise ValueError("Simulator state contains NaN or infinity.")
    return array


def state_sha256(state: np.ndarray | Sequence[float]) -> str:
    """Hash state values and shape independently of host byte order/dtype."""
    array = canonical_state_array(state)
    header = json.dumps(
        {"dtype": "float64-le", "shape": list(array.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def states_match(
    requested: np.ndarray | Sequence[float],
    actual: np.ndarray | Sequence[float],
    *,
    atol: float = 1e-7,
) -> bool:
    requested_array = canonical_state_array(requested)
    actual_array = canonical_state_array(actual)
    return requested_array.shape == actual_array.shape and bool(
        np.allclose(requested_array, actual_array, rtol=0.0, atol=float(atol))
    )


def filter_libero_noops(
    actions: np.ndarray | Sequence[Sequence[float]],
    *,
    threshold: float = 1e-4,
) -> np.ndarray:
    """Match the no-op removal used by LIBERO dataset regeneration.

    A stationary action is kept when it changes the gripper command.  The
    previous *kept* action is used, matching the released no-noops converter.
    """
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"LIBERO actions must be [T,7], got {array.shape}.")
    kept: list[np.ndarray] = []
    for action in array:
        previous = kept[-1] if kept else None
        stationary = float(np.linalg.norm(action[:-1])) < float(threshold)
        same_gripper = previous is None or bool(action[-1] == previous[-1])
        if stationary and same_gripper:
            continue
        kept.append(action)
    if not kept:
        return np.empty((0, 7), dtype=np.float32)
    return np.ascontiguousarray(np.stack(kept).astype(np.float32, copy=False))


def _normalise_stem(value: str) -> str:
    value = Path(value).stem.casefold()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    for suffix in ("_demo", "_demonstration"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value


def demo_file_candidates(
    demo_root: str | Path,
    record: Mapping[str, Any],
) -> list[Path]:
    """Find HDF5 files whose name matches the target task/BDDL stem."""
    root = Path(demo_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LIBERO demonstration root not found: {root}")

    desired_stems = {
        _normalise_stem(str(record.get("counterfactual_bddl_file", ""))),
        _normalise_stem(str(record.get("counterfactual_task_name", ""))),
        _normalise_stem(str(record.get("counterfactual_instruction", ""))),
    }
    desired_stems.discard("")
    if not desired_stems:
        raise ValueError(
            f"Manifest pair {record.get('pair_id')!r} has no target BDDL/task name."
        )

    matches = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in {".hdf5", ".h5"}:
            continue
        candidate_stem = _normalise_stem(path.name)
        if candidate_stem in desired_stems:
            matches.append(path.resolve())
    matches = sorted(set(matches))
    target_suite = str(record.get("counterfactual_task_suite_name", "")).casefold()
    if len(matches) > 1 and target_suite:
        suite_matches = [
            path
            for path in matches
            if target_suite in {part.casefold() for part in path.parts}
        ]
        if suite_matches:
            matches = suite_matches
    return matches


def resolve_demo_file(
    demo_root: str | Path,
    record: Mapping[str, Any],
) -> Path:
    matches = demo_file_candidates(demo_root, record)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one target demo file for pair "
            f"{record.get('pair_id')!r}, found {len(matches)}: "
            f"{[str(path) for path in matches]}"
        )
    return matches[0]


def _demo_sort_key(group_name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(group_name))
    return (int(match.group(1)) if match else 2**31 - 1, str(group_name))


def iter_libero_hdf5_demos(path: str | Path) -> Iterator[LiberoDemo]:
    """Yield standard LIBERO ``states``/``actions`` demonstration groups."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - server dependency check
        raise ImportError("PGC data building requires h5py.") from exc

    path = Path(path)
    with h5py.File(path, "r") as handle:
        root = handle["data"] if "data" in handle else handle
        group_names = sorted(root.keys(), key=_demo_sort_key)
        found = 0
        for group_name in group_names:
            group = root[group_name]
            if not hasattr(group, "keys") or not {"states", "actions"}.issubset(
                set(group.keys())
            ):
                continue
            states = np.asarray(group["states"])
            actions = np.asarray(group["actions"], dtype=np.float32)
            if states.ndim != 2 or states.shape[0] == 0:
                raise ValueError(
                    f"{path}:{group_name}/states must be non-empty [T,D], "
                    f"got {states.shape}."
                )
            if actions.ndim != 2 or actions.shape[0] == 0:
                raise ValueError(
                    f"{path}:{group_name}/actions must be non-empty [T,A], "
                    f"got {actions.shape}."
                )
            if not np.isfinite(actions).all():
                raise ValueError(f"{path}:{group_name}/actions contains NaN or infinity.")
            found += 1
            yield LiberoDemo(
                group_name=str(group_name),
                initial_state=np.asarray(states[0]).copy(),
                actions=np.ascontiguousarray(actions),
            )
        if found == 0:
            raise ValueError(f"No LIBERO states/actions demos found in {path}.")


def validate_manifest_record(record: Mapping[str, Any]) -> None:
    required = {
        "pair_id",
        "task_suite_name",
        "task_id",
        "correct_instruction",
        "counterfactual_instruction",
        "counterfactual_goal_state",
        "counterfactual_bddl_file",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(
            f"Manifest pair {record.get('pair_id')!r} is missing {sorted(missing)}."
        )
    if str(record["task_suite_name"]) not in LIBERO_SUITES:
        raise ValueError(
            f"Unsupported source suite {record['task_suite_name']!r}."
        )
    if not 0 <= int(record["task_id"]) < 10:
        raise ValueError(
            f"Pair {record['pair_id']!r} source task ID must be in [0,9]."
        )
    if not str(record["counterfactual_instruction"]).strip():
        raise ValueError(f"Pair {record['pair_id']!r} has an empty target instruction.")
    goal = record["counterfactual_goal_state"]
    if not isinstance(goal, list) or not goal:
        raise ValueError(f"Pair {record['pair_id']!r} has no alternate goal state.")
    transfer_mode = str(record.get("state_transfer_mode", "flat_exact"))
    if transfer_mode not in PGC_STATE_TRANSFER_MODES:
        raise ValueError(
            f"Pair {record['pair_id']!r} has unsupported state_transfer_mode "
            f"{transfer_mode!r}; expected one of {PGC_STATE_TRANSFER_MODES}."
        )
    if (
        record.get("counterfactual_goal_changed") is False
        and str(record["task_suite_name"]) != "libero_spatial"
    ):
        raise ValueError(
            f"Pair {record['pair_id']!r} may keep the terminal goal only for "
            "LIBERO-Spatial state-grounded supervision."
        )


def provenance_pair(record: Mapping[str, Any]) -> dict[str, Any]:
    validate_manifest_record(record)
    return {
        "pair_id": str(record["pair_id"]),
        "source_suite": str(record["task_suite_name"]),
        "source_task_id": int(record["task_id"]),
        "source_instruction": str(record["correct_instruction"]).strip(),
        "counterfactual_instruction": str(
            record["counterfactual_instruction"]
        ).strip(),
        "counterfactual_goal_state": record["counterfactual_goal_state"],
        "counterfactual_task_suite_name": str(
            record.get(
                "counterfactual_task_suite_name", record["task_suite_name"]
            )
        ),
        "counterfactual_task_id": int(
            record.get("counterfactual_task_id", record["task_id"])
        ),
        "source_bddl_file": str(record.get("source_bddl_file", "")),
        "counterfactual_bddl_file": str(record["counterfactual_bddl_file"]),
        "state_transfer_mode": str(
            record.get("state_transfer_mode", "flat_exact")
        ),
        "counterfactual_goal_changed": bool(
            record.get("counterfactual_goal_changed", True)
        ),
        "counterfactual_state_changed": bool(
            record.get("counterfactual_state_changed", True)
        ),
    }


def build_provenance(
    records: Sequence[Mapping[str, Any]],
    *,
    successful_episode_count: int = 0,
) -> dict[str, Any]:
    if not records:
        raise ValueError("A PGC dataset requires at least one intervention pair.")
    pairs = [provenance_pair(record) for record in records]
    pair_ids = [pair["pair_id"] for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("PGC manifest contains duplicate pair IDs.")
    suites = sorted({pair["source_suite"] for pair in pairs})
    return {
        "format": PGC_DATA_FORMAT,
        "benchmark": "libero",
        "action_supervision": PGC_ACTION_SUPERVISION,
        "collection_method": (
            "audited_target_demo_replay_with_exact_or_named_joint_state_transfer"
        ),
        "state_aligned": True,
        "state_match_tolerance": 1e-7,
        "state_catalog": "meta/pgc_initial_states/episode_{episode_index:06d}.npy",
        "successful_only": True,
        "successful_episode_count": int(successful_episode_count),
        "source_suites": suites,
        "pairs": pairs,
    }


def libero_lerobot_features(resolution: int = 512) -> dict[str, dict[str, Any]]:
    resolution = int(resolution)
    if resolution <= 0:
        raise ValueError("Camera resolution must be positive.")
    return {
        "observation.images.image": {
            "dtype": "video",
            "shape": (3, resolution, resolution),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": (3, resolution, resolution),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "eef_x",
                "eef_y",
                "eef_z",
                "eef_axis_x",
                "eef_axis_y",
                "eef_axis_z",
                "gripper_left",
                "gripper_right",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "delta_x",
                "delta_y",
                "delta_z",
                "delta_axis_x",
                "delta_axis_y",
                "delta_axis_z",
                "gripper",
            ],
        },
    }
