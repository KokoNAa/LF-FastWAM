#!/usr/bin/env python3
"""Build hash-audited PGC V9 entity--relation supervision sidecars.

The generated files contain privileged simulator labels only.  They are kept
outside the LeRobot dataset and are never opened by the inference path.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from fastwam.datasets.pgc_libero import (  # noqa: E402
    PGC_ENTITY_RELATION_ARRAY_NAMES,
    PGC_ENTITY_RELATION_FORMAT,
    PGC_ENTITY_RELATION_INDEX,
    array_sha256,
    atomic_write_json,
    canonical_state_array,
    filter_libero_noops,
    iter_libero_hdf5_demos,
    libero_problem_entity_catalog,
    parse_libero_goal_clauses,
    read_jsonl,
    resolve_demo_file,
    state_sha256,
)
from scripts.build_pgc_libero_data import (  # noqa: E402
    PGC_EPISODES,
    PGC_PROVENANCE,
    _get_inner_env,
    _load_suite_manifest,
    _make_source_env,
    _reset_exact_state,
    _set_counterfactual_goal,
)


CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")
ARRAY_NAMES = PGC_ENTITY_RELATION_ARRAY_NAMES


def _entity_id(name: str) -> int:
    """Stable positive int64 entity ID shared by every generated sidecar."""
    digest = hashlib.sha256(str(name).strip().casefold().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--hdf5-root",
        type=Path,
        help="Required for a native (non-PGC) LeRobot dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-height", type=int, default=56)
    parser.add_argument("--mask-width", type=int, default=112)
    parser.add_argument("--max-clauses", type=int, default=4)
    parser.add_argument("--state-atol", type=float, default=1.0e-7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument(
        "--workspace-min", nargs=3, type=float, default=(-0.8, -0.8, 0.0)
    )
    parser.add_argument(
        "--workspace-max", nargs=3, type=float, default=(0.8, 0.8, 1.2)
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mask_height <= 0 or args.mask_width <= 0 or args.mask_width % 2:
        parser.error("mask dimensions must be positive and width must be even")
    if args.max_clauses != 4:
        parser.error("the PGC v9 contract requires --max-clauses=4")
    lower = np.asarray(args.workspace_min, dtype=np.float32)
    upper = np.asarray(args.workspace_max, dtype=np.float32)
    if np.any(upper <= lower):
        parser.error("workspace max must be greater than min on every axis")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _problem_for_task(suite_name: str, task_id: int) -> tuple[Any, dict[str, Any], Path]:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import bddl_utils as BDDLUtils

    suite = benchmark.get_benchmark_dict()[str(suite_name)]()
    task = suite.get_task(int(task_id))
    path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return task, BDDLUtils.robosuite_parse_problem(str(path)), path


def _problem_from_path(path: str | Path) -> dict[str, Any]:
    from libero.libero.envs import bddl_utils as BDDLUtils

    return BDDLUtils.robosuite_parse_problem(str(Path(path)))


def _record_roles(
    record: Mapping[str, Any], *, max_clauses: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    _, source_problem, _ = _problem_for_task(
        str(record["task_suite_name"]), int(record["task_id"])
    )
    target_path = Path(str(record["counterfactual_bddl_file"]))
    target_problem = _problem_from_path(target_path)
    source = parse_libero_goal_clauses(
        source_problem["goal_state"],
        regions=source_problem.get("regions", {}),
        max_clauses=max_clauses,
        instruction=str(record["correct_instruction"]),
        entity_catalog=libero_problem_entity_catalog(source_problem),
    )
    target = parse_libero_goal_clauses(
        target_problem["goal_state"],
        regions=target_problem.get("regions", {}),
        max_clauses=max_clauses,
        instruction=str(record["counterfactual_instruction"]),
        entity_catalog=libero_problem_entity_catalog(target_problem),
    )
    return target, source, target_problem, source_problem


def _native_record(record: Mapping[str, Any]) -> dict[str, Any]:
    _, _, source_bddl = _problem_for_task(
        str(record["task_suite_name"]), int(record["task_id"])
    )
    cloned = dict(record)
    cloned.update(
        {
            "pair_id": f"native_{record['task_suite_name']}_{int(record['task_id']):02d}",
            "counterfactual_instruction": str(record["correct_instruction"]),
            "counterfactual_task_suite_name": str(record["task_suite_name"]),
            "counterfactual_task_id": int(record["task_id"]),
            "counterfactual_bddl_file": str(source_bddl),
            "counterfactual_goal_changed": False,
        }
    )
    return cloned


def _episode_actions(dataset: Any, episode_index: int) -> np.ndarray:
    episode = dataset.get_episode_data(int(episode_index))
    action = episode.get("action")
    if action is None:
        raise KeyError(f"Episode {episode_index} has no action column.")
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 7 or not np.isfinite(action).all():
        raise ValueError(f"Episode {episode_index} actions must be finite [T,7].")
    return np.ascontiguousarray(action)


def _episode_instruction(dataset: Any, episode_index: int) -> str:
    episode = dataset.get_episode_data(int(episode_index))
    task_index = np.asarray(episode["task_index"])[0]
    if hasattr(task_index, "item"):
        task_index = task_index.item()
    return str(dataset.meta.tasks[int(task_index)]).strip()


def _match_native_demo(
    *,
    record: Mapping[str, Any],
    actions: np.ndarray,
    hdf5_root: Path,
    used: set[tuple[str, str]],
) -> tuple[np.ndarray, str, str]:
    lookup = dict(record)
    # ``record`` may describe a cross-suite counterfactual pair (for example a
    # LIBERO-10 source with a LIBERO-90 donor).  Native episodes must always be
    # audited against the *source* suite/task.  Leaving the counterfactual
    # suite in the lookup can resolve an identically named LIBERO-90 task and
    # then make every otherwise-valid native action audit fail.
    lookup["counterfactual_task_suite_name"] = str(record["task_suite_name"])
    lookup["counterfactual_task_id"] = int(record["task_id"])
    lookup["counterfactual_task_name"] = str(record["correct_instruction"])
    lookup["counterfactual_instruction"] = str(record["correct_instruction"])
    _, _, source_bddl = _problem_for_task(
        str(record["task_suite_name"]), int(record["task_id"])
    )
    lookup["counterfactual_bddl_file"] = str(source_bddl)
    demo_path = resolve_demo_file(hdf5_root, lookup)
    candidates: list[tuple[np.ndarray, str]] = []
    for demo in iter_libero_hdf5_demos(demo_path):
        filtered = filter_libero_noops(demo.actions)
        if filtered.shape == actions.shape and np.allclose(
            filtered, actions, rtol=0.0, atol=2.0e-5
        ):
            candidates.append((demo.initial_state, demo.group_name))
    available = [
        item for item in candidates if (str(demo_path), item[1]) not in used
    ]
    if len(available) != 1:
        raise RuntimeError(
            "Native LeRobot/HDF5 action audit did not find one unused exact "
            f"match for {record['correct_instruction']!r}: {len(available)}."
        )
    state, group_name = available[0]
    used.add((str(demo_path), group_name))
    return np.asarray(state).copy(), str(demo_path), group_name


def _instance_geom_ids(env: Any, entity_names: Sequence[str]) -> dict[str, np.ndarray]:
    mapping = getattr(_get_inner_env(env).model, "instances_to_ids", None)
    if not isinstance(mapping, Mapping):
        raise RuntimeError("robosuite model has no instances_to_ids mapping.")
    result: dict[str, np.ndarray] = {}
    for name in entity_names:
        instance = mapping.get(name)
        if not isinstance(instance, Mapping):
            stem = name.rsplit("_", 1)[0]
            candidates = [
                value
                for key, value in mapping.items()
                if key == stem or key.startswith(stem + "_")
            ]
            if len(candidates) == 1:
                instance = candidates[0]
        ids = [] if not isinstance(instance, Mapping) else instance.get("geom", [])
        result[name] = np.asarray(ids, dtype=np.int32).reshape(-1)
    return result


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(
        image.resize((width, height), resample=Image.Resampling.NEAREST),
        dtype=np.uint8,
    ) > 0


def _entity_masks(
    obs: Mapping[str, Any],
    geom_ids: Mapping[str, np.ndarray],
    *,
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    half = width // 2
    result: dict[str, list[np.ndarray]] = {name: [] for name in geom_ids}
    for camera in CAMERA_NAMES:
        segmentation = np.asarray(obs[f"{camera}_segmentation_element"])
        if segmentation.ndim == 3:
            segmentation = segmentation[..., 0]
        segmentation = np.ascontiguousarray(segmentation[::-1, ::-1])
        for name, ids in geom_ids.items():
            result[name].append(
                _resize_mask(np.isin(segmentation, ids), height, half)
            )
    return {
        name: np.concatenate(camera_masks, axis=-1)
        for name, camera_masks in result.items()
    }


def _per_view_mask_geometry(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-camera visibility and local normalized (x,y) mask centers."""
    if mask.ndim != 2 or mask.shape[1] % len(CAMERA_NAMES):
        raise ValueError("ERAF concatenated role mask has invalid camera geometry.")
    view_masks = np.stack(np.split(mask, len(CAMERA_NAMES), axis=-1), axis=0)
    visible = view_masks.reshape(len(CAMERA_NAMES), -1).any(axis=-1)
    centers = np.zeros((len(CAMERA_NAMES), 2), dtype=np.float32)
    height, width = view_masks.shape[-2:]
    for camera_index in np.flatnonzero(visible):
        rows, columns = np.nonzero(view_masks[int(camera_index)])
        centers[int(camera_index)] = np.asarray(
            [
                2.0 * float(columns.mean()) / max(1, width - 1) - 1.0,
                2.0 * float(rows.mean()) / max(1, height - 1) - 1.0,
            ],
            dtype=np.float32,
        )
    return visible.astype(np.bool_), centers


def _body_position(env: Any, name: str) -> tuple[np.ndarray, bool]:
    inner = _get_inner_env(env)
    body_ids = getattr(inner, "obj_body_id", {})
    body_id = body_ids.get(name) if isinstance(body_ids, Mapping) else None
    if body_id is None:
        try:
            body_id = inner.sim.model.body_name2id(name)
        except Exception:
            return np.zeros(3, dtype=np.float32), False
    return np.asarray(inner.sim.data.body_xpos[int(body_id)], dtype=np.float32), True


def _site_position(env: Any, name: str) -> tuple[np.ndarray, bool]:
    """Resolve a LIBERO region site without guessing its coordinate frame.

    BDDL region keys normally match MuJoCo site names, but robosuite object
    namespaces can add a prefix.  Prefer the environment's structural
    ``object_sites_dict`` entry and only use a suffix match when it is unique.
    """
    inner = _get_inner_env(env)
    model = inner.sim.model
    candidates = [str(name)]
    object_sites = getattr(inner, "object_sites_dict", {})
    site = object_sites.get(str(name)) if isinstance(object_sites, Mapping) else None
    if site is not None:
        if isinstance(site, str):
            candidates.append(site)
        for attribute in ("name", "site_name"):
            value = getattr(site, attribute, None)
            if value:
                candidates.append(str(value))
    site_names = [str(value) for value in getattr(model, "site_names", ())]
    suffix_matches = [
        value
        for value in site_names
        if value == str(name) or value.endswith("_" + str(name))
    ]
    if len(suffix_matches) == 1:
        candidates.append(suffix_matches[0])
    for candidate in dict.fromkeys(candidates):
        try:
            site_id = int(model.site_name2id(candidate))
            if site_id >= 0:
                return (
                    np.asarray(inner.sim.data.site_xpos[site_id], dtype=np.float32),
                    True,
                )
        except Exception:
            continue
    return np.zeros(3, dtype=np.float32), False


def _workspace_offset(env: Any) -> tuple[np.ndarray, bool]:
    """Return the world-frame origin used by LIBERO's table region sampler."""
    value = np.asarray(
        getattr(_get_inner_env(env), "workspace_offset", ()), dtype=np.float32
    ).reshape(-1)
    if value.size < 3 or not np.isfinite(value[:3]).all():
        return np.zeros(3, dtype=np.float32), False
    return value[:3].copy(), True


def _normalize_position(
    position: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return np.clip(2.0 * (position - lower) / (upper - lower) - 1.0, -1.0, 1.0)


def _region_anchor(
    env: Any,
    clause: Mapping[str, Any],
    problem: Mapping[str, Any],
) -> tuple[np.ndarray, bool]:
    region_name = clause.get("reference_region") or clause.get("subject_region")
    if region_name:
        site_position, site_valid = _site_position(env, str(region_name))
        if site_valid:
            return site_position, True
        region = (problem.get("regions") or {}).get(str(region_name), {})
        ranges = region.get("ranges") if isinstance(region, Mapping) else None
        if isinstance(ranges, Sequence) and ranges:
            values = np.asarray(ranges[0], dtype=np.float32).reshape(-1)
            offset, offset_valid = _workspace_offset(env)
            if values.size >= 4 and offset_valid and np.isfinite(values[:4]).all():
                # LIBERO's InitialSceneTemplates create these samplers with
                # reference_pos=self.workspace_offset.  The BDDL range is
                # therefore local to that origin, not already in world space.
                return offset + np.asarray(
                    [
                        (values[0] + values[2]) / 2,
                        (values[1] + values[3]) / 2,
                        0.0,
                    ],
                    dtype=np.float32,
                ), True
    return _body_position(env, str(clause["reference"]))


def _clause_truth(env: Any, raw_clause: Sequence[Any]) -> bool:
    inner = _get_inner_env(env)
    original = inner.parsed_problem.get("goal_state")
    try:
        inner.parsed_problem["goal_state"] = [list(raw_clause)]
        for method_name in ("_check_success", "check_success"):
            method = getattr(inner, method_name, None)
            if callable(method):
                return bool(method())
    finally:
        inner.parsed_problem["goal_state"] = original
    raise RuntimeError("LIBERO environment exposes no predicate success check.")


def _empty_role_arrays(
    frame_count: int, max_clauses: int, height: int, width: int
) -> dict[str, np.ndarray]:
    return {
        "predicate_ids": np.zeros((frame_count, max_clauses), np.int64),
        "clause_valid": np.zeros((frame_count, max_clauses), np.bool_),
        "subject_entity_ids": np.full(
            (frame_count, max_clauses), -1, np.int64
        ),
        "reference_entity_ids": np.full(
            (frame_count, max_clauses), -1, np.int64
        ),
        "subject_masks": np.zeros((frame_count, max_clauses, height, width), np.bool_),
        "reference_masks": np.zeros((frame_count, max_clauses, height, width), np.bool_),
        "subject_mask_valid": np.zeros((frame_count, max_clauses), np.bool_),
        "reference_mask_valid": np.zeros((frame_count, max_clauses), np.bool_),
        "subject_view_visible": np.zeros(
            (frame_count, max_clauses, len(CAMERA_NAMES)), np.bool_
        ),
        "reference_view_visible": np.zeros(
            (frame_count, max_clauses, len(CAMERA_NAMES)), np.bool_
        ),
        "subject_view_centers": np.zeros(
            (frame_count, max_clauses, len(CAMERA_NAMES), 2), np.float32
        ),
        "reference_view_centers": np.zeros(
            (frame_count, max_clauses, len(CAMERA_NAMES), 2), np.float32
        ),
        "subject_positions": np.zeros((frame_count, max_clauses, 3), np.float32),
        "reference_positions": np.zeros((frame_count, max_clauses, 3), np.float32),
        "subject_position_valid": np.zeros((frame_count, max_clauses), np.bool_),
        "reference_position_valid": np.zeros((frame_count, max_clauses), np.bool_),
        "grasp_anchors": np.zeros((frame_count, max_clauses, 3), np.float32),
        "grasp_anchor_valid": np.zeros((frame_count, max_clauses), np.bool_),
        "goal_anchors": np.zeros((frame_count, max_clauses, 3), np.float32),
        "goal_anchor_valid": np.zeros((frame_count, max_clauses), np.bool_),
        "interaction_anchors": np.zeros((frame_count, max_clauses, 3), np.float32),
        "interaction_anchor_valid": np.zeros((frame_count, max_clauses), np.bool_),
        "predicate_truth": np.zeros((frame_count, max_clauses), np.float32),
        "predicate_truth_valid": np.zeros((frame_count, max_clauses), np.bool_),
        "phase_ids": np.zeros((frame_count, max_clauses), np.int64),
        "phase_valid": np.zeros((frame_count, max_clauses), np.bool_),
    }


def _replay_labels(
    env: Any,
    *,
    initial_state: np.ndarray,
    actions: np.ndarray,
    target_clauses: list[dict[str, Any]],
    source_clauses: list[dict[str, Any]],
    target_problem: Mapping[str, Any],
    source_problem: Mapping[str, Any],
    state_atol: float,
    settle_steps: int,
    height: int,
    width: int,
    max_clauses: int,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    source_execution_supervision: bool,
) -> tuple[dict[str, np.ndarray], str]:
    obs, _ = _reset_exact_state(
        env, initial_state, state_atol=state_atol, settle_steps=settle_steps
    )
    entities = sorted(
        {
            str(clause[role])
            for clauses in (target_clauses, source_clauses)
            for clause in clauses
            for role in ("subject", "reference")
        }
    )
    geom_ids = _instance_geom_ids(env, entities)
    labels = {
        role: _empty_role_arrays(len(actions), max_clauses, height, width)
        for role in ("target", "source")
    }
    subject_tracks: dict[tuple[str, int], list[np.ndarray]] = {}
    eef_track: list[np.ndarray] = []
    gripper_track: list[float] = []
    state_digest = hashlib.sha256()
    for frame_index, action in enumerate(actions):
        inner = _get_inner_env(env)
        state = canonical_state_array(inner.sim.get_state().flatten())
        state_digest.update(state.tobytes(order="C"))
        masks = _entity_masks(obs, geom_ids, height=height, width=width)
        eef_track.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
        gripper_track.append(float(action[-1]))
        for role, clauses, problem in (
            ("target", target_clauses, target_problem),
            ("source", source_clauses, source_problem),
        ):
            arrays = labels[role]
            for clause_index, clause in enumerate(clauses):
                arrays["predicate_ids"][frame_index, clause_index] = int(
                    clause["predicate_id"]
                )
                arrays["clause_valid"][frame_index, clause_index] = True
                for entity_role in ("subject", "reference"):
                    entity = str(clause[entity_role])
                    arrays[f"{entity_role}_entity_ids"][
                        frame_index, clause_index
                    ] = _entity_id(entity)
                    mask = masks[entity]
                    arrays[f"{entity_role}_masks"][
                        frame_index, clause_index
                    ] = mask
                    arrays[f"{entity_role}_mask_valid"][
                        frame_index, clause_index
                    ] = bool(mask.any())
                    view_visible, view_centers = _per_view_mask_geometry(mask)
                    arrays[f"{entity_role}_view_visible"][
                        frame_index, clause_index
                    ] = view_visible
                    arrays[f"{entity_role}_view_centers"][
                        frame_index, clause_index
                    ] = view_centers
                    position, valid = _body_position(env, entity)
                    arrays[f"{entity_role}_position_valid"][
                        frame_index, clause_index
                    ] = valid
                    if valid:
                        arrays[f"{entity_role}_positions"][
                            frame_index, clause_index
                        ] = _normalize_position(
                            position, workspace_min, workspace_max
                        )
                    if entity_role == "subject":
                        subject_tracks.setdefault(
                            (role, clause_index), []
                        ).append(position.copy())
                anchor, anchor_valid = _region_anchor(env, clause, problem)
                arrays["goal_anchor_valid"][frame_index, clause_index] = anchor_valid
                if anchor_valid:
                    arrays["goal_anchors"][
                        frame_index, clause_index
                    ] = _normalize_position(
                        anchor, workspace_min, workspace_max
                    )
                arrays["predicate_truth"][frame_index, clause_index] = float(
                    _clause_truth(env, clause["raw"])
                )
                arrays["predicate_truth_valid"][frame_index, clause_index] = True
        obs, _, _, _ = env.step(np.asarray(action, dtype=np.float32).tolist())

    eef = np.stack(eef_track)
    gripper = np.asarray(gripper_track)
    for role, clauses in (("target", target_clauses), ("source", source_clauses)):
        arrays = labels[role]
        for clause_index, clause in enumerate(clauses):
            truth = arrays["predicate_truth"][:, clause_index] > 0.5
            truth_steps = np.flatnonzero(truth)
            track = np.stack(subject_tracks[(role, clause_index)])
            displacement = np.linalg.norm(track - track[0], axis=-1)
            moved = np.flatnonzero(displacement > 0.015)
            closed = np.flatnonzero(gripper > 0.0)
            if clause["predicate"] in {"open", "close", "turnon", "turnoff"} and truth_steps.size:
                interaction_step = int(truth_steps[0])
            elif moved.size:
                interaction_step = int(moved[0])
            elif closed.size:
                interaction_step = int(closed[0])
            else:
                interaction_step = 0
            interaction_anchor = _normalize_position(
                eef[interaction_step], workspace_min, workspace_max
            )
            arrays["interaction_anchors"][:, clause_index] = interaction_anchor
            arrays["interaction_anchor_valid"][:, clause_index] = True
            if clause["predicate"] in {"in", "on", "left", "right", "front", "back"}:
                grasp_step = int(moved[0]) if moved.size else interaction_step
                arrays["grasp_anchors"][:, clause_index] = _normalize_position(
                    eef[grasp_step], workspace_min, workspace_max
                )
                arrays["grasp_anchor_valid"][:, clause_index] = True
            completion_step = int(truth_steps[0]) if truth_steps.size else len(actions)
            arrays["phase_ids"][:, clause_index] = 0
            arrays["phase_ids"][interaction_step:completion_step, clause_index] = 1
            if completion_step < len(actions):
                arrays["phase_ids"][completion_step:, clause_index] = 2
            arrays["phase_valid"][:, clause_index] = True
            if role == "source" and not source_execution_supervision:
                # A direct counterfactual episode executes the target
                # instruction.  Its states still provide valid same-state
                # source entity/relation/truth and static goal-anchor labels,
                # but its EEF path cannot supervise how the source instruction
                # should be grasped or phased.
                arrays["grasp_anchor_valid"][:, clause_index] = False
                arrays["interaction_anchor_valid"][:, clause_index] = False
                arrays["phase_valid"][:, clause_index] = False
    flattened = {
        f"{role}_{name}": array
        for role, role_arrays in labels.items()
        for name, array in role_arrays.items()
    }
    return flattened, state_digest.hexdigest()


def _save_episode(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    records = _load_suite_manifest(args.manifest.expanduser().resolve(), args.suite)
    by_pair = {str(record["pair_id"]): record for record in records}
    by_instruction = {
        str(record["correct_instruction"]).strip().casefold(): record
        for record in records
    }
    provenance_path = dataset_root / PGC_PROVENANCE
    is_counterfactual = provenance_path.is_file()
    if not is_counterfactual and args.hdf5_root is None:
        raise ValueError("Native sidecar construction requires --hdf5-root.")
    audits = (
        sorted(
            read_jsonl(dataset_root / PGC_EPISODES),
            key=lambda item: int(item["episode_index"]),
        )
        if is_counterfactual
        else None
    )

    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)
    episode_count = len(audits) if audits is not None else int(dataset.num_episodes)
    if output_root.exists() and (output_root / "index.json").exists() and not args.overwrite:
        raise FileExistsError(
            f"V9 sidecar already exists: {output_root}; use --overwrite deliberately."
        )
    entries: list[dict[str, Any]] = []
    entity_vocabulary: dict[str, int] = {}
    used_native_demos: set[tuple[str, str]] = set()
    workspace_min = np.asarray(args.workspace_min, dtype=np.float32)
    workspace_max = np.asarray(args.workspace_max, dtype=np.float32)
    for episode_index in range(episode_count):
        actions = _episode_actions(dataset, episode_index)
        if audits is not None:
            audit = audits[episode_index]
            if int(audit["episode_index"]) != episode_index:
                raise ValueError("PGC episode audit indices must be dense.")
            pair_id = str(audit["pair_id"])
            record = by_pair[pair_id]
            initial_state_path = dataset_root / str(audit["source_initial_state_catalog"])
            initial_state = np.load(initial_state_path, allow_pickle=False)
            demo_source = str(initial_state_path)
            demo_group = "pgc_audited_state"
        else:
            instruction = _episode_instruction(dataset, episode_index)
            try:
                record = _native_record(by_instruction[instruction.casefold()])
            except KeyError as exc:
                raise KeyError(
                    f"Native instruction is absent from the suite manifest: {instruction!r}."
                ) from exc
            pair_id = str(record["pair_id"])
            initial_state, demo_source, demo_group = _match_native_demo(
                record=record,
                actions=actions,
                hdf5_root=args.hdf5_root.expanduser().resolve(),
                used=used_native_demos,
            )
        target_clauses, source_clauses, target_problem, source_problem = _record_roles(
            record, max_clauses=args.max_clauses
        )
        for clause in (*target_clauses, *source_clauses):
            for role in ("subject", "reference"):
                entity = str(clause[role])
                entity_vocabulary[entity] = _entity_id(entity)
        env = None
        try:
            env, _ = _make_source_env(
                record,
                resolution=512,
                seed=args.seed + episode_index,
                camera_segmentations="element",
            )
            if is_counterfactual:
                _set_counterfactual_goal(env, record)
            arrays, replay_state_sha = _replay_labels(
                env,
                initial_state=initial_state,
                actions=actions,
                target_clauses=target_clauses,
                source_clauses=source_clauses,
                target_problem=target_problem,
                source_problem=source_problem,
                state_atol=args.state_atol,
                settle_steps=args.settle_steps,
                height=args.mask_height,
                width=args.mask_width,
                max_clauses=args.max_clauses,
                workspace_min=workspace_min,
                workspace_max=workspace_max,
                source_execution_supervision=not is_counterfactual,
            )
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        relpath = Path("episodes") / f"episode_{episode_index:06d}.npz"
        episode_path = output_root / relpath
        _save_episode(episode_path, arrays)
        entries.append(
            {
                "episode_index": episode_index,
                "pair_id": pair_id,
                "file": str(relpath),
                "sha256": _sha256(episode_path),
                "state_sha256": replay_state_sha,
                "initial_state_sha256": state_sha256(initial_state),
                "action_sha256": array_sha256(actions),
                "frame_count": int(actions.shape[0]),
                "source_demo": demo_source,
                "source_demo_group": demo_group,
                "target_clauses": target_clauses,
                "source_clauses": source_clauses,
            }
        )
        atomic_write_json(
            output_root / PGC_ENTITY_RELATION_INDEX,
            {
                "format": PGC_ENTITY_RELATION_FORMAT,
                "benchmark": "libero",
                "suite": args.suite,
                "dataset": str(dataset_root),
                "dataset_kind": "counterfactual" if is_counterfactual else "native",
                "privileged_supervision": "training_only",
                "deployment_inputs": "rgb_language_proprio",
                "max_clauses": args.max_clauses,
                "predicate_vocabulary": [
                    "pad", "in", "on", "left", "right", "front", "back",
                    "open", "close", "turnon", "turnoff",
                ],
                "entity_id_scheme": "sha256_63bit",
                "entity_vocabulary": dict(sorted(entity_vocabulary.items())),
                "camera_names": list(CAMERA_NAMES),
                "view_center_coordinate_system": "per_camera_normalized_xy",
                "mask_size": [args.mask_height, args.mask_width],
                "workspace_min": workspace_min.tolist(),
                "workspace_max": workspace_max.tolist(),
                "episode_count": len(entries),
                "episodes": entries,
            },
        )
        print(
            f"SAVED ERAF episode={episode_index} pair={pair_id} "
            f"frames={len(actions)} progress={len(entries)}/{episode_count}",
            flush=True,
        )
    if len(entries) != episode_count:
        raise RuntimeError("PGC v9 entity-relation construction is incomplete.")
    print(output_root / PGC_ENTITY_RELATION_INDEX)


if __name__ == "__main__":
    main()
