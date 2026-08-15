#!/usr/bin/env python3
"""Build state-aligned successful PGC trajectories from LIBERO HDF5 demos.

For every audited source->counterfactual pair this program takes a successful
demonstration of the counterfactual task, transfers that demonstration into the
*source* environment, installs the requested success predicate, and replays the
actions.  Identical simulator layouts use exact flat-state restoration; layouts
with different distractor inventories use audited named-joint transfer.  The
trajectory is written only if:

1. the source simulator reports the transferred source state was restored;
2. replay satisfies the alternate predicate; and
3. a second recording pass is also successful.

This is action supervision, not instruction relabeling.  The output is a
FastWAM-readable LeRobot v2.1 directory with dataset- and episode-level PGC
audit metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from fastwam.datasets.pgc_libero import (  # noqa: E402
    append_jsonl,
    atomic_write_json,
    build_provenance,
    canonical_state_array,
    filter_libero_noops,
    iter_libero_hdf5_demos,
    libero_lerobot_features,
    read_jsonl,
    resolve_demo_file,
    state_sha256,
    states_match,
    validate_manifest_record,
)


LOGGER = logging.getLogger("pgc_libero_data")
PGC_PROVENANCE = Path("meta/pgc_provenance.json")
PGC_EPISODES = Path("meta/pgc_episodes.jsonl")
PGC_PENDING = Path("meta/pgc_pending_episode.json")
PGC_STATE_DIR = Path("meta/pgc_initial_states")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay successful target-task HDF5 demos in paired source "
            "environments and export direct PGC counterfactual supervision."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--demo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--episodes-per-pair", type=int, default=5)
    parser.add_argument("--max-demos-per-pair", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera-resolution", type=int, default=512)
    parser.add_argument(
        "--video-codec",
        choices=("h264", "hevc", "libsvtav1", "h264_nvenc"),
        default="h264",
    )
    parser.add_argument("--state-atol", type=float, default=1e-7)
    parser.add_argument("--min-actions", type=int, default=16)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=10,
        help="Dummy actions after restoring the HDF5 initial state.",
    )
    parser.add_argument(
        "--keep-noops",
        action="store_true",
        help="Do not apply LIBERO's standard no-op action filter.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Resolve demo files and print coverage without starting MuJoCo.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing output without duplicating donor demos.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Keep a dataset even if some manifest pairs lack enough successes.",
    )
    args = parser.parse_args()
    if args.episodes_per_pair <= 0:
        parser.error("--episodes-per-pair must be positive")
    if args.max_demos_per_pair < args.episodes_per_pair:
        parser.error("--max-demos-per-pair must be >= --episodes-per-pair")
    if args.fps <= 0 or args.camera_resolution <= 0:
        parser.error("--fps and --camera-resolution must be positive")
    if args.min_actions <= 0:
        parser.error("--min-actions must be positive")
    if args.settle_steps < 0:
        parser.error("--settle-steps must be non-negative")
    if args.state_atol < 0:
        parser.error("--state-atol must be non-negative")
    if not args.plan_only and args.output is None:
        parser.error("--output is required unless --plan-only is set")
    return args


def _load_suite_manifest(path: Path, suite_name: str) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    selected = [
        record
        for record in records
        if str(record.get("task_suite_name", "")) == suite_name
    ]
    if not selected:
        raise ValueError(f"No {suite_name!r} records found in {path}.")
    for record in selected:
        validate_manifest_record(record)
    pair_ids = [str(record["pair_id"]) for record in selected]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError(f"Manifest {path} contains duplicate pair IDs.")
    source_ids = [int(record["task_id"]) for record in selected]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(
            f"Manifest {path} contains multiple pairs for one source task."
        )
    return sorted(selected, key=lambda record: int(record["task_id"]))


def _plan_demo_files(
    records: list[dict[str, Any]], demo_root: Path
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    failures: list[str] = []
    for record in records:
        pair_id = str(record["pair_id"])
        try:
            demo_path = resolve_demo_file(demo_root, record)
            demo_count = sum(1 for _ in iter_libero_hdf5_demos(demo_path))
        except Exception as exc:
            failures.append(f"{pair_id}: {exc}")
            continue
        resolved[pair_id] = demo_path
        print(
            f"PLAN {pair_id}: target={record['counterfactual_instruction']!r} "
            f"demos={demo_count} file={demo_path}",
            flush=True,
        )
    if failures:
        raise ValueError(
            "Could not resolve raw demonstrations for every pair:\n  "
            + "\n  ".join(failures)
        )
    return resolved


def _get_inner_env(env: Any) -> Any:
    inner = getattr(env, "env", None)
    if inner is None:
        raise TypeError("Expected a LIBERO environment wrapper with `.env`.")
    return inner


def _sim_state(env: Any) -> np.ndarray:
    inner = _get_inner_env(env)
    sim = getattr(inner, "sim", None)
    if sim is None or not hasattr(sim, "get_state"):
        raise RuntimeError("LIBERO simulator does not expose sim.get_state().")
    state = sim.get_state()
    if hasattr(state, "flatten"):
        state = state.flatten()
    return np.asarray(state).copy()


def _set_counterfactual_goal(env: Any, record: Mapping[str, Any]) -> list[list[Any]]:
    from libero.libero.envs import bddl_utils as BDDLUtils

    from experiments.libero.language_interventions import canonical_goal_state

    target_bddl = Path(str(record["counterfactual_bddl_file"]))
    if not target_bddl.is_file():
        raise FileNotFoundError(
            f"Counterfactual BDDL file does not exist: {target_bddl}"
        )
    target_problem = BDDLUtils.robosuite_parse_problem(str(target_bddl))
    inner = _get_inner_env(env)
    source_problem = inner.parsed_problem
    source_problem_name = str(source_problem.get("problem_name", ""))
    target_problem_name = str(target_problem.get("problem_name", ""))
    if source_problem_name != target_problem_name:
        raise ValueError(
            "Counterfactual donor uses a different LIBERO environment class: "
            f"{source_problem_name!r} != {target_problem_name!r}."
        )
    computed_goal = target_problem.get("goal_state", [])
    if not isinstance(computed_goal, list) or not computed_goal:
        raise ValueError("Counterfactual donor has no non-empty goal_state.")
    computed_goal = [list(predicate) for predicate in computed_goal]
    goal_changed = canonical_goal_state(source_problem.get("goal_state", [])) != (
        canonical_goal_state(computed_goal)
    )
    declared_goal_changed = bool(record.get("counterfactual_goal_changed", True))
    if goal_changed != declared_goal_changed:
        raise ValueError(
            f"Manifest pair {record['pair_id']!r} counterfactual_goal_changed="
            f"{declared_goal_changed} but BDDL comparison gives {goal_changed}."
        )
    if not goal_changed and str(record.get("task_suite_name")) != "libero_spatial":
        raise ValueError(
            "An unchanged terminal goal is permitted only for LIBERO-Spatial, "
            "where the language distinction is the initial object placement."
        )
    declared_goal = record["counterfactual_goal_state"]
    if canonical_goal_state(computed_goal) != canonical_goal_state(declared_goal):
        raise ValueError(
            f"Manifest pair {record['pair_id']!r} goal does not match target BDDL."
        )
    runtime_entities = set(getattr(inner, "object_states_dict", {}))
    missing = sorted(
        {
            str(entity)
            for predicate in computed_goal
            for entity in predicate[1:]
        }
        - runtime_entities
    )
    if missing:
        raise ValueError(
            f"Pair {record['pair_id']!r} target entities are absent: {missing}."
        )
    inner.parsed_problem["goal_state"] = computed_goal
    return computed_goal


def _make_source_env(
    record: Mapping[str, Any], *, resolution: int, seed: int
) -> tuple[Any, Any]:
    from libero.libero import benchmark

    from experiments.libero.libero_utils import get_libero_env

    suite_name = str(record["task_suite_name"])
    task_id = int(record["task_id"])
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    if str(task.language).strip().casefold() != str(
        record["correct_instruction"]
    ).strip().casefold():
        raise ValueError(
            f"Source task text changed for {suite_name}/{task_id}: "
            f"{task.language!r} != {record['correct_instruction']!r}."
        )
    env, _ = get_libero_env(task, resolution, seed, env_num=1)
    _set_counterfactual_goal(env, record)
    return env, task


def _make_target_env(
    record: Mapping[str, Any], *, resolution: int, seed: int
) -> tuple[Any, Any]:
    from libero.libero import benchmark

    from experiments.libero.libero_utils import get_libero_env

    suite_name = str(record["counterfactual_task_suite_name"])
    task_id = int(record["counterfactual_task_id"])
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    expected = str(record["counterfactual_instruction"]).strip().casefold()
    if str(task.language).strip().casefold() != expected:
        raise ValueError(
            f"Target task text changed for {suite_name}/{task_id}: "
            f"{task.language!r} != {record['counterfactual_instruction']!r}."
        )
    env, _ = get_libero_env(task, resolution, seed, env_num=1)
    return env, task


def _joint_names(sim: Any) -> list[str]:
    model = sim.model
    names = getattr(model, "joint_names", None)
    if names is not None:
        return [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in names
            if name is not None
        ]
    resolved = []
    for joint_id in range(int(model.njnt)):
        name = None
        method = getattr(model, "joint_id2name", None)
        if callable(method):
            name = method(joint_id)
        if name is None:
            method = getattr(model, "id2name", None)
            if callable(method):
                try:
                    name = method(joint_id, "joint")
                except TypeError:
                    name = None
        if name is None:
            try:
                import mujoco

                name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                )
            except Exception:
                name = None
        if name:
            resolved.append(str(name))
    if not resolved:
        raise RuntimeError("MuJoCo model exposes no named joints for state transfer.")
    return resolved


def _problem_object_names(problem: Mapping[str, Any]) -> set[str]:
    mapping = problem.get("objects", {})
    if not isinstance(mapping, Mapping):
        return set()
    return {
        str(value)
        for values in mapping.values()
        for value in (values if isinstance(values, (list, tuple, set)) else [values])
    }


def _copy_joint_value(
    *, source_data: Any, target_data: Any, joint_name: str, quantity: str
) -> None:
    getter = getattr(target_data, f"get_joint_{quantity}")
    setter = getattr(source_data, f"set_joint_{quantity}")
    value = np.asarray(getter(joint_name)).copy()
    setter(joint_name, value.item() if value.ndim == 0 else value)


def _named_joint_transfer_state(
    source_env: Any,
    target_env: Any,
    donor_state: np.ndarray,
    record: Mapping[str, Any],
    *,
    state_atol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a valid source flat state by copying shared donor joints by name."""
    target_env.reset()
    target_env.set_init_state(np.asarray(donor_state))
    actual_target_state = _sim_state(target_env)
    if not states_match(donor_state, actual_target_state, atol=state_atol):
        raise ValueError(
            "Target simulator could not restore its own HDF5 donor state exactly."
        )

    source_env.reset()
    source_inner = _get_inner_env(source_env)
    target_inner = _get_inner_env(target_env)
    source_sim = source_inner.sim
    target_sim = target_inner.sim
    source_joints = set(_joint_names(source_sim))
    target_joints = set(_joint_names(target_sim))
    shared_joints = sorted(source_joints & target_joints)
    if not shared_joints:
        raise ValueError("Source and target environments have no shared named joints.")

    goal_entities = {
        str(entity)
        for predicate in record["counterfactual_goal_state"]
        for entity in predicate[1:]
    }
    source_objects = _problem_object_names(source_inner.parsed_problem)
    goal_objects = sorted(goal_entities & source_objects)
    missing_goal_object_joints = [
        entity
        for entity in goal_objects
        if not any(
            joint == entity or joint.startswith(f"{entity}_")
            for joint in shared_joints
        )
    ]
    if missing_goal_object_joints:
        raise ValueError(
            "Named-joint transfer cannot locate goal objects in both models: "
            f"{missing_goal_object_joints}."
        )

    for joint_name in shared_joints:
        _copy_joint_value(
            source_data=source_sim.data,
            target_data=target_sim.data,
            joint_name=joint_name,
            quantity="qpos",
        )
        try:
            _copy_joint_value(
                source_data=source_sim.data,
                target_data=target_sim.data,
                joint_name=joint_name,
                quantity="qvel",
            )
        except (AttributeError, KeyError, ValueError):
            # Some fixed or compatibility-wrapper joints expose qpos only.
            pass
    source_sim.forward()
    source_inner._post_process()
    source_inner._update_observables(force=True)
    transferred_state = _sim_state(source_env)
    source_env.set_init_state(transferred_state)
    if not states_match(transferred_state, _sim_state(source_env), atol=state_atol):
        raise ValueError("Named-joint transferred source state is not round-trip stable.")
    return transferred_state, {
        "state_transfer_mode": "named_joint_remap",
        "donor_initial_state_sha256": state_sha256(donor_state),
        "transferred_initial_state_sha256": state_sha256(transferred_state),
        "shared_joint_count": len(shared_joints),
        "goal_object_joint_count": len(goal_objects),
    }


def _prepare_source_initial_state(
    source_env: Any,
    target_env: Any | None,
    donor_state: np.ndarray,
    record: Mapping[str, Any],
    *,
    state_atol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(record.get("state_transfer_mode", "flat_exact"))
    if mode == "flat_exact":
        return np.asarray(donor_state).copy(), {
            "state_transfer_mode": mode,
            "donor_initial_state_sha256": state_sha256(donor_state),
            "transferred_initial_state_sha256": state_sha256(donor_state),
        }
    if mode == "named_joint_remap":
        if target_env is None:
            raise ValueError("named_joint_remap requires a target environment.")
        return _named_joint_transfer_state(
            source_env,
            target_env,
            donor_state,
            record,
            state_atol=state_atol,
        )
    raise ValueError(f"Unsupported state_transfer_mode={mode!r}.")


def _goal_satisfied(env: Any, *, done: bool) -> bool:
    inner = _get_inner_env(env)
    for owner in (inner, env):
        for method_name in ("_check_success", "check_success"):
            method = getattr(owner, method_name, None)
            if callable(method):
                result = method()
                if isinstance(result, Mapping):
                    result = all(bool(value) for value in result.values())
                return bool(result)
    # Older wrappers expose only `done`; current LIBERO exposes _check_success.
    return bool(done)


def _frame_from_obs(obs: Mapping[str, Any], action: np.ndarray) -> dict[str, np.ndarray]:
    from experiments.libero.libero_utils import get_libero_image, quat2axisangle

    images = get_libero_image(obs)
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return {
        "observation.images.image": np.asarray(images["image"], dtype=np.uint8),
        "observation.images.wrist_image": np.asarray(
            images["wrist_image"], dtype=np.uint8
        ),
        "observation.state": state,
        "action": np.asarray(action, dtype=np.float32),
    }


def _reset_exact_state(
    env: Any,
    initial_state: np.ndarray,
    *,
    state_atol: float,
    settle_steps: int,
) -> tuple[Mapping[str, Any], np.ndarray]:
    env.reset()
    obs = env.set_init_state(np.asarray(initial_state))
    actual_state = _sim_state(env)
    if not states_match(initial_state, actual_state, atol=state_atol):
        requested = canonical_state_array(initial_state)
        actual = canonical_state_array(actual_state)
        max_error = None
        if requested.shape == actual.shape:
            max_error = float(np.max(np.abs(requested - actual)))
        raise ValueError(
            "Source simulator did not restore the prepared initial state exactly: "
            f"requested_shape={requested.shape} actual_shape={actual.shape} "
            f"max_abs_error={max_error}."
        )
    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    for _ in range(int(settle_steps)):
        obs, _, _, _ = env.step(dummy_action)
    return obs, actual_state


def _replay(
    env: Any,
    initial_state: np.ndarray,
    actions: np.ndarray,
    *,
    state_atol: float,
    settle_steps: int,
    dataset: Any | None = None,
    instruction: str | None = None,
) -> dict[str, Any]:
    obs, actual_state = _reset_exact_state(
        env,
        initial_state,
        state_atol=state_atol,
        settle_steps=settle_steps,
    )
    if _goal_satisfied(env, done=False):
        return {
            "success": False,
            "initial_goal_satisfied": True,
            "actions": 0,
            "actual_initial_state": actual_state,
        }
    done = False
    used_actions = 0
    for action in actions:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"LIBERO action must be shape (7,), got {action.shape}.")
        if dataset is not None:
            if instruction is None:
                raise ValueError("Recording requires the executed instruction.")
            dataset.add_frame(
                _frame_from_obs(obs, action),
                task=[instruction, instruction, instruction, instruction],
            )
        obs, _, done, _ = env.step(action.tolist())
        used_actions += 1
        if _goal_satisfied(env, done=bool(done)):
            done = True
            break
    return {
        "success": bool(done),
        "initial_goal_satisfied": False,
        "actions": int(used_actions),
        "actual_initial_state": actual_state,
    }


def _discard_episode(dataset: Any) -> None:
    episode_index = int(dataset.episode_buffer["episode_index"])
    for key in dataset.meta.video_keys:
        image_dir = dataset._get_image_file_path(episode_index, key, 0).parent
        if image_dir.is_dir():
            shutil.rmtree(image_dir)
    dataset.episode_buffer = dataset.create_episode_buffer()


def _pairs_by_source(
    pairs: list[dict[str, Any]], *, label: str
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError(f"{label} contains a non-object pair record.")
        source_key = (
            str(pair.get("source_suite", "")),
            int(pair.get("source_task_id", -1)),
        )
        if source_key in indexed:
            raise ValueError(f"{label} duplicates source task {source_key}.")
        indexed[source_key] = pair
    return indexed


def _merge_resume_provenance_pairs(
    provenance: Mapping[str, Any],
    expected: Mapping[str, Any],
    episode_audits: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Safely replace donor pairs that have never produced an episode.

    Successful pair definitions are immutable because their audit records and
    language/action data depend on them. A pair that exhausted every donor demo
    with zero successes may be replaced by a later ranked candidate without
    discarding unrelated successful episodes.
    """
    old_pairs = list(provenance.get("pairs") or [])
    new_pairs = list(expected.get("pairs") or [])
    old_by_source = _pairs_by_source(old_pairs, label="saved provenance")
    new_by_source = _pairs_by_source(new_pairs, label="new manifest")
    if set(old_by_source) != set(new_by_source):
        raise ValueError(
            "Cannot resume with a different set of source tasks: "
            f"saved={sorted(old_by_source)}, new={sorted(new_by_source)}."
        )

    known_old_pair_ids = {
        str(pair.get("pair_id", "")) for pair in old_pairs
    }
    used_pair_ids = {
        str(audit.get("pair_id", "")) for audit in episode_audits
    }
    unknown_audits = sorted(used_pair_ids - known_old_pair_ids)
    if unknown_audits:
        raise ValueError(
            "PGC episode audits reference pairs absent from saved provenance: "
            f"{unknown_audits}."
        )

    replacements: list[dict[str, Any]] = []
    for source_key in sorted(old_by_source):
        old_pair = old_by_source[source_key]
        new_pair = new_by_source[source_key]
        if old_pair == new_pair:
            continue
        old_pair_id = str(old_pair.get("pair_id", ""))
        if old_pair_id in used_pair_ids:
            raise ValueError(
                "Cannot replace a PGC donor pair that already produced "
                f"successful episodes: source={source_key}, pair={old_pair_id}."
            )
        replacements.append(
            {
                "source_suite": source_key[0],
                "source_task_id": source_key[1],
                "old_pair_id": old_pair_id,
                "new_pair_id": str(new_pair.get("pair_id", "")),
            }
        )

    merged = dict(provenance)
    merged["pairs"] = new_pairs
    merged["source_suites"] = list(expected.get("source_suites") or [])
    if replacements:
        history = list(merged.get("unproductive_pair_replacements") or [])
        history.extend(replacements)
        merged["unproductive_pair_replacements"] = history
    return merged, replacements


def _dataset_create_or_resume(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    *,
    episode_audits: list[dict[str, Any]] | None = None,
):
    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

    output = args.output.expanduser().resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"Output already exists: {output}. Use --resume or choose a new path."
        )
    if not output.exists():
        dataset = LeRobotDataset.create(
            repo_id=output.name,
            root=output,
            fps=args.fps,
            robot_type="panda",
            features=libero_lerobot_features(args.camera_resolution),
            use_videos=True,
            video_codec=args.video_codec,
            is_compute_episode_stats_image=False,
        )
        provenance = build_provenance(records, successful_episode_count=0)
        provenance["state_match_tolerance"] = float(args.state_atol)
        provenance["settle_steps"] = int(args.settle_steps)
        provenance["noop_filter_enabled"] = not bool(args.keep_noops)
        provenance["fps"] = int(args.fps)
        provenance["camera_resolution"] = int(args.camera_resolution)
        atomic_write_json(output / PGC_PROVENANCE, provenance)
        return dataset, provenance

    provenance_path = output / PGC_PROVENANCE
    if not provenance_path.is_file():
        raise ValueError(f"Cannot resume: missing {provenance_path}.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = build_provenance(
        records,
        successful_episode_count=int(provenance.get("successful_episode_count", 0)),
    )
    provenance, replacements = _merge_resume_provenance_pairs(
        provenance,
        expected,
        list(episode_audits or []),
    )
    resume_contract = {
        "state_match_tolerance": float(args.state_atol),
        "settle_steps": int(args.settle_steps),
        "noop_filter_enabled": not bool(args.keep_noops),
        "fps": int(args.fps),
        "camera_resolution": int(args.camera_resolution),
    }
    mismatched = {
        key: (provenance.get(key), value)
        for key, value in resume_contract.items()
        if provenance.get(key) != value
    }
    if mismatched:
        raise ValueError(
            f"Cannot resume with different collection settings: {mismatched}."
        )
    dataset = LeRobotDataset(repo_id=output.name, root=output)
    dataset.image_writer = None
    dataset.video_codec = args.video_codec
    dataset.is_compute_episode_stats_image = False
    dataset.episode_buffer = dataset.create_episode_buffer()
    if replacements:
        atomic_write_json(provenance_path, provenance)
        for replacement in replacements:
            LOGGER.warning(
                "Replaced unproductive PGC pair for %s/%d: %s -> %s.",
                replacement["source_suite"],
                replacement["source_task_id"],
                replacement["old_pair_id"],
                replacement["new_pair_id"],
            )
    return dataset, provenance


def _existing_audits(output: Path) -> list[dict[str, Any]]:
    path = output / PGC_EPISODES
    return read_jsonl(path) if path.is_file() else []


def _episode_files_complete(output: Path, episode_index: int) -> bool:
    info_path = output / "meta/info.json"
    if not info_path.is_file():
        return False
    info = json.loads(info_path.read_text(encoding="utf-8"))
    chunk = episode_index // int(info.get("chunks_size", 1000))
    data_path = output / str(info["data_path"]).format(
        episode_chunk=chunk, episode_index=episode_index
    )
    if not data_path.is_file():
        return False
    for key, feature in info.get("features", {}).items():
        if feature.get("dtype") != "video":
            continue
        video_path = output / str(info["video_path"]).format(
            episode_chunk=chunk,
            episode_index=episode_index,
            video_key=key,
        )
        if not video_path.is_file():
            return False
    return True


def _recover_pending(output: Path) -> None:
    pending_path = output / PGC_PENDING
    if not pending_path.is_file():
        return
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    episode_index = int(pending["episode_index"])
    audits = _existing_audits(output)
    already_audited = any(int(item["episode_index"]) == episode_index for item in audits)
    if already_audited:
        provenance_path = output / PGC_PROVENANCE
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["successful_episode_count"] = len(audits)
        atomic_write_json(provenance_path, provenance)
        pending_path.unlink()
        return
    if not _episode_files_complete(output, episode_index):
        raise RuntimeError(
            f"Found an incomplete pending episode {episode_index} in {output}. "
            "Preserve the directory for diagnosis and restart with a new output path."
        )
    append_jsonl(output / PGC_EPISODES, pending)
    provenance_path = output / PGC_PROVENANCE
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["successful_episode_count"] = len(audits) + 1
    atomic_write_json(provenance_path, provenance)
    pending_path.unlink()
    LOGGER.warning("Recovered committed pending episode %d.", episode_index)


def _save_successful_episode(
    dataset: Any,
    provenance: dict[str, Any],
    *,
    args: argparse.Namespace,
    record: Mapping[str, Any],
    demo_path: Path,
    demo_group: str,
    initial_state: np.ndarray,
    used_actions: int,
    raw_action_count: int,
    filtered_action_count: int,
    transfer_audit: Mapping[str, Any],
) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    episode_index = int(dataset.meta.total_episodes)
    state_relpath = PGC_STATE_DIR / f"episode_{episode_index:06d}.npy"
    state_path = output / state_relpath
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(state_path, canonical_state_array(initial_state), allow_pickle=False)
    audit = {
        "episode_index": episode_index,
        "pair_id": str(record["pair_id"]),
        "source_suite": str(record["task_suite_name"]),
        "source_task_id": int(record["task_id"]),
        "source_initial_state_index": episode_index,
        "source_initial_state_catalog": str(state_relpath),
        "initial_state_sha256": state_sha256(initial_state),
        "initial_state_match": True,
        "counterfactual_goal_satisfied": True,
        "counterfactual_instruction": str(
            record["counterfactual_instruction"]
        ).strip(),
        "counterfactual_goal_state": record["counterfactual_goal_state"],
        "counterfactual_goal_changed": bool(
            record.get("counterfactual_goal_changed", True)
        ),
        "donor_demo_file": str(demo_path),
        "donor_demo_group": str(demo_group),
        "donor_demo_key": f"{demo_path.resolve()}::{demo_group}",
        "recorded_action_count": int(used_actions),
        "donor_raw_action_count": int(raw_action_count),
        "donor_filtered_action_count": int(filtered_action_count),
        "collection_method": (
            "audited_target_demo_replay_with_exact_or_named_joint_state_transfer"
        ),
        **dict(transfer_audit),
    }
    atomic_write_json(output / PGC_PENDING, audit)
    dataset.save_episode(raw_file_name=audit["donor_demo_key"])
    append_jsonl(output / PGC_EPISODES, audit)
    provenance["successful_episode_count"] = int(dataset.meta.total_episodes)
    atomic_write_json(output / PGC_PROVENANCE, provenance)
    (output / PGC_PENDING).unlink()
    return audit


def _collect(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    demo_paths: dict[str, Path],
) -> None:
    output = args.output.expanduser().resolve()
    if output.exists() and args.resume:
        _recover_pending(output)
    audits = _existing_audits(output)
    dataset, provenance = _dataset_create_or_resume(
        args,
        records,
        episode_audits=audits,
    )
    if int(dataset.meta.total_episodes) != len(audits):
        raise RuntimeError(
            "LeRobot episode count and PGC audit count differ: "
            f"episodes={dataset.meta.total_episodes} audits={len(audits)}."
        )
    if int(provenance.get("successful_episode_count", -1)) != len(audits):
        provenance["successful_episode_count"] = len(audits)
        atomic_write_json(output / PGC_PROVENANCE, provenance)
    counts = Counter(str(item["pair_id"]) for item in audits)
    used_demos = {
        (str(item["pair_id"]), str(item.get("donor_demo_key", "")))
        for item in audits
    }
    failures: dict[str, list[str]] = {str(record["pair_id"]): [] for record in records}

    for pair_offset, record in enumerate(records):
        pair_id = str(record["pair_id"])
        required = int(args.episodes_per_pair)
        if counts[pair_id] >= required:
            LOGGER.info("%s already has %d/%d episodes.", pair_id, counts[pair_id], required)
            continue
        demo_path = demo_paths[pair_id]
        env = None
        target_env = None
        attempted = 0
        try:
            env, _ = _make_source_env(
                record,
                resolution=args.camera_resolution,
                seed=args.seed + pair_offset,
            )
            if str(record.get("state_transfer_mode", "flat_exact")) == (
                "named_joint_remap"
            ):
                target_env, _ = _make_target_env(
                    record,
                    # Target is used only to decode named simulator joints;
                    # small camera buffers avoid doubling collector EGL memory.
                    resolution=min(args.camera_resolution, 64),
                    seed=args.seed + pair_offset + 100_000,
                )
            for demo in iter_libero_hdf5_demos(demo_path):
                donor_key = f"{demo_path.resolve()}::{demo.group_name}"
                if (pair_id, donor_key) in used_demos:
                    continue
                if attempted >= int(args.max_demos_per_pair):
                    break
                attempted += 1
                try:
                    replay_actions = (
                        demo.actions
                        if args.keep_noops
                        else filter_libero_noops(demo.actions)
                    )
                    source_initial_state, transfer_audit = (
                        _prepare_source_initial_state(
                            env,
                            target_env,
                            demo.initial_state,
                            record,
                            state_atol=args.state_atol,
                        )
                    )
                    validation = _replay(
                        env,
                        source_initial_state,
                        replay_actions,
                        state_atol=args.state_atol,
                        settle_steps=args.settle_steps,
                    )
                    if not validation["success"]:
                        reason = (
                            "target already satisfied before donor actions"
                            if validation["initial_goal_satisfied"]
                            else "target not satisfied"
                        )
                        failures[pair_id].append(f"{demo.group_name}: {reason}")
                        continue
                    if int(validation["actions"]) < int(args.min_actions):
                        failures[pair_id].append(
                            f"{demo.group_name}: only {validation['actions']} actions"
                        )
                        continue

                    instruction = str(record["counterfactual_instruction"]).strip()
                    recording = _replay(
                        env,
                        source_initial_state,
                        replay_actions,
                        state_atol=args.state_atol,
                        settle_steps=args.settle_steps,
                        dataset=dataset,
                        instruction=instruction,
                    )
                    if not recording["success"]:
                        _discard_episode(dataset)
                        failures[pair_id].append(
                            f"{demo.group_name}: nondeterministic recording replay"
                        )
                        continue
                    audit = _save_successful_episode(
                        dataset,
                        provenance,
                        args=args,
                        record=record,
                        demo_path=demo_path,
                        demo_group=demo.group_name,
                        initial_state=source_initial_state,
                        used_actions=int(recording["actions"]),
                        raw_action_count=int(demo.actions.shape[0]),
                        filtered_action_count=int(replay_actions.shape[0]),
                        transfer_audit=transfer_audit,
                    )
                    counts[pair_id] += 1
                    used_demos.add((pair_id, donor_key))
                    print(
                        f"SAVED episode={audit['episode_index']} pair={pair_id} "
                        f"actions={audit['recorded_action_count']} "
                        f"progress={counts[pair_id]}/{required}",
                        flush=True,
                    )
                    if counts[pair_id] >= required:
                        break
                except Exception as exc:
                    if dataset.episode_buffer["size"]:
                        _discard_episode(dataset)
                    failures[pair_id].append(
                        f"{demo.group_name}: {type(exc).__name__}: {exc}"
                    )
                    LOGGER.warning(
                        "Rejected %s/%s: %s\n%s",
                        pair_id,
                        demo.group_name,
                        exc,
                        traceback.format_exc(),
                    )
        finally:
            if target_env is not None:
                close = getattr(target_env, "close", None)
                if callable(close):
                    close()
            if env is not None:
                close = getattr(env, "close", None)
                if callable(close):
                    close()

    missing = {
        pair_id: int(args.episodes_per_pair) - counts[pair_id]
        for pair_id in (str(record["pair_id"]) for record in records)
        if counts[pair_id] < int(args.episodes_per_pair)
    }
    summary = {
        "output": str(output),
        "successful_episodes": int(sum(counts.values())),
        "episodes_per_pair": dict(sorted(counts.items())),
        "missing_episodes_per_pair": missing,
        "recent_rejections": {
            pair_id: reasons[-5:]
            for pair_id, reasons in failures.items()
            if reasons
        },
    }
    atomic_write_json(output / "meta/pgc_collection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if missing and not args.allow_partial:
        raise RuntimeError(
            "PGC collection is incomplete. Inspect meta/pgc_collection_summary.json; "
            "rerun with --resume after adding compatible demos."
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    records = _load_suite_manifest(args.manifest, args.suite)
    demo_paths = _plan_demo_files(records, args.demo_root)
    if args.plan_only:
        print(
            f"PGC plan passed: suite={args.suite} pairs={len(records)} "
            f"demo_files={len(set(demo_paths.values()))}",
            flush=True,
        )
        return
    _collect(args, records, demo_paths)


if __name__ == "__main__":
    main()
