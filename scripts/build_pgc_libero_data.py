#!/usr/bin/env python3
"""Build state-aligned successful PGC trajectories from LIBERO HDF5 demos.

For every audited source->counterfactual pair this program takes a successful
demonstration of the counterfactual task, restores that demonstration's exact
MuJoCo state in the *source* environment, installs the alternate success
predicate, and replays the actions.  The trajectory is written only if:

1. the source simulator reports the requested state was restored;
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

    from experiments.libero.language_interventions import (
        canonical_goal_state,
        validate_counterfactual_problem,
    )

    target_bddl = Path(str(record["counterfactual_bddl_file"]))
    if not target_bddl.is_file():
        raise FileNotFoundError(
            f"Counterfactual BDDL file does not exist: {target_bddl}"
        )
    target_problem = BDDLUtils.robosuite_parse_problem(str(target_bddl))
    inner = _get_inner_env(env)
    source_problem = inner.parsed_problem
    computed_goal = validate_counterfactual_problem(source_problem, target_problem)
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
            "Source simulator did not restore the donor initial state exactly: "
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


def _dataset_create_or_resume(args: argparse.Namespace, records: list[dict[str, Any]]):
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
    if provenance.get("pairs") != expected["pairs"]:
        raise ValueError("Cannot resume with a different intervention manifest.")
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
        "donor_demo_file": str(demo_path),
        "donor_demo_group": str(demo_group),
        "donor_demo_key": f"{demo_path.resolve()}::{demo_group}",
        "recorded_action_count": int(used_actions),
        "donor_raw_action_count": int(raw_action_count),
        "donor_filtered_action_count": int(filtered_action_count),
        "collection_method": "target_demo_replay_in_paired_source_environment",
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
    dataset, provenance = _dataset_create_or_resume(args, records)
    audits = _existing_audits(output)
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
        attempted = 0
        try:
            env, _ = _make_source_env(
                record,
                resolution=args.camera_resolution,
                seed=args.seed + pair_offset,
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
                    validation = _replay(
                        env,
                        demo.initial_state,
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
                        demo.initial_state,
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
                        initial_state=demo.initial_state,
                        used_actions=int(recording["actions"]),
                        raw_action_count=int(demo.actions.shape[0]),
                        filtered_action_count=int(replay_actions.shape[0]),
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
