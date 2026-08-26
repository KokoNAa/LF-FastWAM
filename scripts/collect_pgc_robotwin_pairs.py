#!/usr/bin/env python3
"""Collect same-scene left/right RoboTwin expert counterfactual trajectories.

The source task owns scene initialization.  The opposite directional expert is
executed in that *same source scene*, first for planning and then for RGB/qpos
recording.  This avoids the invalid shortcut of pairing independently sampled
``place_a2b_left`` and ``place_a2b_right`` datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.robotwin.pgc_data import (
    PGC_ROBOTWIN_PAIR_FORMAT,
    PGC_ROBOTWIN_RAW_FORMAT,
    array_sha256,
    direction_from_task,
    opposite_direction,
    scene_state_vector,
    validate_action_array,
    validate_pair_record,
)
from fastwam.datasets.pgc_libero import PGC_DATA_FORMAT


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _load_actions(path: Path) -> np.ndarray:
    candidates = (
        "joint_action/vector",
        "observation/joint_action/vector",
        "action",
        "actions",
    )
    with h5py.File(path, "r") as handle:
        for key in candidates:
            if key in handle:
                return validate_action_array(np.asarray(handle[key]))
        matches: list[str] = []

        def visit(name: str, value: Any) -> None:
            if isinstance(value, h5py.Dataset) and value.ndim == 2 and value.shape[-1] == 14:
                matches.append(name)

        handle.visititems(visit)
        if len(matches) != 1:
            raise ValueError(
                f"Expected one [T,14] qpos dataset in {path}, found {matches}."
            )
        return validate_action_array(np.asarray(handle[matches[0]]))


def _load_robotwin_args(
    *, robotwin_root: Path, task_name: str, task_config: str, output_root: Path
) -> tuple[Any, dict[str, Any]]:
    os.chdir(robotwin_root)
    if str(robotwin_root) not in sys.path:
        sys.path.insert(0, str(robotwin_root))
    from script.collect_data import class_decorator, get_embodiment_config
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    task = class_decorator(task_name)
    config_path = robotwin_root / "task_config" / f"{task_config}.yml"
    with config_path.open("r", encoding="utf-8") as handle:
        args = yaml.load(handle.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config
    embodiment = args["embodiment"]
    embodiment_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"
    with embodiment_path.open("r", encoding="utf-8") as handle:
        embodiment_types = yaml.load(handle.read(), Loader=yaml.FullLoader)

    def robot_file(name: str) -> str:
        value = embodiment_types[name]["file_path"]
        if value is None:
            raise ValueError(f"Embodiment {name!r} has no file_path.")
        return value

    if len(embodiment) == 1:
        args["left_robot_file"] = robot_file(embodiment[0])
        args["right_robot_file"] = robot_file(embodiment[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment[0])
    elif len(embodiment) == 3:
        args["left_robot_file"] = robot_file(embodiment[0])
        args["right_robot_file"] = robot_file(embodiment[1])
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment[0]}+{embodiment[1]}"
    else:
        raise ValueError(f"Invalid embodiment config: {embodiment!r}.")
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["save_path"] = str(output_root)
    return task, args


def _close(task: Any, *, clear_cache: bool = False) -> None:
    try:
        task.close_env(clear_cache=clear_cache)
    except TypeError:
        task.close_env()


def _capture_data_type(args: dict[str, Any]) -> dict[str, Any]:
    data_type = dict(args.get("data_type") or {})
    data_type.update(
        rgb=True,
        qpos=True,
        actor_segmentation_ids=True,
        pgc_entity_state=True,
    )
    return data_type


def _write_provenance(
    *,
    run_root: Path,
    pair_id: str,
    source_task: str,
    target_task: str,
    source_direction: str,
    target_direction: str,
    dataset_kind: str,
    episode_count: int,
) -> None:
    provenance = {
        "format": PGC_DATA_FORMAT,
        "raw_format": PGC_ROBOTWIN_RAW_FORMAT,
        "platform": "robotwin",
        "dataset_kind": dataset_kind,
        "state_aligned": True,
        "action_dim": 14,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "successful_episode_count": episode_count,
        "pairs": [
            {
                "pair_id": pair_id,
                "source_suite": "robotwin",
                "source_task_id": 0,
                "source_task": source_task,
                "counterfactual_task": target_task,
                "source_instruction": f"Place object A to the {source_direction} of object B.",
                "counterfactual_instruction": f"Place object A to the {target_direction} of object B.",
                "strict_conflict": True,
                "strict_conflict_type": "mutually_exclusive_spatial_direction",
            }
        ],
    }
    (run_root / "meta" / "pgc_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _replay_dataset(
    *,
    task: Any,
    base_args: dict[str, Any],
    run_root: Path,
    accepted: list[tuple[int, np.ndarray]],
    execution_direction: str,
    dataset_kind: str,
    pair_id: str,
    source_task: str,
    target_task: str,
    source_direction: str,
    target_direction: str,
) -> None:
    audit_path = run_root / "meta" / "pgc_episodes.jsonl"
    audit_path.write_text("", encoding="utf-8")
    replay_args = dict(base_args)
    replay_args.update(
        save_path=str(run_root),
        data_type=_capture_data_type(base_args),
        need_plan=False,
        save_data=True,
        render_freq=0,
    )
    for episode_index, (scene_seed, planned_state) in enumerate(accepted):
        task.setup_demo(now_ep_num=episode_index, seed=scene_seed, **replay_args)
        replay_state = scene_state_vector(task)
        if not np.array_equal(replay_state, planned_state):
            _close(task)
            raise RuntimeError(
                f"Scene replay drift for {pair_id}/{dataset_kind} "
                f"episode={episode_index}."
            )
        trajectory = task.load_tran_data(episode_index)
        replay_args["left_joint_path"] = trajectory["left_joint_path"]
        replay_args["right_joint_path"] = trajectory["right_joint_path"]
        task.set_path_lst(replay_args)
        info = task.play_once_direction(execution_direction)
        if not task.check_direction_success(execution_direction):
            _close(task)
            raise RuntimeError(
                f"{dataset_kind} replay failed for {pair_id} "
                f"episode={episode_index}."
            )
        _close(
            task,
            clear_cache=(
                (episode_index + 1) % int(base_args["clear_cache_freq"]) == 0
            ),
        )
        task.merge_pkl_to_hdf5_video()
        task.remove_data_cache()
        hdf5_path = run_root / "data" / f"episode{episode_index}.hdf5"
        actions = _load_actions(hdf5_path)
        state_relpath = (
            Path("meta") / "initial_states" / f"episode{episode_index}.npy"
        )
        record = validate_pair_record(
            {
                "format": PGC_ROBOTWIN_PAIR_FORMAT,
                "episode_index": episode_index,
                "pair_id": pair_id,
                "dataset_kind": dataset_kind,
                "source_task": source_task,
                "counterfactual_task": target_task,
                "source_direction": source_direction,
                "counterfactual_direction": target_direction,
                "executed_direction": execution_direction,
                "scene_seed": scene_seed,
                "initial_state_sha256": array_sha256(planned_state),
                "source_initial_state_catalog": state_relpath.as_posix(),
                "action_sha256": array_sha256(actions),
                "action_count": int(actions.shape[0]),
                "raw_hdf5": hdf5_path.relative_to(run_root).as_posix(),
                "source_instruction": f"Place object A to the {source_direction} of object B.",
                "counterfactual_instruction": f"Place object A to the {target_direction} of object B.",
                "scene_info": info,
            }
        )
        _append_jsonl(audit_path, record)
        print(
            f"[record] completed {pair_id}/{dataset_kind} "
            f"episode={episode_index} seed={scene_seed}",
            flush=True,
        )
    _write_provenance(
        run_root=run_root,
        pair_id=pair_id,
        source_task=source_task,
        target_task=target_task,
        source_direction=source_direction,
        target_direction=target_direction,
        dataset_kind=dataset_kind,
        episode_count=len(accepted),
    )


def collect_direction(
    *,
    robotwin_root: Path,
    output_root: Path,
    source_task: str,
    task_config: str,
    episodes: int,
    start_seed: int,
) -> Path:
    source_direction = direction_from_task(source_task)
    target_direction = opposite_direction(source_direction)
    target_task = f"place_a2b_{target_direction}"
    pair_id = f"{source_task}_to_{target_task}"
    pair_root = output_root / pair_id
    native_root = pair_root / "native"
    counterfactual_root = pair_root / "counterfactual"
    task, args = _load_robotwin_args(
        robotwin_root=robotwin_root,
        task_name=source_task,
        task_config=task_config,
        output_root=pair_root,
    )
    for run_root in (native_root, counterfactual_root):
        (run_root / "meta" / "initial_states").mkdir(parents=True, exist_ok=True)
    accepted: list[tuple[int, np.ndarray]] = []
    seed = int(start_seed)

    plan_args = dict(args)
    plan_args["data_type"] = _capture_data_type(args)
    plan_args.update(need_plan=True, save_data=False, render_freq=0)
    while len(accepted) < episodes:
        episode_index = len(accepted)
        try:
            target_plan_args = dict(plan_args, save_path=str(counterfactual_root))
            task.setup_demo(now_ep_num=episode_index, seed=seed, **target_plan_args)
            target_initial_state = scene_state_vector(task)
            task.play_once_direction(target_direction)
            target_ok = bool(
                task.plan_success
                and task.check_direction_success(target_direction)
            )
            if target_ok:
                task.save_traj_data(episode_index)
            _close(task)
            if not target_ok:
                print(f"[plan] rejected target {pair_id} seed={seed}", flush=True)
                seed += 1
                continue

            native_plan_args = dict(plan_args, save_path=str(native_root))
            task.setup_demo(now_ep_num=episode_index, seed=seed, **native_plan_args)
            native_initial_state = scene_state_vector(task)
            if not np.array_equal(native_initial_state, target_initial_state):
                raise RuntimeError(
                    f"Joint seed selection changed scene state for {pair_id} seed={seed}."
                )
            task.play_once_direction(source_direction)
            native_ok = bool(
                task.plan_success
                and task.check_direction_success(source_direction)
            )
            if native_ok:
                task.save_traj_data(episode_index)
            _close(task)
            if not native_ok:
                print(f"[plan] rejected native {pair_id} seed={seed}", flush=True)
                seed += 1
                continue

            for run_root in (native_root, counterfactual_root):
                state_path = (
                    run_root
                    / "meta"
                    / "initial_states"
                    / f"episode{episode_index}.npy"
                )
                np.save(state_path, target_initial_state, allow_pickle=False)
            accepted.append((seed, target_initial_state))
            print(
                f"[plan] jointly accepted {pair_id} episode={episode_index} "
                f"seed={seed}",
                flush=True,
            )
        except Exception:
            print(traceback.format_exc(), flush=True)
            _close(task)
        seed += 1
    _replay_dataset(
        task=task,
        base_args=args,
        run_root=native_root,
        accepted=accepted,
        execution_direction=source_direction,
        dataset_kind="native",
        pair_id=pair_id,
        source_task=source_task,
        target_task=target_task,
        source_direction=source_direction,
        target_direction=target_direction,
    )
    _replay_dataset(
        task=task,
        base_args=args,
        run_root=counterfactual_root,
        accepted=accepted,
        execution_direction=target_direction,
        dataset_kind="counterfactual",
        pair_id=pair_id,
        source_task=source_task,
        target_task=target_task,
        source_direction=source_direction,
        target_direction=target_direction,
    )
    return pair_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, default=PROJECT_ROOT / "third_party/RoboTwin")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=4300000)
    parser.add_argument(
        "--source-tasks",
        nargs="+",
        default=("place_a2b_left", "place_a2b_right"),
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.start_seed < 0:
        parser.error("--episodes must be positive and --start-seed non-negative")
    robotwin_root = args.robotwin_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for offset, source_task in enumerate(args.source_tasks):
        collect_direction(
            robotwin_root=robotwin_root,
            output_root=output_root,
            source_task=source_task,
            task_config=args.task_config,
            episodes=args.episodes,
            start_seed=args.start_seed + offset * 1_000_000,
        )


if __name__ == "__main__":
    main()
