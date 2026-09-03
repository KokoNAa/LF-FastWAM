#!/usr/bin/env python3
"""Collect matched RoboTwin expert trajectories with ERAF supervision.

The source task owns scene initialization.  Native and strict-counterfactual
experts are planned and recorded in that exact source scene.  These captures
are grounding supervision, not source-directed failure replans and therefore
must never be labelled as final full-goal corrective data.
"""

from __future__ import annotations

import argparse
import hashlib
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
    ROBOTWIN_ERAF_PAIR_SPECS,
    RoboTwinPairSpec,
    array_sha256,
    pair_spec_from_source_task,
    scene_state_vector,
    validate_action_array,
    validate_pair_record,
)
from experiments.robotwin.pgc_task_variants import (
    check_variant,
    install_pgc_task_contract,
    play_variant,
)
from fastwam.datasets.pgc_libero import PGC_DATA_FORMAT


COLLECTION_PROFILES = {
    "grounding": {
        "native": {
            "artifact_role": "eraf_grounding_supervision",
            "allowed_training_stages": ["grounding"],
            "forbidden_training_stages": [
                "no_eraf",
                "joint",
                "final_short_lora",
            ],
        },
        "counterfactual": {
            "artifact_role": "eraf_grounding_supervision",
            "allowed_training_stages": ["grounding"],
            "forbidden_training_stages": [
                "no_eraf",
                "joint",
                "final_short_lora",
            ],
        },
    },
    "no_eraf_historical": {
        "native": {
            "artifact_role": "offline_native",
            "allowed_training_stages": [
                "no_eraf",
                "joint",
                "final_short_lora",
            ],
            "forbidden_training_stages": ["grounding"],
        },
        "counterfactual": {
            "artifact_role": "historical_cf",
            "allowed_training_stages": [
                "no_eraf",
                "joint",
                "final_short_lora",
            ],
            "forbidden_training_stages": ["grounding"],
        },
    },
    "no_eraf_strict": {
        # The source replay is retained only as the same-scene strict audit.
        # It is not added to the optimizer's offline-native pool.
        "native": {
            "artifact_role": "strict_source_native_audit",
            "allowed_training_stages": [],
            "forbidden_training_stages": [
                "grounding",
                "no_eraf",
                "joint",
                "final_short_lora",
            ],
        },
        "counterfactual": {
            "artifact_role": "strict_cf",
            "allowed_training_stages": [
                "no_eraf",
                "joint",
                "final_short_lora",
            ],
            "forbidden_training_stages": ["grounding"],
        },
    },
}


def collection_contract(profile: str, dataset_kind: str) -> dict[str, Any]:
    try:
        return dict(COLLECTION_PROFILES[str(profile)][str(dataset_kind)])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported RoboTwin collection profile/kind: "
            f"{profile!r}/{dataset_kind!r}."
        ) from exc


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
            if (
                isinstance(value, h5py.Dataset)
                and value.ndim == 2
                and value.shape[-1] == 14
            ):
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
    spec: RoboTwinPairSpec,
    dataset_kind: str,
    episode_count: int,
    task_config: str,
    collection_profile: str,
) -> None:
    contract = collection_contract(collection_profile, dataset_kind)
    provenance = {
        "format": PGC_DATA_FORMAT,
        "raw_format": PGC_ROBOTWIN_RAW_FORMAT,
        "platform": "robotwin",
        "artifact_role": contract["artifact_role"],
        "collection_profile": collection_profile,
        "allowed_training_stages": contract["allowed_training_stages"],
        "forbidden_training_stages": contract["forbidden_training_stages"],
        "full_goal_usage": "not_present",
        "dataset_kind": dataset_kind,
        "task_config": task_config,
        "state_aligned": True,
        "action_dim": 14,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "successful_episode_count": episode_count,
        "goal_verification_contract": "same_scene_expert_goal_replay",
        "full_goal_verified": False,
        "pairs": [
            {
                "pair_id": spec.pair_id,
                "source_suite": "robotwin",
                "source_task_id": 0,
                "source_task": spec.source_task,
                "counterfactual_task": spec.counterfactual_task,
                "source_variant": spec.source_variant,
                "counterfactual_variant": spec.counterfactual_variant,
                "source_instruction": spec.source_instruction,
                "counterfactual_instruction": spec.counterfactual_instruction,
                "strict_conflict": True,
                "strict_conflict_type": spec.strict_conflict_type,
                "entity_names": list(spec.entity_names),
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
    execution_variant: str,
    dataset_kind: str,
    spec: RoboTwinPairSpec,
    task_config: str,
    collection_profile: str,
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
        task._pgc_active_variant = None
        task.setup_demo(now_ep_num=episode_index, seed=scene_seed, **replay_args)
        replay_state = scene_state_vector(task)
        if not np.array_equal(replay_state, planned_state):
            _close(task)
            raise RuntimeError(
                f"Scene replay drift for {spec.pair_id}/{dataset_kind} "
                f"episode={episode_index}."
            )
        trajectory = task.load_tran_data(episode_index)
        replay_args["left_joint_path"] = trajectory["left_joint_path"]
        replay_args["right_joint_path"] = trajectory["right_joint_path"]
        task.set_path_lst(replay_args)
        info = play_variant(task, spec, execution_variant)
        if not check_variant(task, spec, execution_variant):
            _close(task)
            raise RuntimeError(
                f"{dataset_kind} replay failed for {spec.pair_id} "
                f"episode={episode_index}."
            )
        _close(
            task,
            clear_cache=((episode_index + 1) % int(base_args["clear_cache_freq"]) == 0),
        )
        task.merge_pkl_to_hdf5_video()
        task.remove_data_cache()
        hdf5_path = run_root / "data" / f"episode{episode_index}.hdf5"
        actions = _load_actions(hdf5_path)
        state_relpath = Path("meta") / "initial_states" / f"episode{episode_index}.npy"
        state_digest = array_sha256(planned_state)
        action_digest = array_sha256(actions)
        verification_kind = (
            "source_goal" if dataset_kind == "native" else "counterfactual_goal"
        )
        capture_id = hashlib.sha256(
            (
                f"{spec.pair_id}\0{dataset_kind}\0{scene_seed}\0"
                f"{state_digest}\0{action_digest}"
            ).encode("utf-8")
        ).hexdigest()
        record = validate_pair_record(
            {
                "format": PGC_ROBOTWIN_PAIR_FORMAT,
                "episode_index": episode_index,
                "pair_id": spec.pair_id,
                "dataset_kind": dataset_kind,
                "source_task": spec.source_task,
                "counterfactual_task": spec.counterfactual_task,
                "source_variant": spec.source_variant,
                "counterfactual_variant": spec.counterfactual_variant,
                "executed_variant": execution_variant,
                "scene_seed": scene_seed,
                "initial_state_sha256": state_digest,
                "source_initial_state_catalog": state_relpath.as_posix(),
                "action_sha256": action_digest,
                "action_count": int(actions.shape[0]),
                "capture_id": capture_id,
                "capture_state_sha256": state_digest,
                "recorded_action_count": int(actions.shape[0]),
                "verification_kind": verification_kind,
                "verification_step": int(actions.shape[0]),
                "reference_boundary_event": verification_kind,
                "goal_verified": True,
                "source_goal_verified": dataset_kind == "native",
                "counterfactual_goal_verified": dataset_kind == "counterfactual",
                "verification_policy": "expert_goal_replay",
                "capture_origin": "source_scene_expert_replay",
                "source_directed_failure_verified": False,
                "full_goal_verified": False,
                "raw_hdf5": hdf5_path.relative_to(run_root).as_posix(),
                "source_instruction": spec.source_instruction,
                "counterfactual_instruction": spec.counterfactual_instruction,
                "scene_info": info,
            }
        )
        _append_jsonl(audit_path, record)
        print(
            f"[record] completed {spec.pair_id}/{dataset_kind} "
            f"episode={episode_index} seed={scene_seed}",
            flush=True,
        )
    _write_provenance(
        run_root=run_root,
        spec=spec,
        dataset_kind=dataset_kind,
        episode_count=len(accepted),
        task_config=task_config,
        collection_profile=collection_profile,
    )


def collect_pair(
    *,
    robotwin_root: Path,
    output_root: Path,
    source_task: str,
    task_config: str,
    episodes: int,
    start_seed: int,
    max_seed_attempts: int,
    collection_profile: str = "grounding",
) -> Path:
    spec = pair_spec_from_source_task(source_task)
    pair_root = output_root / spec.pair_id
    native_root = pair_root / "native"
    counterfactual_root = pair_root / "counterfactual"
    task, args = _load_robotwin_args(
        robotwin_root=robotwin_root,
        task_name=source_task,
        task_config=task_config,
        output_root=pair_root,
    )
    install_pgc_task_contract(task, spec)
    for run_root in (native_root, counterfactual_root):
        (run_root / "meta" / "initial_states").mkdir(parents=True, exist_ok=True)
    accepted: list[tuple[int, np.ndarray]] = []
    seed = int(start_seed)
    attempts = 0

    plan_args = dict(args)
    plan_args["data_type"] = _capture_data_type(args)
    plan_args.update(need_plan=True, save_data=False, render_freq=0)
    while len(accepted) < episodes and attempts < max_seed_attempts:
        attempts += 1
        episode_index = len(accepted)
        try:
            target_plan_args = dict(plan_args, save_path=str(counterfactual_root))
            task._pgc_active_variant = None
            task.setup_demo(now_ep_num=episode_index, seed=seed, **target_plan_args)
            target_initial_state = scene_state_vector(task)
            play_variant(task, spec, spec.counterfactual_variant)
            target_ok = bool(
                task.plan_success
                and check_variant(task, spec, spec.counterfactual_variant)
            )
            if target_ok:
                task.save_traj_data(episode_index)
            _close(task)
            if not target_ok:
                print(
                    f"[plan] rejected target {spec.pair_id} seed={seed}",
                    flush=True,
                )
                seed += 1
                continue

            native_plan_args = dict(plan_args, save_path=str(native_root))
            task._pgc_active_variant = None
            task.setup_demo(now_ep_num=episode_index, seed=seed, **native_plan_args)
            native_initial_state = scene_state_vector(task)
            if not np.array_equal(native_initial_state, target_initial_state):
                raise RuntimeError(
                    "Joint seed selection changed scene state for "
                    f"{spec.pair_id} seed={seed}."
                )
            play_variant(task, spec, spec.source_variant)
            native_ok = bool(
                task.plan_success and check_variant(task, spec, spec.source_variant)
            )
            if native_ok:
                task.save_traj_data(episode_index)
            _close(task)
            if not native_ok:
                print(
                    f"[plan] rejected native {spec.pair_id} seed={seed}",
                    flush=True,
                )
                seed += 1
                continue

            for run_root in (native_root, counterfactual_root):
                state_path = (
                    run_root / "meta" / "initial_states" / f"episode{episode_index}.npy"
                )
                np.save(state_path, target_initial_state, allow_pickle=False)
            accepted.append((seed, target_initial_state))
            print(
                f"[plan] jointly accepted {spec.pair_id} episode={episode_index} "
                f"seed={seed}",
                flush=True,
            )
        except Exception:
            print(traceback.format_exc(), flush=True)
            _close(task)
        seed += 1
    if len(accepted) != episodes:
        raise RuntimeError(
            f"Accepted {len(accepted)}/{episodes} scenes for {spec.pair_id} "
            f"after {attempts} seed attempts."
        )
    _replay_dataset(
        task=task,
        base_args=args,
        run_root=native_root,
        accepted=accepted,
        execution_variant=spec.source_variant,
        dataset_kind="native",
        spec=spec,
        task_config=task_config,
        collection_profile=collection_profile,
    )
    _replay_dataset(
        task=task,
        base_args=args,
        run_root=counterfactual_root,
        accepted=accepted,
        execution_variant=spec.counterfactual_variant,
        dataset_kind="counterfactual",
        spec=spec,
        task_config=task_config,
        collection_profile=collection_profile,
    )
    return pair_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robotwin-root", type=Path, default=PROJECT_ROOT / "third_party/RoboTwin"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=4300000)
    parser.add_argument(
        "--collection-profile",
        choices=tuple(COLLECTION_PROFILES),
        default="grounding",
        help=(
            "Bind immutable provenance for grounding, the no-ERAF "
            "offline/historical pair, or the no-ERAF strict-CF audit."
        ),
    )
    parser.add_argument(
        "--max-seed-attempts",
        type=int,
        help="Per-pair fail-closed limit (default: max(100, episodes*200)).",
    )
    parser.add_argument(
        "--source-tasks",
        nargs="+",
        default=tuple(spec.source_task for spec in ROBOTWIN_ERAF_PAIR_SPECS),
    )
    args = parser.parse_args()
    max_seed_attempts = args.max_seed_attempts or max(100, args.episodes * 200)
    if args.episodes <= 0 or args.start_seed < 0 or max_seed_attempts <= 0:
        parser.error(
            "--episodes/max-seed-attempts must be positive and "
            "--start-seed non-negative"
        )
    robotwin_root = args.robotwin_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for offset, source_task in enumerate(args.source_tasks):
        collect_pair(
            robotwin_root=robotwin_root,
            output_root=output_root,
            source_task=source_task,
            task_config=args.task_config,
            episodes=args.episodes,
            start_seed=args.start_seed + offset * 1_000_000,
            max_seed_attempts=max_seed_attempts,
            collection_profile=args.collection_profile,
        )


if __name__ == "__main__":
    main()
