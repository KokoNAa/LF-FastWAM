#!/usr/bin/env python3
"""Build the no-ERAF closed-loop-native pool from released-Base captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.robotwin.closed_loop_capture import (
    ALLOWED_STAGES,
    CAPTURE_FORMAT,
    CAPTURE_STATE_DISTRIBUTION,
)
from experiments.robotwin.pgc_data import (
    ROBOTWIN_ERAF_PAIR_SPECS,
    array_sha256,
    pair_spec_from_source_task,
)
from fastwam.datasets.pgc_libero import (
    PGC_ACTION_CONVENTION_ROBOTWIN_QPOS,
    PGC_ACTION_REPLAY_IDENTITY,
    PGC_DATA_FORMAT,
    PGC_ENTITY_RELATION_PREDICATES,
    PGC_ROBOTWIN_ENTITY_RELATION_FORMAT,
    load_pgc_entity_relation_index,
)
from scripts.build_pgc_robotwin_entity_relations import (
    MASK_SIZE,
    WORKSPACE_MAX,
    WORKSPACE_MIN,
    _empty_arrays,
    _entity_id,
    _file_sha256,
    _mosaic_mask,
    _normalize_position,
    _view_geometry,
)
from scripts.convert_pgc_robotwin_to_lerobot import robotwin_lerobot_features


DATA_INDEX = Path("meta/pgc_robotwin_closed_loop_native.json")
FULL_GOAL_INDEX = Path("meta/pgc_robotwin_full_goal/index.json")


def _parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise ValueError(f"Invalid comma-separated value: {value!r}.")
    return result


def _training_pair_id(record: Mapping[str, Any]) -> str:
    language_key = "\0".join(
        (
            str(record["pair_id"]),
            str(record["source_instruction"]).strip().casefold(),
            str(record["counterfactual_instruction"]).strip().casefold(),
        )
    )
    return f"{record['pair_id']}::{hashlib.sha256(language_key.encode()).hexdigest()[:12]}"


def _load_records(
    captures: Path,
    *,
    expected_task_configs: tuple[str, ...],
    max_per_task_config_stage: int,
    seed: int,
) -> list[dict[str, Any]]:
    expected_tasks = {spec.source_task for spec in ROBOTWIN_ERAF_PAIR_SPECS}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for path in sorted(captures.expanduser().resolve().rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("format") != CAPTURE_FORMAT:
            continue
        capture_id = str(record.get("capture_id", ""))
        if len(capture_id) != 64 or capture_id in seen:
            raise ValueError(f"Invalid or duplicate closed-loop capture: {path}.")
        task = str(record.get("source_task", ""))
        config = str(record.get("task_config", ""))
        stage = str(record.get("online_stage_v2", ""))
        if task not in expected_tasks or config not in expected_task_configs:
            continue
        required = {
            "rollout_policy": "immutable_released_base",
            "action_integrity": "selected_equals_immutable_base_exact",
            "state_distribution": CAPTURE_STATE_DISTRIBUTION,
            "privileged_supervision": "training_only",
            "deployment_inputs": "rgb_language_proprio",
            "artifact_role": "closed_loop_native",
            "allowed_training_stages": ["no_eraf", "joint", "final_short_lora"],
            "full_goal_usage": "forbidden_not_present",
            "full_goal_verified": False,
            "condition": "correct",
            "instruction_goal": "source",
            "frame_count": 2,
        }
        mismatches = {
            key: (record.get(key), expected)
            for key, expected in required.items()
            if record.get(key) != expected
        }
        if mismatches or stage not in ALLOWED_STAGES:
            raise ValueError(f"Closed-loop capture contract mismatch at {path}: {mismatches}.")
        if str(record.get("policy_instruction", "")).strip().casefold() != str(
            record.get("source_instruction", "")
        ).strip().casefold():
            raise ValueError(f"Capture is not a Correct/source rollout: {path}.")
        spec = pair_spec_from_source_task(task)
        if (
            str(record.get("pair_id", "")) != spec.pair_id
            or not str(record.get("source_instruction", "")).strip()
            or not str(record.get("counterfactual_instruction", "")).strip()
            or str(record["source_instruction"]).strip().casefold()
            == str(record["counterfactual_instruction"]).strip().casefold()
        ):
            raise ValueError(f"Capture has an invalid language-pair contract: {path}.")
        capture_file = path.parent / str(record.get("capture_file", ""))
        if not capture_file.is_file() or _file_sha256(capture_file) != str(
            record.get("capture_file_sha256", "")
        ):
            raise ValueError(f"Closed-loop capture file hash changed: {capture_file}.")
        with np.load(capture_file, allow_pickle=False) as payload:
            arrays = {key: np.asarray(payload[key]).copy() for key in payload.files}
        for name in ("state", "action", "scene_state"):
            if arrays.get(name, np.empty(0)).shape[0] != 2:
                raise ValueError(f"Closed-loop {name} is not a two-frame segment: {path}.")
        if arrays["state"].shape != (2, 14) or arrays["action"].shape != (2, 14):
            raise ValueError(f"Closed-loop qpos/action shape mismatch: {path}.")
        if array_sha256(arrays["scene_state"]) != str(record["state_sha256"]):
            raise ValueError(f"Closed-loop state hash changed: {path}.")
        if array_sha256(arrays["action"]) != str(record["action_sha256"]):
            raise ValueError(f"Closed-loop action hash changed: {path}.")
        normalized = dict(record)
        normalized["record_path"] = str(path)
        normalized["capture_path"] = str(capture_file)
        normalized["arrays"] = arrays
        normalized["training_pair_id"] = _training_pair_id(normalized)
        grouped[(task, config, stage)].append(normalized)
        seen.add(capture_id)

    missing = [
        f"{task}/{config}"
        for task in sorted(expected_tasks)
        for config in expected_task_configs
        if not any(key[:2] == (task, config) for key in grouped)
    ]
    if missing:
        raise RuntimeError(f"Closed-loop capture lacks task/config coverage: {missing}.")
    checkpoints = {
        str(record["checkpoint"]) for values in grouped.values() for record in values
    }
    if len(checkpoints) != 1:
        raise ValueError(f"Closed-loop captures mix released Base checkpoints: {checkpoints}.")
    rng = np.random.default_rng(int(seed))
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        values = sorted(
            grouped[key],
            key=lambda row: (
                int(row["scene_seed"]),
                int(row["episode_index"]),
                int(row["replan_index"]),
            ),
        )
        if len(values) > max_per_task_config_stage:
            indices = np.sort(
                rng.choice(len(values), size=max_per_task_config_stage, replace=False)
            )
            values = [values[int(index)] for index in indices]
        selected.extend(values)
    selected.sort(
        key=lambda row: (
            str(row["source_task"]),
            str(row["task_config"]),
            int(row["scene_seed"]),
            int(row["replan_index"]),
        )
    )
    if not selected:
        raise RuntimeError("No valid RoboTwin closed-loop-native capture found.")
    return selected


def _clauses(record: Mapping[str, Any], prefix: str) -> list[dict[str, Any]]:
    spec = pair_spec_from_source_task(str(record["source_task"]))
    arrays = record["arrays"]
    valid = np.asarray(arrays[f"snapshot_{prefix}_clause_valid"])[0]
    subjects = np.asarray(arrays[f"snapshot_{prefix}_subject_indices"])[0]
    references = np.asarray(arrays[f"snapshot_{prefix}_reference_indices"])[0]
    predicates = np.asarray(arrays[f"snapshot_{prefix}_predicate_ids"])[0]
    result = []
    for clause_index in np.flatnonzero(valid):
        subject_index = int(subjects[clause_index])
        reference_index = int(references[clause_index])
        predicate_id = int(predicates[clause_index])
        result.append(
            {
                "clause_index": int(clause_index),
                "predicate": PGC_ENTITY_RELATION_PREDICATES[predicate_id],
                "predicate_id": predicate_id,
                "subject": spec.entity_names[subject_index],
                "reference": spec.entity_names[reference_index],
            }
        )
    if not result:
        raise ValueError(f"Closed-loop capture has no {prefix} clause.")
    return result


def _role_arrays(record: Mapping[str, Any], prefix: str) -> dict[str, np.ndarray]:
    spec = pair_spec_from_source_task(str(record["source_task"]))
    captured = record["arrays"]
    frame_count = 2
    positions = np.asarray(captured["snapshot_entity_positions"], dtype=np.float32)
    actor_ids = np.asarray(captured["snapshot_entity_actor_ids"], dtype=np.uint32)
    entity_valid = np.asarray(captured["snapshot_entity_valid"], dtype=np.bool_)
    subject_indices = np.asarray(
        captured[f"snapshot_{prefix}_subject_indices"], dtype=np.int64
    )
    reference_indices = np.asarray(
        captured[f"snapshot_{prefix}_reference_indices"], dtype=np.int64
    )
    predicate_ids = np.asarray(
        captured[f"snapshot_{prefix}_predicate_ids"], dtype=np.int64
    )
    goal_positions = np.asarray(
        captured[f"snapshot_{prefix}_goal_positions"], dtype=np.float32
    )
    truth = np.asarray(
        captured[f"snapshot_{prefix}_predicate_truth"], dtype=np.float32
    )
    clause_valid = np.asarray(
        captured[f"snapshot_{prefix}_clause_valid"], dtype=np.bool_
    )
    arrays = _empty_arrays(frame_count)
    stage = str(record["online_stage_v2"])
    labels = (
        captured["head_actor_ids"],
        captured["left_actor_ids"],
        captured["right_actor_ids"],
    )
    for frame in range(frame_count):
        incomplete = [
            int(index)
            for index in np.flatnonzero(clause_valid[frame])
            if truth[frame, index] < 0.5
        ]
        active_clause = incomplete[0] if incomplete else None
        for clause_index in np.flatnonzero(clause_valid[frame]):
            clause_index = int(clause_index)
            subject_index = int(subject_indices[frame, clause_index])
            reference_index = int(reference_indices[frame, clause_index])
            if not (
                entity_valid[frame, subject_index]
                and entity_valid[frame, reference_index]
            ):
                raise ValueError("Closed-loop clause references an invalid entity.")
            arrays["clause_valid"][frame, clause_index] = True
            arrays["predicate_ids"][frame, clause_index] = int(
                predicate_ids[frame, clause_index]
            )
            arrays["predicate_truth"][frame, clause_index] = truth[
                frame, clause_index
            ]
            arrays["predicate_truth_valid"][frame, clause_index] = True
            phase = 2 if truth[frame, clause_index] >= 0.5 else 0
            if stage in {"holding", "released_unfinished"} and clause_index == active_clause:
                phase = 1
            arrays["phase_ids"][frame, clause_index] = phase
            arrays["phase_valid"][frame, clause_index] = True
            for role, entity_index in (
                ("subject", subject_index),
                ("reference", reference_index),
            ):
                entity_name = f"robotwin:{spec.pair_id}:{spec.entity_names[entity_index]}"
                arrays[f"{role}_entity_ids"][frame, clause_index] = _entity_id(
                    entity_name
                )
                actor_id = int(actor_ids[frame, entity_index])
                mask = _mosaic_mask(
                    labels[0][frame], labels[1][frame], labels[2][frame], actor_id
                )
                visible, centers = _view_geometry(
                    labels[0][frame], labels[1][frame], labels[2][frame], actor_id
                )
                arrays[f"{role}_masks"][frame, clause_index] = mask
                arrays[f"{role}_mask_valid"][frame, clause_index] = bool(mask.any())
                arrays[f"{role}_view_visible"][frame, clause_index] = visible
                arrays[f"{role}_view_centers"][frame, clause_index] = centers
                arrays[f"{role}_positions"][frame, clause_index] = _normalize_position(
                    positions[frame, entity_index]
                )
                arrays[f"{role}_position_valid"][frame, clause_index] = True
            arrays["grasp_anchors"][frame, clause_index] = arrays[
                "subject_positions"
            ][frame, clause_index]
            arrays["goal_anchors"][frame, clause_index] = _normalize_position(
                goal_positions[frame, clause_index]
            )
            arrays["interaction_anchors"][frame, clause_index] = arrays[
                "goal_anchors"
            ][frame, clause_index]
            arrays["grasp_anchor_valid"][frame, clause_index] = True
            arrays["goal_anchor_valid"][frame, clause_index] = True
            arrays["interaction_anchor_valid"][frame, clause_index] = True
    return arrays


def build_dataset(
    records: list[dict[str, Any]],
    *,
    output: Path,
    sidecar: Path,
    fps: int,
    video_codec: str,
    task_configs: tuple[str, ...],
) -> dict[str, Any]:
    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

    output = output.expanduser().resolve()
    sidecar = sidecar.expanduser().resolve()
    for path in (output, sidecar):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite closed-loop output: {path}.")
    first = records[0]["arrays"]["head_rgb"][0]
    height, width = map(int, first.shape[:2])
    dataset = LeRobotDataset.create(
        repo_id=output.name,
        root=output,
        fps=int(fps),
        robot_type="aloha-agilex",
        features=robotwin_lerobot_features(height, width),
        use_videos=True,
        video_codec=video_codec,
        is_compute_episode_stats_image=False,
    )
    episode_dir = sidecar / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=False)
    episode_records = []
    audits = []
    stage_counts: Counter[str] = Counter()
    entity_vocabulary = {
        f"robotwin:{spec.pair_id}:{name}": _entity_id(
            f"robotwin:{spec.pair_id}:{name}"
        )
        for spec in ROBOTWIN_ERAF_PAIR_SPECS
        for name in spec.entity_names
    }
    for episode_index, record in enumerate(records):
        captured = record["arrays"]
        instruction = str(record["source_instruction"])
        for frame in range(2):
            dataset.add_frame(
                {
                    "observation.images.cam_high": captured["head_rgb"][frame],
                    "observation.images.cam_left_wrist": captured["left_rgb"][frame],
                    "observation.images.cam_right_wrist": captured["right_rgb"][frame],
                    "observation.state": captured["state"][frame].astype(np.float32),
                    "action": captured["action"][frame].astype(np.float32),
                },
                task=[instruction, instruction, instruction, instruction],
            )
        dataset.save_episode(raw_file_name=str(record["record_path"]))
        source = _role_arrays(record, "source")
        target = _role_arrays(record, "target")
        payload = {
            **{f"source_{key}": value for key, value in source.items()},
            **{f"target_{key}": value for key, value in target.items()},
        }
        relpath = Path("episodes") / f"episode{episode_index}.npz"
        episode_path = sidecar / relpath
        np.savez_compressed(episode_path, **payload)
        source_clauses = _clauses(record, "source")
        target_clauses = _clauses(record, "target")
        episode_records.append(
            {
                "episode_index": episode_index,
                "pair_id": str(record["training_pair_id"]),
                "file": relpath.as_posix(),
                "sha256": _file_sha256(episode_path),
                "frame_count": 2,
                "state_sha256": str(record["state_sha256"]),
                "initial_state_sha256": str(record["state_sha256"]),
                "action_sha256": str(record["action_sha256"]),
                "capture_id": str(record["capture_id"]),
                "online_stage_v2": str(record["online_stage_v2"]),
                "source_clauses": source_clauses,
                "target_clauses": target_clauses,
            }
        )
        audits.append(
            {
                "episode_index": episode_index,
                "pair_id": str(record["training_pair_id"]),
                "capture_id": str(record["capture_id"]),
                "source_task": str(record["source_task"]),
                "task_config": str(record["task_config"]),
                "scene_seed": int(record["scene_seed"]),
                "replan_index": int(record["replan_index"]),
                "online_stage_v2": str(record["online_stage_v2"]),
                "initial_state_sha256": str(record["state_sha256"]),
                "action_sha256": str(record["action_sha256"]),
                "action_count": 2,
            }
        )
        stage_counts[str(record["online_stage_v2"])] += 1

    task_ids = {
        spec.source_task: index for index, spec in enumerate(ROBOTWIN_ERAF_PAIR_SPECS)
    }
    pairs_by_id = {}
    for record in records:
        pair_id = str(record["training_pair_id"])
        pairs_by_id[pair_id] = {
            "pair_id": pair_id,
            "source_instruction": str(record["source_instruction"]),
            "counterfactual_instruction": str(record["counterfactual_instruction"]),
            "source_suite": "robotwin",
            "source_task_id": task_ids[str(record["source_task"])],
            "strict_conflict": False,
            "strict_conflict_type": pair_spec_from_source_task(
                str(record["source_task"])
            ).strict_conflict_type,
        }
    pairs = [pairs_by_id[key] for key in sorted(pairs_by_id)]
    provenance = {
        "format": PGC_DATA_FORMAT,
        "platform": "robotwin",
        "artifact_role": "closed_loop_native",
        "allowed_training_stages": ["no_eraf", "joint", "final_short_lora"],
        "forbidden_training_stages": ["grounding"],
        "full_goal_usage": "forbidden_not_present",
        "full_goal_verified": False,
        "dataset_kind": "native",
        "state_aligned": True,
        "state_distribution": CAPTURE_STATE_DISTRIBUTION,
        "rollout_policy": "immutable_released_base",
        "action_integrity": "selected_equals_immutable_base_exact",
        "task_configs": list(task_configs),
        "successful_episode_count": len(audits),
        "pairs": pairs,
    }
    (output / "meta/pgc_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "meta/pgc_episodes.jsonl").open("w", encoding="utf-8") as handle:
        for audit in audits:
            handle.write(json.dumps(audit, sort_keys=True) + "\n")
    (output / DATA_INDEX).write_text(
        json.dumps(
            {
                "format": "pgc_robotwin_closed_loop_native_v1",
                "complete": True,
                "episode_count": len(audits),
                "frame_count": 2 * len(audits),
                "stage_counts": dict(stage_counts),
                "state_distribution": CAPTURE_STATE_DISTRIBUTION,
                "full_goal_usage": "forbidden_not_present",
                "episodes": audits,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar_index = {
        "format": PGC_ROBOTWIN_ENTITY_RELATION_FORMAT,
        "artifact_role": "closed_loop_native",
        "allowed_training_stages": ["no_eraf", "joint", "final_short_lora"],
        "full_goal_verified": False,
        "privileged_supervision": "training_only",
        "deployment_inputs": "rgb_language_proprio",
        "dataset": str(output),
        "dataset_kind": "native",
        "state_distribution": CAPTURE_STATE_DISTRIBUTION,
        "dataset_action_convention": PGC_ACTION_CONVENTION_ROBOTWIN_QPOS,
        "simulator_replay_action_transform": PGC_ACTION_REPLAY_IDENTITY,
        "action_dim": 14,
        "entity_id_scheme": "sha256_63bit",
        "predicate_vocabulary": list(PGC_ENTITY_RELATION_PREDICATES),
        "entity_vocabulary": entity_vocabulary,
        "max_clauses": 4,
        "mask_size": list(MASK_SIZE),
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "camera_layout": "robotwin_mosaic",
        "view_center_coordinate_system": "per_camera_normalized_xy",
        "workspace_min": WORKSPACE_MIN.tolist(),
        "workspace_max": WORKSPACE_MAX.tolist(),
        "episode_count": len(episode_records),
        "stage_counts": dict(stage_counts),
        "episodes": episode_records,
    }
    (sidecar / "index.json").write_text(
        json.dumps(sidecar_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    load_pgc_entity_relation_index(sidecar)
    if (output / FULL_GOAL_INDEX).exists():
        raise ValueError("full-goal leaked into closed-loop-native output.")
    return {
        "format": "pgc_robotwin_closed_loop_native_ready_v1",
        "complete": True,
        "dataset": str(output),
        "sidecar": str(sidecar),
        "episodes": len(audits),
        "frames": 2 * len(audits),
        "stage_counts": dict(stage_counts),
        "state_distribution": CAPTURE_STATE_DISTRIBUTION,
        "full_goal_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidecar-output", type=Path, required=True)
    parser.add_argument("--expected-task-configs", default="demo_clean,demo_randomized")
    parser.add_argument("--max-per-task-config-stage", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--video-codec",
        choices=("h264", "hevc", "libsvtav1", "h264_nvenc"),
        default="h264",
    )
    args = parser.parse_args()
    if args.max_per_task_config_stage <= 0 or args.fps <= 0:
        parser.error("max-per-task-config-stage/fps must be positive")
    task_configs = _parse_csv(args.expected_task_configs)
    records = _load_records(
        args.captures,
        expected_task_configs=task_configs,
        max_per_task_config_stage=int(args.max_per_task_config_stage),
        seed=int(args.seed),
    )
    result = build_dataset(
        records,
        output=args.output,
        sidecar=args.sidecar_output,
        fps=int(args.fps),
        video_codec=str(args.video_codec),
        task_configs=task_configs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
