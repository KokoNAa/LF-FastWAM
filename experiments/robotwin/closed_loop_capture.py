"""Audited released-Base closed-loop capture helpers for RoboTwin."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.robotwin.pgc_data import array_sha256, scene_state_vector


CAPTURE_FORMAT = "pgc_robotwin_closed_loop_native_capture_v1"
CAPTURE_STATE_DISTRIBUTION = "immutable_base_closed_loop_replan"
ALLOWED_STAGES = (
    "initial_search",
    "holding",
    "released_unfinished",
    "next_clause_search",
)
CAMERA_GEOMETRY = {
    "head_camera": (16, 20),
    "left_camera": (8, 10),
    "right_camera": (8, 10),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _downsample_labels(labels: np.ndarray, height: int, width: int) -> np.ndarray:
    value = np.asarray(labels, dtype=np.uint32)
    if value.ndim != 2 or value.size == 0:
        raise ValueError(f"Actor segmentation must be a non-empty image, got {value.shape}.")
    rows = np.rint(np.linspace(0, value.shape[0] - 1, height)).astype(np.int64)
    columns = np.rint(np.linspace(0, value.shape[1] - 1, width)).astype(np.int64)
    return np.ascontiguousarray(value[np.ix_(rows, columns)], dtype=np.uint32)


def _actor_labels(task_env: Any, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    observed = observation["observation"]
    if all("actor_segmentation_ids" in observed[name] for name in CAMERA_GEOMETRY):
        labels = {
            name: np.asarray(observed[name]["actor_segmentation_ids"], dtype=np.uint32)
            for name in CAMERA_GEOMETRY
        }
    else:
        labels = task_env.cameras.get_raw_segmentation(level="actor")
    missing = sorted(set(CAMERA_GEOMETRY) - set(labels))
    if missing:
        raise KeyError(f"RoboTwin closed-loop capture is missing cameras: {missing}.")
    return {
        name: _downsample_labels(labels[name], *shape)
        for name, shape in CAMERA_GEOMETRY.items()
    }


def classify_online_stage(
    snapshot: Mapping[str, np.ndarray],
    *,
    replan_index: int,
    left_gripper_open: bool,
    right_gripper_open: bool,
    previous_stage: str | None,
) -> str:
    valid = np.asarray(snapshot["source_clause_valid"], dtype=np.bool_)
    truth = np.asarray(snapshot["source_predicate_truth"], dtype=np.float32)[valid]
    if truth.size == 0:
        raise ValueError("RoboTwin closed-loop snapshot has no valid source clause.")
    if int(replan_index) == 0:
        return "initial_search"
    if not (bool(left_gripper_open) and bool(right_gripper_open)):
        return "holding"
    if bool((truth >= 0.5).any()) and not bool((truth >= 0.5).all()):
        return "next_clause_search"
    if previous_stage == "holding" and not bool((truth >= 0.5).all()):
        return "released_unfinished"
    return "initial_search"


def capture_frame(
    task_env: Any,
    observation: Mapping[str, Any],
    action: np.ndarray,
) -> dict[str, np.ndarray]:
    obs = observation["observation"]
    labels = _actor_labels(task_env, observation)
    snapshot = task_env.pgc_eraf_snapshot()
    action_value = np.ascontiguousarray(action, dtype=np.float32)
    state = np.ascontiguousarray(
        observation["joint_action"]["vector"], dtype=np.float32
    )
    if action_value.shape != (14,) or state.shape != (14,):
        raise ValueError(
            "RoboTwin closed-loop frame requires 14-D qpos state/action, got "
            f"state={state.shape} action={action_value.shape}."
        )
    result = {
        "head_rgb": np.ascontiguousarray(obs["head_camera"]["rgb"], dtype=np.uint8),
        "left_rgb": np.ascontiguousarray(obs["left_camera"]["rgb"], dtype=np.uint8),
        "right_rgb": np.ascontiguousarray(obs["right_camera"]["rgb"], dtype=np.uint8),
        "state": state,
        "action": action_value,
        "scene_state": scene_state_vector(task_env),
        "head_actor_ids": labels["head_camera"],
        "left_actor_ids": labels["left_camera"],
        "right_actor_ids": labels["right_camera"],
    }
    for key, value in snapshot.items():
        result[f"snapshot_{key}"] = np.ascontiguousarray(value)
    return result


def write_capture_segment(
    *,
    capture_root: Path,
    metadata: Mapping[str, Any],
    replan_index: int,
    online_stage: str,
    frames: Sequence[Mapping[str, np.ndarray]],
) -> Path:
    if online_stage not in ALLOWED_STAGES:
        raise ValueError(f"Unsupported RoboTwin online stage: {online_stage!r}.")
    if len(frames) != 2:
        raise ValueError("A closed-loop capture segment must contain exactly two frames.")
    task_name = str(metadata["source_task"])
    task_config = str(metadata["task_config"])
    scene_seed = int(metadata["scene_seed"])
    episode_index = int(metadata["episode_index"])
    if any(set(frame) != set(frames[0]) for frame in frames):
        raise ValueError("Closed-loop capture frame schemas differ.")
    stacked = {
        key: np.stack([np.asarray(frame[key]) for frame in frames], axis=0)
        for key in frames[0]
    }
    state_digest = array_sha256(stacked["scene_state"])
    action_digest = array_sha256(stacked["action"])
    identity = {
        "pair_id": str(metadata["pair_id"]),
        "source_task": task_name,
        "task_config": task_config,
        "scene_seed": scene_seed,
        "episode_index": episode_index,
        "replan_index": int(replan_index),
        "online_stage_v2": online_stage,
        "policy_instruction": str(metadata["policy_instruction"]),
        "state_sha256": state_digest,
        "action_sha256": action_digest,
    }
    capture_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output_dir = (
        Path(capture_root).expanduser().resolve()
        / task_name
        / task_config
        / f"seed_{scene_seed:08d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{episode_index:04d}_replan_{int(replan_index):04d}_{capture_id[:12]}"
    array_path = output_dir / f"{stem}.npz"
    record_path = output_dir / f"{stem}.json"
    if array_path.exists() or record_path.exists():
        if not (array_path.is_file() and record_path.is_file()):
            raise FileExistsError(f"Partial closed-loop capture exists for {stem}.")
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            existing.get("capture_id") != capture_id
            or existing.get("capture_file") != array_path.name
            or existing.get("capture_file_sha256") != _file_sha256(array_path)
            or existing.get("state_sha256") != state_digest
            or existing.get("action_sha256") != action_digest
        ):
            raise FileExistsError(f"Closed-loop capture collision for {stem}.")
        return record_path
    temp_array = output_dir / f".{stem}.npz.tmp"
    with temp_array.open("wb") as handle:
        np.savez_compressed(handle, **stacked)
    temp_array.replace(array_path)
    record = {
        "format": CAPTURE_FORMAT,
        "capture_id": capture_id,
        "capture_file": array_path.name,
        "capture_file_sha256": _file_sha256(array_path),
        "frame_count": 2,
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
        "online_stage_v2": online_stage,
        **dict(metadata),
        **identity,
    }
    temp_record = output_dir / f".{stem}.json.tmp"
    temp_record.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temp_record.replace(record_path)
    return record_path
