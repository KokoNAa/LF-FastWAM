"""RoboTwin same-scene counterfactual collection contracts."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np


PGC_ROBOTWIN_RAW_FORMAT = "pgc_robotwin_same_scene_raw_v1"
PGC_ROBOTWIN_PAIR_FORMAT = "pgc_robotwin_direction_pair_v1"
ROBOTWIN_ACTION_DIM = 14
ROBOTWIN_CAMERAS = ("head_camera", "left_camera", "right_camera")


def opposite_direction(direction: str) -> str:
    direction = str(direction).strip().lower()
    if direction == "left":
        return "right"
    if direction == "right":
        return "left"
    raise ValueError(f"Direction must be left or right, got {direction!r}.")


def direction_from_task(task_name: str) -> str:
    task_name = str(task_name).strip().lower()
    for direction in ("left", "right"):
        if task_name == f"place_a2b_{direction}":
            return direction
    raise ValueError(
        "RoboTwin PGC collector supports place_a2b_left/right, got "
        f"{task_name!r}."
    )


def scene_state_vector(task_env: Any) -> np.ndarray:
    """Canonical state used to prove plan/replay scene identity."""
    object_pose = task_env.object.get_pose()
    reference_pose = task_env.target_object.get_pose()
    values = [
        *np.asarray(object_pose.p, dtype=np.float32).tolist(),
        *np.asarray(object_pose.q, dtype=np.float32).tolist(),
        *np.asarray(reference_pose.p, dtype=np.float32).tolist(),
        *np.asarray(reference_pose.q, dtype=np.float32).tolist(),
        *np.asarray(
            task_env.robot.get_left_arm_real_jointState(), dtype=np.float32
        ).tolist(),
        *np.asarray(
            task_env.robot.get_right_arm_real_jointState(), dtype=np.float32
        ).tolist(),
    ]
    result = np.ascontiguousarray(values, dtype=np.float32)
    if result.ndim != 1 or result.size != 28 or not np.isfinite(result).all():
        raise ValueError(
            "RoboTwin canonical scene state must be a finite 28-D vector, "
            f"got {result.shape}."
        )
    return result


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = f"{value.dtype.str}|{value.shape}".encode("utf-8")
    return hashlib.sha256(header + b"\0" + value.tobytes(order="C")).hexdigest()


def validate_pair_record(record: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "pair_id",
        "source_task",
        "counterfactual_task",
        "source_direction",
        "counterfactual_direction",
        "scene_seed",
        "initial_state_sha256",
        "action_sha256",
        "action_count",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"RoboTwin PGC pair record is missing {missing}.")
    source = direction_from_task(str(record["source_task"]))
    target = direction_from_task(str(record["counterfactual_task"]))
    if source != str(record["source_direction"]) or target != str(
        record["counterfactual_direction"]
    ):
        raise ValueError("RoboTwin PGC task/direction labels disagree.")
    if target != opposite_direction(source):
        raise ValueError("RoboTwin PGC pair must reverse left/right semantics.")
    if int(record["scene_seed"]) < 0 or int(record["action_count"]) <= 0:
        raise ValueError("RoboTwin PGC seed/action_count is invalid.")
    for key in ("initial_state_sha256", "action_sha256"):
        digest = str(record[key]).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"RoboTwin PGC {key} is not a SHA256 digest.")
    return dict(record)


def validate_action_array(actions: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(actions, dtype=np.float32)
    if (
        result.ndim != 2
        or result.shape[0] <= 0
        or result.shape[1] != ROBOTWIN_ACTION_DIM
        or not np.isfinite(result).all()
    ):
        raise ValueError(
            "RoboTwin PGC actions must be finite non-empty [T,14], got "
            f"{result.shape}."
        )
    return result
