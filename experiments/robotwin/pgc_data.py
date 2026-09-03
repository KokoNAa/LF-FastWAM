"""RoboTwin same-scene counterfactual collection contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


PGC_ROBOTWIN_RAW_FORMAT = "pgc_robotwin_same_scene_raw_v1"
PGC_ROBOTWIN_PAIR_FORMAT = "pgc_robotwin_direction_pair_v1"
ROBOTWIN_ACTION_DIM = 14
ROBOTWIN_CAMERAS = ("head_camera", "left_camera", "right_camera")


@dataclass(frozen=True)
class RoboTwinPairSpec:
    pair_id: str
    source_task: str
    counterfactual_task: str
    source_variant: str
    counterfactual_variant: str
    source_instruction: str
    counterfactual_instruction: str
    strict_conflict_type: str
    entity_names: tuple[str, ...]
    pair_aliases: tuple[str, ...] = ()


ROBOTWIN_ERAF_PAIR_SPECS = (
    RoboTwinPairSpec(
        pair_id="place_a2b_left_to_right",
        pair_aliases=("place_a2b_left_to_place_a2b_right", "left_to_right"),
        source_task="place_a2b_left",
        counterfactual_task="place_a2b_right",
        source_variant="left",
        counterfactual_variant="right",
        source_instruction="Place object A to the left of object B.",
        counterfactual_instruction="Place object A to the right of object B.",
        strict_conflict_type="mutually_exclusive_spatial_direction",
        entity_names=("object", "target_object"),
    ),
    RoboTwinPairSpec(
        pair_id="place_a2b_right_to_left",
        pair_aliases=("place_a2b_right_to_place_a2b_left", "right_to_left"),
        source_task="place_a2b_right",
        counterfactual_task="place_a2b_left",
        source_variant="right",
        counterfactual_variant="left",
        source_instruction="Place object A to the right of object B.",
        counterfactual_instruction="Place object A to the left of object B.",
        strict_conflict_type="mutually_exclusive_spatial_direction",
        entity_names=("object", "target_object"),
    ),
    RoboTwinPairSpec(
        pair_id="stack_blocks_two_green_on_red_to_red_on_green",
        source_task="stack_blocks_two",
        counterfactual_task="stack_blocks_two_red_on_green",
        source_variant="green_on_red",
        counterfactual_variant="red_on_green",
        source_instruction="Move the blocks to the center, then stack the green block on the red block.",
        counterfactual_instruction="Move the green block to the center, then stack the red block on the green block.",
        strict_conflict_type="mutually_exclusive_stack_order",
        entity_names=("red_block", "green_block"),
    ),
    RoboTwinPairSpec(
        pair_id="blocks_ranking_rgb_to_bgr",
        source_task="blocks_ranking_rgb",
        counterfactual_task="blocks_ranking_bgr",
        source_variant="rgb",
        counterfactual_variant="bgr",
        source_instruction="Arrange the red, green, and blue blocks from left to right in a row.",
        counterfactual_instruction="Arrange the blue, green, and red blocks from left to right in a row.",
        strict_conflict_type="mutually_exclusive_row_order",
        entity_names=("red_block", "green_block", "blue_block"),
    ),
    RoboTwinPairSpec(
        pair_id="place_burger_fries_native_to_swapped_slots",
        source_task="place_burger_fries",
        counterfactual_task="place_burger_fries_swapped_slots",
        source_variant="native_slots",
        counterfactual_variant="swapped_slots",
        source_instruction="Place the hamburger on the left slot of the tray and the french fries on the right slot.",
        counterfactual_instruction="Place the hamburger on the right slot of the tray and the french fries on the left slot.",
        strict_conflict_type="mutually_exclusive_slot_assignment",
        entity_names=("hamburger", "french_fries", "tray"),
    ),
)

ROBOTWIN_ERAF_PAIR_IDS = tuple(spec.pair_id for spec in ROBOTWIN_ERAF_PAIR_SPECS)


def pair_spec_from_source_task(task_name: str) -> RoboTwinPairSpec:
    matches = [
        spec for spec in ROBOTWIN_ERAF_PAIR_SPECS if spec.source_task == task_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one RoboTwin ERAF pair for source task {task_name!r}, "
            f"found {len(matches)}."
        )
    return matches[0]


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
        "RoboTwin PGC collector supports place_a2b_left/right, got " f"{task_name!r}."
    )


def scene_state_vector(task_env: Any) -> np.ndarray:
    """Canonical state used to prove plan/replay scene identity."""
    if hasattr(task_env, "pgc_scene_actors"):
        actors = list(task_env.pgc_scene_actors())
    elif hasattr(task_env, "object") and hasattr(task_env, "target_object"):
        actors = [task_env.object, task_env.target_object]
    else:
        raise ValueError("RoboTwin task does not expose PGC scene actors.")
    if not actors:
        raise ValueError("RoboTwin PGC scene actor list is empty.")
    values: list[float] = []
    for actor in actors:
        pose = actor.get_pose()
        values.extend(np.asarray(pose.p, dtype=np.float32).tolist())
        values.extend(np.asarray(pose.q, dtype=np.float32).tolist())
    values.extend(
        np.asarray(
            task_env.robot.get_left_arm_real_jointState(), dtype=np.float32
        ).tolist()
    )
    values.extend(
        np.asarray(
            task_env.robot.get_right_arm_real_jointState(), dtype=np.float32
        ).tolist()
    )
    if hasattr(task_env, "pgc_eraf_snapshot"):
        snapshot = task_env.pgc_eraf_snapshot()
        for key in (
            "source_subject_indices",
            "source_reference_indices",
            "source_predicate_ids",
            "source_goal_positions",
            "source_clause_valid",
            "target_subject_indices",
            "target_reference_indices",
            "target_predicate_ids",
            "target_goal_positions",
            "target_clause_valid",
        ):
            values.extend(
                np.asarray(snapshot[key], dtype=np.float32).reshape(-1).tolist()
            )
    result = np.ascontiguousarray(values, dtype=np.float32)
    minimum_size = 7 * len(actors) + ROBOTWIN_ACTION_DIM
    if result.ndim != 1 or result.size < minimum_size or not np.isfinite(result).all():
        raise ValueError(
            "RoboTwin canonical scene state has an invalid shape, "
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
        "scene_seed",
        "initial_state_sha256",
        "action_sha256",
        "action_count",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"RoboTwin PGC pair record is missing {missing}.")
    spec = pair_spec_from_source_task(str(record["source_task"]))
    if str(record["counterfactual_task"]) != spec.counterfactual_task:
        if "source_direction" in record or "counterfactual_direction" in record:
            raise ValueError("RoboTwin PGC pair must reverse left/right semantics.")
        raise ValueError("RoboTwin PGC counterfactual task disagrees with pair spec.")
    if str(record["pair_id"]) not in {spec.pair_id, *spec.pair_aliases}:
        raise ValueError("RoboTwin PGC pair_id disagrees with pair spec.")
    source_variant = str(
        record.get("source_variant", record.get("source_direction", ""))
    )
    target_variant = str(
        record.get("counterfactual_variant", record.get("counterfactual_direction", ""))
    )
    if (
        source_variant != spec.source_variant
        or target_variant != spec.counterfactual_variant
    ):
        raise ValueError("RoboTwin PGC task/variant labels disagree.")
    dataset_kind = record.get("dataset_kind")
    if dataset_kind is not None:
        dataset_kind = str(dataset_kind)
        if dataset_kind not in {"native", "counterfactual"}:
            raise ValueError(f"RoboTwin PGC dataset_kind is invalid: {dataset_kind!r}.")
        expected_execution = (
            spec.source_variant
            if dataset_kind == "native"
            else spec.counterfactual_variant
        )
        executed = str(
            record.get("executed_variant", record.get("executed_direction", ""))
        )
        if executed != expected_execution:
            raise ValueError(
                "RoboTwin PGC executed variant does not match dataset_kind."
            )
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
