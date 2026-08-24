"""Privileged closed-loop Cartesian servo for ERAF action-gap ablations.

This controller is evaluation-only.  It consumes live MuJoCo/BDDL oracle
positions and phases and therefore must never be enabled for deployment or
reported as a learned-policy result.  The learned Proposal still supplies
orientation actions; the servo replaces Cartesian translation and, for
placement predicates, the gripper command.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


PLACEMENT_PREDICATE_IDS = frozenset({1, 2, 3, 4, 5, 6})
INTERACTION_PREDICATE_IDS = frozenset({7, 8, 9, 10})


@dataclass(frozen=True)
class OraclePhaseServoConfig:
    enabled: bool = False
    approach_gain: float = 4.0
    transport_gain: float = 4.0
    max_translation_action: float = 0.20
    approach_height_m: float = 0.08
    transport_height_m: float = 0.10
    grasp_offset_m: float = 0.01
    release_height_m: float = 0.04
    horizontal_tolerance_m: float = 0.035
    grasp_distance_m: float = 0.035
    release_distance_m: float = 0.05
    interaction_distance_m: float = 0.045

    def validate(self) -> None:
        positive = {
            "approach_gain": self.approach_gain,
            "transport_gain": self.transport_gain,
            "max_translation_action": self.max_translation_action,
            "approach_height_m": self.approach_height_m,
            "transport_height_m": self.transport_height_m,
            "horizontal_tolerance_m": self.horizontal_tolerance_m,
            "grasp_distance_m": self.grasp_distance_m,
            "release_distance_m": self.release_distance_m,
            "interaction_distance_m": self.interaction_distance_m,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(
                "Oracle phase-servo parameters must be positive: "
                f"{invalid}."
            )
        if not 0 < self.max_translation_action <= 1:
            raise ValueError(
                "Oracle phase-servo max translation action must be in (0, 1]."
            )
        for name in ("grasp_offset_m", "release_height_m"):
            if getattr(self, name) < 0:
                raise ValueError(f"Oracle phase-servo {name} must be nonnegative.")


def _world_position(
    normalized: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> np.ndarray:
    return workspace_min + (normalized + 1.0) * (workspace_max - workspace_min) / 2.0


def _bounded_translation(error: np.ndarray, *, gain: float, limit: float) -> np.ndarray:
    command = np.asarray(error, dtype=np.float32) * float(gain)
    norm = float(np.linalg.norm(command))
    if norm > float(limit):
        command = command * (float(limit) / max(norm, 1.0e-8))
    return np.clip(command, -float(limit), float(limit)).astype(np.float32)


def _first_unfinished_clause(oracle: Mapping[str, Any]) -> int | None:
    valid = np.asarray(oracle["clause_valid"], dtype=np.bool_).reshape(-1)
    truth = np.asarray(oracle["predicate_truth"], dtype=np.bool_).reshape(-1)
    if valid.shape != truth.shape:
        raise ValueError("Oracle clause-valid and predicate-truth shapes disagree.")
    unfinished = np.flatnonzero(valid & ~truth)
    return int(unfinished[0]) if unfinished.size else None


def apply_oracle_phase_servo(
    action_chunk: np.ndarray,
    *,
    obs: Mapping[str, Any],
    oracle: Mapping[str, Any],
    workspace_min: Sequence[float],
    workspace_max: Sequence[float],
    config: OraclePhaseServoConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replace translation/gripper with a live privileged phase servo."""

    actions = np.asarray(action_chunk, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] < 7:
        raise ValueError(
            "Oracle phase servo expects a non-empty [T,D>=7] action chunk."
        )
    if not np.isfinite(actions).all():
        raise ValueError("Oracle phase servo received non-finite actions.")
    output = actions.copy()
    if not config.enabled:
        return output, {"enabled": False, "applied": False, "mode": "disabled"}
    config.validate()

    lower = np.asarray(workspace_min, dtype=np.float32).reshape(3)
    upper = np.asarray(workspace_max, dtype=np.float32).reshape(3)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper <= lower):
        raise ValueError("Oracle phase servo received invalid workspace bounds.")
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
    selected = _first_unfinished_clause(oracle)
    if selected is None:
        return output, {
            "enabled": True,
            "applied": False,
            "mode": "all_clauses_complete",
            "selected_clause": -1,
        }

    predicate_ids = np.asarray(oracle["predicate_ids"], dtype=np.int64).reshape(-1)
    phase_ids = np.asarray(oracle["phase_ids"], dtype=np.int64).reshape(-1)
    subject_valid = np.asarray(
        oracle["subject_position_valid"], dtype=np.bool_
    ).reshape(-1)
    goal_valid = np.asarray(oracle["goal_anchor_valid"], dtype=np.bool_).reshape(-1)
    grasped = np.asarray(
        oracle.get("subject_grasped", np.zeros_like(phase_ids)), dtype=np.bool_
    ).reshape(-1)
    subject_normalized = np.asarray(
        oracle["subject_positions"], dtype=np.float32
    ).reshape(-1, 3)
    goal_normalized = np.asarray(oracle["goal_anchors"], dtype=np.float32).reshape(-1, 3)
    predicate_id = int(predicate_ids[selected])
    requested_phase = int(phase_ids[selected])
    subject_is_grasped = bool(grasped[selected])
    effective_phase = requested_phase
    if requested_phase == 1 and not subject_is_grasped:
        # Released-but-unsatisfied clauses must reacquire the object rather
        # than transporting toward the goal with an empty gripper.
        effective_phase = 0

    subject = (
        _world_position(subject_normalized[selected], lower, upper)
        if bool(subject_valid[selected])
        else None
    )
    goal = (
        _world_position(goal_normalized[selected], lower, upper)
        if bool(goal_valid[selected])
        else None
    )
    original_translation = output[:, :3].copy()
    original_gripper = output[:, -1].copy()
    target: np.ndarray | None = None
    mode = "missing_anchor"
    gripper_command: float | None = None

    if predicate_id in PLACEMENT_PREDICATE_IDS and effective_phase == 0 and subject is not None:
        horizontal = float(np.linalg.norm(eef[:2] - subject[:2]))
        grasp_target = subject + np.asarray([0.0, 0.0, config.grasp_offset_m])
        grasp_distance = float(np.linalg.norm(eef - grasp_target))
        if grasp_distance <= config.grasp_distance_m:
            target = eef.copy()
            mode = "grasp_close"
            gripper_command = 1.0
        elif horizontal > config.horizontal_tolerance_m:
            target = subject + np.asarray([0.0, 0.0, config.approach_height_m])
            mode = "approach_hover"
            gripper_command = -1.0
        else:
            target = grasp_target
            mode = "approach_descend"
            gripper_command = -1.0
    elif predicate_id in PLACEMENT_PREDICATE_IDS and effective_phase == 1 and goal is not None:
        horizontal = float(np.linalg.norm(eef[:2] - goal[:2]))
        release_target = goal + np.asarray([0.0, 0.0, config.release_height_m])
        release_distance = float(np.linalg.norm(eef - release_target))
        if release_distance <= config.release_distance_m:
            target = eef.copy()
            mode = "release_open"
            gripper_command = -1.0
        elif horizontal > config.horizontal_tolerance_m:
            target = goal + np.asarray([0.0, 0.0, config.transport_height_m])
            mode = "transport_hover"
            gripper_command = 1.0
        else:
            target = release_target
            mode = "transport_descend"
            gripper_command = 1.0
    elif predicate_id in INTERACTION_PREDICATE_IDS:
        interaction = goal if goal is not None else subject
        if interaction is not None:
            distance = float(np.linalg.norm(eef - interaction))
            if distance > config.interaction_distance_m:
                target = interaction
                mode = "interaction_approach"
            else:
                # Close to the fixture, retain Proposal translation and
                # orientation so it can perform the non-Cartesian interaction.
                mode = "interaction_proposal"

    if target is not None:
        gain = config.approach_gain if effective_phase == 0 else config.transport_gain
        command = _bounded_translation(
            target - eef,
            gain=gain,
            limit=config.max_translation_action,
        )
        # The rollout executes several open-loop actions before replanning.
        # Taper the command across the chunk to avoid overshooting a live
        # waypoint while retaining the first-step direction for diagnosis.
        taper = np.linspace(1.0, 0.2, output.shape[0], dtype=np.float32)
        output[:, :3] = command[None, :] * taper[:, None]
    if gripper_command is not None:
        output[:, -1] = float(gripper_command)
    if not np.isfinite(output).all():
        raise RuntimeError("Oracle phase servo produced non-finite actions.")

    subject_distance = None if subject is None else float(np.linalg.norm(eef - subject))
    goal_distance = None if goal is None else float(np.linalg.norm(eef - goal))
    action_delta_rms = float(np.sqrt(np.mean((output - actions) ** 2)))
    return output, {
        "enabled": True,
        "applied": bool(target is not None or gripper_command is not None),
        "mode": mode,
        "selected_clause": selected,
        "predicate_id": predicate_id,
        "requested_phase": requested_phase,
        "effective_phase": effective_phase,
        "subject_grasped": subject_is_grasped,
        "eef_position": eef.tolist(),
        "subject_position": None if subject is None else subject.tolist(),
        "goal_position": None if goal is None else goal.tolist(),
        "target_position": None if target is None else target.tolist(),
        "subject_distance_m": subject_distance,
        "goal_distance_m": goal_distance,
        "translation_action": output[0, :3].tolist(),
        "original_translation_action": original_translation[0].tolist(),
        "gripper_action": float(output[0, -1]),
        "original_gripper_action": float(original_gripper[0]),
        "action_delta_rms": action_delta_rms,
    }


def summarize_oracle_phase_servo(
    episode_records: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    records = [record for episode in episode_records for record in episode]
    applied = [record for record in records if bool(record.get("applied", False))]
    modes = Counter(str(record.get("mode", "unknown")) for record in records)
    progress = {"approach": [], "transport": []}
    for episode in episode_records:
        previous: Mapping[str, Any] | None = None
        for record in episode:
            if previous is not None and (
                record.get("selected_clause") == previous.get("selected_clause")
                and record.get("effective_phase") == previous.get("effective_phase")
            ):
                phase = int(record.get("effective_phase", -1))
                key = "approach" if phase == 0 else "transport" if phase == 1 else None
                distance_key = (
                    "subject_distance_m" if phase == 0 else "goal_distance_m"
                )
                before = previous.get(distance_key)
                after = record.get(distance_key)
                if key is not None and before is not None and after is not None:
                    progress[key].append(float(after) < float(before) - 1.0e-4)
            previous = record
    delta = [float(record["action_delta_rms"]) for record in applied]
    return {
        "enabled": True,
        "decisions": len(records),
        "applied_decisions": len(applied),
        "applied_rate": float(len(applied) / max(len(records), 1)),
        "mode_counts": dict(sorted(modes.items())),
        "action_delta_rms_mean": float(np.mean(delta)) if delta else None,
        "action_delta_rms_max": float(np.max(delta)) if delta else None,
        "approach_progress_samples": len(progress["approach"]),
        "approach_progress_rate": (
            float(np.mean(progress["approach"])) if progress["approach"] else None
        ),
        "transport_progress_samples": len(progress["transport"]),
        "transport_progress_rate": (
            float(np.mean(progress["transport"])) if progress["transport"] else None
        ),
    }
