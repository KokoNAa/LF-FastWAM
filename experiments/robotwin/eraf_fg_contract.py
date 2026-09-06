"""Auditable failure-replay and acceptance rules for the ERAF/FG experiment.

This is a new protocol, not the historical V9.39 source-directed-only pool.
No simulator or model dependency is needed to validate its records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

FORMAT = "robotwin_eraf_fg_failure_replay_v1"
TARGET_TASKS = ("place_a2b_left", "blocks_ranking_rgb")
CF_MINIMUM = {
    "place_a2b_left": 1, "place_a2b_right": 2, "place_burger_fries": 1,
    "stack_blocks_two": 2, "blocks_ranking_rgb": 1,
}
FAILURE_ORIGINS = {
    "source_directed": "source_directed_failure_replan",
    "neither_goal": "goal_incomplete_failure_replan",
}
CAMERAS = ("head_camera", "left_camera", "right_camera")


def scene_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return str(row["source_task"]), str(row["task_config"]), int(row["scene_seed"])


def failure_kind(*, source_ever_success: bool, cf_ever_success: bool) -> str:
    if not isinstance(source_ever_success, bool) or not isinstance(cf_ever_success, bool):
        raise ValueError("Failure classification requires explicit boolean goal audits.")
    if cf_ever_success:
        raise ValueError("A successful CF episode is not a failure-correction example.")
    return "source_directed" if source_ever_success else "neither_goal"


def assert_scene_disjoint(
    records: Iterable[Mapping[str, Any]], excluded: Iterable[tuple[str, str, int]],
) -> None:
    blocked = set(excluded)
    overlap = sorted({scene_key(row) for row in records} & blocked)
    if overlap:
        raise ValueError(f"Training/evaluation scene overlap: {overlap}")


def verify_replayed_state(expected: np.ndarray, actual: np.ndarray, atol: float = 1e-7) -> float:
    expected, actual = np.asarray(expected), np.asarray(actual)
    if not 0 <= atol <= 1e-7:
        raise ValueError("Replay tolerance must be in [0, 1e-7].")
    if expected.shape != actual.shape or not expected.size:
        raise ValueError("Replay state shape changed or is empty.")
    if not (np.isfinite(expected).all() and np.isfinite(actual).all()):
        raise ValueError("Non-finite replay state.")
    error = float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64))))
    if error > atol:
        raise ValueError(f"Policy-prefix replay drift: max_abs={error} > {atol}.")
    return error


def candidate_replans(steps: Iterable[int], *, limit: int = 20) -> list[int]:
    """Try late states first while retaining early recoverable states.

    Episode initialization (step zero) is deliberately ineligible for FG.
    Uniform subsampling prevents all candidates clustering after an object
    has irreversibly fallen off the table.
    """
    values = sorted(set(int(step) for step in steps if int(step) > 0))
    if not 1 <= limit <= 20:
        raise ValueError("At most 20 correction candidates are allowed.")
    if len(values) > limit:
        indices = np.rint(np.linspace(0, len(values) - 1, limit)).astype(int)
        values = [values[index] for index in indices]
    return list(reversed(values))


@dataclass(frozen=True)
class ActionWindow:
    start: int
    action: np.ndarray
    valid: np.ndarray


def action_windows(actions: np.ndarray, *, supervision: str = "full", stride: int = 8) -> list[ActionWindow]:
    """Keep the real goal-reaching tail; pad storage only, never supervision.

    The local ablation starts at the identical failure state but may see only
    the first 12 corrective actions. It cannot obtain later windows.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < 12:
        raise ValueError("Corrections require at least 12 real 14-D actions.")
    if not np.isfinite(actions).all() or stride < 1:
        raise ValueError("Non-finite actions or invalid window stride.")
    if supervision not in {"full", "local"}:
        raise ValueError("Expected full or local correction supervision.")
    length = len(actions) if supervision == "full" else 12
    starts = list(range(0, length, stride)) if supervision == "full" else [0]
    result = []
    for start in starts:
        count = min(32, length - start)
        values = np.zeros((32, 14), dtype=np.float32)
        values[:count] = actions[start:start + count]
        valid = np.arange(32) < count
        result.append(ActionWindow(start, values, valid))
    return result


def validate_correction(record: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(record)
    if row.get("format") != FORMAT:
        raise ValueError("Unknown failure-replay protocol.")
    if scene_key(row)[0] not in TARGET_TASKS:
        raise ValueError("This corrective experiment targets left placement and ranking.")
    kind = failure_kind(source_ever_success=row["source_goal_ever_success"],
                        cf_ever_success=row["counterfactual_goal_ever_success"])
    if row.get("failure_kind") != kind or row.get("capture_origin") != FAILURE_ORIGINS[kind]:
        raise ValueError("Failure provenance contradicts the rollout goal audit.")
    if int(row.get("capture_action_index", 0)) <= 0:
        raise ValueError("FG must start after actual policy actions, not at initialization.")
    if int(row.get("prefix_action_count", -1)) != int(row["capture_action_index"]):
        raise ValueError("Incomplete executed policy prefix.")
    for key in ("initial_state_sha256", "capture_state_sha256", "prefix_action_sha256",
                "correction_action_sha256", "rollout_checkpoint_sha256"):
        value = row.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing/invalid identity hash: {key}.")
    if row.get("verification_policy") != "full_goal" or row.get("full_goal_verified") is not True:
        raise ValueError("Partial or unverified corrections are not FG positives.")
    if row.get("counterfactual_goal_final_success") is not True:
        raise ValueError("The complete CF predicate must hold at the verified endpoint.")
    if row.get("both_grippers_open_final") is not True:
        raise ValueError("A held object does not establish full-goal completion.")
    if int(row.get("verified_replay_count", 0)) < 2:
        raise ValueError("A correction must succeed in two independent prefix replays.")
    if not 1 <= int(row.get("captured_state_count", 0)) <= 64:
        raise ValueError("Invalid capture count.")
    if not 1 <= int(row.get("candidate_count", 0)) <= 20:
        raise ValueError("Invalid corrective candidate count.")
    if int(row.get("recorded_action_count", 0)) < 12:
        raise ValueError("Fewer than 12 real corrective actions.")
    atol = float(row.get("state_atol", float("inf")))
    error = float(row.get("replay_state_max_abs", float("inf")))
    if not (0 <= error <= atol <= 1e-7):
        raise ValueError("Replay-state verification failed.")
    if row.get("replay_split") not in {"train", "replay_holdout"}:
        raise ValueError("Corrective data must have a scene-isolated data split.")
    return row


def acceptance(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Use complete episode counts, never a partial success-rate proxy."""
    cells = summary.get("cells", [])
    by_key = {(r["source_task"], r["condition"]): r for r in cells}
    expected = {(task, condition) for task in CF_MINIMUM for condition in ("correct", "counterfactual")}
    if summary.get("complete") is not True or set(by_key) != expected or len(cells) != len(expected):
        raise ValueError("Acceptance needs a complete five-task Correct/CF matrix.")
    for row in cells:
        n, k = int(row["episodes"]), int(row["selected_goal_successes"])
        if n != 3 or not 0 <= k <= n or row.get("task_config") != "demo_clean":
            raise ValueError("Unexpected regression evaluation cell.")
    correct = sum(int(by_key[t, "correct"]["selected_goal_successes"]) for t in CF_MINIMUM)
    cf = {t: int(by_key[t, "counterfactual"]["selected_goal_successes"]) for t in CF_MINIMUM}
    failures = (["correct_below_10"] if correct < 10 else [])
    failures += [f"cf_below_minimum:{t}" for t, minimum in CF_MINIMUM.items() if cf[t] < minimum]
    return {"eligible": not failures, "correct": correct, "counterfactual": sum(cf.values()),
            "cf_by_task": cf, "unmet_requirements": failures}
