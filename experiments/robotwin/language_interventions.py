"""RoboTwin language-intervention and counterfactual-goal contracts.

The module is intentionally independent of SAPIEN so manifests and goal logic
can be validated on a CPU workstation.  Runtime environments only need to
expose actors with ``get_pose().p`` and the standard RoboTwin robot gripper
queries.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MANIFEST_FORMAT = "robotwin_language_interventions_v1"
EPISODE_FORMAT = "robotwin_language_intervention_episode_v1"
VALID_CONDITIONS = ("correct", "shuffled", "counterfactual")
VALID_DIRECTIONS = ("negative", "positive")
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class ManifestError(ValueError):
    """Raised when a RoboTwin intervention manifest is not executable."""


@dataclass(frozen=True)
class InterventionPair:
    pair_id: str
    source_task: str
    counterfactual_task: str
    source_goal: dict[str, Any]
    counterfactual_goal: dict[str, Any]


@dataclass(frozen=True)
class GoalEvaluation:
    success: bool
    details: dict[str, Any]


@dataclass(frozen=True)
class GoalSnapshot:
    source: GoalEvaluation
    counterfactual: GoalEvaluation

    def selected(self, goal_name: str) -> GoalEvaluation:
        if goal_name == "source":
            return self.source
        if goal_name == "counterfactual":
            return self.counterfactual
        raise ValueError(f"Unsupported goal name: {goal_name!r}")


def normalize_condition(value: Any) -> str:
    condition = str(value if value is not None else "correct").strip().lower()
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"Unsupported RoboTwin instruction condition {value!r}; "
            f"expected one of {list(VALID_CONDITIONS)}."
        )
    return condition


def condition_contract(condition: Any) -> tuple[str, str]:
    """Return ``(instruction_goal, selected_success_goal)`` for a condition."""

    normalized = normalize_condition(condition)
    if normalized == "correct":
        return "source", "source"
    if normalized == "shuffled":
        return "counterfactual", "source"
    return "counterfactual", "counterfactual"


def stable_instruction_seed(
    *, scene_seed: int, task_name: str, instruction_type: str
) -> int:
    """Build a process-independent seed for matched language generation."""

    payload = f"{int(scene_seed)}\0{task_name}\0{instruction_type}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _nonempty(record: Mapping[str, Any], field: str, *, context: str) -> str:
    value = str(record.get(field, "")).strip()
    if not value:
        raise ManifestError(f"{context}: missing non-empty `{field}`")
    return value


def _validate_goal_spec(raw: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{context}: goal must be a JSON object")
    goal = dict(raw)
    if goal.get("type") != "relative_pose":
        raise ManifestError(
            f"{context}: unsupported goal type {goal.get('type')!r}; "
            "expected 'relative_pose'"
        )
    for field in ("actor", "reference"):
        _nonempty(goal, field, context=context)
    axis = str(goal.get("axis", "")).strip().lower()
    if axis not in AXIS_INDEX:
        raise ManifestError(f"{context}: `axis` must be one of {sorted(AXIS_INDEX)}")
    if axis == "z":
        raise ManifestError(f"{context}: relative_pose CIS currently requires x/y axis")
    direction = str(goal.get("direction", "")).strip().lower()
    if direction not in VALID_DIRECTIONS:
        raise ManifestError(
            f"{context}: `direction` must be one of {list(VALID_DIRECTIONS)}"
        )
    for field in (
        "min_planar_distance",
        "max_planar_distance",
        "max_orthogonal_distance",
    ):
        try:
            value = float(goal[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"{context}: `{field}` must be numeric") from exc
        if value < 0:
            raise ManifestError(f"{context}: `{field}` must be non-negative")
        goal[field] = value
    if goal["max_planar_distance"] <= goal["min_planar_distance"]:
        raise ManifestError(
            f"{context}: max_planar_distance must exceed min_planar_distance"
        )
    require_open = goal.get("require_both_grippers_open", False)
    if not isinstance(require_open, bool):
        raise ManifestError(
            f"{context}: `require_both_grippers_open` must be a boolean"
        )
    goal["axis"] = axis
    goal["direction"] = direction
    return goal


def validate_manifest_data(raw: Any) -> list[InterventionPair]:
    if not isinstance(raw, Mapping):
        raise ManifestError("manifest root must be a JSON object")
    if raw.get("format") != MANIFEST_FORMAT:
        raise ManifestError(
            f"manifest `format` must be {MANIFEST_FORMAT!r}, got {raw.get('format')!r}"
        )
    raw_pairs = raw.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ManifestError("manifest `pairs` must be a non-empty list")

    pairs: list[InterventionPair] = []
    pair_ids: set[str] = set()
    source_tasks: set[str] = set()
    for index, raw_pair in enumerate(raw_pairs):
        context = f"pairs[{index}]"
        if not isinstance(raw_pair, Mapping):
            raise ManifestError(f"{context}: pair must be a JSON object")
        pair_id = _nonempty(raw_pair, "pair_id", context=context)
        source_task = _nonempty(raw_pair, "source_task", context=context)
        counterfactual_task = _nonempty(
            raw_pair, "counterfactual_task", context=context
        )
        if pair_id in pair_ids:
            raise ManifestError(f"{context}: duplicate pair_id {pair_id!r}")
        if source_task in source_tasks:
            raise ManifestError(f"{context}: duplicate source_task {source_task!r}")
        if source_task == counterfactual_task:
            raise ManifestError(
                f"{context}: counterfactual task must differ from source"
            )
        source_goal = _validate_goal_spec(
            raw_pair.get("source_goal"), context=f"{context}.source_goal"
        )
        counterfactual_goal = _validate_goal_spec(
            raw_pair.get("counterfactual_goal"),
            context=f"{context}.counterfactual_goal",
        )
        if source_goal == counterfactual_goal:
            raise ManifestError(f"{context}: source and counterfactual goals are equal")
        pair_ids.add(pair_id)
        source_tasks.add(source_task)
        pairs.append(
            InterventionPair(
                pair_id=pair_id,
                source_task=source_task,
                counterfactual_task=counterfactual_task,
                source_goal=source_goal,
                counterfactual_goal=counterfactual_goal,
            )
        )

    by_source = {pair.source_task: pair for pair in pairs}
    for pair in pairs:
        reverse = by_source.get(pair.counterfactual_task)
        if reverse is None:
            raise ManifestError(
                f"pair {pair.pair_id!r}: missing reverse source pair for "
                f"{pair.counterfactual_task!r}"
            )
        if reverse.counterfactual_task != pair.source_task:
            raise ManifestError(
                f"pair {pair.pair_id!r}: reverse pair does not point back to "
                f"{pair.source_task!r}"
            )
        if reverse.source_goal != pair.counterfactual_goal:
            raise ManifestError(
                f"pair {pair.pair_id!r}: reverse source goal does not match "
                "the forward counterfactual goal"
            )
        if reverse.counterfactual_goal != pair.source_goal:
            raise ManifestError(
                f"pair {pair.pair_id!r}: reverse counterfactual goal does not "
                "match the forward source goal"
            )
    return pairs


def load_intervention_manifest(
    path: str | Path, *, robotwin_root: str | Path | None = None
) -> list[InterventionPair]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    pairs = validate_manifest_data(raw)
    if robotwin_root is not None:
        root = Path(robotwin_root).expanduser().resolve()
        missing: list[Path] = []
        task_names = {
            task
            for pair in pairs
            for task in (pair.source_task, pair.counterfactual_task)
        }
        for task_name in sorted(task_names):
            for relative in (
                Path("envs") / f"{task_name}.py",
                Path("description") / "task_instruction" / f"{task_name}.json",
            ):
                candidate = root / relative
                if not candidate.is_file():
                    missing.append(candidate)
        if missing:
            formatted = "\n".join(str(path) for path in missing)
            raise ManifestError(
                f"manifest references missing RoboTwin files:\n{formatted}"
            )
        for pair in pairs:
            source_path = (
                root / "description" / "task_instruction" / f"{pair.source_task}.json"
            )
            counterfactual_path = (
                root
                / "description"
                / "task_instruction"
                / f"{pair.counterfactual_task}.json"
            )
            with source_path.open("r", encoding="utf-8") as handle:
                source_instructions = json.load(handle)
            with counterfactual_path.open("r", encoding="utf-8") as handle:
                counterfactual_instructions = json.load(handle)
            for instruction_type in ("seen", "unseen"):
                source_placeholder_sets = {
                    frozenset(re.findall(r"{([^}]+)}", str(template)))
                    for template in source_instructions.get(instruction_type, [])
                }
                counterfactual_placeholder_sets = {
                    frozenset(re.findall(r"{([^}]+)}", str(template)))
                    for template in counterfactual_instructions.get(
                        instruction_type, []
                    )
                }
                if not (source_placeholder_sets & counterfactual_placeholder_sets):
                    raise ManifestError(
                        f"pair {pair.pair_id!r}: source/counterfactual "
                        f"{instruction_type} templates have no compatible "
                        "placeholder set"
                    )
    return pairs


def select_intervention_pair(
    pairs: list[InterventionPair], *, source_task: str
) -> InterventionPair:
    matches = [pair for pair in pairs if pair.source_task == source_task]
    if len(matches) != 1:
        raise ManifestError(
            f"expected exactly one intervention pair for source task "
            f"{source_task!r}, found {len(matches)}"
        )
    return matches[0]


def load_matched_episode_records(
    path: str | Path,
    *,
    expected_pair_id: str,
    expected_source_task: str,
    expected_counterfactual_task: str,
    expected_task_config: str,
    expected_instruction_type: str,
    expected_episodes: int,
) -> list[dict[str, Any]]:
    """Load canonical Correct records used to replay matched CIS scenes."""

    records_path = Path(path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid matched episode JSON at {records_path}:{line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Matched episode at {records_path}:{line_number} must be an object"
                )
            records.append(record)

    if len(records) != int(expected_episodes):
        raise ValueError(
            f"Matched episode file {records_path} has {len(records)} records; "
            f"expected {expected_episodes}"
        )

    expected_fields = {
        "format": EPISODE_FORMAT,
        "pair_id": str(expected_pair_id),
        "source_task": str(expected_source_task),
        "counterfactual_task": str(expected_counterfactual_task),
        "task_config": str(expected_task_config),
        "condition": "correct",
        "instruction_type": str(expected_instruction_type),
    }
    seen_seeds: set[int] = set()
    for index, record in enumerate(records):
        for field, expected in expected_fields.items():
            if record.get(field) != expected:
                raise ValueError(
                    f"Matched episode {index} field {field!r} is "
                    f"{record.get(field)!r}; expected {expected!r}"
                )
        try:
            seed = int(record["scene_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Matched episode {index} has an invalid scene_seed"
            ) from exc
        if seed in seen_seeds:
            raise ValueError(f"Matched episode file contains duplicate seed {seed}")
        seen_seeds.add(seed)
        if record.get("episode_index") != index:
            raise ValueError(
                f"Matched episode {index} has episode_index="
                f"{record.get('episode_index')!r}"
            )
        for field in ("source_instruction", "counterfactual_instruction"):
            if not str(record.get(field, "")).strip():
                raise ValueError(f"Matched episode {index} has empty {field}")
        if record.get("initial_source_goal_success") is not False:
            raise ValueError(f"Matched episode {index} has a true initial source goal")
        if record.get("initial_counterfactual_goal_success") is not False:
            raise ValueError(
                f"Matched episode {index} has a true initial counterfactual goal"
            )
        if record.get("instruction_goal") != "source":
            raise ValueError(
                f"Matched episode {index} is not a Correct/source instruction record"
            )
        if record.get("selected_goal") != "source":
            raise ValueError(
                f"Matched episode {index} is not a Correct/source goal record"
            )
        if record.get("policy_instruction") != record.get("source_instruction"):
            raise ValueError(
                f"Matched episode {index} does not use its source instruction"
            )
    return records


def _actor_position(env: Any, attribute: str) -> tuple[float, float, float]:
    if not hasattr(env, attribute):
        raise AttributeError(
            f"RoboTwin environment {type(env).__name__} has no actor {attribute!r}"
        )
    actor = getattr(env, attribute)
    pose = actor.get_pose()
    position = pose.p
    if len(position) < 3:
        raise ValueError(f"Actor {attribute!r} returned an invalid pose: {position!r}")
    return float(position[0]), float(position[1]), float(position[2])


def evaluate_goal(env: Any, goal: Mapping[str, Any]) -> GoalEvaluation:
    """Evaluate one validated declarative goal against a RoboTwin environment."""

    if goal.get("type") != "relative_pose":
        raise ValueError(f"Unsupported goal type at runtime: {goal.get('type')!r}")
    actor_position = _actor_position(env, str(goal["actor"]))
    reference_position = _actor_position(env, str(goal["reference"]))
    delta = tuple(
        actor_position[index] - reference_position[index] for index in range(3)
    )
    axis = str(goal["axis"])
    axis_index = AXIS_INDEX[axis]
    orthogonal_axis = "y" if axis == "x" else "x"
    orthogonal_index = AXIS_INDEX[orthogonal_axis]
    planar_distance = math.hypot(delta[0], delta[1])
    direction_ok = (
        delta[axis_index] < 0
        if goal["direction"] == "negative"
        else delta[axis_index] > 0
    )
    grippers_open = True
    if goal.get("require_both_grippers_open", False):
        robot = getattr(env, "robot", None)
        if robot is None:
            raise AttributeError("RoboTwin environment has no `robot` for gripper goal")
        grippers_open = bool(
            robot.is_left_gripper_open() and robot.is_right_gripper_open()
        )
    success = bool(
        direction_ok
        and float(goal["min_planar_distance"])
        < planar_distance
        < float(goal["max_planar_distance"])
        and abs(delta[orthogonal_index]) < float(goal["max_orthogonal_distance"])
        and grippers_open
    )
    return GoalEvaluation(
        success=success,
        details={
            "actor_position": list(actor_position),
            "reference_position": list(reference_position),
            "delta": list(delta),
            "axis": axis,
            "direction": str(goal["direction"]),
            "axis_delta": float(delta[axis_index]),
            "orthogonal_delta": float(delta[orthogonal_index]),
            "planar_distance": float(planar_distance),
            "grippers_open": bool(grippers_open),
        },
    )


class GoalObserver:
    """Track source and alternate goal achievement throughout one rollout."""

    def __init__(self, env: Any, pair: InterventionPair):
        self.env = env
        self.pair = pair
        self._native_check_success = env.check_success
        self._selected_goal_installed = False
        self.evaluation_calls = 0
        self.source_ever_success = False
        self.counterfactual_ever_success = False
        self.last_snapshot: GoalSnapshot | None = None

    def update(self) -> GoalSnapshot:
        snapshot = GoalSnapshot(
            source=evaluate_goal(self.env, self.pair.source_goal),
            counterfactual=evaluate_goal(self.env, self.pair.counterfactual_goal),
        )
        self.evaluation_calls += 1
        self.source_ever_success = bool(
            self.source_ever_success or snapshot.source.success
        )
        self.counterfactual_ever_success = bool(
            self.counterfactual_ever_success or snapshot.counterfactual.success
        )
        self.last_snapshot = snapshot
        return snapshot

    def install_selected_goal(self, goal_name: str) -> None:
        if goal_name not in {"source", "counterfactual"}:
            raise ValueError(f"Unsupported selected goal: {goal_name!r}")
        if self._selected_goal_installed:
            raise RuntimeError("A selected RoboTwin goal is already installed")

        def selected_check_success() -> bool:
            return self.update().selected(goal_name).success

        self.env.check_success = selected_check_success
        self._selected_goal_installed = True

    def restore_native_goal(self) -> None:
        """Restore the environment's native success predicate after rollout."""

        if self._selected_goal_installed:
            self.env.check_success = self._native_check_success
            self._selected_goal_installed = False

    def episode_diagnostics(self, *, selected_goal: str) -> dict[str, Any]:
        snapshot = self.update()
        return {
            "selected_goal": selected_goal,
            "selected_goal_success": bool(
                self.source_ever_success
                if selected_goal == "source"
                else self.counterfactual_ever_success
            ),
            "source_goal_ever_success": bool(self.source_ever_success),
            "counterfactual_goal_ever_success": bool(self.counterfactual_ever_success),
            "source_goal_final_success": bool(snapshot.source.success),
            "counterfactual_goal_final_success": bool(snapshot.counterfactual.success),
            "goal_evaluation_calls": int(self.evaluation_calls),
            "final_source_goal": snapshot.source.details,
            "final_counterfactual_goal": snapshot.counterfactual.details,
        }
