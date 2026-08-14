"""Behavior-level diagnostics for counterfactual LIBERO rollouts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


COUNTERFACTUAL_BEHAVIOR_CATEGORIES = (
    "counterfactual_goal_success",
    "source_goal_success",
    "target_object_manipulated_placement_failure",
    "source_object_manipulated_no_completion",
    "other_object_manipulated",
    "no_object_manipulated",
)


def goal_subjects(goal_state: Sequence[Sequence[Any]]) -> set[str]:
    """Return the subject entity of each simple LIBERO goal predicate."""
    subjects: set[str] = set()
    for predicate in goal_state:
        if len(predicate) not in (2, 3):
            raise ValueError(
                "Only unary and binary LIBERO goal predicates are supported, "
                f"got {predicate!r}."
            )
        subjects.add(str(predicate[1]))
    return subjects


def classify_counterfactual_behavior(
    *,
    counterfactual_goal_achieved: bool,
    source_goal_achieved: bool,
    manipulated_objects: Iterable[str],
    counterfactual_target_objects: Iterable[str],
    source_target_objects: Iterable[str],
) -> str:
    """Assign one mutually exclusive behavioral outcome to an episode.

    Goal completion takes precedence over intermediate manipulation. When no
    goal is achieved, target-object interaction is kept separate from source
    persistence and unrelated-object mistakes so a zero CIS score can still be
    decomposed into useful failure modes.
    """
    manipulated = {str(name) for name in manipulated_objects}
    counterfactual_targets = {
        str(name) for name in counterfactual_target_objects
    }
    source_targets = {str(name) for name in source_target_objects}

    if counterfactual_goal_achieved:
        return "counterfactual_goal_success"
    if source_goal_achieved:
        return "source_goal_success"
    if manipulated & counterfactual_targets:
        return "target_object_manipulated_placement_failure"
    if manipulated & source_targets:
        return "source_object_manipulated_no_completion"
    if manipulated:
        return "other_object_manipulated"
    return "no_object_manipulated"


def empty_behavior_counts() -> dict[str, int]:
    """Return a stable zero-initialized counter for every category."""
    return {category: 0 for category in COUNTERFACTUAL_BEHAVIOR_CATEGORIES}


class CounterfactualEpisodeTracker:
    """Observe LIBERO predicates and object interactions without acting."""

    def __init__(
        self,
        env: Any,
        *,
        source_goal_state: list[list[Any]],
        counterfactual_goal_state: list[list[Any]],
        lift_threshold_m: float,
    ) -> None:
        inner_env = getattr(env, "env", None)
        if inner_env is None:
            raise TypeError(
                "Counterfactual diagnostics require a LIBERO wrapper with an "
                "inner environment."
            )
        if not hasattr(inner_env, "_eval_predicate"):
            raise TypeError(
                "Counterfactual diagnostics require LIBERO predicate evaluation."
            )
        if lift_threshold_m <= 0:
            raise ValueError(
                "EVALUATION.counterfactual_lift_threshold_m must be positive, "
                f"got {lift_threshold_m}."
            )

        self.inner_env = inner_env
        self.source_goal_state = [list(x) for x in source_goal_state]
        self.counterfactual_goal_state = [
            list(x) for x in counterfactual_goal_state
        ]
        self.source_target_objects = goal_subjects(self.source_goal_state)
        self.counterfactual_target_objects = goal_subjects(
            self.counterfactual_goal_state
        )
        self.lift_threshold_m = float(lift_threshold_m)

        objects_dict = getattr(inner_env, "objects_dict", {})
        obj_body_id = getattr(inner_env, "obj_body_id", {})
        self.trackable_objects = sorted(
            str(name) for name in objects_dict if name in obj_body_id
        )
        self.initial_object_z = {
            name: self._object_z(name) for name in self.trackable_objects
        }
        self.max_lift_delta_m = {name: 0.0 for name in self.trackable_objects}
        self.first_grasp_step: dict[str, int] = {}
        self.grasped_objects: set[str] = set()
        self.lifted_objects: set[str] = set()
        self.source_goal_achieved = False
        self.counterfactual_goal_achieved = False
        self.source_goal_final = False
        self.counterfactual_goal_final = False

    def _object_z(self, name: str) -> float:
        body_id = self.inner_env.obj_body_id[name]
        return float(self.inner_env.sim.data.body_xpos[body_id][2])

    def _goal_holds(self, goal_state: list[list[Any]]) -> bool:
        return all(
            bool(self.inner_env._eval_predicate(predicate))
            for predicate in goal_state
        )

    def _robot_grippers(self) -> list[Any]:
        gripper = self.inner_env.robots[0].gripper
        if isinstance(gripper, dict):
            return list(gripper.values())
        return [gripper]

    def _is_grasped(self, name: str) -> bool:
        object_model = self.inner_env.objects_dict[name]
        return any(
            bool(
                self.inner_env._check_grasp(
                    gripper=gripper,
                    object_geoms=object_model,
                )
            )
            for gripper in self._robot_grippers()
        )

    def observe(self, policy_step: int) -> None:
        self.source_goal_final = self._goal_holds(self.source_goal_state)
        self.counterfactual_goal_final = self._goal_holds(
            self.counterfactual_goal_state
        )
        self.source_goal_achieved |= self.source_goal_final
        self.counterfactual_goal_achieved |= self.counterfactual_goal_final

        for name in self.trackable_objects:
            lift_delta = self._object_z(name) - self.initial_object_z[name]
            self.max_lift_delta_m[name] = max(
                self.max_lift_delta_m[name],
                float(lift_delta),
            )
            if lift_delta >= self.lift_threshold_m:
                self.lifted_objects.add(name)
            if self._is_grasped(name):
                self.grasped_objects.add(name)
                self.first_grasp_step.setdefault(name, int(policy_step))

    def result(self, *, episode_idx: int) -> dict[str, Any]:
        manipulated_objects = self.grasped_objects | self.lifted_objects
        category = classify_counterfactual_behavior(
            counterfactual_goal_achieved=self.counterfactual_goal_achieved,
            source_goal_achieved=self.source_goal_achieved,
            manipulated_objects=manipulated_objects,
            counterfactual_target_objects=self.counterfactual_target_objects,
            source_target_objects=self.source_target_objects,
        )
        return {
            "episode": int(episode_idx),
            "category": category,
            "counterfactual_goal_achieved": bool(
                self.counterfactual_goal_achieved
            ),
            "source_goal_achieved": bool(self.source_goal_achieved),
            "counterfactual_goal_final": bool(self.counterfactual_goal_final),
            "source_goal_final": bool(self.source_goal_final),
            "counterfactual_target_objects": sorted(
                self.counterfactual_target_objects
            ),
            "source_target_objects": sorted(self.source_target_objects),
            "grasped_objects": sorted(self.grasped_objects),
            "lifted_objects": sorted(self.lifted_objects),
            "manipulated_objects": sorted(manipulated_objects),
            "first_grasp_step": dict(sorted(self.first_grasp_step.items())),
            "max_lift_delta_m": {
                name: float(delta)
                for name, delta in sorted(self.max_lift_delta_m.items())
                if delta > 0
            },
        }
