"""Caller-owned policy-guard state controls for LIBERO evaluation.

The stateless mode is an evaluation ablation.  It deliberately prevents a
checkpoint from observing its own previous policy state at the next replan,
without changing the checkpoint or any model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


RECURRENT_POLICY_STATE_MODE = "recurrent"
STATELESS_REPLAN_POLICY_STATE_MODE = "reset_each_replan"
COMPLETION_ONLY_POLICY_STATE_MODE = "completion_only"

PENDING_STATE_ID = 0
COMPLETED_STATE_ID = 3


def _clone_tensor_like(value: Any) -> Any:
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(value, "copy", None)
    if callable(copy):
        return copy()
    raise TypeError("Policy state values must provide clone() or copy().")


def _as_bool_tensor_like(value: Any) -> Any:
    bool_method = getattr(value, "bool", None)
    if callable(bool_method):
        return bool_method()
    astype = getattr(value, "astype", None)
    if callable(astype):
        return astype(bool, copy=False)
    raise TypeError("Policy-state validity must be tensor-like.")


def _as_long_tensor_like(value: Any) -> Any:
    long_method = getattr(value, "long", None)
    if callable(long_method):
        return long_method()
    astype = getattr(value, "astype", None)
    if callable(astype):
        return astype("int64", copy=False)
    raise TypeError("Policy-state IDs must be tensor-like.")


@dataclass
class PolicyGuardStateController:
    """Own policy state for exactly one rollout episode."""

    reset_each_replan: bool = False
    completion_only: bool = False
    _state: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.reset_each_replan and self.completion_only:
            raise ValueError(
                "Policy state cannot be both reset-each-replan and completion-only."
            )

    @property
    def mode(self) -> str:
        if self.reset_each_replan:
            return STATELESS_REPLAN_POLICY_STATE_MODE
        if self.completion_only:
            return COMPLETION_ONLY_POLICY_STATE_MODE
        return RECURRENT_POLICY_STATE_MODE

    def state_for_replan(self) -> Optional[dict[str, Any]]:
        """Return the state visible to the next model invocation."""
        return None if self.reset_each_replan else self._state

    def accept_model_state(self, state: Optional[dict[str, Any]]) -> None:
        """Retain a model output only when recurrent state is enabled."""
        if self.reset_each_replan or state is None:
            self._state = None
            return
        if not self.completion_only:
            self._state = state
            return
        self._state = self._completion_only_state(state)

    def _completion_only_state(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        state_ids = state.get("phase_safe_memory_state_ids")
        state_valid = state.get("phase_safe_memory_valid")
        if state_ids is None or state_valid is None:
            raise ValueError(
                "Completion-only policy state requires state IDs and validity."
            )
        sanitized_ids = _clone_tensor_like(state_ids)
        sanitized_valid = _clone_tensor_like(state_valid)
        completed = _as_bool_tensor_like(state_valid) & (
            _as_long_tensor_like(state_ids) == COMPLETED_STATE_ID
        )
        if self._state is not None:
            previous_ids = self._state["phase_safe_memory_state_ids"]
            previous_valid = self._state["phase_safe_memory_valid"]
            completed = completed | (
                _as_bool_tensor_like(previous_valid)
                & (_as_long_tensor_like(previous_ids) == COMPLETED_STATE_ID)
            )
        sanitized_ids[...] = PENDING_STATE_ID
        sanitized_valid[...] = False
        sanitized_ids[completed] = COMPLETED_STATE_ID
        sanitized_valid[completed] = True
        return {
            **state,
            "phase_safe_memory_state_ids": sanitized_ids,
            "phase_safe_memory_valid": sanitized_valid,
        }

    def reset_episode(self) -> None:
        self._state = None
