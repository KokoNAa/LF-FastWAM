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


@dataclass
class PolicyGuardStateController:
    """Own policy state for exactly one rollout episode."""

    reset_each_replan: bool = False
    _state: Optional[dict[str, Any]] = None

    @property
    def mode(self) -> str:
        return (
            STATELESS_REPLAN_POLICY_STATE_MODE
            if self.reset_each_replan
            else RECURRENT_POLICY_STATE_MODE
        )

    def state_for_replan(self) -> Optional[dict[str, Any]]:
        """Return the state visible to the next model invocation."""
        return None if self.reset_each_replan else self._state

    def accept_model_state(self, state: Optional[dict[str, Any]]) -> None:
        """Retain a model output only when recurrent state is enabled."""
        self._state = None if self.reset_each_replan else state

    def reset_episode(self) -> None:
        self._state = None
