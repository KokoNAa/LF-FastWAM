import unittest

import numpy as np

from experiments.libero.policy_guard_state import PolicyGuardStateController
from experiments.libero.completion_only_memory_audit import (
    build_completion_only_memory_report,
)
from experiments.libero.stateless_replan_audit import (
    build_stateless_replan_report,
)


class PolicyGuardStateControllerTest(unittest.TestCase):
    def test_recurrent_mode_returns_the_previous_model_state(self):
        controller = PolicyGuardStateController(reset_each_replan=False)
        state = {"phase_safe_memory_state_ids": [1, 0]}
        controller.accept_model_state(state)
        self.assertIs(controller.state_for_replan(), state)
        self.assertEqual(controller.mode, "recurrent")
        controller.reset_episode()
        self.assertIsNone(controller.state_for_replan())

    def test_stateless_mode_discards_every_model_state(self):
        controller = PolicyGuardStateController(reset_each_replan=True)
        controller.accept_model_state({"phase_safe_memory_state_ids": [1, 0]})
        self.assertIsNone(controller.state_for_replan())
        self.assertEqual(controller.mode, "reset_each_replan")

    def test_state_ablation_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "both reset-each-replan"):
            PolicyGuardStateController(
                reset_each_replan=True,
                completion_only=True,
            )

    def test_completion_only_keeps_a_monotonic_completed_bitset(self):
        controller = PolicyGuardStateController(completion_only=True)
        model_state = {
            "phase_safe_memory_state_ids": np.asarray([[3, 1, 2, 0]]),
            "phase_safe_memory_valid": np.ones((1, 4), dtype=bool),
        }
        controller.accept_model_state(model_state)
        np.testing.assert_array_equal(
            model_state["phase_safe_memory_state_ids"],
            np.asarray([[3, 1, 2, 0]]),
        )
        np.testing.assert_array_equal(
            model_state["phase_safe_memory_valid"],
            np.ones((1, 4), dtype=bool),
        )
        first = controller.state_for_replan()
        np.testing.assert_array_equal(
            first["phase_safe_memory_state_ids"],
            np.asarray([[3, 0, 0, 0]]),
        )
        np.testing.assert_array_equal(
            first["phase_safe_memory_valid"],
            np.asarray([[True, False, False, False]]),
        )

        controller.accept_model_state(
            {
                "phase_safe_memory_state_ids": np.asarray([[0, 3, 1, 2]]),
                "phase_safe_memory_valid": np.ones((1, 4), dtype=bool),
            }
        )
        second = controller.state_for_replan()
        np.testing.assert_array_equal(
            second["phase_safe_memory_state_ids"],
            np.asarray([[3, 3, 0, 0]]),
        )
        np.testing.assert_array_equal(
            second["phase_safe_memory_valid"],
            np.asarray([[True, True, False, False]]),
        )
        self.assertEqual(controller.mode, "completion_only")


class StatelessReplanReportTest(unittest.TestCase):
    @staticmethod
    def _records(previous_state_valid: bool = False):
        return [
            {
                "extended_diagnostics": {
                    "clauses": [
                        {
                            "phase_safe_memory_available": True,
                            "phase_safe_memory_previous_state_valid": (
                                previous_state_valid
                            ),
                        }
                    ]
                }
            }
            for _ in range(3)
        ]

    def test_report_requires_a_proven_cut_state_channel(self):
        report = build_stateless_replan_report(
            self._records(),
            enabled=True,
            phase_safe_memory={
                "postgrasp_clause_scheduler_accuracy": 0.95,
                "geometry_max_abs": 0.0,
            },
            action_integrity={
                "chunks": 3,
                "exact_rate": 1.0,
                "max_abs_error": 0.0,
            },
        )
        self.assertTrue(report["state_input_channel_cut"])
        self.assertTrue(report["passed"])

        leaked = build_stateless_replan_report(
            self._records(previous_state_valid=True),
            enabled=True,
            phase_safe_memory={
                "postgrasp_clause_scheduler_accuracy": 0.95,
                "geometry_max_abs": 0.0,
            },
            action_integrity={
                "chunks": 3,
                "exact_rate": 1.0,
                "max_abs_error": 0.0,
            },
        )
        self.assertFalse(leaked["state_input_channel_cut"])
        self.assertFalse(leaked["passed"])


class CompletionOnlyMemoryReportTest(unittest.TestCase):
    @staticmethod
    def _records(previous_states):
        return [
            {
                "extended_diagnostics": {
                    "clauses": [
                        {
                            "phase_safe_memory_available": True,
                            "phase_safe_memory_previous_state_valid": (
                                state is not None
                            ),
                            "phase_safe_memory_previous_state": state,
                            "status": (
                                "completed" if state == 3 else "initial_search"
                            ),
                        }
                    ]
                }
            }
            for state in previous_states
        ]

    def test_report_accepts_only_completed_recurrent_states(self):
        report = build_completion_only_memory_report(
            self._records([None, 3, 3]),
            enabled=True,
            phase_safe_memory={
                "postgrasp_clause_scheduler_accuracy": 0.95,
                "geometry_max_abs": 0.0,
                "completed_sticky_violation_rate": 0.0,
            },
            action_integrity={
                "chunks": 3,
                "exact_rate": 1.0,
                "max_abs_error": 0.0,
            },
        )
        self.assertEqual(report["completed_previous_state_count"], 2)
        self.assertEqual(
            report["noncompleted_previous_state_leakage_rate"], 0.0
        )
        self.assertTrue(report["passed"])

        leaked = build_completion_only_memory_report(
            self._records([None, 1, 3]),
            enabled=True,
            phase_safe_memory={
                "postgrasp_clause_scheduler_accuracy": 0.95,
                "geometry_max_abs": 0.0,
                "completed_sticky_violation_rate": 0.0,
            },
            action_integrity={
                "chunks": 3,
                "exact_rate": 1.0,
                "max_abs_error": 0.0,
            },
        )
        self.assertFalse(leaked["passed"])


if __name__ == "__main__":
    unittest.main()
