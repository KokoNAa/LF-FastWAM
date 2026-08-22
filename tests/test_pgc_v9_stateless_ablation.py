import unittest

from experiments.libero.policy_guard_state import PolicyGuardStateController
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


if __name__ == "__main__":
    unittest.main()
