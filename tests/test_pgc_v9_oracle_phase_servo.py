import unittest

import numpy as np

from experiments.libero.oracle_phase_servo import (
    OraclePhaseServoConfig,
    apply_oracle_phase_servo,
    summarize_oracle_phase_servo,
)


LOWER = np.zeros(3, dtype=np.float32)
UPPER = np.ones(3, dtype=np.float32)


def normalized(position):
    return np.asarray(position, dtype=np.float32) * 2.0 - 1.0


def oracle(*, subject, goal, phase=0, grasped=False, predicate=1):
    return {
        "clause_valid": np.asarray([True, False]),
        "predicate_truth": np.asarray([False, False]),
        "predicate_ids": np.asarray([predicate, 0]),
        "phase_ids": np.asarray([phase, 0]),
        "subject_positions": np.stack([normalized(subject), np.zeros(3)]),
        "goal_anchors": np.stack([normalized(goal), np.zeros(3)]),
        "subject_position_valid": np.asarray([True, False]),
        "goal_anchor_valid": np.asarray([True, False]),
        "subject_grasped": np.asarray([grasped, False]),
    }


class OraclePhaseServoTest(unittest.TestCase):
    def setUp(self):
        self.actions = np.zeros((5, 7), dtype=np.float32)
        self.actions[:, 3:6] = np.asarray([0.2, -0.1, 0.3])
        self.actions[:, -1] = 0.25
        self.config = OraclePhaseServoConfig(enabled=True)

    def apply(self, *, eef, payload, config=None):
        return apply_oracle_phase_servo(
            self.actions,
            obs={"robot0_eef_pos": np.asarray(eef, dtype=np.float32)},
            oracle=payload,
            workspace_min=LOWER,
            workspace_max=UPPER,
            config=self.config if config is None else config,
        )

    def test_disabled_is_exact(self):
        output, diagnostics = self.apply(
            eef=[0.1, 0.1, 0.2],
            payload=oracle(subject=[0.5, 0.5, 0.2], goal=[0.8, 0.8, 0.2]),
            config=OraclePhaseServoConfig(enabled=False),
        )
        self.assertTrue(np.array_equal(output, self.actions))
        self.assertFalse(diagnostics["applied"])

    def test_approach_moves_toward_subject_with_open_gripper(self):
        output, diagnostics = self.apply(
            eef=[0.1, 0.1, 0.2],
            payload=oracle(subject=[0.5, 0.5, 0.2], goal=[0.8, 0.8, 0.2]),
        )
        self.assertEqual(diagnostics["mode"], "approach_hover")
        self.assertGreater(float(output[0, 0]), 0)
        self.assertGreater(float(output[0, 1]), 0)
        self.assertLessEqual(
            float(np.linalg.norm(output[0, :3])),
            self.config.max_translation_action + 1.0e-6,
        )
        self.assertTrue(np.all(output[:, -1] == -1.0))
        np.testing.assert_array_equal(output[:, 3:6], self.actions[:, 3:6])

    def test_close_gripper_at_grasp_target(self):
        subject = np.asarray([0.5, 0.5, 0.2])
        output, diagnostics = self.apply(
            eef=subject + np.asarray([0.0, 0.0, self.config.grasp_offset_m]),
            payload=oracle(subject=subject, goal=[0.8, 0.8, 0.2]),
        )
        self.assertEqual(diagnostics["mode"], "grasp_close")
        np.testing.assert_allclose(output[:, :3], 0.0)
        self.assertTrue(np.all(output[:, -1] == 1.0))

    def test_holding_transports_with_closed_gripper(self):
        output, diagnostics = self.apply(
            eef=[0.2, 0.2, 0.3],
            payload=oracle(
                subject=[0.2, 0.2, 0.2],
                goal=[0.8, 0.8, 0.2],
                phase=1,
                grasped=True,
            ),
        )
        self.assertEqual(diagnostics["mode"], "transport_hover")
        self.assertTrue(np.all(output[:, -1] == 1.0))
        self.assertGreater(float(output[0, 0]), 0)
        self.assertGreater(float(output[0, 1]), 0)

    def test_release_opens_gripper_near_goal(self):
        goal = np.asarray([0.8, 0.8, 0.2])
        output, diagnostics = self.apply(
            eef=goal + np.asarray([0.0, 0.0, self.config.release_height_m]),
            payload=oracle(
                subject=[0.2, 0.2, 0.2], goal=goal, phase=1, grasped=True
            ),
        )
        self.assertEqual(diagnostics["mode"], "release_open")
        self.assertTrue(np.all(output[:, -1] == -1.0))

    def test_transport_proposal_release_preserves_acquisition(self):
        output, diagnostics = self.apply(
            eef=[0.1, 0.1, 0.2],
            payload=oracle(subject=[0.5, 0.5, 0.2], goal=[0.8, 0.8, 0.2]),
            config=OraclePhaseServoConfig(
                enabled=True, scope="transport_proposal_release"
            ),
        )
        self.assertEqual(diagnostics["mode"], "proposal_acquisition")
        self.assertFalse(diagnostics["applied"])
        self.assertTrue(np.array_equal(output, self.actions))

    def test_transport_proposal_release_holds_then_returns_release_to_proposal(self):
        config = OraclePhaseServoConfig(
            enabled=True, scope="transport_proposal_release"
        )
        goal = np.asarray([0.8, 0.8, 0.2])
        output, diagnostics = self.apply(
            eef=[0.2, 0.2, 0.3],
            payload=oracle(
                subject=[0.2, 0.2, 0.2], goal=goal, phase=1, grasped=True
            ),
            config=config,
        )
        self.assertEqual(diagnostics["mode"], "transport_hover")
        self.assertTrue(np.all(output[:, -1] == 1.0))

        output, diagnostics = self.apply(
            eef=goal + np.asarray([0.0, 0.0, config.release_height_m]),
            payload=oracle(
                subject=[0.2, 0.2, 0.2], goal=goal, phase=1, grasped=True
            ),
            config=config,
        )
        self.assertEqual(diagnostics["mode"], "release_proposal")
        self.assertFalse(diagnostics["applied"])
        self.assertTrue(np.array_equal(output, self.actions))

    def test_transport_oracle_release_preserves_acquisition_and_opens(self):
        config = OraclePhaseServoConfig(
            enabled=True, scope="transport_oracle_release"
        )
        subject = np.asarray([0.5, 0.5, 0.2])
        goal = np.asarray([0.8, 0.8, 0.2])
        output, diagnostics = self.apply(
            eef=subject,
            payload=oracle(subject=subject, goal=goal),
            config=config,
        )
        self.assertEqual(diagnostics["mode"], "proposal_acquisition")
        self.assertTrue(np.array_equal(output, self.actions))

        output, diagnostics = self.apply(
            eef=goal + np.asarray([0.0, 0.0, config.release_height_m]),
            payload=oracle(
                subject=subject, goal=goal, phase=1, grasped=True
            ),
            config=config,
        )
        self.assertEqual(diagnostics["mode"], "release_open")
        self.assertTrue(np.all(output[:, -1] == -1.0))

    def test_transport_scopes_leave_interactions_to_proposal(self):
        output, diagnostics = self.apply(
            eef=[0.1, 0.1, 0.2],
            payload=oracle(
                subject=[0.5, 0.5, 0.2], goal=[0.8, 0.8, 0.2], predicate=7
            ),
            config=OraclePhaseServoConfig(
                enabled=True, scope="transport_oracle_release"
            ),
        )
        self.assertEqual(diagnostics["mode"], "proposal_interaction")
        self.assertFalse(diagnostics["applied"])
        self.assertTrue(np.array_equal(output, self.actions))

    def test_invalid_scope_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scope must be one of"):
            self.apply(
                eef=[0.1, 0.1, 0.2],
                payload=oracle(
                    subject=[0.5, 0.5, 0.2], goal=[0.8, 0.8, 0.2]
                ),
                config=OraclePhaseServoConfig(enabled=True, scope="unknown"),
            )

    def test_released_unfinished_retries_approach(self):
        subject = np.asarray([0.5, 0.5, 0.2])
        output, diagnostics = self.apply(
            eef=subject + np.asarray([0.0, 0.0, self.config.grasp_offset_m]),
            payload=oracle(
                subject=subject,
                goal=[0.8, 0.8, 0.2],
                phase=1,
                grasped=False,
            ),
        )
        self.assertEqual(diagnostics["requested_phase"], 1)
        self.assertEqual(diagnostics["effective_phase"], 0)
        self.assertEqual(diagnostics["mode"], "grasp_close")
        self.assertTrue(np.all(output[:, -1] == 1.0))

    def test_interaction_near_fixture_preserves_proposal(self):
        output, diagnostics = self.apply(
            eef=[0.5, 0.5, 0.2],
            payload=oracle(
                subject=[0.5, 0.5, 0.2],
                goal=[0.5, 0.5, 0.2],
                predicate=7,
            ),
        )
        self.assertEqual(diagnostics["mode"], "interaction_proposal")
        self.assertTrue(np.array_equal(output, self.actions))

    def test_summary_reports_phase_progress(self):
        summary = summarize_oracle_phase_servo(
            [[
                {
                    "applied": True,
                    "mode": "approach_hover",
                    "scope": "full",
                    "selected_clause": 0,
                    "effective_phase": 0,
                    "subject_distance_m": 0.4,
                    "goal_distance_m": 0.8,
                    "action_delta_rms": 0.1,
                },
                {
                    "applied": True,
                    "mode": "approach_hover",
                    "scope": "full",
                    "selected_clause": 0,
                    "effective_phase": 0,
                    "subject_distance_m": 0.3,
                    "goal_distance_m": 0.8,
                    "action_delta_rms": 0.2,
                },
            ]]
        )
        self.assertEqual(summary["decisions"], 2)
        self.assertEqual(summary["approach_progress_samples"], 1)
        self.assertEqual(summary["approach_progress_rate"], 1.0)
        self.assertAlmostEqual(summary["action_delta_rms_mean"], 0.15)
        self.assertEqual(summary["scope_counts"], {"full": 2})


if __name__ == "__main__":
    unittest.main()
