import json
import tempfile
import unittest
from pathlib import Path

from experiments.libero.counterfactual_diagnostics import (
    CounterfactualEpisodeTracker,
)
from scripts.summarize_counterfactual_behaviors import summarize


class _FakeSimData:
    def __init__(self):
        self.body_xpos = {
            0: [0.0, 0.0, 0.10],
            1: [0.0, 0.0, 0.10],
            2: [0.0, 0.0, 0.10],
        }


class _FakeSim:
    def __init__(self):
        self.data = _FakeSimData()


class _FakeRobot:
    gripper = "fake_gripper"


class _FakeInnerEnv:
    def __init__(self):
        self.sim = _FakeSim()
        self.robots = [_FakeRobot()]
        self.objects_dict = {
            "source_1": "source_model",
            "target_1": "target_model",
            "other_1": "other_model",
        }
        self.obj_body_id = {
            "source_1": 0,
            "target_1": 1,
            "other_1": 2,
        }
        self.predicate_values = {}
        self.grasped_models = set()

    def _eval_predicate(self, predicate):
        return self.predicate_values.get(tuple(predicate), False)

    def _check_grasp(self, *, gripper, object_geoms):
        del gripper
        return object_geoms in self.grasped_models


class _FakeWrapper:
    def __init__(self):
        self.env = _FakeInnerEnv()


class CounterfactualEpisodeTrackerTest(unittest.TestCase):
    def setUp(self):
        self.source_goal = [["in", "source_1", "basket_region"]]
        self.counterfactual_goal = [["in", "target_1", "basket_region"]]
        self.env = _FakeWrapper()

    def test_tracks_source_goal_grasp_and_lift(self):
        tracker = CounterfactualEpisodeTracker(
            self.env,
            source_goal_state=self.source_goal,
            counterfactual_goal_state=self.counterfactual_goal,
            lift_threshold_m=0.04,
        )
        tracker.observe(policy_step=0)

        inner = self.env.env
        inner.grasped_models.add("source_model")
        inner.sim.data.body_xpos[0][2] = 0.16
        inner.predicate_values[tuple(self.source_goal[0])] = True
        tracker.observe(policy_step=12)
        result = tracker.result(episode_idx=3)

        self.assertEqual(result["episode"], 3)
        self.assertEqual(result["category"], "source_goal_success")
        self.assertTrue(result["source_goal_achieved"])
        self.assertFalse(result["counterfactual_goal_achieved"])
        self.assertEqual(result["grasped_objects"], ["source_1"])
        self.assertEqual(result["lifted_objects"], ["source_1"])
        self.assertEqual(
            result["counterfactual_graspable_target_objects"], ["target_1"]
        )
        self.assertEqual(result["first_grasp_step"], {"source_1": 12})
        self.assertAlmostEqual(result["max_lift_delta_m"]["source_1"], 0.06)

    def test_unary_fixture_goal_is_not_counted_as_graspable(self):
        tracker = CounterfactualEpisodeTracker(
            self.env,
            source_goal_state=self.source_goal,
            counterfactual_goal_state=[["open", "cabinet_1_middle_region"]],
            lift_threshold_m=0.04,
        )
        tracker.observe(policy_step=0)
        result = tracker.result(episode_idx=0)

        self.assertEqual(
            result["counterfactual_target_objects"],
            ["cabinet_1_middle_region"],
        )
        self.assertEqual(result["counterfactual_graspable_target_objects"], [])

    def test_classifies_target_manipulation_without_goal(self):
        tracker = CounterfactualEpisodeTracker(
            self.env,
            source_goal_state=self.source_goal,
            counterfactual_goal_state=self.counterfactual_goal,
            lift_threshold_m=0.04,
        )
        self.env.env.grasped_models.add("target_model")
        tracker.observe(policy_step=8)
        result = tracker.result(episode_idx=0)

        self.assertEqual(
            result["category"],
            "target_object_manipulated_placement_failure",
        )
        self.assertEqual(result["manipulated_objects"], ["target_1"])


class CounterfactualBehaviorSummaryTest(unittest.TestCase):
    def test_summarizes_episode_categories_and_events(self):
        result = {
            "task_id": 0,
            "pair_id": "object_00_to_01",
            "task_description": "pick source",
            "policy_instruction": "pick target",
            "total_episodes": 2,
            "counterfactual_episode_diagnostics": [
                {
                    "episode": 0,
                    "category": "source_goal_success",
                    "source_goal_achieved": True,
                    "counterfactual_goal_achieved": False,
                    "source_target_objects": ["source_1"],
                    "counterfactual_target_objects": ["target_1"],
                    "grasped_objects": ["source_1"],
                    "lifted_objects": ["source_1"],
                    "manipulated_objects": ["source_1"],
                    "first_grasp_step": {"source_1": 20},
                    "max_lift_delta_m": {"source_1": 0.1},
                    "policy_steps": 120,
                    "horizon_timeout": False,
                },
                {
                    "episode": 1,
                    "category": "other_object_manipulated",
                    "source_goal_achieved": False,
                    "counterfactual_goal_achieved": False,
                    "source_target_objects": ["source_1"],
                    "counterfactual_target_objects": ["target_1"],
                    "grasped_objects": ["other_1"],
                    "lifted_objects": [],
                    "manipulated_objects": ["other_1"],
                    "first_grasp_step": {"other_1": 40},
                    "max_lift_delta_m": {"other_1": 0.01},
                    "policy_steps": 600,
                    "horizon_timeout": True,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gpu0_task0_results.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            summary = summarize(Path(tmpdir), expected_episodes=2)

        self.assertEqual(summary["total_episodes"], 2)
        self.assertEqual(summary["behavior_counts"]["source_goal_success"], 1)
        self.assertEqual(
            summary["behavior_counts"]["other_object_manipulated"], 1
        )
        self.assertEqual(summary["event_counts"]["source_goal_achieved"], 1)
        self.assertEqual(summary["event_counts"]["other_object_grasped"], 1)
        self.assertEqual(summary["event_counts"]["horizon_timeout"], 1)


if __name__ == "__main__":
    unittest.main()
