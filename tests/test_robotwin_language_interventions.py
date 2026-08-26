import json
import tempfile
import unittest
from pathlib import Path

from experiments.robotwin.language_interventions import (
    EPISODE_FORMAT,
    GoalObserver,
    ManifestError,
    condition_contract,
    evaluate_goal,
    load_matched_episode_records,
    load_intervention_manifest,
    stable_instruction_seed,
    validate_manifest_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "configs" / "eval" / "robotwin_cis_spatial.json"


class _Pose:
    def __init__(self, position):
        self.p = list(position)


class _Actor:
    def __init__(self, position):
        self.position = list(position)

    def get_pose(self):
        return _Pose(self.position)


class _Robot:
    def __init__(self, *, left_open=True, right_open=True):
        self.left_open = left_open
        self.right_open = right_open

    def is_left_gripper_open(self):
        return self.left_open

    def is_right_gripper_open(self):
        return self.right_open


class _Env:
    def __init__(self, object_position, target_position):
        self.object = _Actor(object_position)
        self.target_object = _Actor(target_position)
        self.robot = _Robot()

    def check_success(self):
        raise AssertionError("native check should be replaced")


class RoboTwinManifestTest(unittest.TestCase):
    def test_checked_in_manifest_is_bidirectional_and_executable(self):
        pairs = load_intervention_manifest(
            MANIFEST,
            robotwin_root=PROJECT_ROOT / "third_party" / "RoboTwin",
        )
        self.assertEqual(
            [(pair.source_task, pair.counterfactual_task) for pair in pairs],
            [
                ("place_a2b_left", "place_a2b_right"),
                ("place_a2b_right", "place_a2b_left"),
            ],
        )

    def test_manifest_rejects_missing_reverse_pair(self):
        raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
        raw["pairs"] = raw["pairs"][:1]
        with self.assertRaisesRegex(ManifestError, "missing reverse"):
            validate_manifest_data(raw)

    def test_condition_contracts_match_dtl_and_cis_definitions(self):
        self.assertEqual(condition_contract("correct"), ("source", "source"))
        self.assertEqual(condition_contract("shuffled"), ("counterfactual", "source"))
        self.assertEqual(
            condition_contract("counterfactual"),
            ("counterfactual", "counterfactual"),
        )

    def test_instruction_seed_is_stable_and_task_specific(self):
        first = stable_instruction_seed(
            scene_seed=100042,
            task_name="place_a2b_right",
            instruction_type="unseen",
        )
        second = stable_instruction_seed(
            scene_seed=100042,
            task_name="place_a2b_right",
            instruction_type="unseen",
        )
        different = stable_instruction_seed(
            scene_seed=100042,
            task_name="place_a2b_left",
            instruction_type="unseen",
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def _write_matched_records(self, records):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "episodes.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _matched_record(index, seed):
        source_instruction = f"put object left {index}"
        return {
            "format": EPISODE_FORMAT,
            "pair_id": "place_a2b_left_to_right",
            "source_task": "place_a2b_left",
            "counterfactual_task": "place_a2b_right",
            "task_config": "demo_clean",
            "condition": "correct",
            "episode_index": index,
            "scene_seed": seed,
            "instruction_type": "unseen",
            "source_instruction": source_instruction,
            "counterfactual_instruction": f"put object right {index}",
            "policy_instruction": source_instruction,
            "instruction_goal": "source",
            "selected_goal": "source",
            "initial_source_goal_success": False,
            "initial_counterfactual_goal_success": False,
        }

    def _load_matched(self, path, expected_episodes=2):
        return load_matched_episode_records(
            path,
            expected_pair_id="place_a2b_left_to_right",
            expected_source_task="place_a2b_left",
            expected_counterfactual_task="place_a2b_right",
            expected_task_config="demo_clean",
            expected_instruction_type="unseen",
            expected_episodes=expected_episodes,
        )

    def test_matched_episode_loader_preserves_canonical_order(self):
        records = [
            self._matched_record(0, 4300000),
            self._matched_record(1, 4300004),
        ]
        loaded = self._load_matched(self._write_matched_records(records))
        self.assertEqual(
            [record["scene_seed"] for record in loaded],
            [4300000, 4300004],
        )

    def test_matched_episode_loader_rejects_contract_drift(self):
        records = [
            self._matched_record(0, 4300000),
            self._matched_record(1, 4300004),
        ]
        records[1]["condition"] = "shuffled"
        with self.assertRaisesRegex(ValueError, "condition"):
            self._load_matched(self._write_matched_records(records))

        records[1] = self._matched_record(1, 4300000)
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            self._load_matched(self._write_matched_records(records))


class RoboTwinGoalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.left_pair, cls.right_pair = load_intervention_manifest(MANIFEST)

    def test_left_and_right_goals_are_mutually_exclusive(self):
        env = _Env((-0.13, -0.1, 0.75), (0.0, -0.1, 0.75))
        left = evaluate_goal(env, self.left_pair.source_goal)
        right = evaluate_goal(env, self.left_pair.counterfactual_goal)
        self.assertTrue(left.success)
        self.assertFalse(right.success)

        env.object.position[0] = 0.13
        left = evaluate_goal(env, self.left_pair.source_goal)
        right = evaluate_goal(env, self.left_pair.counterfactual_goal)
        self.assertFalse(left.success)
        self.assertTrue(right.success)

    def test_neutral_initial_scene_satisfies_neither_goal(self):
        env = _Env((-0.13, 0.05, 0.75), (0.0, -0.1, 0.75))
        self.assertFalse(evaluate_goal(env, self.left_pair.source_goal).success)
        self.assertFalse(evaluate_goal(env, self.left_pair.counterfactual_goal).success)

    def test_goal_requires_open_grippers(self):
        env = _Env((-0.13, -0.1, 0.75), (0.0, -0.1, 0.75))
        env.robot.left_open = False
        self.assertFalse(evaluate_goal(env, self.left_pair.source_goal).success)

    def test_observer_tracks_ever_success_and_installs_selected_goal(self):
        env = _Env((-0.13, 0.05, 0.75), (0.0, -0.1, 0.75))
        observer = GoalObserver(env, self.left_pair)
        observer.install_selected_goal("counterfactual")
        self.assertFalse(env.check_success())
        env.object.position[:] = [0.13, -0.1, 0.75]
        self.assertTrue(env.check_success())
        env.object.position[:] = [0.5, 0.5, 0.75]
        diagnostics = observer.episode_diagnostics(selected_goal="counterfactual")
        self.assertTrue(diagnostics["selected_goal_success"])
        self.assertTrue(diagnostics["counterfactual_goal_ever_success"])
        self.assertFalse(diagnostics["counterfactual_goal_final_success"])
        observer.restore_native_goal()
        with self.assertRaisesRegex(AssertionError, "native check"):
            env.check_success()

    def test_observer_rejects_double_install_and_restore_is_idempotent(self):
        env = _Env((-0.13, 0.05, 0.75), (0.0, -0.1, 0.75))
        observer = GoalObserver(env, self.left_pair)
        observer.install_selected_goal("source")
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            observer.install_selected_goal("counterfactual")
        observer.restore_native_goal()
        observer.restore_native_goal()
        with self.assertRaisesRegex(AssertionError, "native check"):
            env.check_success()


if __name__ == "__main__":
    unittest.main()
