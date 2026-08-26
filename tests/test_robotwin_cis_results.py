import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_robotwin_cis import load_job_output, summarize_run


def _episode(condition, seed, *, selected, source, counterfactual):
    policy_instruction = (
        "put A left of B" if condition == "correct" else "put A right of B"
    )
    return {
        "format": "robotwin_language_intervention_episode_v1",
        "pair_id": "place_a2b_left_to_right",
        "source_task": "place_a2b_left",
        "counterfactual_task": "place_a2b_right",
        "task_config": "demo_clean",
        "condition": condition,
        "episode_index": 0,
        "scene_seed": seed,
        "instruction_type": "unseen",
        "checkpoint": "/checkpoints/model.pt",
        "source_instruction": "put A left of B",
        "counterfactual_instruction": "put A right of B",
        "policy_instruction": policy_instruction,
        "instruction_goal": "source" if condition == "correct" else "counterfactual",
        "initial_source_goal_success": False,
        "initial_counterfactual_goal_success": False,
        "selected_goal": (
            "counterfactual" if condition == "counterfactual" else "source"
        ),
        "selected_goal_success": selected,
        "source_goal_ever_success": source,
        "counterfactual_goal_ever_success": counterfactual,
    }


def _write_job(root, condition, records):
    directory = root / "place_a2b_left" / "demo_clean" / condition
    directory.mkdir(parents=True)
    with (directory / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    total = len(records)
    selected = sum(record["selected_goal_success"] for record in records)
    source = sum(record["source_goal_ever_success"] for record in records)
    counterfactual = sum(
        record["counterfactual_goal_ever_success"] for record in records
    )
    summary = {
        "format": "robotwin_language_intervention_summary_v1",
        "pair_id": "place_a2b_left_to_right",
        "source_task": "place_a2b_left",
        "counterfactual_task": "place_a2b_right",
        "condition": condition,
        "task_config": "demo_clean",
        "instruction_type": "unseen",
        "checkpoint": "/checkpoints/model.pt",
        "total_episodes": total,
        "selected_goal_successes": selected,
        "selected_goal_success_rate": selected / total,
        "source_goal_successes": source,
        "source_goal_success_rate": source / total,
        "counterfactual_goal_successes": counterfactual,
        "counterfactual_goal_success_rate": counterfactual / total,
        "scene_seeds": [record["scene_seed"] for record in records],
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return directory


class RoboTwinCISResultsTest(unittest.TestCase):
    def test_full_matrix_audits_matching_and_computes_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed = 100042
            _write_job(
                root,
                "correct",
                [
                    _episode(
                        "correct",
                        seed,
                        selected=True,
                        source=True,
                        counterfactual=False,
                    )
                ],
            )
            _write_job(
                root,
                "shuffled",
                [
                    _episode(
                        "shuffled",
                        seed,
                        selected=True,
                        source=True,
                        counterfactual=False,
                    )
                ],
            )
            _write_job(
                root,
                "counterfactual",
                [
                    _episode(
                        "counterfactual",
                        seed,
                        selected=True,
                        source=False,
                        counterfactual=True,
                    )
                ],
            )
            payload = summarize_run(
                root,
                expected_episodes=1,
                expected_tasks=["place_a2b_left"],
                expected_task_configs=["demo_clean"],
                expected_conditions=["correct", "shuffled", "counterfactual"],
                require_complete=True,
            )
        self.assertTrue(payload["complete"])
        self.assertEqual(len(payload["matched_seed_instruction_audits"]), 1)
        self.assertEqual(payload["metrics"][0]["correct_sr"], 1.0)
        self.assertEqual(payload["metrics"][0]["dtl_shuffle"], 1.0)
        self.assertEqual(payload["metrics"][0]["cis"], 1.0)

    def test_rejects_shuffle_counterfactual_instruction_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed = 100042
            _write_job(
                root,
                "correct",
                [
                    _episode(
                        "correct",
                        seed,
                        selected=True,
                        source=True,
                        counterfactual=False,
                    )
                ],
            )
            shuffled = _episode(
                "shuffled", seed, selected=False, source=False, counterfactual=True
            )
            _write_job(root, "shuffled", [shuffled])
            counterfactual = _episode(
                "counterfactual", seed, selected=True, source=False, counterfactual=True
            )
            counterfactual["policy_instruction"] = "a different instruction"
            _write_job(root, "counterfactual", [counterfactual])
            with self.assertRaisesRegex(ValueError, "policy instruction mismatch"):
                summarize_run(
                    root,
                    expected_episodes=1,
                    expected_tasks=["place_a2b_left"],
                    expected_task_configs=["demo_clean"],
                    expected_conditions=["correct", "shuffled", "counterfactual"],
                    require_complete=True,
                )

    def test_resume_validation_rejects_partial_episode_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            directory = _write_job(
                root,
                "correct",
                [
                    _episode(
                        "correct",
                        100042,
                        selected=True,
                        source=True,
                        counterfactual=False,
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "expected 2"):
                load_job_output(directory, expected_episodes=2)


if __name__ == "__main__":
    unittest.main()
