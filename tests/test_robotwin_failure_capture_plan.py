import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_pgc_robotwin_failure_capture_plan import build_capture_plan


def _record(*, pair_id, task_config, scene_seed, source=True, target=False):
    return {
        "format": "robotwin_language_intervention_episode_v1",
        "pair_id": pair_id,
        "source_task": "source_task",
        "counterfactual_task": "target_task",
        "task_config": task_config,
        "condition": "counterfactual",
        "episode_index": scene_seed,
        "scene_seed": scene_seed,
        "source_instruction": "do source",
        "counterfactual_instruction": "do target",
        "source_goal_ever_success": source,
        "counterfactual_goal_ever_success": target,
    }


class FailureCapturePlanTest(unittest.TestCase):
    def _run_root(self, records_by_domain):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "cis_summary.json").write_text(
            json.dumps({"complete": True, "checkpoint": "/checkpoint.pt"}),
            encoding="utf-8",
        )
        for domain, records in records_by_domain.items():
            directory = root / "source_task" / domain / "counterfactual"
            directory.mkdir(parents=True)
            (directory / "episodes.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
        return root

    def test_selects_balanced_source_directed_failures(self):
        root = self._run_root(
            {
                "demo_clean": [
                    _record(pair_id="pair", task_config="demo_clean", scene_seed=seed)
                    for seed in range(5)
                ],
                "demo_randomized": [
                    _record(
                        pair_id="pair",
                        task_config="demo_randomized",
                        scene_seed=seed,
                    )
                    for seed in range(10, 15)
                ],
            }
        )
        payload = build_capture_plan(root, episodes_per_pair=5)
        pair = payload["pairs"]["pair"]
        selected = pair["selected_capture_candidates"]
        self.assertEqual(
            [record["task_config"] for record in selected],
            [
                "demo_clean",
                "demo_randomized",
                "demo_clean",
                "demo_randomized",
                "demo_clean",
            ],
        )
        self.assertEqual(payload["required_successful_full_goal_episodes"], 5)
        self.assertEqual(payload["full_goal_usage"], "final_short_lora_only")
        self.assertEqual(payload["allowed_training_stages"], [])

    def test_rejects_insufficient_failure_coverage(self):
        root = self._run_root(
            {
                "demo_clean": [
                    _record(pair_id="pair", task_config="demo_clean", scene_seed=seed)
                    for seed in range(4)
                ]
            }
        )
        with self.assertRaisesRegex(ValueError, "only 4"):
            build_capture_plan(root, episodes_per_pair=5)

    def test_excludes_target_successes_and_neither_goal(self):
        root = self._run_root(
            {
                "demo_clean": [
                    _record(
                        pair_id="pair",
                        task_config="demo_clean",
                        scene_seed=0,
                        source=True,
                        target=True,
                    ),
                    _record(
                        pair_id="pair",
                        task_config="demo_clean",
                        scene_seed=1,
                        source=False,
                        target=False,
                    ),
                ]
            }
        )
        with self.assertRaisesRegex(ValueError, "No source-directed failures"):
            build_capture_plan(root, episodes_per_pair=1)


if __name__ == "__main__":
    unittest.main()
