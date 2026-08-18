import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_pgc_v9_evaluation import (
    RunSpec,
    build_report,
    mcnemar_exact_p,
    wilson_interval,
)


class PGCV9EvaluationStatisticsTest(unittest.TestCase):
    def test_wilson_and_exact_mcnemar(self):
        lower, upper = wilson_interval(50, 50)
        self.assertLess(lower, 1.0)
        self.assertEqual(upper, 1.0)
        self.assertAlmostEqual(mcnemar_exact_p(11, 1), 0.00634765625)
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)

    def _write_run(
        self,
        root: Path,
        *,
        model: str,
        condition: str,
        successes: dict[int, list[int]],
    ) -> RunSpec:
        run = root / model / condition
        suite = run / "libero_10"
        suite.mkdir(parents=True)
        for task_id, successful_trials in successes.items():
            diagnostics = []
            for trial in range(2):
                achieved = trial in successful_trials
                diagnostics.append(
                    {
                        "episode": trial,
                        "counterfactual_goal_achieved": achieved,
                        "counterfactual_target_objects": ["target"],
                        "grasped_objects": ["target"] if achieved else [],
                        "lifted_objects": ["target"] if achieved else [],
                    }
                )
            payload = {
                "task_id": task_id,
                "total_episodes": 2,
                "successes": len(successful_trials),
                "success_episodes": successful_trials,
            }
            if condition != "correct":
                payload["counterfactual_episode_diagnostics"] = diagnostics
            (suite / f"gpu0_task{task_id}_results.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        return RunSpec(model=model, condition=condition, seed=42, path=run)

    def test_three_condition_report_is_strictly_paired(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = []
            for condition in ("correct", "raw_cis", "strict_cis"):
                runs.append(
                    self._write_run(
                        root,
                        model="Base",
                        condition=condition,
                        successes={0: [0], 1: [0]},
                    )
                )
                runs.append(
                    self._write_run(
                        root,
                        model="V9",
                        condition=condition,
                        successes={0: [0, 1], 1: [0, 1]},
                    )
                )
            report = build_report(
                runs, expected_seeds=1, expected_tasks=2, trials_per_task=2
            )
            self.assertEqual(report["conditions"]["raw_cis"]["V9"]["successes"], 4)
            self.assertEqual(
                report["conditions"]["strict_cis"]["paired"]["v9_only"], 2
            )
            self.assertEqual(
                report["conditions"]["raw_cis"]["per_seed"]["42"]["V9"][
                    "successes"
                ],
                4,
            )
            self.assertIn("strict_cis_significantly_above_base", report["admission_checks"])

    def test_rejects_unpaired_episode_sets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = []
            for condition in ("correct", "raw_cis", "strict_cis"):
                runs.append(
                    self._write_run(
                        root,
                        model="Base",
                        condition=condition,
                        successes={0: [0], 1: [0]},
                    )
                )
                v9 = self._write_run(
                    root,
                    model="V9",
                    condition=condition,
                    successes={0: [0], 1: [0]},
                )
                runs.append(v9)
            missing = root / "V9" / "strict_cis" / "libero_10" / "gpu0_task1_results.json"
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "episodes, expected"):
                build_report(
                    runs, expected_seeds=1, expected_tasks=2, trials_per_task=2
                )

    def test_grasp_rate_excludes_unary_fixture_goals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = []
            for condition in ("correct", "raw_cis", "strict_cis"):
                for model in ("Base", "V9"):
                    spec = self._write_run(
                        root,
                        model=model,
                        condition=condition,
                        successes={0: [0, 1], 1: [0, 1]},
                    )
                    if condition != "correct":
                        result = (
                            spec.path
                            / "libero_10"
                            / "gpu0_task1_results.json"
                        )
                        payload = json.loads(result.read_text(encoding="utf-8"))
                        for diagnostic in payload[
                            "counterfactual_episode_diagnostics"
                        ]:
                            diagnostic[
                                "counterfactual_graspable_target_objects"
                            ] = []
                            diagnostic["grasped_objects"] = []
                            diagnostic["lifted_objects"] = []
                        result.write_text(json.dumps(payload), encoding="utf-8")
                    runs.append(spec)

            report = build_report(
                runs, expected_seeds=1, expected_tasks=2, trials_per_task=2
            )
            strict_v9 = report["conditions"]["strict_cis"]["V9"]
            self.assertEqual(strict_v9["grasp_eligible_episodes"], 2)
            self.assertEqual(strict_v9["target_grasp_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
