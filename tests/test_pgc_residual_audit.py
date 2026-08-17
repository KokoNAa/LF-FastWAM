import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_pgc_residual_gap import (
    diagnose_residual_gap,
    summarize_offline_records,
    summarize_rollout_results,
    summarize_values,
)


class PGCResidualAuditTest(unittest.TestCase):
    def test_summarize_values(self):
        summary = summarize_values([1.0, 2.0, 3.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 3.0)

    def test_rollout_summary_reads_nested_replan_decisions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = {
                "task_id": 2,
                "task_description": "pick target",
                "total_episodes": 1,
                "successes": 1,
                "policy_guard_episode_diagnostics": [
                    {
                        "episode": 0,
                        "decisions": [
                            {
                                "selected_counterfactual": True,
                                "candidate_delta_rms": 0.1,
                                "candidate_saturation_fraction": 0.0,
                                "target_binding_top1_mass": 0.5,
                            },
                            {
                                "selected_counterfactual": False,
                                "candidate_delta_rms": 0.2,
                                "candidate_saturation_fraction": 0.1,
                                "target_binding_top1_mass": 0.7,
                            },
                        ],
                    }
                ],
            }
            path = root / "gpu0_task2_results.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            summary = summarize_rollout_results(root)

        self.assertEqual(summary["result_files"], 1)
        self.assertEqual(summary["decisions"], 2)
        self.assertEqual(summary["overrides"], 1)
        self.assertAlmostEqual(summary["candidate_delta_rms"]["mean"], 0.15)
        self.assertAlmostEqual(summary["target_binding_top1_mass"]["mean"], 0.6)

    def test_offline_summary_keeps_cached_and_live_paths_separate(self):
        records = []
        for value in (0.2, 0.4):
            mode = {
                "candidate_delta_rms": value,
                "candidate_saturation_fraction": 0.0,
                "base_target_prefix_mse": 1.0,
                "candidate_target_prefix_mse": 0.5,
                "candidate_mse_improvement": 0.5,
                "target_binding_top1_mass": 0.4,
                "target_binding_entropy": 0.5,
                "target_binding_similarity_max": 0.6,
                "delta_rms_by_action_dim": [value] * 7,
            }
            records.append(
                {
                    "prompt": "target A",
                    "cached_text": dict(mode),
                    "live_text": {**mode, "candidate_delta_rms": value / 2},
                }
            )
        summary = summarize_offline_records(records)
        self.assertAlmostEqual(
            summary["cached_text"]["candidate_delta_rms"]["mean"], 0.3
        )
        self.assertAlmostEqual(
            summary["live_text"]["candidate_delta_rms"]["mean"], 0.15
        )

    def test_diagnoses_closed_loop_distribution_shift(self):
        offline = {
            "cached_text": {
                "candidate_delta_rms": {"mean": 0.25},
                "candidate_mse_improvement": {"mean": 0.4},
            },
            "live_text": {
                "candidate_delta_rms": {"mean": 0.24},
                "candidate_mse_improvement": {"mean": 0.35},
            },
        }
        rollout = {"candidate_delta_rms": {"mean": 0.0055}}
        text = {"cosine_similarity": {"mean": 0.999}}
        result = diagnose_residual_gap(
            checkpoint_state={"exact_match": True},
            offline=offline,
            rollout=rollout,
            text_context=text,
        )
        self.assertEqual(result["diagnosis"], "closed_loop_state_distribution_shift")
        self.assertLess(result["closed_loop_to_live_delta_ratio"], 0.03)

    def test_diagnoses_live_text_collapse_before_state_shift(self):
        offline = {
            "cached_text": {
                "candidate_delta_rms": {"mean": 0.3},
                "candidate_mse_improvement": {"mean": 0.4},
            },
            "live_text": {
                "candidate_delta_rms": {"mean": 0.02},
                "candidate_mse_improvement": {"mean": 0.1},
            },
        }
        result = diagnose_residual_gap(
            checkpoint_state={"exact_match": True},
            offline=offline,
            rollout={"candidate_delta_rms": {"mean": 0.005}},
            text_context={"cosine_similarity": {"mean": 0.8}},
        )
        self.assertEqual(result["diagnosis"], "cached_live_text_path_mismatch")


if __name__ == "__main__":
    unittest.main()
