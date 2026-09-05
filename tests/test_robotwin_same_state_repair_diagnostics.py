"""CPU-only checks for diagnosing the saved repair artifacts."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from experiments.robotwin.same_state_repair import action_rows, checkpoint_score
from scripts.inspect_robotwin_same_state_repair import FORMAT, inspect, paired_components, render


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ComponentTest(unittest.TestCase):
    def test_pair_error_separates_common_bias_from_reversed_language(self):
        source, target = np.zeros((24, 14)), np.ones((24, 14))
        common = paired_components(source + .2, target + .2, source, target, source, target)
        self.assertAlmostEqual(common["pair_mse"], .04)
        self.assertAlmostEqual(common["common_mse"], .04)
        self.assertAlmostEqual(common["conditional_mse"], 0)
        self.assertAlmostEqual(common["conditional_update_mse"], 0)
        reversed_pair = paired_components(target, source, source, target, source, target)
        self.assertAlmostEqual(reversed_pair["pair_mse"], 1)
        self.assertAlmostEqual(reversed_pair["common_mse"], 0)
        self.assertAlmostEqual(reversed_pair["conditional_mse"], 1)
        self.assertAlmostEqual(reversed_pair["delta_projection"], -1)
        self.assertAlmostEqual(reversed_pair["delta_cosine"], -1)

    def test_zero_expert_or_prediction_difference_has_no_cosine(self):
        zeros, ones = np.zeros((24, 14)), np.ones((24, 14))
        self.assertIsNone(paired_components(zeros, zeros, zeros, ones, zeros, zeros)["delta_cosine"])
        self.assertIsNone(paired_components(zeros, ones, zeros, zeros, zeros, zeros)["delta_projection"])


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.plan = {"format": FORMAT, "initial_model": "step1000", "steps": 4,
                     "evaluation_steps": [0, 2, 4], "eval_seeds": [42, 43], "anchor_weight": .25,
                     "arms": ["paired_flow", "paired_flow_anchor"], "states": [
                         {"id": "fit", "split": "repair", "pair_id": "stack", "profile": "historical"},
                         {"id": "guard", "split": "guard", "pair_id": "stack", "profile": "strict"}]}
        write(self.root / "plan.json", self.plan)
        write(self.root / "summary.json", {"format": FORMAT, "complete": True})
        for arm in self.plan["arms"]:
            folder = self.root / arm
            (folder / "actions").mkdir(parents=True)
            training = []
            for step in range(1, 5):
                term = {"id": "fit", "flow_source_mse": step, "flow_target_mse": 2 * step}
                if arm == "paired_flow_anchor":
                    term.update(anchor_source_mse=2., anchor_target_mse=1.)
                training.append({"step": step, "draws": [{"id": "fit", "time": 100 * step,
                    "scheduler_weight": step, "noise_seed": 17000 + step, "noise_sha256": str(step) * 64}],
                    "terms": [term], "gradient_norm_before_clip": step / 2, "parameters_with_grad": 8})
            (folder / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in training))
            scores = []
            for step in self.plan["evaluation_steps"]:
                rows = []
                for state in self.plan["states"]:
                    for seed in self.plan["eval_seeds"]:
                        r, q = np.zeros((32, 14)), np.ones((32, 14))
                        # The seed perturbation makes averaging squared errors
                        # measurably different from squaring mean RMSE.
                        offset = .02 * (seed - 42)
                        initial_s, initial_t = r + .6 + offset, q - .6 + offset
                        s, t = r + .6 - step * .125 + offset, q - .6 + step * .125 + offset
                        np.savez_compressed(folder / "actions" / f"repair{step:06d}_{state['id']}_seed{seed}.npz",
                                            source=s, target=t, source_reference=r, target_reference=q)
                        rows += [{**state, "arm": arm, "repair_step": step, "noise_seed": seed, **row}
                                 for row in action_rows(s, t, r, q, initial_s, initial_t)]
                score = {**checkpoint_score(rows), "repair_step": step}
                scores.append(score)
                write(folder / f"evaluation_{step:06d}.json", {"rows": rows, "score": score})
            write(folder / "complete.json", {"format": FORMAT, "complete": True, "arm": arm,
                "steps": 4, "final_production_replay_passed": True,
                "frozen_parameter_versions_unchanged": True, "scores": scores, "best": None})

    def test_saved_artifacts_produce_matched_seed_decomposition_and_weighted_losses(self):
        report = inspect(self.root)
        self.assertTrue(report["training_inputs_match_across_arms"])
        self.assertEqual(report["repair_states"], 1)
        self.assertEqual(report["guard_states"], 1)
        early = report["training_windows"][0]
        self.assertEqual((early["first_step"], early["last_step"]), (1, 2))
        self.assertEqual(early["weighted_flow_source_mse"], 2.5)  # mean(1*1, 2*2)
        self.assertEqual(early["source_objective_contribution"], 1.25)
        self.assertAlmostEqual(early["source_objective_share"], 1 / 3)
        anchor = next(row for row in report["training_windows"]
                      if row["arm"] == "paired_flow_anchor" and row["window"] == "early")
        self.assertAlmostEqual(anchor["source_objective_contribution"], 1.5)
        endpoint = next(row for row in report["repair_endpoint_components"] if row["repair_step"] == 4)
        self.assertAlmostEqual(endpoint["pair_mse"], .0102)
        self.assertAlmostEqual(endpoint["common_mse"], .0002)
        self.assertAlmostEqual(endpoint["conditional_mse"], .01)
        self.assertAlmostEqual(endpoint["conditional_update_fraction"], 1)
        self.assertEqual(endpoint["both_correct"], 2)
        self.assertEqual(report["gradient_summary"][0]["fraction_clipped_at_1"], .5)
        self.assertIn("[repair_endpoints]", render(report))
        self.assertIn("paired_flow/training.jsonl", report["input_sha256"])
        json.dumps(report, allow_nan=False)

    def test_incomplete_evaluation_and_duplicate_training_terms_fail(self):
        path = self.root / "paired_flow/evaluation_000002.json"
        original = json.loads(path.read_text())
        broken = copy.deepcopy(original)
        broken["rows"].pop()
        write(path, broken)
        with self.assertRaisesRegex(ValueError, "Incomplete/duplicate evaluations"):
            inspect(self.root)
        write(path, original)
        log = self.root / "paired_flow/training.jsonl"
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        rows[0]["terms"].append(rows[0]["terms"][0])
        log.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "duplicate/extra states"):
            inspect(self.root)

    def test_changed_noise_and_nonfinite_training_loss_fail(self):
        log = self.root / "paired_flow_anchor/training.jsonl"
        original = log.read_text()
        rows = [json.loads(line) for line in original.splitlines()]
        rows[0]["draws"][0]["noise_seed"] += 1
        log.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "draws differ"):
            inspect(self.root)
        rows = [json.loads(line) for line in original.splitlines()]
        rows[0]["terms"][0]["anchor_source_mse"] = float("nan")
        log.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "Invalid numeric value"):
            inspect(self.root)

    def test_action_files_are_checked_against_logged_rmse(self):
        path = self.root / "paired_flow/actions/repair000004_fit_seed42.npz"
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["source"] += .3
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "actions disagree"):
            inspect(self.root)

    def test_cli_writes_reports_without_importing_torch_or_touching_sources(self):
        repo = Path(__file__).resolve().parents[1]
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        command = (
            "import runpy, sys; "
            "runpy.run_path('scripts/inspect_robotwin_same_state_repair.py', run_name='__main__'); "
            "assert 'torch' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", command, str(self.root)], cwd=repo,
                                capture_output=True, text=True, check=True)
        self.assertIn("[training_windows]", result.stdout)
        self.assertTrue((self.root / "repair_diagnostics.json").is_file())
        self.assertEqual((self.root / "repair_diagnostics.txt").read_text(), result.stdout)
        self.assertTrue(all(path.read_bytes() == data for path, data in before.items()))


if __name__ == "__main__":
    unittest.main()
