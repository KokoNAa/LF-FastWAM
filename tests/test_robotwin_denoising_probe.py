import argparse
import contextlib
import io
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.robotwin.denoising_probe import capture_action_cache, denoising_metrics
from scripts.probe_robotwin_denoising import build_plan, resolve_source_probe, summarize
from scripts.probe_robotwin_no_eraf import write_json
from tests import test_robotwin_no_eraf_probe as probe_fixtures


class DenoisingMathTest(unittest.TestCase):
    def test_flow_error_is_distinct_from_noise_scaled_reconstruction_error(self):
        clean = np.zeros((32, 14))
        noise = np.ones_like(clean) * .3
        for sigma in (.1, .5, .9, 1.):
            noisy = (1 - sigma) * clean + sigma * noise
            target = noise - clean
            metrics = denoising_metrics(clean, noisy, target, target, target + 1, sigma)
            self.assertEqual([r["horizon"] for r in metrics], [24, 32])
            for row in metrics:
                self.assertAlmostEqual(row["correct_x0_rmse"], 0.)
                self.assertAlmostEqual(row["wrong_flow_rmse"], 1.)
                self.assertAlmostEqual(row["wrong_x0_rmse"], sigma)
                self.assertAlmostEqual(row["wrong_minus_correct_flow_rmse"], 1.)

    def test_rounding_floor_and_shape_checks(self):
        clean = np.zeros((32, 14))
        rows = denoising_metrics(clean, clean + .01, clean, clean, clean, .5)
        self.assertAlmostEqual(rows[-1]["x0_dtype_rounding_floor_rmse"], .01)
        with self.assertRaises(ValueError):
            denoising_metrics(clean, clean[:24], clean, clean, clean, .5)
        with self.assertRaises(ValueError):
            denoising_metrics(clean, clean, clean, clean, clean, 0.)

    def test_cache_capture_preserves_forward_output_and_restores_method(self):
        class Model:
            def _predict_action_noise_with_cache(self, **kwargs):
                return kwargs["latents_action"] + kwargs["timestep_action"]

        model = Model()
        common = {"context": object(), "context_mask": object(), "video_kv_cache": object(),
                  "attention_mask": object(), "video_seq_len": 12}

        def run():
            model._predict_action_noise_with_cache(latents_action=1, timestep_action=2, **common)
            return model._predict_action_noise_with_cache(latents_action=3, timestep_action=4, **common)

        result, cache, calls = capture_action_cache(model, run)
        self.assertEqual(result, 7)
        self.assertEqual(calls, 2)
        self.assertEqual(cache, common)
        self.assertNotIn("_predict_action_noise_with_cache", vars(model))
        with self.assertRaisesRegex(RuntimeError, "failed rollout"):
            capture_action_cache(model, lambda: (_ for _ in ()).throw(RuntimeError("failed rollout")))
        self.assertNotIn("_predict_action_noise_with_cache", vars(model))

    def test_preexisting_predictor_override_is_restored(self):
        class Model:
            pass

        model = Model()
        override = lambda **kwargs: 1
        model._predict_action_noise_with_cache = override
        with self.assertRaisesRegex(ValueError, "did not expose"):
            capture_action_cache(model, lambda: 0)
        self.assertIs(model._predict_action_noise_with_cache, override)


class DenoisingPipelineTest(unittest.TestCase):
    def setUp(self):
        fixture = probe_fixtures.PreparationTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with contextlib.redirect_stdout(io.StringIO()):
            self.source, self.states = fixture.prepare()
        self.root = Path(self.source["output"])
        for model in self.source["checkpoints"]:
            (self.root / model).mkdir()
            rows = []
            for state in self.states:
                path = self.root / model / f"{state['id']}.npz"
                np.savez(path, source=np.zeros((32, 14)), target=np.ones((32, 14)))
                rows.append({**state, "actions_file": str(path)})
            (self.root / model / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            write_json(self.root / model / "complete.json", {"states": len(self.states)})
        write_json(self.root / "summary.json", {"complete": True, "format": self.source["format"]})
        self.args = argparse.Namespace(source_probe=str(self.root), output=str(fixture.root / "denoise"),
                                       pairs=[fixture.pair], sigmas=[.1, 1.], noise_seeds=[42, 43], gpus=[0, 1, 2])

    def test_plan_selects_only_shared_initial_state_without_writing(self):
        plan = build_plan(self.args)
        self.assertEqual(len(plan["states"]), 1)
        self.assertEqual(plan["states"][0]["frame_index"], 0)
        self.assertFalse(Path(plan["output"]).exists())
        (self.root / "step500/complete.json").write_text('{"states": 0}')
        with self.assertRaisesRegex(ValueError, "Incomplete source model"):
            build_plan(self.args)

    def test_latest_ignores_failed_or_unrelated_runs(self):
        unrelated = self.root.parent / "unrelated"
        unrelated.mkdir()
        write_json(unrelated / "summary.json", {"complete": True, "format": "unrelated"})
        self.assertEqual(resolve_source_probe("latest", self.root.parent), self.root)
        write_json(self.root / "summary.json", {"complete": False, "format": self.source["format"]})
        with self.assertRaisesRegex(ValueError, "No completed"):
            resolve_source_probe("latest", self.root.parent)

    def test_summary_requires_complete_paired_noise_and_keeps_both_horizons(self):
        plan = build_plan(self.args)
        root = Path(plan["output"])
        root.mkdir()
        write_json(root / "plan.json", plan)
        with self.assertRaises(FileNotFoundError):
            summarize(root)
        for model in plan["models"]:
            (root / model).mkdir()
            rows = []
            for state in plan["states"]:
                for kind in ("source", "target"):
                    for sigma in plan["sigmas"]:
                        for seed in plan["noise_seeds"]:
                            clean = np.zeros((32, 14)) + (kind == "target")
                            noise = np.random.default_rng(seed).standard_normal(clean.shape)
                            noisy = (1 - sigma) * clean + sigma * noise
                            target = noise - clean
                            metrics = denoising_metrics(clean, noisy, target, target, target + .1, sigma)
                            rows.append({"id": state["id"], "pair_id": state["pair_id"], "reference": kind,
                                "sigma": sigma, "noise_seed": seed,
                                "noisy_action_sha256": f"{seed}_{sigma}_{kind if sigma < 1 else 'same'}",
                                "metrics": metrics})
            (root / model / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            write_json(root / model / "complete.json", {"records": len(rows)})
        with contextlib.redirect_stdout(io.StringIO()):
            summarize(root)
        summary = json.loads((root / "summary.json").read_text())
        self.assertEqual(len(summary["rows"]), 3 * 2 * 2 * 2)
        self.assertTrue(all(row["noise_draws"] == 2 for row in summary["rows"]))
        path = root / "step1000/records.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        next(r for r in rows if r["sigma"] == 1 and r["reference"] == "target")["noisy_action_sha256"] = "changed"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "Noisy actions differ"):
            summarize(root)


if __name__ == "__main__":
    unittest.main()
