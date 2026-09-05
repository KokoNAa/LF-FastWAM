"""CPU tests for actual HDF5 preparation and same-state comparison contracts."""

import argparse
import ast
import io
import json
import shutil
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.robotwin.no_eraf_probe import (
    CAMERAS, difference, frame_positions, last_equal_qpos_prefix,
    observations_equal, paired_action_details, reference_metrics, require_pair, training_episode_ids, typed_hash,
)
from scripts.probe_robotwin_no_eraf import (
    audit_actions, build_plan, inference_bootstrap_configs, prepare_states, summarize, write_json,
)


class BootstrapConfigTest(unittest.TestCase):
    def test_temporal_v2_bootstrap_passes_real_lora_validator_without_changing_saved_config(self):
        from omegaconf import OmegaConf

        # Execute the production pure-Python config validators, excluding the
        # torch-dependent tensor implementation so this regression runs on CPU
        # hosts without torch. Do not duplicate the validation rules in tests.
        path = Path(__file__).resolve().parents[1] / "src/fastwam/models/wan22/lora.py"
        tree = ast.parse(path.read_text())
        names = {"normalize_lora_config", "normalize_paired_language_control_config"}
        nodes = [node for node in tree.body if (
            isinstance(node, ast.ImportFrom) and node.module == "__future__"
            or isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id.startswith("DEFAULT_") for target in node.targets)
            or isinstance(node, ast.FunctionDef) and node.name in names
        )]
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        normalize = namespace["normalize_lora_config"]
        cfg = OmegaConf.create({"dim": 14, "model": {"lora": {
            "enabled": True, "rank": 16, "extra_trainable_patterns": [],
            "paired_language_control": {key: True for key in (
                "enabled", "bidirectional_supervision", "deployment_matched_action_cache",
                "correct_branch_action_ranking")}}},
            "data": {"train": {"processor": {"action_output_dim": "${dim}"}}}})
        before = OmegaConf.to_container(cfg, resolve=True)
        broken = OmegaConf.to_container(cfg.model.lora, resolve=True)
        broken["enabled"] = False
        broken["paired_language_control"].update(enabled=False, bidirectional_supervision=False)
        with self.assertRaisesRegex(ValueError, "Deployment-matched Action ranking requires"):
            normalize(broken)
        model_cfg, processor_cfg = inference_bootstrap_configs(cfg)
        bootstrap = normalize(OmegaConf.to_container(model_cfg.lora, resolve=True))
        self.assertFalse(bootstrap["enabled"])
        for key in ("enabled", "bidirectional_supervision", "deployment_matched_action_cache",
                    "correct_branch_action_ranking"):
            self.assertFalse(bootstrap["paired_language_control"][key])
        self.assertEqual(processor_cfg.action_output_dim, 14)
        self.assertEqual(OmegaConf.to_container(cfg, resolve=True), before)
        # This is the saved config that _load_lora_adapter passes to configure_lora.
        restored = normalize(before["model"]["lora"])
        self.assertTrue(restored["enabled"])
        self.assertTrue(restored["paired_language_control"]["correct_branch_action_ranking"])


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.source = np.zeros((32, 14), dtype=np.float32)
        self.target = np.ones_like(self.source)

    def test_true_redirection_and_opposite_redirection(self):
        good = reference_metrics(self.source, self.target, self.source, "native", self.source, self.target)
        self.assertAlmostEqual(good["dual_reference"]["language_delta_projection_on_expert_delta"], 1.)
        self.assertTrue(good["dual_reference"]["target_language_prefers_target_expert"])
        bad = reference_metrics(self.target, self.source, self.source, "native", self.source, self.target)
        self.assertAlmostEqual(bad["dual_reference"]["language_delta_projection_on_expert_delta"], -1.)
        self.assertFalse(bad["dual_reference"]["target_language_prefers_target_expert"])

    def test_target_only_preference_is_not_bidirectional_switching(self):
        args = (self.source, self.target, self.source, self.source)
        both_target = paired_action_details(self.target, self.target, *args)[-1]
        self.assertEqual(both_target["preference"], "both_target")
        self.assertFalse(both_target["both_languages_prefer_own_expert"])
        self.assertEqual(both_target["common_update_vs_base_rms"], 1.)
        self.assertEqual(both_target["language_delta_update_vs_base_rms"], 0.)
        reversed_choice = paired_action_details(self.target, self.source, *args)[-1]
        self.assertEqual(reversed_choice["preference"], "reversed")
        tied = paired_action_details(self.target * .5, self.target * .5, *args)[-1]
        self.assertEqual(tied["preference"], "tie")

    def test_action_audit_distinguishes_executed_prefix_and_gripper_energy(self):
        target = self.source.copy()
        target[24:, 6] = 1
        windows = paired_action_details(self.source, target, self.source, target, self.source, self.source)
        self.assertEqual([w["horizon"] for w in windows], [8, 16, 24, 32])
        self.assertEqual(windows[2]["preference"], "indistinguishable_references")
        self.assertIsNone(windows[2]["both_languages_prefer_own_expert"])
        last = windows[-1]
        self.assertEqual(last["preference"], "both_correct")
        self.assertEqual(last["source_coordinate_on_reference_axis"], 0.)
        self.assertEqual(last["target_coordinate_on_reference_axis"], 1.)
        self.assertEqual(last["language_delta_top_dimensions"][0]["dimension"], "left_gripper")
        self.assertEqual(last["language_delta_top_dimensions"][0]["energy_fraction"], 1.)

    def test_large_difference_can_be_orthogonal_to_goal_reference(self):
        target = self.source.copy()
        target[:, 0] = 1
        orthogonal = self.source.copy()
        orthogonal[:, 1] = 10
        metrics = reference_metrics(self.source, orthogonal, self.source, "native", self.source, target)
        self.assertGreater(metrics["language_delta"]["rms"], 1.)
        self.assertEqual(metrics["dual_reference"]["language_delta_projection_on_expert_delta"], 0.)

    def test_identical_expert_prefix_is_not_a_language_failure_label(self):
        metrics = reference_metrics(self.source, self.target, self.source, "native", self.source, self.source)
        self.assertFalse(metrics["dual_reference"]["expert_references_distinguishable"])
        self.assertIsNone(metrics["dual_reference"]["target_language_prefers_target_expert"])

    def test_counterfactual_own_reference_and_unmatched_state(self):
        metrics = reference_metrics(self.source, self.target, self.target, "counterfactual")
        self.assertEqual(metrics["correct_language_reference_rmse"], 0.)
        self.assertEqual(metrics["wrong_minus_correct_reference_rmse"], 1.)
        self.assertIsNone(metrics["dual_reference"])
        with self.assertRaises(ValueError):
            reference_metrics(self.source, self.target, self.source, "native", self.source)

    def test_shape_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            difference(self.source[:1], self.target)
        with self.assertRaises(ValueError):
            difference(self.source, self.target * np.nan)

    def test_full_action_windows_and_prefix_boundary(self):
        self.assertEqual(frame_positions(40, 32, [.25, .5, .75]), [0, 2, 4, 6])
        with self.assertRaises(ValueError):
            frame_positions(31, 32, [.5])
        source = np.zeros((100, 14))
        target = source.copy()
        target[40:] = 1
        self.assertEqual(last_equal_qpos_prefix(source, target, 32), 39)
        target[0] = 1
        self.assertIsNone(last_equal_qpos_prefix(source, target, 32))

    def test_split_matches_dataset_rng_and_typed_hash(self):
        shuffled = list(range(25))
        np.random.default_rng(42).shuffle(shuffled)
        self.assertEqual(training_episode_ids(25, .05), set(shuffled[:23]))
        self.assertEqual(training_episode_ids(3, 0), {0, 1, 2})
        self.assertNotEqual(typed_hash(self.source), typed_hash(self.source.astype(np.float64)))

    def test_same_robot_qpos_is_not_enough_for_dual_reference(self):
        obs = {"state": np.zeros(14), **{c: np.zeros((4, 4, 3), dtype=np.uint8) for c in CAMERAS}}
        other = {key: value.copy() for key, value in obs.items()}
        self.assertTrue(observations_equal(obs, other))
        other[CAMERAS[0]][0, 0] = 1
        self.assertFalse(observations_equal(obs, other))


class PreparationTest(unittest.TestCase):
    def setUp(self):
        try:
            import h5py
            from omegaconf import OmegaConf
            from PIL import Image
        except ImportError:
            self.skipTest("HDF5/config preparation tests need h5py, omegaconf and Pillow")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pair = "place_a2b_left_to_right"
        self.domain = self.root / "expert/historical/demo_clean"
        self.train = self.root / "train"
        self.train.mkdir()
        self.base, self.stats = self.root / "base.pt", self.root / "stats.json"
        self.base.write_bytes(b"fixture base; planning must not deserialize this")
        self.stats.write_text("{}")
        (self.train / "checkpoints/weights").mkdir(parents=True)
        for step in (500, 1000):
            (self.train / f"checkpoints/weights/step_{step:06d}.pt").write_bytes(b"fixture")
        self.source = np.zeros((100, 14), dtype=np.float32)
        self.target = self.source.copy()
        self.target[40:] = 1
        for kind, actions in (("native", self.source), ("counterfactual", self.target)):
            raw = self.domain / "raw" / self.pair / kind
            dataset = self.domain / "lerobot" / self.pair / kind
            (raw / "meta/initial_states").mkdir(parents=True)
            (raw / "data").mkdir()
            (dataset / "meta").mkdir(parents=True)
            rows = []
            for episode in range(3):
                initial = np.array([episode, 2., 3.], dtype=np.float64)
                np.save(raw / f"meta/initial_states/episode{episode}.npy", initial)
                with h5py.File(raw / f"data/episode{episode}.hdf5", "w") as handle:
                    handle.create_dataset("joint_action/vector", data=actions)
                    for camera in CAMERAS:
                        encoded = []
                        for frame in range(100):
                            color = 40 if kind == "counterfactual" and frame >= 40 else 10
                            buffer = io.BytesIO()
                            Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8)).save(buffer, format="JPEG")
                            encoded.append(buffer.getvalue())
                        handle.create_dataset(f"observation/{camera}/rgb", data=np.array(encoded, dtype="S1000"))
                rows.append({"pair_id": self.pair, "episode_index": episode,
                             "dataset_kind": kind, "scene_seed": 100 + episode,
                             "initial_state_sha256": typed_hash(initial), "action_sha256": typed_hash(actions),
                             "action_count": len(actions), "raw_hdf5": f"data/episode{episode}.hdf5",
                             "source_initial_state_catalog": f"meta/initial_states/episode{episode}.npy",
                             "source_instruction": "left", "counterfactual_instruction": "right",
                             "source_goal_verified": kind == "native",
                             "counterfactual_goal_verified": kind == "counterfactual", "full_goal_verified": False})
            text = "".join(json.dumps(row) + "\n" for row in rows)
            (raw / "meta/pgc_episodes.jsonl").write_text(text)
            (dataset / "meta/pgc_episodes.jsonl").write_text(text)
            write_json(dataset / "meta/info.json", {"total_episodes": 3})
        self.cfg = {"resume": str(self.base), "model": {
            "policy_guard": {"enabled": False}, "langforce_mvp": {"enabled": False},
            "transition_contract": {"enabled": False}, "action_dit_config": {"use_latent_action_queries": False},
            "lora": {"paired_language_control": {key: True for key in (
                "enabled", "bidirectional_supervision", "deployment_matched_action_cache", "correct_branch_action_ranking")}}},
            "data": {"train": {"num_frames": 33, "processor": {"action_output_dim": 14},
                "pretrained_norm_stats": str(self.stats), "val_set_proportion": .05,
                "dataset_dirs": [str(self.domain / "lerobot" / self.pair / "native")],
                "pgc_counterfactual_dataset_dirs": [str(self.domain / "lerobot" / self.pair / "counterfactual")]}}}
        write_json(self.train / "config.yaml", self.cfg)
        self.args = argparse.Namespace(train_run=str(self.train), expert_root=str(self.root / "expert"),
            base_checkpoint=str(self.base), stats_path=str(self.stats), output=str(self.root / "output"),
            steps=[500, 1000], profiles=["historical"], pairs=None, task_config="demo_clean",
            episodes_per_pair=1, fractions=[.25, .5, .75], inference_steps=10, seed=42, gpus=[0, 1, 2])

    def prepare(self):
        plan = build_plan(self.args)
        output = Path(plan["output"])
        output.mkdir()
        write_json(output / "plan.json", plan)
        return plan, prepare_states(plan)

    def test_planning_is_read_only_and_uses_training_split(self):
        plan = build_plan(self.args)
        self.assertFalse(Path(self.args.output).exists())
        self.assertIn(plan["pairs"][0]["episode_index"], training_episode_ids(3, .05))
        self.assertNotIn(plan["pairs"][0]["episode_index"], set(range(3)) - training_episode_ids(3, .05))

    def test_real_hdf5_preparation_selects_informative_shared_prefix(self):
        plan, states = self.prepare()
        candidates = [s for s in states if s["is_prefix_boundary_candidate"]]
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["dual_reference_valid"])
        self.assertEqual(candidates[0]["frame_index"], 39)
        with np.load(candidates[0]["file"]) as data:
            self.assertEqual(data["own_reference"].shape, (32, 14))
            self.assertGreater(difference(data["source_reference"], data["target_reference"])["rms"], 0)
        later = [s for s in states if s["frame_index"] == 51]
        self.assertEqual(len(later), 2)
        self.assertTrue(all(not s["dual_reference_valid"] for s in later))
        self.assertEqual(len({s["id"] for s in states}), len(states))

    def test_strict_native_is_raw_audit_only_without_converted_dataset(self):
        strict = self.root / "expert/strict/demo_clean"
        shutil.copytree(self.domain, strict)
        shutil.rmtree(strict / "lerobot" / self.pair / "native")
        self.cfg["data"]["train"]["pgc_counterfactual_dataset_dirs"].append(
            str(strict / "lerobot" / self.pair / "counterfactual"))
        write_json(self.train / "config.yaml", self.cfg)
        self.args.profiles = ["strict"]
        _, states = self.prepare()
        self.assertTrue(any(s["expert_kind"] == "native" for s in states))
        self.assertTrue(all(s["own_reference_in_training_split"] ==
                            (s["expert_kind"] == "counterfactual") for s in states))

    def test_changed_raw_actions_fail_before_model_start(self):
        import h5py
        plan = build_plan(self.args)
        with h5py.File(plan["pairs"][0]["native"]["hdf5"], "r+") as handle:
            handle["joint_action/vector"][2, 1] = 4
        Path(plan["output"]).mkdir()
        with self.assertRaisesRegex(ValueError, "action hash"):
            prepare_states(plan)

    def test_changed_converted_provenance_rejected(self):
        audit = self.domain / "lerobot" / self.pair / "counterfactual/meta/pgc_episodes.jsonl"
        rows = [json.loads(line) for line in audit.read_text().splitlines()]
        rows[0]["scene_seed"] = 999
        audit.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            build_plan(self.args)

    def test_unmatched_pair_cannot_be_silently_reused(self):
        a = {"pair_id": "p", "scene_seed": 0, "initial_state_sha256": "hash",
             "source_instruction": "left", "counterfactual_instruction": "right",
             "dataset_kind": "native", "source_goal_verified": True}
        b = dict(a, dataset_kind="counterfactual", counterfactual_goal_verified=True)
        require_pair(a, b)
        b["scene_seed"] = 1
        with self.assertRaises(ValueError):
            require_pair(a, b)

    def test_summary_requires_complete_models_and_compares_same_states(self):
        plan, states = self.prepare()
        root = Path(plan["output"])
        with self.assertRaises(FileNotFoundError):
            summarize(root)
        with self.assertRaises(FileNotFoundError):
            audit_actions(root)
        for model in plan["checkpoints"]:
            (root / model).mkdir()
            records = []
            for state in states:
                with np.load(state["file"]) as data:
                    ref = data["own_reference"].copy()
                    source, target = ref.copy(), ref.copy()
                    if model != "base":
                        target = target + .1
                    source_ref = data["source_reference"] if state["dual_reference_valid"] else None
                    target_ref = data["target_reference"] if state["dual_reference_valid"] else None
                    metrics = reference_metrics(source, target, ref, state["expert_kind"], source_ref, target_ref)
                action_path = root / model / f"{state['id']}.npz"
                paired_refs = ({"source_reference": source_ref, "target_reference": target_ref}
                               if source_ref is not None else {})
                np.savez(action_path, source=source, target=target, **paired_refs)
                records.append({**state, "model": model, "actions_file": str(action_path),
                                "metrics_normalized": metrics, "metrics_executed_prefix24": metrics})
            (root / model / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            write_json(root / model / "complete.json", {"states": len(states)})
        summarize(root)
        summary = json.loads((root / "summary.json").read_text())
        self.assertEqual(summary["models"]["base"]["language_delta_rms"], 0)
        self.assertAlmostEqual(summary["models"]["step500"]["target_delta_vs_base_rms"], .1, places=6)
        self.assertTrue((root / "comparisons.csv").is_file())
        before = (root / "summary.json").read_bytes()
        audit_summary = audit_actions(root)
        self.assertEqual(audit_summary["matched_states_per_model"], 4)
        self.assertEqual(audit_summary["models"]["step1000"]["32"]["both_source"], 3)
        self.assertEqual((root / "summary.json").read_bytes(), before)
        self.assertTrue((root / "action_audit.csv").is_file())
        # Alter one model's expert reference: a cross-model comparison must fail.
        state = next(s for s in states if s["dual_reference_valid"])
        path = root / "step1000" / f"{state['id']}.npz"
        with np.load(path) as data:
            values = {key: data[key] for key in data.files}
        values["target_reference"] = values["target_reference"] + 1
        np.savez(path, **values)
        with self.assertRaisesRegex(ValueError, "expert reference mismatch"):
            audit_actions(root)


if __name__ == "__main__":
    unittest.main()
