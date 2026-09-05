import argparse
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.lora import (
    DEFAULT_TARGET_MODULES, inject_lora, is_lora_parameter_name, matches_any_pattern,
)
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from experiments.robotwin.same_state_repair import (
    action_rows, anchor_gradient_audit, audit_frozen, backward_paired_anchor,
    backward_paired_flow, checkpoint_score, configure_action_lora, fixed_flow_rows,
    frozen_versions, move_cache, noise_tensor, paired_velocity_losses,
    predict_with_grad, repair_payload, sample_cached_actions,
)
from scripts import train_robotwin_same_state_repair as runner
from scripts.probe_robotwin_no_eraf import sha256, write_json
from tests import test_robotwin_no_eraf_probe as probe_fixtures


class TinyModel(nn.Module):
    """Real VideoDiT/ActionDiT/MoT with production FastWAM method bodies.

    Extracting only these methods avoids importing pretrained-model downloaders
    and tokenizers. No attention, scheduler, adapter, or backward math is mocked.
    """

    def __init__(self):
        super().__init__()
        self.video_expert = WanVideoDiT(
            hidden_dim=16, in_dim=2, ffn_dim=32, out_dim=2, text_dim=10, freq_dim=8,
            eps=1e-6, patch_size=(1, 1, 1), num_heads=2, attn_head_dim=4, num_layers=2,
            has_image_input=False, seperated_timestep=True, require_vae_embedding=False,
            require_clip_embedding=False, fuse_vae_embedding_in_latents=True,
            action_conditioned=False, video_attention_mask_mode="first_frame_causal")
        self.action_expert = ActionDiT(
            hidden_dim=12, action_dim=14, ffn_dim=24, text_dim=10, freq_dim=8, eps=1e-6,
            num_heads=2, attn_head_dim=4, num_layers=2, use_latent_action_queries=False)
        self.mot = MoT({"video": self.video_expert, "action": self.action_expert}, mot_checkpoint_mixed_attn=False)
        self.proprio_encoder = nn.Linear(14, 10)
        for expert in (self.video_expert, self.action_expert):
            inject_lora(expert, target_modules=DEFAULT_TARGET_MODULES, rank=2, alpha=2, dropout=.05)
            with torch.no_grad():
                for name, parameter in expert.named_parameters():
                    if name.endswith(".lora_B"):
                        parameter.normal_(std=.02)
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.lora_enabled = True
        self.transition_contract_enabled = False
        self.uses_transition_queries = False
        self.lora_config = {"enabled": True, "extra_trainable_patterns": [], "experts": ["video", "action"]}
        self.lora_base_checkpoint = "/tmp/repair-test-base.pt"
        self.train_action_scheduler = WanContinuousFlowMatchScheduler()
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler()
        self.eval()

    def cache(self, language):
        generator = torch.Generator().manual_seed(10 if language == "source" else 11)
        context = torch.randn((1, 3, 10), generator=generator).to(dtype=self.torch_dtype)
        mask = torch.ones((1, 3), dtype=torch.bool)
        with torch.no_grad():
            pre = self.video_expert.pre_dit(
                x=torch.ones((1, 2, 1, 2, 2), dtype=self.torch_dtype),
                timestep=torch.zeros(1, dtype=self.torch_dtype), context=context,
                context_mask=mask, action=None, fuse_vae_embedding_in_latents=True)
            length = pre["tokens"].shape[1]
            cache = self.mot.prefill_video_cache(
                video_tokens=pre["tokens"], video_freqs=pre["freqs"], video_t_mod=pre["t_mod"],
                video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
                video_attention_mask=torch.ones((length, length), dtype=torch.bool))
        return {"context": context, "context_mask": mask, "state_only_context_mask": mask,
                "video_kv_cache": cache, "attention_mask": torch.ones((length + 32, length + 32), dtype=torch.bool),
                "video_seq_len": length, "routed_transition_tokens": None}

    def payload(self):
        return {"format": "fastwam_lora_adapter_v1", "step": 1000, "lora_config": self.lora_config,
                "mot_trainable": {k: v.clone() for k, v in self._lora_adapter_state_dict().items()}}


tree = ast.parse((runner.REPO / "src/fastwam/models/wan22/fastwam.py").read_text())
fastwam = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FastWAM")
for method_name in ("_prepare_action_tokens", "_predict_action_noise_with_cache",
                    "_adapter_parameter_ids", "_lora_adapter_state_dict"):
    method = next(node for node in fastwam.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = {"torch": torch, "Optional": Optional, "Any": object,
                 "is_lora_parameter_name": is_lora_parameter_name, "matches_any_pattern": matches_any_pattern}
    exec(compile(ast.Module(body=[copy.deepcopy(method)], type_ignores=[]), "production_fastwam_methods", "exec"), namespace)
    setattr(TinyModel, method_name, namespace[method_name])


class RepairAutogradTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(2)
        self.model = TinyModel()
        self.initial = self.model.payload()
        self.selected = configure_action_lora(self.model)
        self.protected = frozen_versions(self.model)
        self.caches = {kind: self.model.cache(kind) for kind in ("source", "target")}
        self.noise = noise_tensor((1, 32, 14), 91, self.model)
        self.time = torch.tensor([1000.])
        self.refs = {"source": torch.zeros_like(self.noise), "target": torch.ones_like(self.noise) * .7}

    def test_real_production_predictor_is_identical_with_and_without_autograd(self):
        cache = self.caches["source"]
        ordinary = self.model._predict_action_noise_with_cache(
            latents_action=self.noise, timestep_action=self.time, **cache)
        differentiable = predict_with_grad(self.model, cache, self.noise, self.time)
        self.assertFalse(ordinary.requires_grad)
        self.assertTrue(differentiable.requires_grad)
        torch.testing.assert_close(ordinary, differentiable, rtol=0, atol=0)
        differentiable.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.selected.values()))
        audit_frozen(self.model, self.protected)
        self.assertNotIn("_predict_action_noise_with_cache", vars(self.model))

    def test_explicit_endpoint_weight_trains_real_action_lora_and_preserves_video(self):
        scheduler_weight = float(self.model.train_action_scheduler.training_weight(self.time))
        self.assertAlmostEqual(scheduler_weight, 0., places=6)
        before = {name: value.detach().clone() for name, value in self.selected.items()}
        errors = backward_paired_flow(self.model, self.caches, self.refs, self.noise, self.time, .25)
        self.assertTrue(all(np.isfinite(value) for value in errors.values()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.selected.values()))
        optimizer = torch.optim.AdamW(list(self.selected.values()), lr=1e-3, weight_decay=0)
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in self.selected.items()))
        audit_frozen(self.model, self.protected)
        payload = repair_payload(self.model, self.initial, 1, {"arm": "paired_flow_anchor"})
        self.assertEqual(payload["step"], 1001)
        for name, value in self.initial["mot_trainable"].items():
            if name.startswith("mixtures.video."):
                torch.testing.assert_close(payload["mot_trainable"][name], value, rtol=0, atol=0)

    def test_paired_backward_matches_equal_weight_joint_objective(self):
        backward_paired_flow(self.model, self.caches, self.refs, self.noise, self.time, .25)
        actual = {name: p.grad.clone() if p.grad is not None else None for name, p in self.selected.items()}
        self.model.zero_grad(set_to_none=True)
        losses = []
        for language in ("source", "target"):
            prediction = predict_with_grad(self.model, self.caches[language], self.noise, self.time, checkpoint=False)
            losses.append((prediction - (self.noise - self.refs[language])).square().mean())
        ((losses[0] + losses[1]) * .25 / 2).backward()
        for name, parameter in self.selected.items():
            if actual[name] is not None:
                torch.testing.assert_close(actual[name], parameter.grad, rtol=1e-5, atol=1e-7)

    def test_anchor_gain_one_matches_original_positive_loss_gradients(self):
        old_errors = backward_paired_flow(self.model, self.caches, self.refs, self.noise, self.time, .25)
        expected = {name: p.grad.clone() if p.grad is not None else None for name, p in self.selected.items()}
        self.model.zero_grad(set_to_none=True)
        errors = backward_paired_anchor(self.model, self.caches, self.refs, self.noise, .25)
        self.assertAlmostEqual(errors["common_mse"] + errors["conditional_mse"],
                               (old_errors["source"] + old_errors["target"]) / 2, places=6)
        for name, parameter in self.selected.items():
            if expected[name] is None:
                self.assertIsNone(parameter.grad)
            else:
                torch.testing.assert_close(expected[name], parameter.grad, rtol=0, atol=0)

    def test_difference_loss_corrects_reversed_velocity_with_both_branch_gradients(self):
        # source action=0, CF action=1, common noise=0 => velocities 0 and -1.
        # Start with the velocities reversed. Descent must raise source and lower CF.
        source, target = nn.Parameter(torch.tensor([[[-1.]]])), nn.Parameter(torch.tensor([[[0.]]]))
        losses = paired_velocity_losses({"source": source, "target": target},
                                        {"source": torch.zeros_like(source), "target": -torch.ones_like(target)})
        (losses["common_mse"] + 4 * losses["conditional_mse"]).backward()
        self.assertAlmostEqual(float(source.grad), -4.)
        self.assertAlmostEqual(float(target.grad), 4.)
        self.assertAlmostEqual(float(losses["common_mse"].detach()), 0.)
        self.assertAlmostEqual(float(losses["conditional_mse"].detach()), 1.)

    def test_initial_component_gradient_audit_does_not_populate_optimizer_grads(self):
        versions = frozen_versions(self.model)
        before = {key: p.detach().clone() for key, p in self.selected.items()}
        audit = anchor_gradient_audit(self.model, self.caches, self.refs, self.noise)
        self.assertGreater(audit["common_grad_norm"], 0)
        self.assertGreater(audit["conditional_grad_norm"], 0)
        self.assertTrue(all(p.grad is None for p in self.selected.values()))
        self.assertTrue(all(torch.equal(before[key], p) for key, p in self.selected.items()))
        audit_frozen(self.model, versions)

    def test_fixed_flow_is_repeatable_and_matches_the_production_endpoint(self):
        before = torch.get_rng_state()
        rows = fixed_flow_rows(self.model, self.caches, self.refs, [91, 92], [.5, 1.])
        self.assertEqual(rows, fixed_flow_rows(self.model, self.caches, self.refs, [91, 92], [.5, 1.]))
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(len(rows), 8)
        row = next(r for r in rows if r["noise_seed"] == 91 and r["sigma"] == 1 and r["horizon"] == 32)
        prediction = self.model._predict_action_noise_with_cache(
            latents_action=self.noise, timestep_action=self.time, **self.caches["source"])
        self.assertAlmostEqual(row["source_mse"], float((prediction - (self.noise - self.refs["source"])).square().mean()))
        self.assertTrue(all(p.grad is None for p in self.selected.values()))

    def test_bfloat16_backbone_keeps_float32_action_adapter_gradients(self):
        self.model.to(dtype=torch.bfloat16)
        self.model.torch_dtype = torch.bfloat16
        selected = configure_action_lora(self.model)
        caches = {kind: self.model.cache(kind) for kind in ("source", "target")}
        refs = {kind: value.to(torch.bfloat16) for kind, value in self.refs.items()}
        backward_paired_flow(self.model, caches, refs, self.noise.to(torch.bfloat16),
                             self.time.to(torch.bfloat16), .25)
        gradients = [p.grad for p in selected.values() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(g.abs().sum() > 0 for g in gradients))
        expected = {key: p.grad.clone() if p.grad is not None else None for key, p in selected.items()}
        self.model.zero_grad(set_to_none=True)
        backward_paired_anchor(self.model, caches, refs, self.noise.to(torch.bfloat16), .25, conditional_gain=1.)
        for key, p in selected.items():
            if expected[key] is None:
                self.assertIsNone(p.grad)
            else:
                torch.testing.assert_close(p.grad, expected[key], rtol=0, atol=0)
        self.model.zero_grad(set_to_none=True)
        backward_paired_anchor(self.model, caches, refs, self.noise.to(torch.bfloat16), .25, conditional_gain=4.)
        gradients = [p.grad for p in selected.values() if p.grad is not None]
        self.assertTrue(all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(g.abs().sum() > 0 for g in gradients))

    def test_checkpoint_keeps_inherited_video_and_round_trips_predictions(self):
        backward_paired_flow(self.model, self.caches, self.refs, self.noise, self.time, .25)
        torch.optim.AdamW(list(self.selected.values()), lr=1e-3).step()
        payload = repair_payload(self.model, self.initial, 1, {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repair.pt"
            torch.save(payload, path)
            restored = torch.load(path, weights_only=False)
        clone = copy.deepcopy(self.model)
        clone.mot.load_state_dict(restored["mot_trainable"], strict=False)
        actual = sample_cached_actions(self.model, self.caches["source"], 42, 10)
        expected = sample_cached_actions(clone, clone.cache("source"), 42, 10)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        with patch.object(self.model, "_lora_adapter_state_dict", return_value={
            k: v for k, v in payload["mot_trainable"].items() if k.startswith("mixtures.action.")}):
            with self.assertRaisesRegex(ValueError, "lost inherited"):
                repair_payload(self.model, self.initial, 1, {})

    def test_frozen_mutation_is_detected(self):
        with torch.no_grad():
            self.model.proprio_encoder.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, "frozen parameter changed"):
            audit_frozen(self.model, self.protected)

    def test_cache_clone_can_cross_an_inference_mode_boundary(self):
        with torch.inference_mode():
            cache = {"x": torch.ones(2), "nested": [torch.zeros(1), None]}
        cloned = move_cache(cache, "cpu", clone=True)
        parameter = nn.Parameter(torch.ones(2))
        (cloned["x"] * parameter).sum().backward()
        torch.testing.assert_close(parameter.grad, torch.ones(2))


class RepairScoringTest(unittest.TestCase):
    def rows(self, source, target, split="repair"):
        source_ref, target_ref = np.zeros((32, 14)), np.ones((32, 14))
        return [{"id": "s", "split": split, **row} for row in action_rows(
            source, target, source_ref, target_ref, source_ref + .4, target_ref - .4)]

    def test_both_target_and_reversed_predictions_cannot_pass(self):
        source, target = np.zeros((32, 14)), np.ones((32, 14))
        for rows in (self.rows(target, target), self.rows(target, source)):
            self.assertFalse(checkpoint_score(rows)["fit_pass"])

    def test_fit_pass_requires_each_correct_branch_to_improve(self):
        source, target = np.zeros((32, 14)), np.ones((32, 14))
        rows = self.rows(source + .1, target - .1)
        self.assertTrue(checkpoint_score(rows)["fit_pass"])
        rows[0]["initial_source_rmse"] = .01
        self.assertFalse(checkpoint_score(rows)["fit_pass"])

    def test_regression_blocks_checkpoint_selection_and_no_guard_is_not_a_pass(self):
        source, target = np.zeros((32, 14)), np.ones((32, 14))
        fit = self.rows(source + .1, target - .1)
        self.assertFalse(checkpoint_score(fit)["eligible_for_closed_loop_check"])
        guard = self.rows(source + .1, target - .1, split="guard")
        self.assertTrue(checkpoint_score(fit + guard)["eligible_for_closed_loop_check"])
        guard[0]["source_correct_rmse"] = 1.
        score = checkpoint_score(fit + guard)
        self.assertFalse(score["guard_pass"])
        self.assertEqual(score["guard_regressions"][0]["language"], "source")


class RepairPipelineTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)
        fixture = probe_fixtures.PreparationTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        with contextlib.redirect_stdout(io.StringIO()):
            self.source, self.states = fixture.prepare()
        self.root = Path(self.source["output"])
        torch.manual_seed(3)
        self.model = TinyModel()
        self.model.lora_base_checkpoint = str(fixture.base)
        # A synthetic paired window with shared initial command and distinct
        # remaining actions, used only for this CPU pipeline integration test.
        state = next(s for s in self.states if s["frame_index"] == 0)
        with np.load(state["file"]) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["target_reference"][1:] = .5
        np.savez_compressed(state["file"], **arrays)
        state["sha256"] = sha256(state["file"])
        write_json(self.root / "states.json", self.states)
        torch.save(self.model.payload(), self.source["checkpoints"]["step1000"])
        self.source["checkpoint_sha256"] = {k: sha256(v) for k, v in self.source["checkpoints"].items()}
        write_json(self.root / "plan.json", self.source)
        write_json(self.root / "summary.json", {"format": self.source["format"], "complete": True})
        output = self.root / "step1000"
        output.mkdir()
        rows = []
        predictions = {kind: sample_cached_actions(self.model, self.model.cache(kind), 42, 10)
                       for kind in ("source", "target")}
        for entry in self.states:
            path = output / f"{entry['id']}.npz"
            with np.load(entry["file"]) as data:
                refs = {name: data[name] for name in ("source_reference", "target_reference") if name in data.files}
            np.savez_compressed(path, **predictions, **refs)
            rows.append({**entry, "actions_file": str(path)})
        (output / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        write_json(output / "complete.json", {"states": len(self.states)})
        self.args = argparse.Namespace(source_probe=str(self.root), output=str(fixture.root / "repair"),
            initial_model="step1000", pairs=[fixture.pair], arms=["paired_flow_anchor"], gpus=[0],
            steps=2, eval_every=1, train_seed=17000, eval_seeds=[42, 43], learning_rate=1e-3,
            anchor_weight=.25, minimum_improvement=.05, guard_relative=.1, guard_absolute=.005)

    def test_plan_is_read_only_and_rejects_unauthorized_or_overlapping_training(self):
        plan = runner.build_plan(self.args)
        self.assertEqual(sum(s["split"] == "repair" for s in plan["states"]), 1)
        self.assertFalse(Path(self.args.output).exists())
        self.args.eval_seeds = [17001]
        with self.assertRaisesRegex(ValueError, "overlap"):
            runner.build_plan(self.args)
        self.args.eval_seeds = [42]
        self.source["pairs"][0]["native_in_training_split"] = False
        write_json(self.root / "plan.json", self.source)
        with self.assertRaisesRegex(ValueError, "authorized historical"):
            runner.build_plan(self.args)

    def test_source_repair_reuses_original_probe_and_rejects_changed_inputs(self):
        previous = runner.build_plan(self.args)
        previous_root = self.fixture.root / "previous-repair"
        previous_root.mkdir()
        write_json(previous_root / "plan.json", previous)
        write_json(previous_root / "summary.json", {"format": runner.FORMAT, "complete": True})
        folder = previous_root / "paired_flow_anchor"
        folder.mkdir()
        selected_id = next(s["id"] for s in previous["states"] if s["split"] == "repair")
        records = [{"step": step, "draws": [{"id": selected_id, "noise_seed": 17000 + step,
            "time": 500., "scheduler_weight": 1., "noise_sha256": "a" * 64}]} for step in (1, 2)]
        (folder / "training.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        self.args.source_repair = str(previous_root)
        self.args.source_probe = "/not-the-selected-source"
        plan = runner.build_plan(self.args)
        self.assertEqual(plan["source_probe"], previous["source_probe"])
        self.assertEqual(plan["initial_model"], "step1000")
        self.assertIn(str((previous_root / "plan.json").resolve()), plan["source_artifact_sha256"])
        self.assertEqual(plan["training_draws"]["1"][selected_id]["noise_seed"], 17001)
        self.assertFalse(Path(self.args.output).exists())
        source = copy.deepcopy(self.source)
        source["changed_since_repair"] = True
        write_json(self.root / "plan.json", source)
        with self.assertRaisesRegex(ValueError, "Source artifact changed"):
            runner.build_plan(self.args)

    def test_cpu_worker_runs_real_backward_saves_checkpoints_and_summarizes(self):
        class Policy:
            seed = 42
            policy_guard_state = None
            processor = SimpleNamespace(shape_meta={"action": [{"key": "default"}]}, normalizer=SimpleNamespace(
                normalizers={"action": {"default": SimpleNamespace(
                    scale=torch.ones(14), offset=torch.zeros(14), forward=lambda value: value)}}))

            def _infer_action_chunk(self, obs, instruction):
                kind = "source" if instruction == "left" else "target"
                return sample_cached_actions(self.model, self.model.cache(kind), self.seed, 10)

        policy = Policy()
        self.args.arms = list(runner.ARMS)
        self.args.fixed_flow_sigmas = [.5, 1.]
        self.args.audit_anchor_gradients = True
        plan = runner.build_plan(self.args)
        plan["normalization_state_count"] = 2
        # Keep a two-state-style noise schedule even though this tiny fixture
        # has one active state. A subset restart must not fall back to n=1 seeds.
        from experiments.robotwin.no_eraf_probe import typed_hash
        fit_id = next(s["id"] for s in plan["states"] if s["split"] == "repair")
        scheduler = self.model.train_action_scheduler
        plan["training_draws"] = {}
        for step in (1, 2):
            seed = plan["train_seed"] + 2 * step + 1
            u = torch.rand((1,), generator=torch.Generator().manual_seed(1_000_000_000 + seed))
            time = scheduler._phi(u, scheduler.shift) * scheduler.num_train_timesteps
            noise = noise_tensor((1, 32, 14), seed, self.model)
            plan["training_draws"][str(step)] = {fit_id: {"id": fit_id, "noise_seed": seed,
                "time": float(time.item()), "scheduler_weight": float(scheduler.training_weight(time)),
                "noise_sha256": typed_hash(noise[0].numpy())}}
        plan["git_commit"] = "cpu-fixture"
        root = Path(plan["output"])
        root.mkdir()
        write_json(root / "plan.json", plan)
        args = argparse.Namespace(plan=str(root / "plan.json"), arm=None, gpu=0)
        with contextlib.redirect_stdout(io.StringIO()):
            for arm in plan["arms"]:
                args.arm = arm
                policy.model = copy.deepcopy(self.model)
                with patch.object(runner, "load_probe_policy", return_value=(policy, {})):
                    runner.worker(args)
            runner.summarize(root)
        summary = json.loads((root / "summary.json").read_text())
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["fixed_flow_summary"])
        self.assertEqual(set(summary["anchor_gradient_audit"]), set(runner.ARMS))
        from scripts.inspect_robotwin_same_state_repair import inspect
        diagnostics = inspect(root)
        self.assertTrue(diagnostics["training_inputs_match_across_arms"])
        self.assertEqual(diagnostics["repair_states"], 1)
        json.dumps(diagnostics, allow_nan=False)
        final_eval = json.loads((root / args.arm / "evaluation_000002.json").read_text())
        self.assertEqual(len(final_eval["fixed_flow_rows"]), 8)
        self.assertTrue(all(row["sigma"] in (.5, 1.) for row in final_eval["fixed_flow_rows"]))
        checkpoint = torch.load(root / args.arm / "checkpoints/repair_000002.pt", weights_only=False)
        self.assertEqual(checkpoint["step"], 1002)
        self.assertEqual(checkpoint["robotwin_same_state_repair"]["optimizer_steps"], 2)
        self.assertTrue(any(key.startswith("mixtures.video.") for key in checkpoint["mot_trainable"]))
        log = (root / args.arm / "training.jsonl").read_text().splitlines()
        self.assertEqual(len(log), 2)
        self.assertTrue(all(json.loads(row)["gradient_norm_before_clip"] > 0 for row in log))
        # A restart with the recorded draws and denominator reproduces the
        # control exactly, with a fresh optimizer and the original weights.
        child_args = copy.deepcopy(self.args)
        child_args.source_repair = str(root)
        child_args.output = str(root.parent / "restart-control")
        child_args.arms = ["paired_flow_anchor"]
        child_plan = runner.build_plan(child_args)
        self.assertEqual(child_plan["normalization_state_count"], 2)
        child_plan["git_commit"] = "cpu-fixture"
        child_root = Path(child_plan["output"])
        child_root.mkdir()
        write_json(child_root / "plan.json", child_plan)
        policy.model = copy.deepcopy(self.model)
        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(runner, "load_probe_policy", return_value=(policy, {})):
                runner.worker(argparse.Namespace(plan=str(child_root / "plan.json"), arm="paired_flow_anchor", gpu=0))
        original_payload = torch.load(root / "paired_flow_anchor/checkpoints/repair_000002.pt", weights_only=False)
        restarted_payload = torch.load(child_root / "paired_flow_anchor/checkpoints/repair_000002.pt", weights_only=False)
        for key, value in original_payload["mot_trainable"].items():
            torch.testing.assert_close(value, restarted_payload["mot_trainable"][key], rtol=0, atol=0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.report("latest", root.parent), root)
        training_path = root / args.arm / "training.jsonl"
        altered = [json.loads(row) for row in log]
        altered[0]["draws"][0]["time"] += 1
        training_path.write_text("".join(json.dumps(row) + "\n" for row in altered))
        with self.assertRaisesRegex(ValueError, "draws differ"):
            runner.summarize(root)
        training_path.write_text("\n".join(log) + "\n")
        flow_path = root / args.arm / "evaluation_000001.json"
        original_flow = json.loads(flow_path.read_text())
        broken_flow = copy.deepcopy(original_flow)
        broken_flow["fixed_flow_rows"].pop()
        write_json(flow_path, broken_flow)
        with self.assertRaisesRegex(ValueError, "fixed-flow evaluations"):
            runner.summarize(root)
        write_json(flow_path, original_flow)
        (root / args.arm / "evaluation_000001.json").unlink()
        with self.assertRaises(FileNotFoundError):
            runner.summarize(root)


if __name__ == "__main__":
    unittest.main()
