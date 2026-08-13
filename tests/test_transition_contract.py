import tempfile
import unittest

import torch

from fastwam.models.wan22.transition_contract import ContrastiveContractLoss
from test_langforce_mvp import tiny_fastwam


LORA_CONFIG = {
    "enabled": True,
    "rank": 2,
    "alpha": 4,
    "dropout": 0.0,
    "experts": ["video", "action"],
}


class TransitionContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model = tiny_fastwam(transition_contract=True)

    def test_router_shapes_language_path_and_no_action_shortcuts(self):
        video_tokens = torch.randn(2, 8, 16)
        context = torch.randn(2, 4, 10)
        full_mask = torch.ones(2, 4, dtype=torch.bool)
        routed_a, z_a, metrics = self.model.encode_intended_transition(
            video_tokens=video_tokens,
            video_tokens_per_frame=4,
            context=context,
            full_context_mask=full_mask,
            route_scale=1.0,
        )
        routed_b, z_b, _ = self.model.encode_intended_transition(
            video_tokens=video_tokens,
            video_tokens_per_frame=4,
            context=context + 0.5,
            full_context_mask=full_mask,
            route_scale=1.0,
        )
        self.assertEqual(tuple(routed_a.shape), (2, 2, 12))
        self.assertEqual(tuple(z_a.shape), (2, 8))
        self.assertGreater((routed_a - routed_b).abs().max().item(), 1.0e-7)
        self.assertGreater((z_a - z_b).abs().max().item(), 1.0e-7)
        torch.testing.assert_close(z_a.norm(dim=-1), torch.ones(2))
        self.assertIn("router_attention_entropy", metrics)

        action_pre = self.model._prepare_action_tokens(
            action_tokens=torch.randn(2, 3, 3),
            timestep=torch.tensor([0.2, 0.4]),
            context=context,
            full_context_mask=full_mask,
            state_only_context_mask=torch.zeros_like(full_mask),
            mode="posterior",
            transition_query_tokens=routed_a,
        )
        self.assertFalse(action_pre["context_mask"].any())

        attention = self.model._build_mot_attention_mask(
            video_seq_len=8,
            action_seq_len=5,
            video_tokens_per_frame=4,
            device=torch.device("cpu"),
            num_queries=2,
            action_reads_raw_video=False,
            queries_read_raw_video=False,
        )
        self.assertFalse(attention[8:, :8].any())
        self.assertTrue(attention[10:, 8:10].all())

    def test_zero_scale_recovers_legacy_posterior_query_interface(self):
        video_tokens = torch.randn(2, 8, 16)
        context = torch.randn(2, 4, 10)
        full_mask = torch.ones(2, 4, dtype=torch.bool)
        routed, _, metrics = self.model.encode_intended_transition(
            video_tokens=video_tokens,
            video_tokens_per_frame=4,
            context=context,
            full_context_mask=full_mask,
            route_scale=0.0,
        )
        expected = self.model.action_expert.latent_action_queries.expand(2, -1, -1)
        torch.testing.assert_close(routed, expected, rtol=0, atol=0)
        self.assertEqual(float(metrics["router_route_scale"]), 0.0)

        state_mask = torch.zeros_like(full_mask)
        recovered = self.model._prepare_action_tokens(
            action_tokens=torch.randn(2, 3, 3),
            timestep=torch.tensor([0.2, 0.4]),
            context=context,
            full_context_mask=full_mask,
            state_only_context_mask=state_mask,
            mode="posterior",
            transition_query_tokens=routed,
            policy_recovery=True,
        )
        self.assertTrue(recovered["context_mask"][:, :2].all())
        self.assertFalse(recovered["context_mask"][:, 2:].any())

    def test_video_prefill_exposes_final_hidden_for_router(self):
        video_pre = self.model.video_expert.pre_dit(
            x=torch.randn(2, 2, 2, 2, 2),
            timestep=torch.tensor([0.2, 0.3]),
            context=torch.randn(2, 4, 10),
            context_mask=torch.ones(2, 4, dtype=torch.bool),
            action=torch.randn(2, 3, 3),
            fuse_vae_embedding_in_latents=True,
        )
        cache, final_hidden = self.model._run_video_expert_to_final_hidden(video_pre)
        self.assertEqual(len(cache), 2)
        self.assertEqual(final_hidden.shape, video_pre["tokens"].shape)
        self.assertGreater(
            (final_hidden - video_pre["tokens"]).abs().max().item(), 1.0e-7
        )

    def test_recovery_start_matches_joint_m1_policy_numerically(self):
        self.model.eval()
        self.model.set_training_progress(0, 100)
        video_pre = self.model.video_expert.pre_dit(
            x=torch.randn(2, 2, 2, 2, 2),
            timestep=torch.tensor([0.2, 0.3]),
            context=torch.randn(2, 4, 10),
            context_mask=torch.ones(2, 4, dtype=torch.bool),
            action=torch.randn(2, 3, 3),
            fuse_vae_embedding_in_latents=True,
        )
        context = torch.randn(2, 4, 10)
        full_mask = torch.ones(2, 4, dtype=torch.bool)
        state_mask = torch.zeros_like(full_mask)
        noisy_action = torch.randn(2, 3, 3)
        timestep_action = torch.tensor([0.4, 0.6])

        recovered, final_hidden, _, metrics = self.model._forward_tc_v2_train(
            video_pre=video_pre,
            action_tokens=noisy_action,
            timestep_action=timestep_action,
            context=context,
            full_context_mask=full_mask,
            state_only_context_mask=state_mask,
        )

        action_pre = self.model._prepare_action_tokens(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            full_context_mask=full_mask,
            state_only_context_mask=state_mask,
            mode="posterior",
            transition_query_tokens=(
                self.model.action_expert.transition_queries.expand(2, -1, -1)
            ),
            policy_recovery=True,
        )
        attention = self.model._build_mot_attention_mask(
            video_seq_len=int(video_pre["tokens"].shape[1]),
            action_seq_len=int(action_pre["tokens"].shape[1]),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=torch.device("cpu"),
            num_queries=2,
            action_reads_raw_video=False,
            queries_read_raw_video=True,
        )
        joint = self.model.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        expected = self.model.action_expert.post_dit(joint["action"], action_pre)
        torch.testing.assert_close(recovered, expected, rtol=0, atol=0)
        self.assertEqual(float(metrics["router_route_scale"]), 0.0)
        self.assertEqual(float(metrics["policy_recovery_joint_m1"]), 1.0)
        torch.testing.assert_close(final_hidden, joint["video"], rtol=0, atol=0)

    def test_recovery_start_preserves_joint_m1_dropout_rng(self):
        dropout_config = {**LORA_CONFIG, "dropout": 0.5}
        self.model.configure_lora(dropout_config)
        with torch.no_grad():
            for module in self.model.modules():
                if hasattr(module, "lora_B"):
                    module.lora_B.normal_(mean=0.0, std=0.05)
        self.model.train()
        self.model.set_training_progress(0, 100)

        video_pre = self.model.video_expert.pre_dit(
            x=torch.randn(2, 2, 2, 2, 2),
            timestep=torch.tensor([0.2, 0.3]),
            context=torch.randn(2, 4, 10),
            context_mask=torch.ones(2, 4, dtype=torch.bool),
            action=torch.randn(2, 3, 3),
            fuse_vae_embedding_in_latents=True,
        )
        noisy_action = torch.randn(2, 3, 3)
        timestep_action = torch.tensor([0.4, 0.6])
        context = torch.randn(2, 4, 10)
        full_mask = torch.ones(2, 4, dtype=torch.bool)
        state_mask = torch.zeros_like(full_mask)

        rng_state = torch.random.get_rng_state()
        recovered, final_hidden, _, _ = self.model._forward_tc_v2_train(
            video_pre=video_pre,
            action_tokens=noisy_action,
            timestep_action=timestep_action,
            context=context,
            full_context_mask=full_mask,
            state_only_context_mask=state_mask,
        )
        torch.random.set_rng_state(rng_state)
        expected, _, expected_video = (
            self.model._run_joint_m1_policy_with_video_cache(
                video_pre=video_pre,
                action_tokens=noisy_action,
                timestep_action=timestep_action,
                context=context,
                full_context_mask=full_mask,
                state_only_context_mask=state_mask,
            )
        )
        torch.testing.assert_close(recovered, expected, rtol=0, atol=0)
        torch.testing.assert_close(final_hidden, expected_video, rtol=0, atol=0)

    def test_recovery_ramp_blends_complete_policy_outputs(self):
        self.model.eval()
        self.model.set_training_progress(20, 100)
        video_pre = self.model.video_expert.pre_dit(
            x=torch.randn(2, 2, 2, 2, 2),
            timestep=torch.tensor([0.2, 0.3]),
            context=torch.randn(2, 4, 10),
            context_mask=torch.ones(2, 4, dtype=torch.bool),
            action=torch.randn(2, 3, 3),
            fuse_vae_embedding_in_latents=True,
        )
        _, _, _, metrics = self.model._forward_tc_v2_train(
            video_pre=video_pre,
            action_tokens=torch.randn(2, 3, 3),
            timestep_action=torch.tensor([0.4, 0.6]),
            context=torch.randn(2, 4, 10),
            full_context_mask=torch.ones(2, 4, dtype=torch.bool),
            state_only_context_mask=torch.zeros(2, 4, dtype=torch.bool),
        )
        self.assertAlmostEqual(float(metrics["router_route_scale"]), 0.5)
        self.assertGreater(float(metrics["policy_recovery_output_gap"]), 0.0)

    def test_policy_recovery_and_router_schedule(self):
        self.model.train()
        self.model.set_training_progress(0, 100)
        self.assertEqual(self.model._transition_router_scale(), 0.0)
        self.model.set_training_progress(20, 100)
        self.assertAlmostEqual(self.model._transition_router_scale(), 0.5)
        self.model.set_training_progress(29, 100)
        self.assertLess(self.model._transition_router_scale(), 1.0)
        self.model.set_training_progress(30, 100)
        self.assertEqual(self.model._transition_router_scale(), 1.0)
        self.model.set_training_progress(10, 100)
        self.assertEqual(self.model._transition_contract_scale(), 1.0)
        evaluation_model = tiny_fastwam(transition_contract=True).eval()
        self.assertEqual(evaluation_model._transition_router_scale(), 1.0)

    def test_contract_loss_batch_one_is_finite(self):
        z_l = torch.randn(1, 8, requires_grad=True)
        z_f = torch.randn(1, 8, requires_grad=True)
        loss, metrics = ContrastiveContractLoss()(z_l, z_f)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(metrics["contract_candidate_count"]), 1.0)
        loss.backward()

    def test_realized_transition_uses_clean_current_and_future(self):
        latents = torch.randn(2, 2, 2, 2, 2)
        context = torch.randn(2, 4, 10)
        z_future = self.model.encode_realized_transition(
            clean_input_latents=latents,
            context=context,
            full_context_mask=torch.ones(2, 4, dtype=torch.bool),
            action=torch.randn(2, 3, 3),
            fuse_vae_embedding_in_latents=True,
        )
        self.assertEqual(tuple(z_future.shape), (2, 8))
        self.assertTrue(torch.isfinite(z_future).all())
        torch.testing.assert_close(z_future.norm(dim=-1), torch.ones(2))

    def test_contract_retrieval_detects_aligned_embeddings(self):
        z_l = torch.eye(4)
        z_f = torch.eye(4)
        loss, metrics = ContrastiveContractLoss(temperature=0.07)(z_l, z_f)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(metrics["contract_retrieval_acc"]), 1.0)
        self.assertGreater(float(metrics["sim_LF_margin"]), 0.0)

    def test_lora_exposes_and_round_trips_transition_modules(self):
        self.model.configure_lora(LORA_CONFIG)
        report = self.model.prepare_trainable_parameters()
        trainable = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(
            any(name.startswith("transition_contract_modules.") for name in trainable)
        )
        self.assertLess(report["trainable"], report["total"])

        with torch.no_grad():
            parameter = next(self.model.transition_contract_modules.parameters())
            parameter.add_(0.123)
            expected = parameter.detach().clone()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tc_adapter.pt"
            self.model.save_checkpoint(path, step=5)
            payload = torch.load(path, map_location="cpu")
            self.assertEqual(
                payload["architecture_metadata"]["architecture"], "tc_fastwam"
            )
            self.assertEqual(
                payload["architecture_metadata"]["transition_contract_version"],
                2,
            )
            self.assertEqual(
                payload["architecture_metadata"]["router_visual_source"],
                "video_expert_final_hidden",
            )
            self.assertEqual(
                payload["architecture_metadata"]["policy_recovery_blend"],
                "action_flow_velocity",
            )
            self.assertEqual(
                payload["architecture_metadata"]["policy_recovery_source"],
                "joint_mot_posterior",
            )
            self.assertNotIn(
                "transition_contract_modules.*",
                payload["lora_config"]["extra_trainable_patterns"],
            )
            restored = tiny_fastwam(transition_contract=True)
            restored.load_checkpoint(path)
        actual = next(restored.transition_contract_modules.parameters())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_legacy_m1_checkpoint_loads_into_tc_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = f"{tmpdir}/base.pt"
            path = f"{tmpdir}/m1.pt"
            legacy = tiny_fastwam()
            legacy.save_checkpoint(base_path)
            legacy.configure_lora(LORA_CONFIG)
            legacy.load_checkpoint(base_path)
            legacy.save_checkpoint(path)
            self.model.load_checkpoint(path)
            self.assertEqual(self.model.transition_policy_init_checkpoint, path)
            self.assertEqual(self.model.lora_base_checkpoint, base_path)
            tc_path = f"{tmpdir}/tc_v2.pt"
            self.model.save_checkpoint(tc_path)
            tc_payload = torch.load(tc_path, map_location="cpu")
            self.assertEqual(tc_payload["base_checkpoint"], base_path)
            self.assertEqual(
                tc_payload["architecture_metadata"]["policy_init_checkpoint"],
                path,
            )
        torch.testing.assert_close(
            self.model.action_expert.latent_action_queries,
            legacy.action_expert.latent_action_queries,
        )

    def test_v1_tc_adapter_is_rejected_before_mutating_v2_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tc_v1.pt"
            self.model.configure_lora(LORA_CONFIG)
            self.model.save_checkpoint(path)
            payload = torch.load(path, map_location="cpu")
            payload["architecture_metadata"]["transition_contract_version"] = 1
            torch.save(payload, path)

            restored = tiny_fastwam(transition_contract=True)
            with self.assertRaisesRegex(ValueError, "version mismatch"):
                restored.load_checkpoint(path)
            self.assertFalse(restored.lora_enabled)


if __name__ == "__main__":
    unittest.main()
