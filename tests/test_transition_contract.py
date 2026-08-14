import json
import os
import tempfile
import unittest

import torch

from fastwam.datasets.counterfactual import (
    load_counterfactual_instruction_map,
    stable_instruction_id,
)
from fastwam.models.wan22.transition_contract import (
    ContrastiveContractLoss,
    CounterfactualActionPrototypeBank,
    CounterfactualRankingLoss,
)
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

        recovered, final_hidden, _, _, metrics = self.model._forward_tc_v2_train(
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
        recovered, final_hidden, _, _, _ = self.model._forward_tc_v2_train(
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
        _, _, _, _, metrics = self.model._forward_tc_v2_train(
            video_pre=video_pre,
            action_tokens=torch.randn(2, 3, 3),
            timestep_action=torch.tensor([0.4, 0.6]),
            context=torch.randn(2, 4, 10),
            full_context_mask=torch.ones(2, 4, dtype=torch.bool),
            state_only_context_mask=torch.zeros(2, 4, dtype=torch.bool),
        )
        self.assertAlmostEqual(float(metrics["router_route_scale"]), 0.5)
        self.assertGreater(
            float(metrics["policy_recovery_output_gap"].detach()),
            0.0,
        )

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

    def test_v3_freezes_m1_policy_and_exposes_only_contract_parameters(self):
        model = tiny_fastwam(
            transition_contract=True,
            transition_contract_version=3,
        )
        model.configure_lora(LORA_CONFIG)
        report = model.prepare_trainable_parameters()
        trainable = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertGreater(report["trainable"], 0)
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("transition_contract_modules.")
                for name in trainable
            )
        )
        self.assertFalse(model.mot.training)
        self.assertTrue(model.transition_contract_modules.training)
        metadata = model._transition_contract_metadata()
        self.assertEqual(metadata["transition_contract_version"], 3)
        self.assertEqual(
            metadata["student_policy_path"],
            "pure_router_from_step_zero",
        )
        self.assertEqual(
            metadata["policy_protection"],
            "requires_grad_false_and_optimizer_exclusion",
        )

    def test_v3_distills_frozen_joint_m1_and_trains_lf_representation(self):
        model = tiny_fastwam(
            transition_contract=True,
            transition_contract_version=3,
        )
        model.configure_lora(LORA_CONFIG)
        model.prepare_trainable_parameters()
        policy_before = {
            name: parameter.detach().clone()
            for name, parameter in model.mot.named_parameters()
        }
        router_before = {
            name: parameter.detach().clone()
            for name, parameter in model.transition_contract_modules[
                "router"
            ].named_parameters()
        }
        optimizer = torch.optim.SGD(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            lr=1.0e-2,
        )
        # The v2 recovery schedule is still at zero, but v3 must train the
        # pure Router student from the first optimizer step.
        model.set_training_progress(0, 100)

        context = torch.randn(2, 4, 10)
        full_mask = torch.ones(2, 4, dtype=torch.bool)
        state_mask = torch.zeros_like(full_mask)
        action = torch.randn(2, 3, 3)
        video_pre = model.video_expert.pre_dit(
            x=torch.randn(2, 2, 2, 2, 2),
            timestep=torch.tensor([0.2, 0.3]),
            context=context,
            context_mask=full_mask,
            action=action,
            fuse_vae_embedding_in_latents=True,
        )
        student, _, z_language, teacher, metrics = model._forward_tc_v2_train(
            video_pre=video_pre,
            action_tokens=torch.randn(2, 3, 3),
            timestep_action=torch.tensor([0.4, 0.6]),
            context=context,
            full_context_mask=full_mask,
            state_only_context_mask=state_mask,
        )
        self.assertIsNotNone(teacher)
        self.assertFalse(teacher.requires_grad)
        self.assertTrue(student.requires_grad)
        self.assertTrue(z_language.requires_grad)
        self.assertEqual(
            float(metrics["policy_recovery_joint_m1"].detach()),
            1.0,
        )
        self.assertEqual(
            float(metrics["router_recovery_schedule_scale"].detach()),
            0.0,
        )
        self.assertEqual(float(metrics["router_route_scale"].detach()), 1.0)
        self.assertGreater(
            float(metrics["policy_recovery_output_gap"].detach()),
            0.0,
        )

        z_future = model.encode_realized_transition(
            clean_input_latents=torch.randn(2, 2, 2, 2, 2),
            context=context,
            full_context_mask=full_mask,
            action=action,
            fuse_vae_embedding_in_latents=True,
        )
        loss_contract, _ = model.compute_transition_contract_loss(
            z_language, z_future
        )
        loss = torch.nn.functional.mse_loss(student.float(), teacher.float())
        loss = loss + loss_contract
        loss.backward()

        router_grads = [
            parameter.grad
            for parameter in model.transition_contract_modules["router"].parameters()
        ]
        outcome_grads = [
            parameter.grad
            for parameter in model.transition_contract_modules[
                "outcome_encoder"
            ].parameters()
        ]
        self.assertTrue(any(grad is not None for grad in router_grads))
        self.assertTrue(any(grad is not None for grad in outcome_grads))
        self.assertTrue(
            all(parameter.grad is None for parameter in model.mot.parameters())
        )
        optimizer.step()
        for name, parameter in model.mot.named_parameters():
            torch.testing.assert_close(
                parameter,
                policy_before[name],
                rtol=0,
                atol=0,
            )
        self.assertTrue(
            any(
                not torch.equal(parameter, router_before[name])
                for name, parameter in model.transition_contract_modules[
                    "router"
                ].named_parameters()
            )
        )

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

    def test_contract_masks_same_task_false_negatives(self):
        embeddings = torch.eye(3)
        loss, metrics = ContrastiveContractLoss(temperature=0.07)(
            embeddings,
            embeddings,
            group_ids=torch.tensor([5, 5, 9]),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(
            float(metrics["contract_same_task_negative_fraction_LF"]),
            1.0 / 3.0,
        )
        self.assertAlmostEqual(
            float(metrics["contract_effective_negative_count_LF"]),
            4.0 / 3.0,
        )

    def test_tc_full_action_future_contract_is_finite_and_trainable(self):
        model = tiny_fastwam(
            transition_contract=True,
            transition_contract_version=4,
        )
        z_action = model.encode_action_effect_transition(
            current_video_hidden=torch.randn(3, 4, 16),
            action=torch.randn(3, 5, 3),
            proprio=None,
            action_is_pad=torch.tensor(
                [
                    [False, False, False, False, False],
                    [False, False, False, True, True],
                    [False, False, False, False, True],
                ]
            ),
        )
        z_future = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
        loss, metrics = model.compute_action_future_contract_loss(
            z_action, z_future
        )
        self.assertEqual(tuple(z_action.shape), (3, 8))
        torch.testing.assert_close(z_action.norm(dim=-1), torch.ones(3))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("sim_AF_positive", metrics)
        self.assertIn("contract_retrieval_acc_AF", metrics)
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.transition_contract_modules[
                    "action_effect_encoder"
                ].parameters()
            )
        )

    def test_counterfactual_action_bank_uses_target_task_positives(self):
        bank = CounterfactualActionPrototypeBank(
            num_slots=4,
            num_queries=2,
            query_dim=3,
            action_effect_dim=3,
            momentum=0.0,
        )
        queries = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )
        actions = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        bank.update(
            task_ids=torch.tensor([10, 20]),
            query_residuals=queries,
            action_effects=actions,
        )
        loss, metrics = bank.positive_loss(
            counterfactual_task_ids=torch.tensor([20, 10]),
            query_residuals=queries.flip(0),
            action_intents=actions.flip(0),
            valid_mask=torch.ones(2, dtype=torch.bool),
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertEqual(
            float(metrics["counterfactual_action_prototype_retrieval_acc"]),
            1.0,
        )
        self.assertEqual(
            float(metrics["counterfactual_action_prototype_count"]), 2.0
        )

        mismatched, _ = bank.positive_loss(
            counterfactual_task_ids=torch.tensor([20, 10]),
            query_residuals=queries,
            action_intents=actions,
            valid_mask=torch.ones(2, dtype=torch.bool),
        )
        self.assertGreater(float(mismatched), 0.0)

    def test_counterfactual_action_separation_is_bounded(self):
        model = tiny_fastwam(
            transition_contract=True,
            transition_contract_version=5,
        )
        loss, metrics = model.compute_counterfactual_action_separation_loss(
            positive_error=torch.tensor([0.10, 0.20]),
            counterfactual_source_error=torch.tensor([0.20, 0.21]),
            valid_mask=torch.ones(2, dtype=torch.bool),
        )
        self.assertAlmostEqual(float(loss), 0.02, places=6)
        self.assertAlmostEqual(
            float(
                metrics[
                    "counterfactual_action_separation_satisfied_fraction"
                ]
            ),
            0.5,
        )

    def test_counterfactual_ranking_uses_only_valid_examples(self):
        z_future = torch.eye(3)
        z_positive = torch.eye(3)
        z_negative = -torch.eye(3)
        loss, metrics = CounterfactualRankingLoss(margin=0.2)(
            z_positive,
            z_negative,
            z_future,
            valid_mask=torch.tensor([True, False, True]),
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(
            float(metrics["counterfactual_margin_satisfied_fraction"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(metrics["counterfactual_valid_fraction"]), 2.0 / 3.0
        )

        violated, _ = CounterfactualRankingLoss(margin=0.2)(
            -z_future,
            z_future,
            z_future,
            valid_mask=torch.ones(3, dtype=torch.bool),
        )
        self.assertGreater(float(violated), 0.0)

    def test_tc_full_build_inputs_requires_and_preserves_negative_context(self):
        model = tiny_fastwam(
            transition_contract=True,
            transition_contract_version=4,
        )
        model._encode_video_latents = lambda *_args, **_kwargs: torch.randn(
            2, 2, 2, 2, 2
        )
        sample = {
            "video": torch.randn(2, 3, 5, 16, 16),
            "action": torch.randn(2, 4, 3),
            "context": torch.randn(2, 4, 10),
            "context_mask": torch.ones(2, 4, dtype=torch.bool),
            "negative_context": torch.randn(2, 4, 10),
            "negative_context_mask": torch.ones(2, 4, dtype=torch.bool),
            "negative_valid": torch.tensor([True, False]),
            "transition_task_id": torch.tensor([10, 11]),
        }
        inputs = model.build_inputs(sample)
        self.assertEqual(tuple(inputs["negative_context"].shape), (2, 4, 10))
        self.assertEqual(inputs["negative_valid"].tolist(), [True, False])

        sample.pop("negative_context")
        with self.assertRaisesRegex(ValueError, "negative_context"):
            model.build_inputs(sample)

    def test_tc_v5_build_inputs_requires_counterfactual_task_id(self):
        model = tiny_fastwam(
            transition_contract=True,
            transition_contract_version=5,
        )
        model._encode_video_latents = lambda *_args, **_kwargs: torch.randn(
            2, 2, 2, 2, 2
        )
        sample = {
            "video": torch.randn(2, 3, 5, 16, 16),
            "action": torch.randn(2, 4, 3),
            "context": torch.randn(2, 4, 10),
            "context_mask": torch.ones(2, 4, dtype=torch.bool),
            "negative_context": torch.randn(2, 4, 10),
            "negative_context_mask": torch.ones(2, 4, dtype=torch.bool),
            "negative_valid": torch.ones(2, dtype=torch.bool),
            "transition_task_id": torch.tensor([10, 20]),
        }
        with self.assertRaisesRegex(ValueError, "counterfactual_task_id"):
            model.build_inputs(sample)

        sample["counterfactual_task_id"] = torch.tensor([20, 10])
        inputs = model.build_inputs(sample)
        self.assertEqual(
            inputs["counterfactual_task_id"].tolist(), [20, 10]
        )

    def test_tc_full_training_loss_updates_only_transition_modules(self):
        model = tiny_fastwam(
            transition_contract=True,
            transition_contract_version=4,
        )
        model.configure_lora(LORA_CONFIG)
        model.prepare_trainable_parameters()
        model._encode_video_latents = lambda *_args, **_kwargs: torch.randn(
            2, 2, 2, 2, 2
        )
        model.set_training_progress(10, 100)
        sample = {
            "video": torch.randn(2, 3, 5, 16, 16),
            "action": torch.randn(2, 4, 3),
            "context": torch.randn(2, 4, 10),
            "context_mask": torch.ones(2, 4, dtype=torch.bool),
            "negative_context": torch.randn(2, 4, 10),
            "negative_context_mask": torch.ones(2, 4, dtype=torch.bool),
            "negative_valid": torch.tensor([True, False]),
            "transition_task_id": torch.tensor([10, 11]),
        }
        loss, metrics = model.training_loss(sample)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_action_future_contract", metrics)
        self.assertIn("loss_counterfactual_ranking", metrics)
        self.assertIn("sim_AF_margin", metrics)
        self.assertIn("sim_CF_margin", metrics)
        loss.backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in model.mot.parameters())
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.transition_contract_modules.parameters()
            )
        )

    def test_tc_v5_action_positive_supervision_reaches_policy_queries(self):
        model = tiny_fastwam(
            transition_contract=True,
            transition_contract_version=5,
        )
        model.configure_lora(LORA_CONFIG)
        model.prepare_trainable_parameters()
        model._encode_video_latents = lambda *_args, **_kwargs: torch.randn(
            2, 2, 2, 2, 2
        )
        model.set_training_progress(10, 100)
        sample = {
            "video": torch.randn(2, 3, 5, 16, 16),
            "action": torch.randn(2, 4, 3),
            "context": torch.randn(2, 4, 10),
            "context_mask": torch.ones(2, 4, dtype=torch.bool),
            "negative_context": torch.randn(2, 4, 10),
            "negative_context_mask": torch.ones(2, 4, dtype=torch.bool),
            "negative_valid": torch.ones(2, dtype=torch.bool),
            "transition_task_id": torch.tensor([10, 20]),
            "counterfactual_task_id": torch.tensor([20, 10]),
        }
        loss, metrics = model.training_loss(sample)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            metrics["counterfactual_action_policy_branch_active"], 1.0
        )
        self.assertIn("loss_counterfactual_action_positive", metrics)
        self.assertIn("loss_counterfactual_action_separation", metrics)
        self.assertIn("sim_CAP_query_positive", metrics)
        self.assertIn(
            "counterfactual_action_prototype_retrieval_acc", metrics
        )
        loss.backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in model.mot.parameters())
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.transition_contract_modules[
                    "router"
                ].parameters()
            )
        )

    def test_counterfactual_manifest_loader_rejects_unsafe_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "manifest.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "correct_instruction": "pick up soup",
                            "counterfactual_instruction": "pick up milk",
                            "counterfactual_is_executable": True,
                        }
                    )
                    + "\n"
                )
            mapping = load_counterfactual_instruction_map(path)
            self.assertEqual(mapping["pick up soup"], "pick up milk")
            self.assertEqual(
                stable_instruction_id(" Pick Up Soup "),
                stable_instruction_id("pick up soup"),
            )

            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "correct_instruction": "same",
                            "counterfactual_instruction": "same",
                            "counterfactual_is_executable": True,
                        }
                    )
                    + "\n"
                )
            with self.assertRaisesRegex(ValueError, "positive instruction"):
                load_counterfactual_instruction_map(path)

    def test_tc_v3_checkpoint_migrates_to_tc_full_v4(self):
        source = tiny_fastwam(
            transition_contract=True,
            transition_contract_version=3,
        )
        source.configure_lora(LORA_CONFIG)
        with torch.no_grad():
            expected = next(
                source.transition_contract_modules["router"].parameters()
            )
            expected.add_(0.321)
            expected = expected.detach().clone()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tc_v3.pt")
            source.save_checkpoint(path, step=4000)
            restored = tiny_fastwam(
                transition_contract=True,
                transition_contract_version=4,
            )
            restored.load_checkpoint(path)
            actual = next(
                restored.transition_contract_modules["router"].parameters()
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertIn(
                "action_effect_encoder", restored.transition_contract_modules
            )
            metadata = restored._transition_contract_metadata()
            self.assertTrue(metadata["use_action_effect"])
            self.assertTrue(metadata["use_cf_ranking"])
            self.assertEqual(
                metadata["contract_visual_source"],
                "language_neutral_video_patch_tokens",
            )
            restored.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in restored.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all(
                    name.startswith("transition_contract_modules.")
                    for name in trainable
                )
            )

    def test_tc_v4_checkpoint_migrates_to_action_positive_v5(self):
        source = tiny_fastwam(
            transition_contract=True,
            transition_contract_version=4,
        )
        source.configure_lora(LORA_CONFIG)
        with torch.no_grad():
            expected = next(
                source.transition_contract_modules["router"].parameters()
            )
            expected.add_(0.456)
            expected = expected.detach().clone()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tc_v4.pt")
            source.save_checkpoint(path, step=4000)
            restored = tiny_fastwam(
                transition_contract=True,
                transition_contract_version=5,
            )
            restored.load_checkpoint(path)
        actual = next(
            restored.transition_contract_modules["router"].parameters()
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        metadata = restored._transition_contract_metadata()
        self.assertEqual(metadata["transition_contract_version"], 5)
        self.assertTrue(metadata["use_cf_action_positive"])

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
            self.assertEqual(
                os.path.realpath(self.model.transition_policy_init_checkpoint),
                os.path.realpath(path),
            )
            self.assertEqual(
                os.path.realpath(self.model.lora_base_checkpoint),
                os.path.realpath(base_path),
            )
            tc_path = f"{tmpdir}/tc_v2.pt"
            self.model.save_checkpoint(tc_path)
            tc_payload = torch.load(tc_path, map_location="cpu")
            self.assertEqual(
                os.path.realpath(tc_payload["base_checkpoint"]),
                os.path.realpath(base_path),
            )
            self.assertEqual(
                tc_payload["architecture_metadata"]["policy_init_checkpoint"],
                self.model.transition_policy_init_checkpoint,
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
