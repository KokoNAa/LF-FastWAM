import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.policy_guard import (
    ActionOutcomeVerifier,
    GoalActionAlignmentLoss,
    GoalGraphEncoder,
    GoalResidualAdapter,
)
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


class TinyVAE(nn.Module):
    temporal_downsample_factor = 1
    upsampling_factor = 1

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)


def tiny_pgc_fastwam(
    *, configure_lora: bool = True, version: int = 2
) -> FastWAM:
    video = WanVideoDiT(
        hidden_dim=16,
        in_dim=2,
        ffn_dim=32,
        out_dim=2,
        text_dim=10,
        freq_dim=8,
        eps=1.0e-6,
        patch_size=(1, 1, 1),
        num_heads=2,
        attn_head_dim=4,
        num_layers=2,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        action_conditioned=False,
        video_attention_mask_mode="first_frame_causal",
    )
    action = ActionDiT(
        hidden_dim=12,
        action_dim=3,
        ffn_dim=24,
        text_dim=10,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=2,
        attn_head_dim=4,
        num_layers=2,
        use_latent_action_queries=False,
    )
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    )
    model = FastWAM(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=TinyVAE(),
        text_dim=10,
        device="cpu",
        torch_dtype=torch.float32,
        langforce_mvp_config={"enabled": False},
        transition_contract_config={"enabled": False},
        policy_guard_config={
            "enabled": True,
            "version": version,
            "num_action_queries": 2,
            "query_rope_offset": 16,
            "num_goal_tokens": 3,
            "hidden_dim": 8,
            "projection_dim": 8,
            "num_heads": 2,
            "verifier_hidden_dim": 8,
            "counterfactual_action_weight": 1.0,
            "native_distillation_weight": 1.0,
            "goal_residual_scale": 1.0,
            "verifier_weight": 0.25,
            "goal_action_alignment_weight": 0.1,
            "verifier_margin": 0.2,
            "gate_threshold": 0.2,
            "min_counterfactual_score": 0.6,
        },
    )
    if configure_lora:
        model.configure_lora(
            {
                "enabled": True,
                "rank": 2,
                "alpha": 4,
                "dropout": 0.0,
                "experts": ["action"],
                "extra_trainable_patterns": [],
            }
        )
    return model


class PolicyGuardModuleTest(unittest.TestCase):
    def test_goal_graph_is_null_language_safe(self):
        module = GoalGraphEncoder(
            text_dim=10,
            video_dim=16,
            action_dim=12,
            hidden_dim=8,
            projection_dim=8,
            num_goal_tokens=3,
            num_heads=2,
        )
        routed, goal, metrics = module(
            base_queries=torch.randn(2, 2, 12),
            language_hidden=torch.randn(2, 4, 10),
            language_mask=torch.zeros(2, 4, dtype=torch.bool),
            current_video_hidden=torch.randn(2, 5, 16),
        )
        self.assertEqual(tuple(routed.shape), (2, 2, 12))
        self.assertEqual(tuple(goal.shape), (2, 8))
        self.assertTrue(torch.isfinite(routed).all())
        self.assertTrue(torch.isfinite(goal).all())
        self.assertIn("pgc_query_pairwise_cosine", metrics)

    def test_goal_residual_is_exactly_zero_initialized(self):
        adapter = GoalResidualAdapter(
            action_dim=12,
            num_heads=2,
            residual_scale=1.0,
        )
        action = torch.randn(2, 4, 12)
        output, metrics = adapter(action, torch.randn(2, 3, 12))
        self.assertTrue(torch.equal(output, action))
        self.assertEqual(float(metrics["pgc_goal_residual_norm"]), 0.0)
        self.assertEqual(float(metrics["pgc_goal_residual_max_abs"]), 0.0)

    def test_verifier_and_alignment_shapes(self):
        verifier = ActionOutcomeVerifier(
            action_dim=3, video_dim=16, goal_dim=8, hidden_dim=8
        )
        logits, goal_state, action_embedding = verifier(
            current_video_hidden=torch.randn(3, 5, 16),
            goal_embedding=torch.randn(3, 8),
            action=torch.randn(3, 4, 3),
            action_is_pad=torch.tensor(
                [[False] * 4, [False, False, True, True], [False] * 4]
            ),
        )
        self.assertEqual(tuple(logits.shape), (3,))
        loss, metrics = GoalActionAlignmentLoss(temperature=0.1)(
            goal_state,
            action_embedding,
            group_ids=torch.tensor([1, 1, 2]),
        )
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("pgc_goal_action_retrieval_acc", metrics)
        self.assertEqual(float(metrics["pgc_goal_action_candidate_count"]), 3.0)
        self.assertGreater(
            float(metrics["pgc_goal_action_effective_negative_count"]), 0.0
        )

    def test_alignment_uses_cross_rank_negatives_for_local_batch_one(self):
        goal_state = torch.tensor([[1.0, 0.0]], requires_grad=True)
        action_embedding = torch.tensor([[1.0, 0.0]], requires_grad=True)
        remote_goal = torch.tensor([[0.0, 1.0]])
        remote_action = torch.tensor([[1.0, 0.0]])
        gathered_goal = torch.cat([goal_state, remote_goal], dim=0)
        gathered_action = torch.cat([action_embedding, remote_action], dim=0)

        with (
            patch(
                "fastwam.models.wan22.policy_guard._gather_with_grad",
                side_effect=[(gathered_goal, 0), (gathered_action, 0)],
            ),
            patch(
                "fastwam.models.wan22.policy_guard._gather_without_grad",
                return_value=torch.tensor([10, 20]),
            ),
        ):
            loss, metrics = GoalActionAlignmentLoss(temperature=0.1)(
                goal_state,
                action_embedding,
                group_ids=torch.tensor([10]),
            )

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertEqual(float(metrics["pgc_goal_action_candidate_count"]), 2.0)
        self.assertEqual(
            float(metrics["pgc_goal_action_effective_negative_count"]), 1.0
        )
        loss.backward()
        self.assertIsNotNone(goal_state.grad)
        self.assertIsNotNone(action_embedding.grad)

    def test_alignment_single_candidate_is_a_differentiable_noop(self):
        goal_state = torch.randn(1, 8, requires_grad=True)
        action_embedding = torch.randn(1, 8, requires_grad=True)
        loss, metrics = GoalActionAlignmentLoss(temperature=0.1)(
            goal_state,
            action_embedding,
            group_ids=torch.tensor([3]),
        )
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(metrics["pgc_goal_action_candidate_count"]), 1.0)
        self.assertEqual(
            float(metrics["pgc_goal_action_effective_negative_count"]), 0.0
        )
        loss.backward()
        self.assertIsNotNone(goal_state.grad)
        self.assertIsNotNone(action_embedding.grad)


class PolicyGuardIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.model = tiny_pgc_fastwam()

    def test_only_independent_branch_is_trainable(self):
        report = self.model.prepare_trainable_parameters()
        self.assertGreater(report["trainable"], 0)
        trainable = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("policy_guard_action_expert.")
                or name.startswith("policy_guard_modules.")
                for name in trainable
            )
        )
        action_trainable = {
            name.removeprefix("policy_guard_action_expert.")
            for name in trainable
            if name.startswith("policy_guard_action_expert.")
        }
        self.assertTrue(action_trainable)
        self.assertTrue(
            all(
                name.endswith(".lora_A")
                or name.endswith(".lora_B")
                for name in action_trainable
            )
        )
        self.assertFalse(
            self.model.policy_guard_action_expert.use_latent_action_queries
        )
        self.assertFalse(any(p.requires_grad for p in self.model.mot.parameters()))
        self.assertFalse(self.model.mot.training)
        self.assertTrue(self.model.policy_guard_action_expert.training)
        self.assertTrue(self.model.policy_guard_modules.training)

    def test_optimizer_step_cannot_mutate_protected_base(self):
        self.model.prepare_trainable_parameters()
        before = {
            name: value.detach().clone()
            for name, value in self.model.mot.state_dict().items()
        }
        frozen_counterfactual = {
            name: parameter.detach().clone()
            for name, parameter in self.model.policy_guard_action_expert.named_parameters()
            if not parameter.requires_grad
        }
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=1.0e-3,
        )
        loss = sum(
            parameter.float().square().mean()
            for parameter in self.model.policy_guard_action_expert.parameters()
        )
        loss.backward()
        optimizer.step()
        for name, expected in before.items():
            self.assertTrue(torch.equal(self.model.mot.state_dict()[name], expected))
        for name, expected in frozen_counterfactual.items():
            actual = dict(
                self.model.policy_guard_action_expert.named_parameters()
            )[name]
            self.assertTrue(torch.equal(actual, expected), name)

    def test_pgc_refuses_full_action_expert_training(self):
        model = tiny_pgc_fastwam(configure_lora=False)
        with self.assertRaisesRegex(ValueError, "full Action-Expert fine-tuning"):
            model.prepare_trainable_parameters()

    def test_pgc_refuses_video_or_extra_lora_targets(self):
        model = tiny_pgc_fastwam(configure_lora=False)
        with self.assertRaisesRegex(ValueError, "target only"):
            model.configure_lora(
                {
                    "enabled": True,
                    "experts": ["video", "action"],
                    "extra_trainable_patterns": [],
                }
            )
        with self.assertRaisesRegex(ValueError, "does not accept"):
            model.configure_lora(
                {
                    "enabled": True,
                    "experts": ["action"],
                    "extra_trainable_patterns": ["action_expert.head.*"],
                }
            )

    def test_counterfactual_action_forward_has_gradients(self):
        self.model.prepare_trainable_parameters()
        self.model.mot.mot_checkpoint_mixed_attn = True
        self.model.policy_guard_action_expert.use_gradient_checkpointing = True
        self.assertFalse(self.model.mot.training)
        self.assertTrue(self.model.policy_guard_action_expert.training)
        batch_size = 2
        context = torch.randn(batch_size, 4, 10)
        context_mask = torch.ones(batch_size, 4, dtype=torch.bool)
        state_only = torch.zeros_like(context_mask)
        current_video = torch.randn(batch_size, 5, 16)
        routed, _, _ = self.model._encode_policy_guard_goal(
            final_video_hidden=current_video,
            video_tokens_per_frame=5,
            context=context,
            context_mask=context_mask,
        )
        video_cache = [
            {
                "k": torch.randn(batch_size, 5, 8),
                "v": torch.randn(batch_size, 5, 8),
            }
            for _ in range(2)
        ]
        output = self.model._forward_policy_guard_action_from_cache(
            action_tokens=torch.randn(batch_size, 4, 3),
            timestep_action=torch.tensor([0.3, 0.7]),
            context=context,
            full_context_mask=context_mask,
            state_only_context_mask=state_only,
            video_kv_cache=video_cache,
            video_seq_len=5,
            video_tokens_per_frame=5,
            routed_goal_queries=routed,
        )
        self.assertEqual(tuple(output.shape), (batch_size, 4, 3))
        output.square().mean().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in self.model.policy_guard_action_expert.parameters()
            )
        )
        self.assertFalse(any(p.grad is not None for p in self.model.mot.parameters()))

    def test_v2_counterfactual_branch_starts_equal_to_base(self):
        self.model.eval()
        batch_size = 2
        video_seq_len = 5
        action_horizon = 4
        context = torch.randn(batch_size, 4, 10)
        full_context_mask = torch.ones(batch_size, 4, dtype=torch.bool)
        state_only_context_mask = torch.zeros_like(full_context_mask)
        current_video = torch.randn(batch_size, video_seq_len, 16)
        routed, _, _ = self.model._encode_policy_guard_goal(
            final_video_hidden=current_video,
            video_tokens_per_frame=video_seq_len,
            context=context,
            context_mask=full_context_mask,
        )
        video_cache = [
            {
                "k": torch.randn(batch_size, video_seq_len, 8),
                "v": torch.randn(batch_size, video_seq_len, 8),
            }
            for _ in range(2)
        ]
        noisy_action = torch.randn(batch_size, action_horizon, 3)
        timestep = torch.tensor([0.3, 0.7])
        base_pre = self.model.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep,
            context=context,
            context_mask=full_context_mask,
            use_queries=False,
        )
        attention_mask = self.model._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_horizon,
            video_tokens_per_frame=video_seq_len,
            device=noisy_action.device,
            num_queries=0,
            action_reads_raw_video=True,
        )
        base_hidden = self.model.mot.forward_action_with_video_cache(
            action_tokens=base_pre["tokens"],
            action_freqs=base_pre["freqs"],
            action_t_mod=base_pre["t_mod"],
            action_context_payload={
                "context": base_pre["context"],
                "mask": base_pre["context_mask"],
            },
            video_kv_cache=video_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        base_output = self.model.action_expert.post_dit(
            base_hidden, base_pre
        )
        counterfactual_output = (
            self.model._forward_policy_guard_action_from_cache(
                action_tokens=noisy_action,
                timestep_action=timestep,
                context=context,
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
                video_kv_cache=video_cache,
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_seq_len,
                routed_goal_queries=routed,
            )
        )
        self.assertTrue(torch.equal(counterfactual_output, base_output))

    def test_v2_native_distillation_and_cf_action_masks_are_disjoint(self):
        predicted = torch.tensor(
            [
                [[2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0]],
            ]
        )
        teacher = torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[100.0, 0.0, 0.0]],
            ]
        )
        target = torch.tensor(
            [
                [[100.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
            ]
        )
        cf_loss, native_loss, metrics = (
            self.model._compute_policy_guard_v2_action_losses(
                predicted_action=predicted,
                base_action_teacher=teacher,
                target_action=target,
                action_weight=torch.ones(2),
                action_is_pad=None,
                is_counterfactual=torch.tensor([False, True]),
                direct_action_valid=torch.tensor([True, True]),
            )
        )
        self.assertAlmostEqual(float(native_loss), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(cf_loss), 4.0 / 3.0, places=6)
        self.assertAlmostEqual(float(metrics["pgc_native_fraction"]), 0.5)
        self.assertAlmostEqual(
            float(metrics["pgc_counterfactual_fraction"]), 0.5
        )

    def test_conservative_gate_preserves_base_exactly(self):
        base = torch.randn(2, 4, 3)
        counterfactual = torch.randn_like(base)
        selected, mask = self.model._select_policy_guard_action(
            base_action=base,
            counterfactual_action=counterfactual,
            base_score=torch.tensor([0.8, 0.4]),
            counterfactual_score=torch.tensor([0.9, 0.9]),
        )
        self.assertEqual(mask.tolist(), [False, True])
        self.assertTrue(torch.equal(selected[0], base[0]))
        self.assertTrue(torch.equal(selected[1], counterfactual[1]))

        self.model.policy_guard_gate_mode = "base"
        selected, mask = self.model._select_policy_guard_action(
            base_action=base,
            counterfactual_action=counterfactual,
            base_score=torch.zeros(2),
            counterfactual_score=torch.ones(2),
        )
        self.assertEqual(mask.tolist(), [False, False])
        self.assertTrue(torch.equal(selected, base))

    def test_checkpoint_round_trip_keeps_external_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.pt"
            pgc_path = Path(tmpdir) / "pgc.pt"
            torch.save(
                {"format": "fastwam_full_v1", "mot": self.model.mot.state_dict()},
                base_path,
            )
            self.model.load_checkpoint(base_path)
            with torch.no_grad():
                next(self.model.policy_guard_modules.parameters()).add_(0.25)
                for name, parameter in (
                    self.model.policy_guard_action_expert.named_parameters()
                ):
                    if name.endswith(".lora_B"):
                        parameter.add_(0.25)
            expected_guard = {
                key: value.detach().clone()
                for key, value in self.model.policy_guard_modules.state_dict().items()
            }
            expected_action = {
                key: value.detach().clone()
                for key, value in self.model.policy_guard_action_expert.state_dict().items()
            }
            self.model.save_checkpoint(pgc_path, step=17)

            payload = torch.load(pgc_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["format"], "fastwam_policy_guard_v2")
            self.assertEqual(payload["step"], 17)
            self.assertNotIn("mot", payload)
            self.assertNotIn("counterfactual_action_expert", payload)
            self.assertIn("counterfactual_action_adapter", payload)
            self.assertIn("counterfactual_lora_config", payload)
            self.assertTrue(
                all(
                    name.endswith(".lora_A")
                    or name.endswith(".lora_B")
                    for name in payload["counterfactual_action_adapter"]
                )
            )
            self.assertEqual(
                payload["architecture_metadata"]["policy_guard_version"],
                2,
            )
            self.assertEqual(
                payload["architecture_metadata"][
                    "counterfactual_action_interface"
                ],
                "query_free_raw_current_visual",
            )
            self.assertEqual(
                payload["architecture_metadata"]["policy_protection"],
                "immutable_base_plus_conservative_hard_gate",
            )
            self.assertEqual(
                payload["architecture_metadata"]["counterfactual_tuning"],
                "lora",
            )

            restored = tiny_pgc_fastwam()
            restored.load_checkpoint(pgc_path)
            for key, expected in expected_guard.items():
                self.assertTrue(
                    torch.equal(restored.policy_guard_modules.state_dict()[key], expected)
                )
            for key, expected in expected_action.items():
                self.assertTrue(
                    torch.equal(
                        restored.policy_guard_action_expert.state_dict()[key],
                        expected,
                    )
                )
            self.assertEqual(
                restored.policy_guard_base_checkpoint,
                str(base_path.resolve()),
            )

    def test_v1_checkpoint_remains_loadable_for_legacy_evaluation(self):
        legacy = tiny_pgc_fastwam(version=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.pt"
            pgc_path = Path(tmpdir) / "pgc_v1.pt"
            torch.save(
                {"format": "fastwam_full_v1", "mot": legacy.mot.state_dict()},
                base_path,
            )
            legacy.load_checkpoint(base_path)
            with torch.no_grad():
                legacy.policy_guard_action_expert.latent_action_queries.add_(
                    0.5
                )
            expected_queries = (
                legacy.policy_guard_action_expert.latent_action_queries
                .detach()
                .clone()
            )
            legacy.save_checkpoint(pgc_path, step=9)

            payload = torch.load(
                pgc_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["format"], "fastwam_policy_guard_v1")
            self.assertIn(
                "latent_action_queries",
                payload["counterfactual_action_adapter"],
            )

            restored = tiny_pgc_fastwam(version=1)
            restored.load_checkpoint(pgc_path)
            self.assertTrue(
                torch.equal(
                    restored.policy_guard_action_expert.latent_action_queries,
                    expected_queries,
                )
            )


if __name__ == "__main__":
    unittest.main()
