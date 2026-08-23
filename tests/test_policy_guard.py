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
    BoundedActionVelocityResidual,
    GoalActionAlignmentLoss,
    GoalGraphEncoder,
    GoalResidualAdapter,
    LanguageVisualTargetBinder,
    PairwiseActionAdvantageVerifier,
    RolloutAlignedActionProposal,
    SpatialObjectTokenTargetBinder,
    spatial_mask_to_patch_distribution,
)
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


class TinyVAE(nn.Module):
    temporal_downsample_factor = 1
    upsampling_factor = 1

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)


def tiny_pgc_fastwam(
    *,
    configure_lora: bool = True,
    version: int = 2,
    completion_phase_enabled: bool = False,
    v9_stage: str = "grounding",
    v9_entity_only: bool = False,
    v9_use_anchors: bool = True,
    v9_grounding_objective_version: int = 2,
    v9_completion_only_memory: bool = False,
    v9_action_joint_training: bool = False,
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
            "residual_regularization_weight": 0.01,
            "residual_smoothness_weight": 0.01,
            "velocity_residual_max_abs": 1.0,
            "action_chunk_residual_max_abs": 0.5,
            "rollout_num_inference_steps": 2,
            "proposal_hidden_dim": 8,
            "proposal_num_heads": 2,
            "proposal_num_layers": 1,
            "action_gripper_weight": 2.0,
            "verifier_num_heads": 2,
            "verifier_num_layers": 1,
            "advantage_temperature": 0.25,
            "advantage_clip": 4.0,
            "candidate_max_saturation_fraction": 0.25,
            "candidate_max_delta_rms": 0.5,
            "execution_prefix_steps": 2,
            "suffix_loss_weight": 0.1,
            "completion_phase_enabled": completion_phase_enabled,
            "completion_transport_weight": 2.0,
            "completion_release_weight": 3.0,
            "completion_train_proposal_only": True,
            "closed_loop_corrective_enabled": version == 8,
            "closed_loop_corrective_weight": 2.0,
            "offline_acquisition_weight": 1.0,
            "native_guard_weight": 0.1,
            "acquisition_only": True,
            "closed_loop_train_proposal_only": True,
            "same_state_source_zero_weight": 1.0,
            "goal_separation_weight": 0.25,
            "goal_separation_margin": 0.2,
            "residual_separation_weight": 0.25,
            "residual_separation_margin": 0.05,
            "verifier_wrong_language_weight": 0.5,
            "verifier_bad_candidate_weight": 0.5,
            "target_binding_interaction_weight": 1.0,
            "target_binding_prototype_weight": 0.5,
            "target_binding_source_weight": 0.5,
            "target_binding_hard_negative_weight": 0.5,
            "target_binding_separation_weight": 0.25,
            "target_binding_hard_negative_margin": 0.2,
            "target_binding_separation_margin": 0.15,
            "target_binding_teacher_topk": 0.25,
            "target_binding_teacher_temperature": 0.25,
            "target_binding_hidden_dim": 8,
            "target_binding_num_heads": 2,
            "target_binding_temperature": 0.07,
            "target_binding_prototype_slots": 16,
            "target_binding_prototype_momentum": 0.9,
            "target_binding_prototype_temperature": 0.07,
            "target_binding_prototype_topk": 0.25,
            "target_binding_action_start_step": 10,
            "target_binding_action_ramp_steps": 5,
            "target_binding_num_object_tokens": 2,
            "target_binding_camera_count": 2,
            "target_binding_visual_aspect_ratio": 2.0,
            "target_mask_weight": 1.0,
            "source_mask_weight": 0.5,
            "aux_mask_weight": 0.5,
            "mask_mass_weight": 0.5,
            "cross_object_weight": 0.5,
            "cross_object_margin": 0.25,
            "verifier_start_step": 10,
            "verifier_ramp_steps": 5,
            "goal_residual_scale": 1.0,
            "verifier_weight": 0.25,
            "goal_action_alignment_weight": 0.1,
            "verifier_margin": 0.2,
            "gate_threshold": 0.2,
            "min_counterfactual_score": 0.6,
            "entity_relation_grounding": {
                "training_stage": v9_stage,
                "grounding_objective_version": v9_grounding_objective_version,
                "hidden_dim": 8,
                "num_heads": 2,
                "max_clauses": 4,
                "camera_count": 2,
                "visual_aspect_ratio": 2.0,
                "temperature": 0.07,
                "learning_rate": 2.0e-5,
                "grounding_aux_weight": (
                    0.0 if v9_action_joint_training else 0.25
                ),
                "completion_only_memory": v9_completion_only_memory,
                "action_joint_training": v9_action_joint_training,
                "action_grounding_hidden_dim": 8,
                "action_grounding_num_heads": 2,
                "action_grounding_learning_rate": 1.0e-4,
                "action_causal_ranking_weight": 1.0,
                "action_causal_margin": 0.01,
                "mask_weight": 1.0,
                "attention_mask_weight": 2.0,
                "entity_weight": 1.0,
                "relation_weight": 1.0,
                "anchor_weight": 1.0,
                "position_weight": 0.5,
                "role_swap_weight": 2.0,
                "role_overlap_weight": 1.0,
                "role_swap_margin": 0.20,
                "role_assignment_weight": (
                    1.0
                    if v9_grounding_objective_version >= 4
                    else (4.0 if v9_grounding_objective_version >= 3 else 0.0)
                ),
                "role_assignment_temperature": 0.10,
                "role_assignment_hard_weight": (
                    0.5
                    if v9_grounding_objective_version >= 4
                    else (2.0 if v9_grounding_objective_version >= 3 else 0.0)
                ),
                "role_adapter_hidden_dim": 8,
                "structured_assignment_weight": (
                    2.0 if v9_grounding_objective_version >= 5 else 0.0
                ),
                "structured_assignment_temperature": 0.10,
                "structured_assignment_hard_weight": (
                    (
                        1.0
                        if v9_grounding_objective_version >= 6
                        else 2.0
                    )
                    if v9_grounding_objective_version >= 5
                    else 0.0
                ),
                "multi_clause_consistency_weight": (
                    (
                        2.0
                        if v9_grounding_objective_version >= 6
                        else 1.0
                    )
                    if v9_grounding_objective_version >= 5
                    else 0.0
                ),
                "structured_role_adapter_hidden_dim": 8,
                "balanced_role_adapter_hidden_dim": 8,
                "clause_activation_adapter_hidden_dim": 8,
                "clause_activation_residual_max_abs": 4.0,
                "clause_activation_balance_weight": (
                    1.0 if v9_grounding_objective_version >= 9 else 0.0
                ),
                "clause_cardinality_weight": (
                    1.0 if v9_grounding_objective_version >= 9 else 0.0
                ),
                "clause_worst_slot_weight": (
                    2.0 if v9_grounding_objective_version >= 9 else 0.0
                ),
                "clause_multi_group_weight": 1.0,
                "clause_adapter_energy_weight": (
                    0.01 if v9_grounding_objective_version >= 9 else 0.0
                ),
                "view_fusion_adapter_hidden_dim": 8,
                "view_fusion_residual_max_abs": 4.0,
                "view_fusion_weight": (
                    2.0 if v9_grounding_objective_version >= 10 else 0.0
                ),
                "view_fusion_energy_weight": (
                    0.01 if v9_grounding_objective_version >= 10 else 0.0
                ),
                "clause_scheduler_hidden_dim": 8,
                "clause_scheduler_residual_max_abs": 1.0,
                "clause_scheduler_weight": (
                    1.0 if v9_grounding_objective_version >= 10 else 0.0
                ),
                "clause_scheduler_energy_weight": (
                    0.01 if v9_grounding_objective_version >= 10 else 0.0
                ),
                "closed_loop_rebinding_hidden_dim": 8,
                "closed_loop_query_residual_max_abs": 1.0,
                "closed_loop_state_residual_max_abs": 2.0,
                "phase_rebinding_energy_weight": (
                    0.01 if v9_grounding_objective_version == 13 else 0.0
                ),
                "phase_safe_memory_hidden_dim": 8,
                "phase_safe_memory_state_count": 4,
                "phase_safe_memory_routing_residual_max_abs": 1.0,
                "phase_safe_memory_state_weight": (
                    1.0 if v9_grounding_objective_version >= 14 else 0.0
                ),
                "phase_safe_memory_scheduler_weight": (
                    1.0 if v9_grounding_objective_version >= 14 else 0.0
                ),
                "phase_safe_memory_energy_weight": (
                    0.01 if v9_grounding_objective_version >= 14 else 0.0
                ),
                "role_attention_preservation_weight": (
                    (
                        5.0
                        if v9_grounding_objective_version >= 6
                        else 1.0
                    )
                    if v9_grounding_objective_version >= 4
                    else 0.0
                ),
                "role_position_preservation_weight": (
                    (
                        2.0
                        if v9_grounding_objective_version >= 6
                        else 0.5
                    )
                    if v9_grounding_objective_version >= 4
                    else 0.0
                ),
                "role_anchor_preservation_weight": (
                    (
                        10.0
                        if v9_grounding_objective_version >= 6
                        else 1.0
                    )
                    if v9_grounding_objective_version >= 4
                    else 0.0
                ),
                "role_relation_preservation_weight": (
                    (
                        2.0
                        if v9_grounding_objective_version >= 6
                        else 0.5
                    )
                    if v9_grounding_objective_version >= 4
                    else 0.0
                ),
                "role_adapter_energy_weight": (
                    0.01 if v9_grounding_objective_version >= 4 else 0.0
                ),
                "phase_weight": 1.0,
                "entity_only": v9_entity_only,
                "use_anchors": v9_use_anchors,
            },
        },
    )
    if configure_lora and version <= 2:
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
    def test_v7_spatial_mask_distribution_preserves_two_camera_location(self):
        mask = torch.zeros(2, 8, 16)
        mask[0, :4, :4] = 1
        distribution, valid, metrics = spatial_mask_to_patch_distribution(
            mask, token_count=8
        )
        self.assertEqual(tuple(distribution.shape), (2, 8))
        self.assertEqual(valid.tolist(), [True, False])
        self.assertTrue(torch.allclose(distribution.sum(dim=-1), torch.ones(2)))
        self.assertGreater(float(distribution[0, :4].sum()), 0.99)
        self.assertTrue(
            torch.allclose(distribution[1], torch.full((8,), 1.0 / 8.0))
        )
        self.assertEqual(float(metrics["pgc_v7_mask_grid_height"]), 2.0)
        self.assertEqual(float(metrics["pgc_v7_mask_grid_width"]), 4.0)

    def test_v7_object_tokens_start_exactly_at_base_and_are_query_specific(self):
        module = SpatialObjectTokenTargetBinder(
            text_dim=10,
            video_dim=16,
            action_dim=12,
            hidden_dim=8,
            projection_dim=8,
            num_heads=2,
            num_object_tokens=3,
            camera_count=2,
            visual_aspect_ratio=2.0,
        )
        base_queries = torch.randn(2, 4, 12)
        language = torch.randn(2, 5, 10)
        language_mask = torch.ones(2, 5, dtype=torch.bool)
        visual = torch.randn(2, 8, 16)
        output = module(
            base_queries=base_queries,
            language_hidden=language,
            language_mask=language_mask,
            current_video_hidden=visual,
        )
        self.assertTrue(torch.equal(output[0], base_queries))
        self.assertEqual(tuple(output[2].shape), (2, 8))
        self.assertEqual(float(output[-1]["pgc_v7_binding_query_delta_norm"]), 0.0)

        with torch.no_grad():
            module.query_output_projection.weight.normal_(std=0.05)
        changed = module(
            base_queries=base_queries,
            language_hidden=language,
            language_mask=language_mask,
            current_video_hidden=visual,
        )[0]
        delta = changed - base_queries
        self.assertFalse(torch.allclose(delta[:, 0], delta[:, 1]))

    def test_target_binder_is_null_safe_and_starts_from_shared_query_seeds(self):
        module = LanguageVisualTargetBinder(
            text_dim=10,
            video_dim=16,
            action_dim=12,
            hidden_dim=8,
            projection_dim=8,
            num_heads=2,
        )
        base_queries = torch.randn(2, 3, 12)
        language = torch.randn(2, 4, 10)
        language_mask = torch.tensor(
            [[True, True, False, False], [False, False, False, False]]
        )
        (
            binding_queries,
            binding_embedding,
            target_attention,
            visual_features,
            metrics,
        ) = module(
            base_queries=base_queries,
            language_hidden=language,
            language_mask=language_mask,
            current_video_hidden=torch.randn(2, 5, 16),
        )
        self.assertTrue(torch.equal(binding_queries, base_queries))
        self.assertEqual(tuple(binding_embedding.shape), (2, 8))
        self.assertEqual(tuple(target_attention.shape), (2, 5))
        self.assertEqual(tuple(visual_features.shape), (2, 5, 8))
        self.assertTrue(torch.isfinite(binding_embedding).all())
        self.assertTrue(torch.isfinite(target_attention).all())
        self.assertTrue(
            torch.allclose(target_attention.sum(dim=-1), torch.ones(2))
        )
        self.assertEqual(
            float(metrics["pgc_v6_target_binding_query_delta_norm"]), 0.0
        )

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

    def test_bounded_velocity_residual_is_zero_initialized_and_capped(self):
        adapter = BoundedActionVelocityResidual(
            action_hidden_dim=12,
            action_dim=3,
            num_heads=2,
            max_abs=[0.25, 0.5, 0.75],
        )
        action_hidden = torch.randn(2, 4, 12)
        goal = torch.randn(2, 3, 12)
        residual, metrics = adapter(action_hidden, goal)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertEqual(float(metrics["pgc_velocity_residual_norm"]), 0.0)

        with torch.no_grad():
            adapter.output_projection.bias.fill_(100.0)
        residual, metrics = adapter(action_hidden, goal)
        cap = torch.tensor([0.25, 0.5, 0.75]).view(1, 1, 3)
        self.assertTrue(torch.all(residual.abs() <= cap))
        self.assertAlmostEqual(
            float(metrics["pgc_velocity_residual_saturation_fraction"]),
            1.0,
        )

    def test_rollout_aligned_proposal_is_zero_initialized_and_capped(self):
        proposal = RolloutAlignedActionProposal(
            action_dim=3,
            goal_dim=12,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            max_abs=[0.25, 0.5, 0.75],
        )
        base_action = torch.randn(2, 4, 3, requires_grad=True)
        goal_queries = torch.randn(2, 2, 12, requires_grad=True)
        action_is_pad = torch.tensor(
            [[False, False, False, True], [True, True, True, True]]
        )
        candidate, residual, metrics = proposal(
            base_action=base_action,
            goal_queries=goal_queries,
            action_is_pad=action_is_pad,
        )
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertTrue(torch.equal(candidate, base_action.detach()))
        self.assertTrue(torch.isfinite(candidate).all())
        self.assertEqual(float(metrics["pgc_v4_action_residual_rms"]), 0.0)

        with torch.no_grad():
            proposal.output_projection.bias.fill_(100.0)
        candidate, residual, _ = proposal(
            base_action=base_action,
            goal_queries=goal_queries,
            action_is_pad=action_is_pad,
        )
        cap = torch.tensor([0.25, 0.5, 0.75]).view(1, 1, 3)
        self.assertTrue(torch.all(residual.abs() <= cap))
        self.assertTrue(torch.equal(residual[action_is_pad], torch.zeros(5, 3)))
        self.assertTrue(torch.isfinite(candidate).all())

    def test_pairwise_verifier_is_fp32_temporal_and_equal_candidate_safe(self):
        verifier = PairwiseActionAdvantageVerifier(
            action_dim=3,
            video_dim=16,
            goal_dim=8,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
        ).float()
        action = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 1.0, 0.0]]],
            dtype=torch.bfloat16,
        )
        forward_embedding = verifier.encode_action(action)
        reverse_embedding = verifier.encode_action(action.flip(1))
        self.assertFalse(torch.allclose(forward_embedding, reverse_embedding))
        advantage, base_value, candidate_value, *_ = verifier(
            current_video_hidden=torch.randn(1, 5, 16, dtype=torch.bfloat16),
            goal_embedding=torch.randn(1, 8, dtype=torch.bfloat16),
            base_action=action,
            counterfactual_action=action.clone(),
        )
        self.assertEqual(advantage.dtype, torch.float32)
        self.assertEqual(base_value.dtype, torch.float32)
        self.assertTrue(torch.equal(advantage, torch.zeros_like(advantage)))
        self.assertTrue(torch.equal(base_value, candidate_value))

        all_pad = torch.ones(1, 3, dtype=torch.bool)
        padded_embedding = verifier.encode_action(action, all_pad)
        self.assertTrue(torch.isfinite(padded_embedding).all())

    def test_pairwise_verifier_survives_distributed_bf16_parameter_cast(self):
        verifier = PairwiseActionAdvantageVerifier(
            action_dim=3,
            video_dim=16,
            goal_dim=8,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
        ).bfloat16()
        self.assertTrue(
            all(
                parameter.dtype == torch.bfloat16
                for parameter in verifier.parameters()
            )
        )
        action = torch.randn(2, 4, 3, dtype=torch.bfloat16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            advantage, base_value, candidate_value, *_ = verifier(
                current_video_hidden=torch.randn(
                    2, 5, 16, dtype=torch.bfloat16
                ),
                goal_embedding=torch.randn(2, 8, dtype=torch.bfloat16),
                base_action=action,
                counterfactual_action=action + 0.25,
            )
        self.assertEqual(advantage.dtype, torch.float32)
        self.assertEqual(base_value.dtype, torch.float32)
        self.assertEqual(candidate_value.dtype, torch.float32)
        self.assertTrue(torch.isfinite(advantage).all())
        (base_value + candidate_value).sum().backward()
        self.assertIsNotNone(verifier.value_head[0].weight.grad)
        self.assertTrue(torch.isfinite(verifier.value_head[0].weight.grad).all())

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
                action_weight=torch.ones(2, 1, 1),
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

    def test_v2_batch_one_accepts_scalar_scheduler_weight(self):
        cf_loss, native_loss, metrics = (
            self.model._compute_policy_guard_v2_action_losses(
                predicted_action=torch.tensor([[[2.0, 0.0, 0.0]]]),
                base_action_teacher=torch.tensor([[[1.0, 0.0, 0.0]]]),
                target_action=torch.tensor([[[100.0, 0.0, 0.0]]]),
                action_weight=torch.tensor(2.0),
                action_is_pad=None,
                is_counterfactual=torch.tensor([False]),
                direct_action_valid=torch.tensor([True]),
            )
        )
        self.assertEqual(float(cf_loss), 0.0)
        self.assertAlmostEqual(float(native_loss), 2.0 / 3.0, places=6)
        self.assertEqual(float(metrics["pgc_native_fraction"]), 1.0)

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


class PolicyGuardV3IntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = tiny_pgc_fastwam(version=3)

    def test_v3_trains_only_guard_modules_and_rejects_lora(self):
        self.assertIsNone(self.model.policy_guard_action_expert)
        with self.assertRaisesRegex(ValueError, "does not permit LoRA"):
            self.model.configure_lora(
                {
                    "enabled": True,
                    "rank": 2,
                    "alpha": 4,
                    "dropout": 0.0,
                    "experts": ["action"],
                    "extra_trainable_patterns": [],
                }
            )

        report = self.model.prepare_trainable_parameters()
        self.assertGreater(report["trainable"], 0)
        trainable = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("policy_guard_modules.") for name in trainable)
        )
        self.assertFalse(any(p.requires_grad for p in self.model.mot.parameters()))
        self.assertFalse(self.model.mot.training)

    def test_v3_zero_init_is_exact_base_and_gradients_stop_at_base(self):
        self.model.prepare_trainable_parameters()
        frozen_base = {
            name: value.detach().clone()
            for name, value in self.model.mot.state_dict().items()
        }
        base_hidden = torch.randn(2, 4, 12, requires_grad=True)
        base_velocity = torch.randn(2, 4, 3, requires_grad=True)
        goal_queries = torch.randn(2, 2, 12, requires_grad=True)
        output, residual, _ = (
            self.model._apply_policy_guard_v3_velocity_residual(
                base_action_hidden=base_hidden,
                base_action_velocity=base_velocity,
                routed_goal_queries=goal_queries,
            )
        )
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertTrue(torch.equal(output, base_velocity.detach()))
        (output - 1.0).square().mean().backward()
        self.assertIsNone(base_hidden.grad)
        self.assertIsNone(base_velocity.grad)
        self.assertIsNotNone(
            self.model.policy_guard_modules[
                "action_velocity_residual"
            ].output_projection.weight.grad
        )
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=1.0e-3,
        )
        optimizer.step()
        for name, value in frozen_base.items():
            self.assertTrue(
                torch.equal(self.model.mot.state_dict()[name], value), name
            )

    def test_v3_forward_cache_starts_equal_to_frozen_base(self):
        self.model.eval()
        batch_size = 2
        video_seq_len = 5
        action_horizon = 4
        context = torch.randn(batch_size, 4, 10)
        context_mask = torch.ones(batch_size, 4, dtype=torch.bool)
        state_only = torch.zeros_like(context_mask)
        current_video = torch.randn(batch_size, video_seq_len, 16)
        routed, _, _ = self.model._encode_policy_guard_goal(
            final_video_hidden=current_video,
            video_tokens_per_frame=video_seq_len,
            context=context,
            context_mask=context_mask,
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
            context_mask=context_mask,
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
        base_output = self.model.action_expert.post_dit(base_hidden, base_pre)
        counterfactual_output = (
            self.model._forward_policy_guard_action_from_cache(
                action_tokens=noisy_action,
                timestep_action=timestep,
                context=context,
                full_context_mask=context_mask,
                state_only_context_mask=state_only,
                video_kv_cache=video_cache,
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_seq_len,
                routed_goal_queries=routed,
            )
        )
        self.assertTrue(torch.equal(counterfactual_output, base_output))

    def test_v3_residual_losses_separate_native_and_counterfactual(self):
        residual = torch.tensor(
            [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]
        )
        base = torch.tensor(
            [[[10.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]]
        )
        target = torch.tensor(
            [[[100.0, 0.0, 0.0]], [[11.0, 0.0, 0.0]]]
        )
        cf_loss, native_loss, regularization, smoothness, metrics = (
            self.model._compute_policy_guard_v3_action_losses(
                predicted_residual=residual,
                base_action_teacher=base,
                target_action=target,
                action_weight=torch.ones(2),
                action_is_pad=None,
                is_counterfactual=torch.tensor([False, True]),
                direct_action_valid=torch.tensor([True, True]),
            )
        )
        self.assertAlmostEqual(float(native_loss), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(cf_loss), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(regularization), 4.0 / 3.0, places=6)
        self.assertEqual(float(smoothness), 0.0)
        self.assertAlmostEqual(float(metrics["pgc_native_fraction"]), 0.5)

    def test_v3_verifier_schedule_is_delayed_and_exact(self):
        self.model.set_training_progress(10, 100)
        self.assertEqual(self.model._policy_guard_verifier_scale(), 0.0)
        self.model.set_training_progress(11, 100)
        self.assertAlmostEqual(self.model._policy_guard_verifier_scale(), 0.2)
        self.model.set_training_progress(15, 100)
        self.assertEqual(self.model._policy_guard_verifier_scale(), 1.0)

    def test_v3_checkpoint_round_trip_has_no_action_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.pt"
            pgc_path = Path(tmpdir) / "pgc_v3.pt"
            torch.save(
                {"format": "fastwam_full_v1", "mot": self.model.mot.state_dict()},
                base_path,
            )
            self.model.load_checkpoint(base_path)
            with torch.no_grad():
                self.model.policy_guard_modules[
                    "action_velocity_residual"
                ].output_projection.bias.add_(0.25)
            expected = {
                key: value.detach().clone()
                for key, value in self.model.policy_guard_modules.state_dict().items()
            }
            self.model.save_checkpoint(pgc_path, step=23)
            payload = torch.load(pgc_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["format"], "fastwam_policy_guard_v3")
            self.assertEqual(payload["step"], 23)
            self.assertNotIn("mot", payload)
            self.assertNotIn("counterfactual_action_adapter", payload)
            self.assertNotIn("counterfactual_lora_config", payload)
            metadata = payload["architecture_metadata"]
            self.assertEqual(
                metadata["counterfactual_tuning"],
                "bounded_velocity_residual",
            )
            self.assertEqual(
                metadata["policy_protection"],
                "single_immutable_base_plus_conservative_hard_gate",
            )

            restored = tiny_pgc_fastwam(version=3)
            restored.load_checkpoint(pgc_path)
            for key, value in expected.items():
                self.assertTrue(
                    torch.equal(
                        restored.policy_guard_modules.state_dict()[key],
                        value,
                    ),
                    key,
                )
            self.assertEqual(
                restored.policy_guard_base_checkpoint,
                str(base_path.resolve()),
            )


class PolicyGuardV4IntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.model = tiny_pgc_fastwam(version=4)

    def test_v4_trains_only_sidecars_and_starts_as_exact_base(self):
        self.assertIsNone(self.model.policy_guard_action_expert)
        with self.assertRaisesRegex(ValueError, "does not permit LoRA"):
            self.model.configure_lora(
                {
                    "enabled": True,
                    "rank": 2,
                    "alpha": 4,
                    "dropout": 0.0,
                    "experts": ["action"],
                    "extra_trainable_patterns": [],
                }
            )
        report = self.model.prepare_trainable_parameters()
        self.assertGreater(report["trainable"], 0)
        trainable = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("policy_guard_modules.") for name in trainable)
        )
        self.assertFalse(any(p.requires_grad for p in self.model.mot.parameters()))
        self.assertFalse(self.model.mot.training)
        self.assertTrue(
            all(
                parameter.dtype == torch.float32
                for parameter in self.model.policy_guard_modules[
                    "verifier"
                ].parameters()
            )
        )

        base_action = torch.randn(2, 4, 3, requires_grad=True)
        goal_queries = torch.randn(2, 2, 12, requires_grad=True)
        candidate, residual, _ = self.model.policy_guard_modules[
            "action_chunk_proposal"
        ](
            base_action=base_action,
            goal_queries=goal_queries,
        )
        self.assertTrue(torch.equal(candidate, base_action.detach()))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        candidate.square().mean().backward()
        self.assertIsNone(base_action.grad)
        self.assertIsNotNone(
            self.model.policy_guard_modules[
                "action_chunk_proposal"
            ].output_projection.weight.grad
        )

    def test_v4_final_action_losses_separate_native_and_counterfactual(self):
        base = torch.zeros(2, 1, 3)
        residual = torch.tensor(
            [[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]
        )
        proposal = base + residual
        target = torch.tensor(
            [[[100.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]
        )
        cf_loss, native_loss, regularization, smoothness, metrics = (
            self.model._compute_policy_guard_v4_action_losses(
                proposed_action=proposal,
                predicted_residual=residual,
                base_action=base,
                target_action=target,
                action_is_pad=None,
                is_counterfactual=torch.tensor([False, True]),
                direct_action_valid=torch.tensor([True, True]),
            )
        )
        self.assertAlmostEqual(float(native_loss), 0.125, places=6)
        self.assertAlmostEqual(float(cf_loss), 0.125, places=6)
        self.assertAlmostEqual(float(regularization), 1.0 / 3.0, places=6)
        self.assertEqual(float(smoothness), 0.0)
        self.assertAlmostEqual(float(metrics["pgc_native_fraction"]), 0.5)

    def test_v4_guard_requires_raw_advantage_and_candidate_support(self):
        base = torch.zeros(2, 2, 3)
        counterfactual = torch.ones_like(base)
        base_value = torch.zeros(2)
        counterfactual_value = torch.tensor([0.19, 0.30])
        output, selected = self.model._select_policy_guard_action(
            base_action=base,
            counterfactual_action=counterfactual,
            base_score=base_value,
            counterfactual_score=counterfactual_value,
            candidate_supported=torch.tensor([True, False]),
        )
        self.assertTrue(torch.equal(output, base))
        self.assertTrue(torch.equal(selected, torch.tensor([False, False])))

        output, selected = self.model._select_policy_guard_action(
            base_action=base,
            counterfactual_action=counterfactual,
            base_score=base_value,
            counterfactual_score=counterfactual_value,
            candidate_supported=torch.tensor([True, True]),
        )
        self.assertTrue(torch.equal(selected, torch.tensor([False, True])))
        self.assertTrue(torch.equal(output[0], base[0]))
        self.assertTrue(torch.equal(output[1], counterfactual[1]))

    def test_v4_training_rollout_produces_deployed_base_chunk(self):
        self.model.eval()
        context = torch.randn(2, 4, 10)
        context_mask = torch.ones(2, 4, dtype=torch.bool)
        base_action, video_hidden, neutral_visual, tokens_per_frame = (
            self.model._rollout_policy_guard_base_action(
                first_frame_latents=torch.randn(2, 2, 1, 1, 1),
                initial_action_noise=torch.randn(2, 4, 3),
                context=context,
                full_context_mask=context_mask,
                state_only_context_mask=torch.zeros_like(context_mask),
                fuse_vae_embedding_in_latents=True,
                num_inference_steps=2,
            )
        )
        self.assertEqual(tuple(base_action.shape), (2, 4, 3))
        self.assertEqual(video_hidden.shape[0], 2)
        self.assertEqual(neutral_visual.shape[0], 2)
        self.assertGreater(tokens_per_frame, 0)
        self.assertTrue(torch.isfinite(base_action).all())

        goal_queries, _, _ = self.model._encode_policy_guard_goal(
            final_video_hidden=video_hidden,
            video_tokens_per_frame=tokens_per_frame,
            context=context,
            context_mask=context_mask,
        )
        candidate, residual, _ = self.model.policy_guard_modules[
            "action_chunk_proposal"
        ](
            base_action=base_action,
            goal_queries=goal_queries,
        )
        self.assertTrue(torch.equal(candidate, base_action))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_v4_checkpoint_round_trip_is_self_contained_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.pt"
            pgc_path = Path(tmpdir) / "pgc_v4.pt"
            torch.save(
                {"format": "fastwam_full_v1", "mot": self.model.mot.state_dict()},
                base_path,
            )
            self.model.load_checkpoint(base_path)
            with torch.no_grad():
                self.model.policy_guard_modules[
                    "action_chunk_proposal"
                ].output_projection.bias.add_(0.125)
            expected = {
                key: value.detach().clone()
                for key, value in self.model.policy_guard_modules.state_dict().items()
            }
            self.model.save_checkpoint(pgc_path, step=37)
            payload = torch.load(pgc_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["format"], "fastwam_policy_guard_v4")
            self.assertEqual(payload["step"], 37)
            self.assertNotIn("mot", payload)
            self.assertNotIn("counterfactual_action_adapter", payload)
            self.assertNotIn("counterfactual_lora_config", payload)
            metadata = payload["architecture_metadata"]
            self.assertEqual(
                metadata["counterfactual_tuning"],
                "rollout_aligned_final_action_residual",
            )
            self.assertEqual(
                metadata["verifier_margin_space"],
                "raw_fp32_pairwise_advantage",
            )
            self.assertEqual(metadata["rollout_num_inference_steps"], 2)

            restored = tiny_pgc_fastwam(version=4)
            restored.load_checkpoint(pgc_path)
            for key, value in expected.items():
                self.assertTrue(
                    torch.equal(
                        restored.policy_guard_modules.state_dict()[key],
                        value,
                    ),
                    key,
                )
            self.assertEqual(
                restored.policy_guard_base_checkpoint,
                str(base_path.resolve()),
            )


class PolicyGuardV5IntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        self.model = tiny_pgc_fastwam(version=5)

    def test_v5_keeps_base_frozen_and_uses_prefix_weighting(self):
        report = self.model.prepare_trainable_parameters()
        self.assertGreater(report["trainable"], 0)
        self.assertFalse(any(p.requires_grad for p in self.model.mot.parameters()))
        self.assertTrue(
            all(
                name.startswith("policy_guard_modules.")
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            )
        )
        prediction = torch.tensor(
            [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
              [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]]
        )
        weighted = self.model._compute_policy_guard_v5_weighted_action_mse_per_sample(
            prediction=prediction,
            target=torch.zeros_like(prediction),
            action_is_pad=None,
        )
        self.assertAlmostEqual(float(weighted), 10.0, places=5)

    def test_v5_completion_weights_post_grasp_and_freezes_other_sidecars(self):
        model = tiny_pgc_fastwam(
            version=5,
            completion_phase_enabled=True,
        )
        report = model.prepare_trainable_parameters()
        self.assertGreater(report["trainable"], 0)
        trainable_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(
                name.startswith("policy_guard_modules.action_chunk_proposal.")
                for name in trainable_names
            )
        )

        base = torch.zeros(1, 4, 3)
        residual = torch.zeros_like(base)
        proposal = base.clone()
        target = torch.ones_like(base)
        goal = torch.tensor([[1.0, 0.0]])

        def action_loss(phase: int) -> float:
            losses = model._compute_policy_guard_v5_action_losses(
                proposed_action=proposal,
                predicted_residual=residual,
                source_predicted_residual=residual,
                base_action=base,
                target_action=target,
                counterfactual_goal_embedding=goal,
                source_goal_embedding=goal,
                action_is_pad=None,
                is_counterfactual=torch.tensor([True]),
                direct_action_valid=torch.tensor([True]),
                paired_language_valid=torch.tensor([True]),
                completion_phase=torch.tensor([phase]),
                completion_phase_valid=torch.tensor([True]),
            )
            return float(losses[0])

        pregrasp = action_loss(0)
        self.assertAlmostEqual(action_loss(1), 2.0 * pregrasp, places=6)
        self.assertAlmostEqual(action_loss(2), 3.0 * pregrasp, places=6)
        metadata = model._policy_guard_metadata()
        self.assertTrue(metadata["completion_phase_enabled"])
        self.assertEqual(
            metadata["completion_trainable_scope"],
            "action_chunk_proposal_only",
        )

    def test_v5_same_state_pair_has_source_zero_and_goal_separation(self):
        base = torch.zeros(2, 4, 3)
        residual = torch.zeros_like(base)
        residual[1] = 0.1
        source_residual = torch.zeros_like(base)
        source_residual[1] = 0.2
        proposal = base + residual
        target = torch.zeros_like(base)
        target[1] = 1.0
        goal = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        source_goal = goal.clone()
        (
            action_loss,
            native_zero,
            source_zero,
            goal_separation,
            residual_separation,
            regularization,
            smoothness,
            metrics,
        ) = self.model._compute_policy_guard_v5_action_losses(
            proposed_action=proposal,
            predicted_residual=residual,
            source_predicted_residual=source_residual,
            base_action=base,
            target_action=target,
            counterfactual_goal_embedding=goal,
            source_goal_embedding=source_goal,
            action_is_pad=None,
            is_counterfactual=torch.tensor([False, True]),
            direct_action_valid=torch.tensor([True, True]),
            paired_language_valid=torch.tensor([False, True]),
        )
        self.assertGreater(float(action_loss), 0.0)
        self.assertEqual(float(native_zero), 0.0)
        self.assertGreater(float(source_zero), 0.0)
        self.assertAlmostEqual(float(goal_separation), 0.2, places=6)
        self.assertEqual(float(residual_separation), 0.0)
        self.assertGreater(float(regularization), 0.0)
        self.assertEqual(float(smoothness), 0.0)
        self.assertAlmostEqual(
            float(metrics["pgc_v5_paired_language_valid_fraction"]),
            0.5,
        )
        self.assertIn("pgc_v5_prefix_final_action_mse_improvement", metrics)

    def test_v5_zero_initialized_residual_separation_backward_is_finite(self):
        base = torch.zeros(1, 4, 3)
        residual = torch.zeros_like(base, requires_grad=True)
        source_residual = torch.zeros_like(base, requires_grad=True)
        goal = torch.tensor([[1.0, 0.0]], requires_grad=True)
        source_goal = torch.tensor([[1.0, 0.0]], requires_grad=True)
        losses = self.model._compute_policy_guard_v5_action_losses(
            proposed_action=base + residual,
            predicted_residual=residual,
            source_predicted_residual=source_residual,
            base_action=base,
            target_action=torch.ones_like(base),
            counterfactual_goal_embedding=goal,
            source_goal_embedding=source_goal,
            action_is_pad=None,
            is_counterfactual=torch.tensor([True]),
            direct_action_valid=torch.tensor([True]),
            paired_language_valid=torch.tensor([True]),
        )
        sum(losses[:7]).backward()
        for tensor in (residual, source_residual, goal, source_goal):
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_v5_verifier_trains_on_wrong_language_and_bad_candidates(self):
        batch_size = 2
        horizon = 4
        current_video = torch.randn(batch_size, 3, 16)
        goal = torch.randn(batch_size, 8)
        base = torch.zeros(batch_size, horizon, 3)
        proposal = base.clone()
        proposal[1] = 0.5
        source = base.clone()
        target = base.clone()
        target[1] = 0.75
        verifier_loss, alignment_loss, metrics = (
            self.model._compute_policy_guard_v5_verifier_loss(
                current_video_hidden=current_video,
                goal_embedding=goal,
                demonstrated_action=target,
                base_candidate_action=base,
                counterfactual_candidate_action=proposal,
                source_candidate_action=source,
                action_is_pad=None,
                is_counterfactual=torch.tensor([False, True]),
                direct_action_valid=torch.tensor([True, True]),
                paired_language_valid=torch.tensor([False, True]),
                goal_ids=torch.tensor([0, 1]),
            )
        )
        self.assertTrue(torch.isfinite(verifier_loss))
        self.assertTrue(torch.isfinite(alignment_loss))
        self.assertIn("pgc_v5_wrong_language_target_advantage", metrics)
        self.assertIn("pgc_v5_bad_candidate_target_advantage", metrics)

    def test_v5_checkpoint_round_trip_records_paired_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.pt"
            pgc_path = Path(tmpdir) / "pgc_v5.pt"
            torch.save(
                {"format": "fastwam_full_v1", "mot": self.model.mot.state_dict()},
                base_path,
            )
            self.model.load_checkpoint(base_path)
            with torch.no_grad():
                self.model.policy_guard_modules[
                    "action_chunk_proposal"
                ].output_projection.bias.add_(0.125)
            expected = {
                key: value.detach().clone()
                for key, value in self.model.policy_guard_modules.state_dict().items()
            }
            self.model.save_checkpoint(pgc_path, step=41)
            payload = torch.load(pgc_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["format"], "fastwam_policy_guard_v5")
            metadata = payload["architecture_metadata"]
            self.assertEqual(
                metadata["counterfactual_tuning"],
                "paired_language_prefix_aligned_action_residual",
            )
            self.assertEqual(metadata["execution_prefix_steps"], 2)
            self.assertEqual(metadata["suffix_loss_weight"], 0.1)

            restored = tiny_pgc_fastwam(version=5)
            restored.load_checkpoint(pgc_path)
            for key, value in expected.items():
                self.assertTrue(
                    torch.equal(
                        restored.policy_guard_modules.state_dict()[key],
                        value,
                    ),
                    key,
                )
            self.assertEqual(
                restored.policy_guard_base_checkpoint,
                str(base_path.resolve()),
            )


class PolicyGuardV8IntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        self.model = tiny_pgc_fastwam(version=8)

    def test_v8_trains_only_proposal_and_weights_closed_loop_rows(self):
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
                name.startswith(
                    "policy_guard_modules.action_chunk_proposal."
                )
                for name in trainable
            )
        )
        self.assertEqual(self.model._policy_guard_verifier_scale(), 0.0)

        base = torch.zeros(3, 4, 3)
        target = torch.ones_like(base)
        residual = torch.zeros_like(base)
        goal = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        losses = self.model._compute_policy_guard_v5_action_losses(
            proposed_action=base,
            predicted_residual=residual,
            source_predicted_residual=residual,
            base_action=base,
            target_action=target,
            counterfactual_goal_embedding=goal,
            source_goal_embedding=goal,
            action_is_pad=None,
            is_counterfactual=torch.tensor([False, True, True]),
            direct_action_valid=torch.tensor([True, True, True]),
            paired_language_valid=torch.tensor([False, True, True]),
            is_closed_loop_corrective=torch.tensor([False, False, True]),
        )
        metrics = losses[-1]
        self.assertAlmostEqual(
            float(metrics["pgc_v8_closed_loop_fraction"]), 1.0 / 3.0
        )
        self.assertAlmostEqual(
            float(metrics["pgc_v8_offline_counterfactual_fraction"]),
            1.0 / 3.0,
        )
        self.assertAlmostEqual(
            float(metrics["pgc_v8_acquisition_sample_weight"]), 1.5
        )

    def test_v8_strictly_warm_starts_v5_and_round_trips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v5_path = root / "v5.pt"
            v8_path = root / "v8.pt"
            v5 = tiny_pgc_fastwam(version=5)
            torch.save(
                {"format": "fastwam_full_v1", "mot": v5.mot.state_dict()},
                base_path,
            )
            v5.load_checkpoint(base_path)
            with torch.no_grad():
                v5.policy_guard_modules[
                    "action_chunk_proposal"
                ].output_projection.bias.add_(0.25)
            expected = {
                key: value.detach().clone()
                for key, value in v5.policy_guard_modules.state_dict().items()
            }
            v5.save_checkpoint(v5_path, step=4000)

            self.model.load_checkpoint(v5_path)
            for key, value in expected.items():
                self.assertTrue(
                    torch.equal(
                        self.model.policy_guard_modules.state_dict()[key],
                        value,
                    ),
                    key,
                )
            self.model.save_checkpoint(v8_path, step=2000)
            payload = torch.load(
                v8_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["format"], "fastwam_policy_guard_v8")
            metadata = payload["architecture_metadata"]
            self.assertEqual(
                metadata["counterfactual_tuning"],
                "closed_loop_replay_verified_target_acquisition_residual",
            )
            self.assertEqual(
                metadata["closed_loop_trainable_scope"],
                "action_chunk_proposal_only",
            )

            restored = tiny_pgc_fastwam(version=8)
            restored.load_checkpoint(v8_path)
            for key, value in self.model.policy_guard_modules.state_dict().items():
                self.assertTrue(
                    torch.equal(
                        restored.policy_guard_modules.state_dict()[key],
                        value,
                    ),
                    key,
                )


class PolicyGuardV6IntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        self.model = tiny_pgc_fastwam(version=6)

    def test_v6_binding_losses_reward_target_and_reject_source_language(self):
        target_attention = torch.tensor([[0.90, 0.05, 0.03, 0.02]])
        source_attention = torch.tensor([[0.02, 0.03, 0.05, 0.90]])
        interaction_teacher = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        source_prototype = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        valid = torch.tensor([True])
        good = self.model._compute_policy_guard_v6_target_binding_losses(
            target_attention=target_attention,
            source_attention=source_attention,
            interaction_teacher=interaction_teacher,
            interaction_valid=valid,
            target_prototype_attention=interaction_teacher,
            target_prototype_valid=valid,
            source_prototype_attention=source_prototype,
            source_prototype_valid=valid,
            direct_action_valid=valid,
            paired_language_valid=valid,
        )
        bad = self.model._compute_policy_guard_v6_target_binding_losses(
            target_attention=source_attention,
            source_attention=target_attention,
            interaction_teacher=interaction_teacher,
            interaction_valid=valid,
            target_prototype_attention=interaction_teacher,
            target_prototype_valid=valid,
            source_prototype_attention=source_prototype,
            source_prototype_valid=valid,
            direct_action_valid=valid,
            paired_language_valid=valid,
        )
        self.assertLess(float(good[0]), float(bad[0]))
        self.assertLess(float(good[1]), float(bad[1]))
        self.assertEqual(float(good[3]), 0.0)
        self.assertGreater(float(bad[3]), 0.0)
        self.assertIn("pgc_v6_same_state_attention_distance", good[-1])

    def test_v6_trains_only_sidecars_and_uses_visual_bottleneck(self):
        report = self.model.prepare_trainable_parameters()
        self.assertGreater(report["trainable"], 0)
        self.assertFalse(any(p.requires_grad for p in self.model.mot.parameters()))
        trainable_names = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(name.startswith("policy_guard_modules.") for name in trainable_names)
        )
        self.assertTrue(
            any(name.startswith("policy_guard_modules.target_binder.") for name in trainable_names)
        )
        self.assertFalse(
            any(name.startswith("policy_guard_modules.goal_graph.") for name in trainable_names)
        )

        with self.assertRaisesRegex(
            ValueError, "language-neutral current visual tokens"
        ):
            self.model._encode_policy_guard_goal(
                final_video_hidden=torch.randn(1, 4, 16),
                video_tokens_per_frame=4,
                context=torch.randn(1, 3, 10),
                context_mask=torch.ones(1, 3, dtype=torch.bool),
                language_context_len=3,
            )

    def test_v6_stages_binding_then_action_then_verifier(self):
        self.model.set_training_progress(10, 100)
        self.assertEqual(
            self.model._policy_guard_target_binding_action_scale(), 0.0
        )
        self.assertEqual(self.model._policy_guard_verifier_scale(), 0.0)

        self.model.set_training_progress(12, 100)
        self.assertAlmostEqual(
            self.model._policy_guard_target_binding_action_scale(), 0.4
        )
        self.assertEqual(self.model._policy_guard_verifier_scale(), 0.0)

        self.model.set_training_progress(15, 100)
        self.assertEqual(
            self.model._policy_guard_target_binding_action_scale(), 1.0
        )
        self.assertEqual(self.model._policy_guard_verifier_scale(), 0.0)

        self.model.set_training_progress(16, 100)
        self.assertAlmostEqual(self.model._policy_guard_verifier_scale(), 0.2)
        self.model.set_training_progress(20, 100)
        self.assertEqual(self.model._policy_guard_verifier_scale(), 1.0)

    def test_v5_warm_start_then_v6_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.pt"
            v5_path = Path(tmpdir) / "pgc_v5.pt"
            v6_path = Path(tmpdir) / "pgc_v6.pt"
            v5 = tiny_pgc_fastwam(version=5)
            torch.save(
                {"format": "fastwam_full_v1", "mot": v5.mot.state_dict()},
                base_path,
            )
            v5.load_checkpoint(base_path)
            with torch.no_grad():
                v5.policy_guard_modules[
                    "action_chunk_proposal"
                ].output_projection.bias.add_(0.125)
            expected_v5 = {
                key: value.detach().clone()
                for key, value in v5.policy_guard_modules.state_dict().items()
            }
            v5.save_checkpoint(v5_path, step=4000)

            migrated = tiny_pgc_fastwam(version=6)
            migrated.load_checkpoint(v5_path)
            migrated_state = migrated.policy_guard_modules.state_dict()
            for key, value in expected_v5.items():
                self.assertTrue(torch.equal(migrated_state[key], value), key)
            self.assertTrue(
                any(key.startswith("target_binder.") for key in migrated_state)
            )
            query_projection = migrated.policy_guard_modules[
                "target_binder"
            ].binding_query_projection[-1]
            self.assertEqual(float(query_projection.weight.abs().max()), 0.0)

            prototype_bank = migrated.policy_guard_target_prototype_bank
            self.assertIsNotNone(prototype_bank)
            with torch.no_grad():
                prototype_bank.task_ids[0] = 123
                prototype_bank.counts[0] = 7
                prototype_bank.prototypes[0, 0] = 1.0

            migrated.save_checkpoint(v6_path, step=100)
            payload = torch.load(v6_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["format"], "fastwam_policy_guard_v6")
            self.assertEqual(
                int(payload["target_prototype_bank"]["task_ids"][0]), 123
            )
            self.assertEqual(
                int(payload["target_prototype_bank"]["counts"][0]), 7
            )
            metadata = payload["architecture_metadata"]
            self.assertEqual(
                metadata["counterfactual_tuning"],
                "visual_target_bottleneck_paired_action_residual",
            )
            self.assertEqual(
                metadata["target_binding_bottleneck"],
                "visual_only_no_direct_language_residual",
            )
            self.assertEqual(
                metadata["target_binding_visual_source"],
                "pre_dit_language_neutral_current_frame",
            )
            self.assertTrue(metadata["target_prototype_bank_persisted"])

            restored = tiny_pgc_fastwam(version=6)
            restored.load_checkpoint(v6_path)
            for key, value in migrated_state.items():
                self.assertTrue(
                    torch.equal(restored.policy_guard_modules.state_dict()[key], value),
                    key,
                )
            restored_bank = restored.policy_guard_target_prototype_bank
            self.assertIsNotNone(restored_bank)
            self.assertTrue(
                torch.equal(restored_bank.task_ids, prototype_bank.task_ids)
            )
            self.assertTrue(
                torch.equal(restored_bank.counts, prototype_bank.counts)
            )
            self.assertTrue(
                torch.equal(restored_bank.prototypes, prototype_bank.prototypes)
            )


class PolicyGuardV7IntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.model = tiny_pgc_fastwam(version=7)

    def test_v7_mask_losses_reward_correct_language_object_binding(self):
        target = torch.tensor([[0.90, 0.05, 0.03, 0.02]])
        source = torch.tensor([[0.02, 0.03, 0.05, 0.90]])
        aux = torch.tensor([[0.02, 0.90, 0.05, 0.03]])
        target_teacher = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        source_teacher = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        aux_teacher = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        valid = torch.tensor([True])
        good = self.model._compute_policy_guard_v7_target_mask_losses(
            target_attention=target,
            source_attention=source,
            aux_attention=aux,
            target_teacher=target_teacher,
            source_teacher=source_teacher,
            aux_teacher=aux_teacher,
            target_mask_valid=valid,
            source_mask_valid=valid,
            aux_mask_valid=valid,
            direct_action_valid=valid,
            paired_language_valid=valid,
        )
        bad = self.model._compute_policy_guard_v7_target_mask_losses(
            target_attention=source,
            source_attention=target,
            aux_attention=target,
            target_teacher=target_teacher,
            source_teacher=source_teacher,
            aux_teacher=aux_teacher,
            target_mask_valid=valid,
            source_mask_valid=valid,
            aux_mask_valid=valid,
            direct_action_valid=valid,
            paired_language_valid=valid,
        )
        self.assertLess(float(good[0]), float(bad[0]))
        self.assertLess(float(good[1]), float(bad[1]))
        self.assertLess(float(good[2]), float(bad[2]))
        self.assertLess(float(good[3]), float(bad[3]))
        self.assertEqual(float(good[4]), 0.0)
        self.assertGreater(float(bad[4]), 0.0)
        self.assertIn("pgc_v7_target_mask_mass", good[-1])

    def test_v7_trains_only_sidecars_without_v6_prototype_bank(self):
        report = self.model.prepare_trainable_parameters()
        self.assertGreater(report["trainable"], 0)
        self.assertIsNone(self.model.policy_guard_target_prototype_bank)
        self.assertFalse(any(p.requires_grad for p in self.model.mot.parameters()))
        trainable_names = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(
            any(
                name.startswith("policy_guard_modules.target_binder.")
                for name in trainable_names
            )
        )
        self.assertFalse(
            any(
                name.startswith("policy_guard_modules.goal_graph.")
                for name in trainable_names
            )
        )

    def test_v5_warm_start_then_v7_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.pt"
            v5_path = Path(tmpdir) / "pgc_v5.pt"
            v7_path = Path(tmpdir) / "pgc_v7.pt"
            v5 = tiny_pgc_fastwam(version=5)
            torch.save(
                {"format": "fastwam_full_v1", "mot": v5.mot.state_dict()},
                base_path,
            )
            v5.load_checkpoint(base_path)
            with torch.no_grad():
                v5.policy_guard_modules[
                    "action_chunk_proposal"
                ].output_projection.bias.add_(0.125)
            expected_v5 = {
                key: value.detach().clone()
                for key, value in v5.policy_guard_modules.state_dict().items()
            }
            v5.save_checkpoint(v5_path, step=4000)

            migrated = tiny_pgc_fastwam(version=7)
            migrated.load_checkpoint(v5_path)
            migrated_state = migrated.policy_guard_modules.state_dict()
            for key, value in expected_v5.items():
                self.assertTrue(torch.equal(migrated_state[key], value), key)
            self.assertIsNone(migrated.policy_guard_target_prototype_bank)
            self.assertEqual(
                float(
                    migrated.policy_guard_modules[
                        "target_binder"
                    ].query_output_projection.weight.abs().max()
                ),
                0.0,
            )

            migrated.save_checkpoint(v7_path, step=100)
            payload = torch.load(v7_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["format"], "fastwam_policy_guard_v7")
            self.assertNotIn("target_prototype_bank", payload)
            metadata = payload["architecture_metadata"]
            self.assertEqual(
                metadata["counterfactual_tuning"],
                "object_token_mask_grounded_paired_action_residual",
            )
            self.assertEqual(
                metadata["target_binding_bottleneck"],
                "spatial_object_tokens_no_direct_language_residual",
            )
            self.assertEqual(
                metadata["target_mask_supervision"],
                "robosuite_element_current_frame_training_only",
            )

            restored = tiny_pgc_fastwam(version=7)
            restored.load_checkpoint(v7_path)
            for key, value in migrated_state.items():
                self.assertTrue(
                    torch.equal(restored.policy_guard_modules.state_dict()[key], value),
                    key,
                )


if __name__ == "__main__":
    unittest.main()
