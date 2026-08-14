import tempfile
import unittest

import torch
import torch.nn as nn

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


class MaskVideoExpert(nn.Module):
    def build_video_to_video_mask(
        self, video_seq_len, video_tokens_per_frame, device
    ):
        mask = torch.ones(
            video_seq_len, video_seq_len, dtype=torch.bool, device=device
        )
        mask[:video_tokens_per_frame, video_tokens_per_frame:] = False
        return mask


class TinyVAE(nn.Module):
    temporal_downsample_factor = 1
    upsampling_factor = 1

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)


def mask_only_model() -> FastWAM:
    model = FastWAM.__new__(FastWAM)
    nn.Module.__init__(model)
    model.video_expert = MaskVideoExpert()
    return model


def tiny_fastwam(
    *,
    transition_contract: bool = False,
    transition_contract_version: int = 2,
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
        use_latent_action_queries=True,
        num_latent_queries=2,
        query_rope_offset=16,
    )
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    )
    return FastWAM(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=TinyVAE(),
        text_dim=10,
        device="cpu",
        torch_dtype=torch.float32,
        langforce_mvp_config={
            "enabled": not transition_contract,
            "enable_prior": not transition_contract,
            "enable_posterior_advantage": not transition_contract,
            "action_reads_raw_video": False,
            "action_reads_language": False,
            "detach_prior_video_cache": True,
        },
        transition_contract_config={
            "enabled": transition_contract,
            "version": transition_contract_version,
            "projection_dim": 8,
            "temperature": 0.07,
            "contract_weight": 0.05,
            "outcome_stop_gradient": True,
            "use_transition_router": True,
            "router_num_heads": 2,
            "direct_action_video_access": False,
            "direct_action_text_access": False,
            "direct_video_text_access": True,
            "action_conditioned_video": False,
            "use_action_effect": False,
            "use_counterfactual_ranking": False,
            "warmup_ratio": 0.05,
            "ramp_ratio": 0.05,
            "policy_recovery_ratio": 0.10,
            "router_ramp_ratio": 0.20,
            "freeze_m1_during_recovery": True,
            "policy_distillation_enabled": bool(
                transition_contract and transition_contract_version == 3
            ),
            "policy_distillation_weight": 1.0,
            "freeze_m1_policy": bool(
                transition_contract and transition_contract_version == 3
            ),
        },
    )


class LangForceMaskTest(unittest.TestCase):
    def test_joint_attention_structure(self):
        model = mask_only_model()
        mask = model._build_mot_attention_mask(
            video_seq_len=6,
            action_seq_len=7,
            video_tokens_per_frame=2,
            device=torch.device("cpu"),
            num_queries=3,
            action_reads_raw_video=False,
        )
        q = slice(6, 9)
        a = slice(9, 13)
        self.assertTrue(mask[q, :2].all())
        self.assertFalse(mask[q, 2:6].any())
        self.assertTrue(mask[q, q].all())
        self.assertFalse(mask[q, a].any())
        self.assertFalse(mask[a, :6].any())
        self.assertTrue(mask[a, q].all())
        self.assertTrue(mask[a, a].all())

    def test_context_masks_keep_state_but_remove_language(self):
        full = torch.tensor([[True, True, False, False, True]])
        full, state = FastWAM._build_context_masks(
            full_context_mask=full,
            language_context_len=4,
            has_proprio=True,
        )
        self.assertFalse(state[:, :4].any())
        self.assertTrue(state[:, 4:].all())

        posterior = FastWAM._build_action_context_mask(
            full_mask=full,
            state_only_mask=state,
            num_queries=2,
            action_horizon=3,
            mode="posterior",
        )
        prior = FastWAM._build_action_context_mask(
            full_mask=full,
            state_only_mask=state,
            num_queries=2,
            action_horizon=3,
            mode="prior",
        )
        self.assertTrue(torch.equal(posterior[:, :2], full[:, None].expand(-1, 2, -1)))
        self.assertTrue(torch.equal(posterior[:, 2:], state[:, None].expand(-1, 3, -1)))
        self.assertTrue(torch.equal(prior, state[:, None].expand(-1, 5, -1)))


class LangForcePriorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = tiny_fastwam().eval()
        self.first_frame = torch.randn(1, 2, 1, 2, 2)
        self.noisy_action = torch.randn(1, 3, 3)
        self.timestep_action = torch.tensor([0.4])
        self.state_only_mask = torch.zeros(1, 4, dtype=torch.bool)

    def _prior(self, context):
        return self.model._forward_prior_action_train(
            first_frame_latents=self.first_frame,
            noisy_action=self.noisy_action,
            timestep_action=self.timestep_action,
            context=context,
            state_only_context_mask=self.state_only_mask,
            fuse_vae_embedding_in_latents=True,
        )

    def _posterior(self, context):
        full_mask = torch.ones(1, 4, dtype=torch.bool)
        state_mask = torch.zeros_like(full_mask)
        timestep_video = torch.zeros(1)
        video_pre = self.model.video_expert.pre_dit(
            x=self.first_frame,
            timestep=timestep_video,
            context=context,
            context_mask=full_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        action_pre = self.model._prepare_action_tokens(
            action_tokens=self.noisy_action,
            timestep=self.timestep_action,
            context=context,
            full_context_mask=full_mask,
            state_only_context_mask=state_mask,
            mode="posterior",
        )
        mask = self.model._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=video_pre["meta"]["tokens_per_frame"],
            device=torch.device("cpu"),
            num_queries=action_pre["meta"]["num_queries"],
            action_reads_raw_video=False,
        )
        tokens = self.model.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=mask,
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
        return self.model.action_expert.post_dit(tokens["action"], action_pre)

    def test_prior_has_no_language_leakage(self):
        context_a = torch.randn(1, 4, 10)
        context_b = torch.randn(1, 4, 10)
        pred_a = self._prior(context_a)
        pred_b = self._prior(context_b)
        self.assertLess((pred_a - pred_b).abs().max().item(), 1.0e-7)

    def test_posterior_has_a_language_path(self):
        context_a = torch.randn(1, 4, 10)
        context_b = torch.randn(1, 4, 10)
        pred_a = self._posterior(context_a)
        pred_b = self._posterior(context_b)
        self.assertGreater((pred_a - pred_b).abs().max().item(), 1.0e-7)

    def test_query_checkpoint_round_trip(self):
        before = self.model.action_expert.latent_action_queries.detach().clone()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny.pt"
            self.model.save_checkpoint(path)
            restored = tiny_fastwam()
            restored.load_checkpoint(path)
        torch.testing.assert_close(
            restored.action_expert.latent_action_queries,
            before,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
