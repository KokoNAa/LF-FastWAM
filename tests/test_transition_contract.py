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
        )
        routed_b, z_b, _ = self.model.encode_intended_transition(
            video_tokens=video_tokens,
            video_tokens_per_frame=4,
            context=context + 0.5,
            full_context_mask=full_mask,
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
            restored = tiny_fastwam(transition_contract=True)
            restored.load_checkpoint(path)
        actual = next(restored.transition_contract_modules.parameters())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_legacy_m1_checkpoint_loads_into_tc_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/m1.pt"
            legacy = tiny_fastwam()
            legacy.save_checkpoint(path)
            self.model.load_checkpoint(path)
        torch.testing.assert_close(
            self.model.action_expert.latent_action_queries,
            legacy.action_expert.latent_action_queries,
        )


if __name__ == "__main__":
    unittest.main()
