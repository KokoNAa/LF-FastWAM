import io
import unittest

import torch

from fastwam.models.wan22.action_dit import ActionDiT


def make_action_dit(*, use_queries: bool) -> ActionDiT:
    return ActionDiT(
        hidden_dim=16,
        action_dim=3,
        ffn_dim=32,
        text_dim=10,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=2,
        attn_head_dim=4,
        num_layers=2,
        use_gradient_checkpointing=False,
        use_latent_action_queries=use_queries,
        num_latent_queries=3,
        query_rope_offset=16,
    )


class ActionDiTQueryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_query_pack_rope_context_and_output_shape(self):
        model = make_action_dit(use_queries=True).eval()
        action = torch.randn(2, 4, 3)
        timestep = torch.tensor([0.2, 0.7])
        context = torch.randn(2, 5, 10)
        full_mask = torch.ones(2, 5, dtype=torch.bool)
        state_mask = torch.zeros_like(full_mask)
        state_mask[:, -1] = True

        pre = model.pre_dit(
            action_tokens=action,
            timestep=timestep,
            context=context,
            context_mask=full_mask,
            use_queries=True,
            query_context_mask=full_mask,
            action_context_mask=state_mask,
        )

        self.assertEqual(pre["tokens"].shape, (2, 7, 16))
        self.assertEqual(pre["t_mod"].shape, (2, 7, 6, 16))
        self.assertEqual(pre["context_mask"].shape, (2, 7, 5))
        self.assertTrue(pre["context_mask"][:, :3].all())
        self.assertFalse(pre["context_mask"][:, 3:, :-1].any())
        self.assertTrue(pre["context_mask"][:, 3:, -1].all())
        self.assertTrue(
            torch.equal(
                pre["freqs"][:3, 0],
                model.freqs[16:19].to(pre["freqs"].device),
            )
        )
        self.assertTrue(
            torch.equal(
                pre["freqs"][3:, 0],
                model.freqs[:4].to(pre["freqs"].device),
            )
        )
        self.assertTrue(torch.equal(pre["t_mod"][0, 0], pre["t_mod"][1, 0]))

        pred = model.post_dit(pre["tokens"], pre)
        self.assertEqual(pred.shape, (2, 4, 3))

    def test_baseline_path_is_numerically_unchanged(self):
        baseline = make_action_dit(use_queries=False).eval()
        query_capable = make_action_dit(use_queries=True).eval()
        incompatible = query_capable.load_state_dict(
            baseline.state_dict(), strict=False
        )
        self.assertEqual(incompatible.missing_keys, ["latent_action_queries"])
        self.assertEqual(incompatible.unexpected_keys, [])

        action = torch.randn(2, 4, 3)
        timestep = torch.tensor([0.2, 0.7])
        context = torch.randn(2, 5, 10)
        context_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, False, False, False]]
        )
        expected = baseline(action, timestep, context, context_mask)
        actual = query_capable(
            action,
            timestep,
            context,
            context_mask,
            use_queries=False,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_query_parameter_round_trip(self):
        source = make_action_dit(use_queries=True)
        buffer = io.BytesIO()
        torch.save(source.state_dict(), buffer)
        buffer.seek(0)

        restored = make_action_dit(use_queries=True)
        restored.load_state_dict(torch.load(buffer, map_location="cpu"))
        torch.testing.assert_close(
            restored.latent_action_queries,
            source.latent_action_queries,
            rtol=0,
            atol=0,
        )

    def test_old_backbone_key_set_excludes_queries(self):
        model = make_action_dit(use_queries=True)
        backbone_keys = model.backbone_key_set(model.state_dict().keys())
        self.assertNotIn("latent_action_queries", backbone_keys)


if __name__ == "__main__":
    unittest.main()
