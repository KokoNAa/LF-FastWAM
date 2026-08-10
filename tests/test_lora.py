import tempfile
import unittest

import torch

from test_langforce_mvp import tiny_fastwam


LORA_CONFIG = {
    "enabled": True,
    "rank": 2,
    "alpha": 4,
    "dropout": 0.0,
    "experts": ["video", "action"],
}


class FastWAMLoRATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_zero_init_preserves_linear_output_then_updates(self):
        model = tiny_fastwam().eval()
        linear = model.video_expert.blocks[0].self_attn.q
        inputs = torch.randn(2, 3, linear.in_features)
        expected = linear(inputs)

        report = model.configure_lora(LORA_CONFIG)
        self.assertTrue(report["enabled"])
        self.assertGreater(len(report["modules"]), 0)
        self.assertGreater(report["parameters"], 0)

        actual = linear(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        with torch.no_grad():
            linear.lora_B.fill_(0.05)
        updated = linear(inputs)
        self.assertGreater((updated - expected).abs().max().item(), 0.0)

    def test_only_adapters_and_small_action_modules_are_trainable(self):
        model = tiny_fastwam()
        model.configure_lora(LORA_CONFIG)
        report = model.prepare_trainable_parameters()

        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(any(name.endswith(".lora_A") for name in trainable_names))
        self.assertTrue(any(name.endswith(".lora_B") for name in trainable_names))
        self.assertIn("action_expert.latent_action_queries", trainable_names)
        self.assertIn("action_expert.action_encoder.weight", trainable_names)
        self.assertIn("action_expert.head.weight", trainable_names)
        self.assertNotIn("video_expert.patch_embedding.weight", trainable_names)
        self.assertNotIn("action_expert.time_embedding.0.weight", trainable_names)
        self.assertLess(report["trainable"], report["total"])

    def test_adapter_checkpoint_loads_base_and_auto_injects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = f"{tmpdir}/base.pt"
            adapter_path = f"{tmpdir}/adapter.pt"

            base = tiny_fastwam()
            base.save_checkpoint(base_path)

            adapted = tiny_fastwam()
            adapted.configure_lora(LORA_CONFIG)
            adapted.load_checkpoint(base_path)
            adapted.prepare_trainable_parameters()
            with torch.no_grad():
                for name, parameter in adapted.named_parameters():
                    if name.endswith(".lora_B"):
                        parameter.fill_(0.03)
                adapted.action_expert.latent_action_queries.add_(0.2)
            adapted.save_checkpoint(adapter_path, step=7)

            payload = torch.load(adapter_path, map_location="cpu")
            self.assertEqual(payload["format"], "fastwam_lora_adapter_v1")
            self.assertEqual(payload["step"], 7)
            self.assertTrue(payload["mot_trainable"])

            restored = tiny_fastwam()
            restored.load_checkpoint(adapter_path)
            self.assertTrue(restored.lora_enabled)
            restored_state = restored.mot.state_dict()
            for name, value in payload["mot_trainable"].items():
                torch.testing.assert_close(
                    restored_state[name].cpu(), value, rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
