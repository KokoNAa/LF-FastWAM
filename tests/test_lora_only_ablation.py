import tempfile
import unittest
from unittest import mock

import torch

from test_policy_guard import tiny_pgc_fastwam


CONTROL_CONFIG = {
    "enabled": True,
    "rank": 2,
    "alpha": 2,
    "dropout": 0.0,
    "experts": ["video", "action"],
    "extra_trainable_patterns": [],
    "paired_language_control": {
        "enabled": True,
        "bidirectional_supervision": True,
        "deployment_matched_action_cache": True,
        "correct_branch_action_ranking": True,
        "world_language_weight": 0.10,
        "world_language_margin": 0.01,
        "native_action_weight": 1.0,
        "counterfactual_action_weight": 1.0,
        "action_language_weight": 1.0,
        "action_language_margin": 0.01,
        "regularization_weight": 1.0e-6,
    },
}


def tiny_lora_only_control():
    model = tiny_pgc_fastwam(configure_lora=False, version=2)
    # Reuse the query-free tiny Experts while exercising the exact no-PGC
    # control branch. The dormant v2 modules remain frozen and unused.
    model.policy_guard_enabled = False
    model.configure_lora(CONTROL_CONFIG)
    return model


class LoRAOnlyAblationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(97)

    def test_control_rejects_policy_guard(self):
        model = tiny_pgc_fastwam(configure_lora=False, version=2)
        with self.assertRaisesRegex(ValueError, "no-ERAF ablation"):
            model.configure_lora(CONTROL_CONFIG)

    def test_control_loss_has_no_eraf_and_only_lora_gradients(self):
        model = tiny_lora_only_control()
        trainable = model.prepare_trainable_parameters()
        self.assertGreater(trainable["trainable"], 0)
        trainable_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(
                name.endswith(".lora_A") or name.endswith(".lora_B")
                for name in trainable_names
            )
        )

        batch_size = 4
        context = torch.randn(batch_size, 4, 10)
        context_mask = torch.ones(batch_size, 4, dtype=torch.bool)
        input_latents = torch.randn(batch_size, 2, 2, 2, 2)
        inputs = {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": input_latents[:, :, 0:1].clone(),
            "fuse_vae_embedding_in_latents": True,
            "action": torch.randn(batch_size, 3, 3),
            "action_is_pad": None,
            "image_is_pad": None,
            "pgc_is_counterfactual": torch.tensor(
                [False, False, True, True]
            ),
            "pgc_direct_action_valid": torch.ones(
                batch_size, dtype=torch.bool
            ),
            "pgc_paired_language_valid": torch.tensor(
                [False, False, True, True]
            ),
            "pgc_source_context": torch.randn(batch_size, 4, 10),
            "pgc_source_context_mask": context_mask.clone(),
            "pgc_target_context": torch.randn(batch_size, 4, 10),
            "pgc_target_context_mask": context_mask.clone(),
            "pgc_bidirectional_language_valid": torch.ones(
                batch_size, dtype=torch.bool
            ),
        }
        prefetched_caches = []
        action_caches = []
        original_prefill = model.mot.prefill_video_cache
        original_action_forward = model._forward_shared_action_from_cache

        def record_prefill(*args, **kwargs):
            result = original_prefill(*args, **kwargs)
            prefetched_caches.append(result[0])
            return result

        def record_action_forward(**kwargs):
            action_caches.append(kwargs["video_kv_cache"])
            return original_action_forward(**kwargs)

        with mock.patch.object(
            model.mot,
            "prefill_video_cache",
            side_effect=record_prefill,
        ), mock.patch.object(
            model,
            "_forward_shared_action_from_cache",
            side_effect=record_action_forward,
        ):
            loss, metrics = model._training_loss_lora_paired_language_control(
                inputs=inputs,
                full_context_mask=context_mask,
            )
        self.assertEqual(len(prefetched_caches), 2)
        self.assertEqual(len(action_caches), 2)
        self.assertIs(action_caches[0], prefetched_caches[0])
        self.assertIs(action_caches[1], prefetched_caches[1])
        self.assertIsNot(action_caches[0], action_caches[1])
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["lora_only_eraf_enabled"], 0.0)
        self.assertEqual(metrics["lora_only_policy_guard_enabled"], 0.0)
        self.assertEqual(metrics["loss_pgc_v9_eraf"], 0.0)
        self.assertEqual(
            metrics["lora_only_bidirectional_supervision_enabled"], 1.0
        )
        self.assertEqual(
            metrics["lora_only_deployment_matched_action_cache"], 1.0
        )
        self.assertEqual(
            metrics["lora_only_correct_branch_action_ranking"], 1.0
        )
        for name in (
            "loss_lora_only_source_action_flow",
            "loss_lora_only_target_action_flow",
            "loss_lora_only_source_world_language_ranking",
            "loss_lora_only_target_world_language_ranking",
            "loss_lora_only_source_action_language_ranking",
            "loss_lora_only_target_action_language_ranking",
        ):
            self.assertIn(name, metrics)

        loss.backward()
        gradient_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        self.assertTrue(gradient_names)
        self.assertTrue(gradient_names <= trainable_names)
        self.assertFalse(
            any(name.startswith("policy_guard_") for name in gradient_names)
        )

    def test_bidirectional_context_selection_reverses_by_row(self):
        source = torch.tensor([[[1.0]], [[2.0]]])
        target = torch.tensor([[[10.0]], [[20.0]]])
        source_mask = torch.tensor([[True], [False]])
        target_mask = torch.tensor([[False], [True]])
        correct, correct_mask, wrong, wrong_mask = (
            tiny_lora_only_control()._select_bidirectional_language_contexts(
                is_counterfactual=torch.tensor([False, True]),
                source_context=source,
                source_context_mask=source_mask,
                target_context=target,
                target_context_mask=target_mask,
            )
        )
        self.assertTrue(torch.equal(correct[:, 0, 0], torch.tensor([1.0, 20.0])))
        self.assertTrue(torch.equal(wrong[:, 0, 0], torch.tensor([10.0, 2.0])))
        self.assertTrue(torch.equal(correct_mask[:, 0], torch.tensor([True, True])))
        self.assertTrue(torch.equal(wrong_mask[:, 0], torch.tensor([False, False])))

    def test_correct_branch_ranking_detaches_wrong_language_reference(self):
        correct = torch.tensor([1.0], requires_grad=True)
        wrong = torch.tensor([1.0], requires_grad=True)
        loss = tiny_lora_only_control()._paired_action_ranking_per_sample(
            correct_error=correct,
            wrong_error=wrong,
            margin=0.01,
            correct_branch_ranking=True,
        ).sum()
        loss.backward()
        self.assertTrue(torch.equal(correct.grad, torch.ones_like(correct)))
        self.assertIsNone(wrong.grad)

    def test_legacy_ranking_detaches_correct_language_reference(self):
        correct = torch.tensor([1.0], requires_grad=True)
        wrong = torch.tensor([1.0], requires_grad=True)
        loss = tiny_lora_only_control()._paired_action_ranking_per_sample(
            correct_error=correct,
            wrong_error=wrong,
            margin=0.01,
            correct_branch_ranking=False,
        ).sum()
        loss.backward()
        self.assertIsNone(correct.grad)
        self.assertTrue(torch.equal(wrong.grad, -torch.ones_like(wrong)))

    def test_checkpoint_persists_strict_control_contract(self):
        model = tiny_lora_only_control()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = f"{tmpdir}/step_10000.pt"
            model.save_checkpoint(checkpoint, step=10000)
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
        self.assertEqual(payload["format"], "fastwam_lora_adapter_v1")
        self.assertEqual(payload["step"], 10000)
        self.assertTrue(
            payload["lora_config"]["paired_language_control"]["enabled"]
        )
        self.assertTrue(
            payload["lora_config"]["paired_language_control"][
                "bidirectional_supervision"
            ]
        )
        self.assertTrue(
            payload["lora_config"]["paired_language_control"][
                "deployment_matched_action_cache"
            ]
        )
        self.assertTrue(
            payload["lora_config"]["paired_language_control"][
                "correct_branch_action_ranking"
            ]
        )
        self.assertEqual(payload["lora_config"]["experts"], ["video", "action"])
        self.assertEqual(
            payload["lora_config"]["extra_trainable_patterns"], []
        )


if __name__ == "__main__":
    unittest.main()
