import copy
import unittest

import torch

from tests.test_robotwin_same_state_repair import TinyModel
from experiments.robotwin.joint_adapter_repair import (
    build_cache, capture_inputs, configure_adapters, paired_backward, predict,
)
from experiments.robotwin.same_state_repair import audit_frozen, frozen_versions, move_cache


class JointAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        self.model = TinyModel()
        self.noise = torch.randn(1, 32, 14)
        self.time = torch.tensor([700.])

    def capture(self, language):
        def run():
            return self.model._predict_action_noise_with_cache(
                latents_action=self.noise, timestep_action=self.time, **self.model.cache(language))
        return capture_inputs(self.model, run)

    def test_production_capture_replay_and_gradient_scope(self):
        for experts in [('action',), ('video', 'action')]:
            selected = configure_adapters(self.model, experts)
            frozen = frozen_versions(self.model)
            direct, capture, calls = self.capture('source')
            self.assertEqual(calls, 1)
            self.assertNotIn('pre_dit', vars(self.model.video_expert))
            result = predict(self.model, capture, self.noise, self.time)
            torch.testing.assert_close(result, direct, rtol=0, atol=0)
            result.square().mean().backward()
            for expert in experts:
                self.assertGreater(sum(float(p.grad.square().sum()) for n, p in selected.items()
                                       if n.split('.')[1] == expert and p.grad is not None), 0)
            audit_frozen(self.model, frozen)
            self.model.zero_grad(set_to_none=True)

    def test_video_inputs_are_recomputed_after_adapter_update(self):
        selected = configure_adapters(self.model, ('video', 'action'))
        _, capture, _ = self.capture('target')
        original = predict(self.model, capture, self.noise, self.time, checkpoint=False).detach()
        with torch.no_grad():
            for n, p in selected.items():
                if n.startswith('mixtures.video.') and n.endswith('.lora_B'):
                    p.add_(0.1)
        current = predict(self.model, capture, self.noise, self.time, checkpoint=False).detach()
        self.assertGreater(float((current - original).abs().max()), 1e-5)
        fresh, _, _ = self.capture('target')
        torch.testing.assert_close(current, fresh, rtol=0, atol=0)

    def test_paired_loss_identical_before_scope_specific_updates(self):
        template = copy.deepcopy(self.model.state_dict())
        reports = []
        for experts in [('action',), ('video', 'action')]:
            self.model.load_state_dict(template)
            self.model.zero_grad(set_to_none=True)
            selected = configure_adapters(self.model, experts)
            captured = {k: self.capture(k)[1] for k in ('source', 'target')}
            refs = {'source': torch.zeros_like(self.noise), 'target': torch.ones_like(self.noise) * .2}
            reports.append(paired_backward(self.model, captured, refs, self.noise, self.time, .8, .25, 4))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in selected.values() if p.grad is not None))
        self.assertEqual(reports[0], reports[1])


if __name__ == '__main__':
    unittest.main()
