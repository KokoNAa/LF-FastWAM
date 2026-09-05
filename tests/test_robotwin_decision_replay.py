import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tests.test_robotwin_joint_adapter_repair import JointAdapterTest
from experiments.robotwin.decision_replay import (
    DecisionReplay, balanced_order, deployment_modes, endpoint_loss, tensor_digest,
)
from experiments.robotwin.joint_adapter_repair import configure_adapters, predict
from scripts.probe_robotwin_no_eraf import sha256


class DecisionReplayTest(JointAdapterTest):
    def setUp(self):
        super().setUp()
        configure_adapters(self.model, ('video', 'action'))
        self.captured = {k: self.capture(k)[1] for k in ('source', 'target')}
        self.refs = {'source': torch.zeros(1, 32, 14), 'target': torch.ones(1, 32, 14) * .2}
        self.model.mot.train()

    def test_endpoint_checkpoint_restores_training_modes_and_rng_through_backward(self):
        modes = [m.training for m in self.model.modules()]
        rng = torch.get_rng_state().clone()
        gradients = []
        values = []
        for checkpoint in (False, True):
            self.model.zero_grad(set_to_none=True)
            loss = endpoint_loss(self.model, self.captured, self.refs, 123, checkpoint=checkpoint)
            loss.backward()
            values.append(float(loss.detach()))
            gradients.append({n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None})
            self.assertEqual(modes, [m.training for m in self.model.modules()])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(values[0], values[1])
        self.assertEqual(gradients[0].keys(), gradients[1].keys())
        for key in gradients[0]:
            torch.testing.assert_close(gradients[0][key], gradients[1][key], rtol=0, atol=0)
        for expert in ('video', 'action'):
            self.assertGreater(sum(float(p.grad.square().sum()) for n, p in self.model.mot.named_parameters()
                                   if n.startswith('mixtures.' + expert) and p.grad is not None), 0)

    def test_replay_preserves_ordinary_sample_and_rng_and_zero_weight_gradients(self):
        sample = {'action': self.refs['source'], 'prompt': ['instruction']}
        sample_hash = tensor_digest(sample)
        def original(sample):
            noise = torch.randn(1, 32, 14)
            value = predict(self.model, self.captured['source'], noise, self.time)
            return (value - sample['action']).square().mean(), {'ordinary': 1.}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / 'inputs.pt'
            torch.save({'captured': self.captured, 'references': self.refs}, payload)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'complete': True, 'states': [{'id': 'a', 'pair_id': 'p',
                'task_config': 'clean', 'replay_split': 'train', 'payload': str(payload),
                'payload_sha256': sha256(payload)}]}))
            reports = []
            for weight in (None, 0., .25):
                torch.manual_seed(142)
                self.model.zero_grad(set_to_none=True)
                if weight is None:
                    loss, _ = original(sample)
                else:
                    replay = DecisionReplay(str(manifest), sha256(manifest), weight, 27000, root / str(weight))
                    loss, _ = replay.loss(self.model, original, sample)
                after = torch.get_rng_state().clone()
                loss.backward()
                reports.append((float(loss.detach()), after,
                    {n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None}))
                self.assertTrue(torch.equal(after, torch.get_rng_state()))
                if weight is not None:
                    replay.handle.close()
            self.assertEqual(reports[0][0], reports[1][0])
            for key in reports[0][2]:
                torch.testing.assert_close(reports[0][2][key], reports[1][2][key], rtol=0, atol=0)
            self.assertGreater(reports[2][0], reports[0][0])
            self.assertTrue(torch.equal(reports[0][1], reports[2][1]))
            self.assertEqual(sample_hash, tensor_digest(sample))

    def test_balanced_pairs_never_consume_global_rng(self):
        rows = [{'id': f'{p}/{d}/{s}', 'pair_id': p, 'task_config': d}
                for p in ('a', 'b', 'c', 'd', 'e') for d in ('clean', 'random') for s in range(8)]
        before = torch.get_rng_state()
        order = balanced_order(rows, 27)
        self.assertEqual(order, balanced_order(rows, 27))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len({r['id'] for r in order}), 80)
        for i in range(0, 80, 10):
            self.assertEqual(len({(r['pair_id'], r['task_config']) for r in order[i:i+10]}), 10)


if __name__ == '__main__':
    unittest.main()
