import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.prepare_robotwin_native_retention import select_episodes
from scripts.train_robotwin_cf_decision_adapter import mixed_stream


class NativeReplayTest(unittest.TestCase):
    def test_original_holdout_scene_and_benchmark_exclusion(self):
        from experiments.robotwin.no_eraf_probe import training_episode_ids
        rows = [{'episode_index': i, 'scene_seed': i // 3, 'source_task': 'task',
                 'task_config': 'clean'} for i in range(30)]
        original = training_episode_ids(30, .1, 42)
        heldout = {r['scene_seed'] for r in rows if r['episode_index'] not in original}
        selected = select_episodes(rows, .1, 42, {9})
        self.assertTrue(selected)
        self.assertTrue(all(r['episode_index'] in original for r in selected))
        self.assertFalse({r['scene_seed'] for r in selected} & (heldout | {9}))

    def test_native_mix_is_half_independent_of_bank_lengths(self):
        rows = [{'id': f'{kind}_{i}', 'pair_id': 'pair', 'task_config': 'clean',
                 'replay_split': 'train', 'native_retention': kind == 'native'}
                for kind, size in [('native', 2), ('paired', 100)] for i in range(size)]
        drawn = list(itertools.islice(mixed_stream(rows, 42, 1, True), 60))
        self.assertEqual(sum(r['native_retention'] for r in drawn), 30)
        self.assertTrue(all(drawn[i]['native_retention'] == bool(i % 2) for i in range(60)))

    def test_teacher_finishes_before_student_and_native_gradient_is_correct(self):
        import torch
        from experiments.robotwin.native_teacher import retention_backward
        parameter = torch.nn.Parameter(torch.tensor([2.]))
        events = []
        scheduler = SimpleNamespace(num_train_timesteps=1000,
            add_noise=lambda reference, noise, time: noise,
            training_weight=lambda time: torch.tensor(2.))
        model = SimpleNamespace(train_action_scheduler=scheduler, device='cpu', torch_dtype=torch.float32)
        def teacher_predict(*args):
            events.append('teacher')
            return torch.tensor([1.])
        def student_predict(*args):
            events.append('student')
            return parameter * 1.
        with patch('experiments.robotwin.joint_adapter_repair.predict', student_predict):
            terms = retention_backward(model, {}, torch.zeros(1), torch.zeros(1), torch.tensor([500.]),
                SimpleNamespace(predict=teacher_predict), 3., .5)
        self.assertEqual(events, ['teacher', 'teacher', 'student', 'student'])
        torch.testing.assert_close(parameter.grad, torch.tensor([15.]))
        self.assertEqual(terms, {'retention_flow_mse': 1., 'retention_endpoint_mse': 1.})


if __name__ == '__main__':
    unittest.main()
