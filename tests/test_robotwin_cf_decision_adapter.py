import itertools
import unittest
from collections import Counter
from unittest.mock import patch

from scripts.train_robotwin_cf_decision_adapter import FOCUS, average_gradients, pair_stream, small_validation_set


class DecisionSamplingTest(unittest.TestCase):
    def test_training_keeps_all_scenes_and_excludes_replay_holdout(self):
        rows = [{"id": f"{p}/{d}/{s}", "pair_id": p, "task_config": d,
                 "replay_split": "train" if s < 3 else "replay_holdout"}
                for p in (*FOCUS, "delayed") for d in ("clean", "random") for s in range(4)]
        # Two focus pairs get two copies, delayed pair one: 5*2*3 draws.
        drawn = list(itertools.islice(pair_stream(rows, 42, 2), 30))
        counts = Counter(r["id"] for r in drawn)
        self.assertEqual(len(counts), 18)
        for row in rows:
            self.assertEqual(counts[row["id"]], 0 if row["replay_split"] != "train"
                             else 2 if row["pair_id"] in FOCUS else 1)
        self.assertEqual(len(small_validation_set(rows)), 6)
        self.assertEqual(drawn, list(itertools.islice(pair_stream(rows, 42, 2), 30)))

    def test_gradient_average_includes_rank_with_unused_parameter(self):
        import torch
        a = torch.nn.Parameter(torch.zeros(2))
        b = torch.nn.Parameter(torch.zeros(1))
        a.grad = torch.tensor([2., 4.])
        with patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_world_size', return_value=2), \
             patch('torch.distributed.all_reduce', side_effect=lambda flat: flat.add_(torch.tensor([6., 8., 10.]))):
            average_gradients([a, b])
        torch.testing.assert_close(a.grad, torch.tensor([4., 6.]))
        torch.testing.assert_close(b.grad, torch.tensor([5.]))

    def test_initial_sampling_weight_preserves_phase_samples(self):
        rows = [{'id': name, 'pair_id': 'task', 'task_config': 'clean', 'replay_split': 'train',
                 'sampling_weight': weight} for name, weight in [('initial', 4), ('phase', 1)]]
        counts = Counter(r['id'] for r in itertools.islice(pair_stream(rows, 42), 5))
        self.assertEqual(counts, {'initial': 4, 'phase': 1})

    def test_dense_trajectory_length_does_not_change_task_frequency(self):
        rows = [{'id': f'{task}/{i}', 'pair_id': task, 'task_config': 'clean',
                 'replay_split': 'train'} for task, length in [('short', 2), ('long', 8)]
                for i in range(length)]
        rows.append({'id': 'heldout', 'pair_id': 'long', 'task_config': 'clean',
                     'replay_split': 'replay_holdout'})
        drawn = list(itertools.islice(pair_stream(rows, 42, 1, True), 16))
        self.assertEqual(Counter(row['pair_id'] for row in drawn), {'short': 8, 'long': 8})
        self.assertEqual(len({row['id'] for row in drawn if row['pair_id'] == 'long'}), 8)
        self.assertNotIn('heldout', {row['id'] for row in drawn})


if __name__ == "__main__":
    unittest.main()
