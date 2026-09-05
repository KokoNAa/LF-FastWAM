import unittest
import numpy as np
from scripts.prepare_robotwin_shared_decision_replay import shared_decision_frames
from scripts.train_robotwin_cf_decision_adapter import small_validation_set


class SharedDecisionTest(unittest.TestCase):
    def test_only_shared_late_observations_with_distinct_future_are_selected(self):
        source = np.zeros((100, 14))
        target = source.copy()
        target[51:, 0] = 1.
        selected = shared_decision_frames(source, target, lambda f: f <= 48)
        self.assertEqual([r['frame'] for r in selected], [36, 40, 44, 48])
        self.assertTrue(all(r['joint_action_rmse_32'] > .02 for r in selected))
        self.assertEqual(shared_decision_frames(source, source, lambda f: True), [])
        self.assertEqual(shared_decision_frames(source, target, lambda f: False), [])
        target[0, 0] = 1.
        self.assertEqual(shared_decision_frames(source, target, lambda f: True), [])

    def test_initial_and_post_grasp_holdouts_remain_separate(self):
        rows = [{'id': str(i), 'pair_id': 'pair', 'task_config': 'clean', 'replay_split': 'replay_holdout',
                 'validation_phase': phase} for i, phase in enumerate(['initial', 'post_grasp', 'post_grasp'])]
        self.assertEqual([r['id'] for r in small_validation_set(rows)], ['0', '1'])


if __name__ == '__main__':
    unittest.main()
