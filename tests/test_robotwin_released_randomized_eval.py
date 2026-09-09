import unittest

from scripts.run_robotwin_released_randomized_eval import first_ten


class ReleasedRandomizedCatalogTest(unittest.TestCase):
    def rows(self):
        return [dict(scene_seed=100+i, episode_index=i, task_config='demo_randomized',
                     counterfactual_goal_ever_success=i >= 10) for i in range(12)]

    def test_preserves_prefix_without_selecting_successful_scenes(self):
        rows = self.rows()
        self.assertEqual(first_ten(rows), rows[:10])
        self.assertEqual(len(rows), 12)
        self.assertFalse(any(r['counterfactual_goal_ever_success'] for r in first_ten(rows)))

    def test_rejects_incomplete_or_duplicate_catalog(self):
        with self.assertRaises(ValueError): first_ten(self.rows()[:10])
        rows = self.rows()
        rows[-1]['scene_seed'] = rows[0]['scene_seed']
        with self.assertRaises(ValueError): first_ten(rows)

    def test_rejects_wrong_domain_and_reordered_catalog(self):
        rows = self.rows()
        rows[0]['task_config'] = 'demo_clean'
        with self.assertRaises(ValueError): first_ten(rows)
        with self.assertRaises(ValueError): first_ten(list(reversed(self.rows())))
