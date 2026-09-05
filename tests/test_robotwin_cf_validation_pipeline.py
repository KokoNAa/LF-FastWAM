import copy
import unittest
from scripts.run_robotwin_cf_validation_pipeline import initial_comparison


class InitialComparisonTest(unittest.TestCase):
    def fixtures(self):
        return [{'observation_sha256': 'same', 'metadata': {'pair_id': 'pair', 'task_config': 'clean',
            'condition': condition, 'scene_seed': 42, 'source_instruction': 'left',
            'counterfactual_instruction': 'right', 'policy_instruction': instruction}}
            for condition, instruction in [('correct', 'left'), ('counterfactual', 'right')]]

    def test_requires_matching_language_and_observation_in_every_model(self):
        a, b = self.fixtures(), self.fixtures()
        self.assertTrue(initial_comparison({'base': a, 'model': b})['exact_initial_observations_equal'])
        for field in ('observation_sha256', 'policy_instruction'):
            changed = copy.deepcopy(b)
            if field == 'observation_sha256': changed[1][field] = 'different'
            else: changed[1]['metadata'][field] = 'different'
            with self.assertRaises(ValueError): initial_comparison({'base': a, 'model': changed})

    def test_detects_within_model_scene_mismatch_and_duplicate(self):
        a = self.fixtures()
        a[1]['observation_sha256'] = 'different'
        with self.assertRaises(ValueError): initial_comparison({'base': a})
        with self.assertRaises(ValueError): initial_comparison({'base': self.fixtures() * 2})


if __name__ == '__main__':
    unittest.main()
