import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.robotwin.initial_observation_audit import InitialObservationAudit
from experiments.robotwin.no_eraf_probe import CAMERAS


class InitialObservationAuditTest(unittest.TestCase):
    def test_correct_cf_hashes_match_and_only_first_observation_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = InitialObservationAudit(tmp)
            obs = {'joint_action': {'vector': np.zeros(14)},
                   'observation': {c: {'rgb': np.zeros((8, 8, 3), dtype=np.uint8)} for c in CAMERAS}}
            for condition, instruction in [('correct', 'left'), ('counterfactual', 'right')]:
                audit.reset()
                audit.begin(dict(pair_id='pair', source_task='task', task_config='demo_clean',
                    scene_seed=42, episode_index=0, source_instruction='left', counterfactual_instruction='right',
                    policy_instruction=instruction, condition=condition, checkpoint='/check.pt'))
                before = obs['joint_action']['vector'].copy()
                audit.record(obs, instruction)
                np.testing.assert_array_equal(before, obs['joint_action']['vector'])
                audit.record({}, 'later instruction is ignored')
            records = [json.loads(p.read_text()) for p in Path(tmp).glob('*.json')]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]['observation_sha256'], records[1]['observation_sha256'])
            self.assertNotEqual(records[0]['metadata']['policy_instruction'], records[1]['metadata']['policy_instruction'])

    def test_fail_closed_without_episode_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                InitialObservationAudit(tmp).record({}, '')


if __name__ == '__main__':
    unittest.main()
