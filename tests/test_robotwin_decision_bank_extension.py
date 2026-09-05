import json
from pathlib import Path
import tempfile
import unittest

from scripts.extend_robotwin_decision_bank import additional_scenes


class BankExtensionTest(unittest.TestCase):
    def test_existing_replay_holdout_and_original_validation_stay_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            bank = Path(temp)
            states = []
            for domain in ('demo_clean', 'demo_randomized'):
                folder = bank / domain
                folder.mkdir()
                pair = {'pair_id': 'pair'}
                for kind in ('native', 'counterfactual'):
                    root = folder / kind
                    (root / 'meta').mkdir(parents=True)
                    records = [{'episode_index': i, 'scene_seed': i, 'raw_hdf5': f'data/episode{i}.hdf5',
                                'source_instruction': 'source', 'counterfactual_instruction': 'target'} for i in range(5)]
                    (root / 'meta/pgc_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
                    pair[kind] = {'hdf5': str(root / 'data/episode1.hdf5')}
                (folder / 'plan.json').write_text(json.dumps({'pairs': [pair], 'split': {'validation_proportion': .2}}))
                states.extend({'task_config': domain, 'pair_id': 'pair', 'episode_index': i,
                               'replay_split': split} for i, split in ((1, 'train'), (2, 'replay_holdout')))
            # Seed42's original five-episode split holds out episode0.
            result = additional_scenes({'states': states}, bank)
            self.assertEqual({(r['task_config'], r['episode_index']) for r in result},
                             {(d, i) for d in ('demo_clean', 'demo_randomized') for i in (3, 4)})


if __name__ == '__main__':
    unittest.main()
