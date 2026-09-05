import unittest
from pathlib import Path
import json
import tempfile
import ast
import sys
from types import ModuleType
from unittest.mock import patch

from experiments.robotwin.decision_language_replay import build_seen_contexts, bound_spatial_instruction_pairs, replace_language, seen_instruction_pairs


class LanguageReplayTest(unittest.TestCase):
    def test_augmented_contexts_use_deployment_prompt_and_keep_pair_binding(self):
        repo = Path(__file__).resolve().parents[1]
        # Read the authoritative constant without loading the GPU dataset stack.
        tree = ast.parse((repo / 'src/fastwam/datasets/lerobot/robot_video_dataset.py').read_text())
        template = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == 'DEFAULT_PROMPT'
                            for target in node.targets))
        dataset = ModuleType('fastwam.datasets.lerobot.robot_video_dataset')
        dataset.DEFAULT_PROMPT = template

        class Encoder:
            def __init__(self):
                self.prompts = []

            def encode_prompt(self, prompts):
                self.prompts.extend(prompts)
                return prompts, [True] * len(prompts)

        pairs = [{'source': 'Put red on green.', 'target': 'Put green on red.'}]
        rows = [dict(pair_id='test', replay_split='train', seen_instruction_pairs=pairs),
                dict(pair_id='test', replay_split='train', seen_instruction_pairs=pairs),
                dict(pair_id='heldout', replay_split='holdout',
                     seen_instruction_pairs=[{'source': 'SECRET', 'target': 'SECRET'}])]
        model = Encoder()
        with patch.dict(sys.modules, {dataset.__name__: dataset}):
            contexts = build_seen_contexts(model, repo, rows)
        self.assertEqual(set(contexts), {'test'})
        self.assertCountEqual(model.prompts, [template.format(task=text) for text in pairs[0].values()])
        for branch, text in pairs[0].items():
            self.assertEqual(contexts['test'][0][branch], ([template.format(task=text)], [True]))

    def test_spatial_goal_reversal_preserves_actual_object_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            description = root / 'third_party/RoboTwin/description'
            (description / 'task_instruction').mkdir(parents=True)
            (description / 'objects_description').mkdir()
            (description / 'task_instruction/place_a2b_left.json').write_text(json.dumps(
                {'seen': ['Place {A} on the left of {B}.'], 'unseen': ['UNSEEN {A} {B}']}))
            for name, text in [('box', 'box with a left arrow'), ('bottle', 'red bottle')]:
                (description / f'objects_description/{name}.json').write_text(json.dumps({'seen': [text]}))
            pair = bound_spatial_instruction_pairs(root, 'place_a2b_left_to_right', {'{A}': 'box', '{B}': 'bottle'})[0]
            self.assertEqual(pair['source'], 'Place the box with a left arrow on the left of the red bottle.')
            self.assertEqual(pair['target'], 'Place the box with a left arrow on the right of the red bottle.')

    def test_goal_slots_reverse_without_arm_or_unresolved_tokens(self):
        repo = Path(__file__).resolve().parents[1]
        for pair_id in ('blocks_ranking_rgb_to_bgr', 'stack_blocks_two_green_on_red_to_red_on_green'):
            pairs = seen_instruction_pairs(repo, pair_id)
            self.assertEqual(len(pairs), 8)
            for pair in pairs:
                self.assertNotEqual(pair['source'], pair['target'])
                self.assertNotIn('{', pair['source'] + pair['target'])
                self.assertNotIn('arm', pair['source'] + pair['target'])
                expected = pair['source'].replace('red block', 'TEMP').replace(
                    'blue block' if pair_id.startswith('blocks_') else 'green block', 'red block').replace(
                    'TEMP', 'blue block' if pair_id.startswith('blocks_') else 'green block')
                self.assertEqual(pair['target'], expected)

    def test_text_replacement_preserves_state_and_original_cache(self):
        import torch
        inputs = {'context': torch.arange(15.).reshape(1, 5, 3),
                  'context_mask': torch.ones(1, 5, dtype=torch.bool),
                  'state_only_context_mask': torch.tensor([[False, False, False, False, True]])}
        captured = {key: inputs.copy() for key in ('video_inputs', 'action_inputs')}
        text, mask = torch.zeros(1, 4, 3), torch.ones(1, 4, dtype=torch.bool)
        result = replace_language(captured, text, mask)
        for key in captured:
            torch.testing.assert_close(result[key]['context'][:, :4], text)
            torch.testing.assert_close(result[key]['context'][:, 4:], captured[key]['context'][:, 4:])
            self.assertIs(result[key]['state_only_context_mask'], inputs['state_only_context_mask'])
        torch.testing.assert_close(inputs['context'], torch.arange(15.).reshape(1, 5, 3))


if __name__ == '__main__':
    unittest.main()
