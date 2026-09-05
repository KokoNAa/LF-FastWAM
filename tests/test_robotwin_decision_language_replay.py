import unittest
from pathlib import Path

from experiments.robotwin.decision_language_replay import replace_language, seen_instruction_pairs


class LanguageReplayTest(unittest.TestCase):
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
