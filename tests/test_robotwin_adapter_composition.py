import unittest
from scripts.compose_robotwin_language_adapter import compose


class CompositionTest(unittest.TestCase):
    def test_keeps_exact_control_weights_and_replaces_only_language(self):
        import torch
        names = ('mixtures.video.text_embedding.0.lora_A',
                 'mixtures.action.blocks.0.cross_attn.q.lora_B',
                 'mixtures.video.blocks.0.self_attn.q.lora_B',
                 'mixtures.action.blocks.0.ffn.0.lora_A')
        anchor = {'format': 'fastwam_lora_adapter_v1', 'base_checkpoint': '/base.pt', 'lora_config': {'rank': 16},
                  'mot_trainable': {name: torch.zeros(2, 2) for name in names}}
        repair = dict(anchor, mot_trainable={name: torch.ones(2, 2) for name in names})
        result, selected = compose(anchor, repair)
        self.assertEqual(selected, list(names[:2]))
        for name in names:
            expected = repair if name in names[:2] else anchor
            self.assertIs(result['mot_trainable'][name], expected['mot_trainable'][name])
            self.assertEqual(anchor['mot_trainable'][name].sum().item(), 0)
        with self.assertRaisesRegex(ValueError, 'same base'):
            compose(anchor, dict(repair, base_checkpoint='/other.pt'))


if __name__ == '__main__':
    unittest.main()
