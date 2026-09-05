#!/usr/bin/env python3
"""Retain a repair's language adapters and an anchor's remaining control weights."""
from __future__ import annotations
import argparse
import math
from pathlib import Path


def compose(anchor, repair, control_strength=0.):
    if not math.isfinite(control_strength) or not 0 <= control_strength <= 1:
        raise ValueError('Control strength must be between zero and one.')
    if anchor['format'] != 'fastwam_lora_adapter_v1' or repair['format'] != anchor['format']:
        raise ValueError('Expected ordinary LoRA adapter checkpoints.')
    if anchor['base_checkpoint'] != repair['base_checkpoint'] or anchor['lora_config'] != repair['lora_config']:
        raise ValueError('Checkpoints must use the same base and LoRA configuration.')
    source, target = anchor['mot_trainable'], repair['mot_trainable']
    if set(source) != set(target) or any(source[k].shape != target[k].shape for k in source):
        raise ValueError('Adapter parameter names and shapes must match.')
    language = [name for name in source if '.cross_attn.' in name or '.text_embedding.' in name]
    if not language or len(language) == len(source):
        raise ValueError('Expected both language-facing and other control adapters.')
    result = dict(anchor)
    result['mot_trainable'] = {
        name: (target[name] if name in language or control_strength == 1 else
               value if control_strength == 0 else value.lerp(target[name], control_strength))
        for name, value in source.items()}
    result['step'] = repair.get('step', 0)
    return result, language


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--repair', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--control-strength', type=float, default=0.,
                        help='Fraction of non-language repair parameters to retain; zero restores the anchor exactly.')
    args = parser.parse_args()
    result, language = compose(torch.load(args.anchor, map_location='cpu', weights_only=False),
                               torch.load(args.repair, map_location='cpu', weights_only=False), args.control_strength)
    result['decision_repair'] = {'composition': 'repair_language_adapters_with_interpolated_control_parameters',
                                'anchor_checkpoint': args.anchor, 'repair_checkpoint': args.repair,
                                'language_adapter_count': len(language),
                                'control_strength': args.control_strength,
                                'note': 'Uniform checkpoint for every instruction and task; no runtime selection.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save(result, output)
    print(f'[composed] language={len(language)}/{len(result["mot_trainable"])} control_strength={args.control_strength} output={output}')


if __name__ == '__main__':
    main()
