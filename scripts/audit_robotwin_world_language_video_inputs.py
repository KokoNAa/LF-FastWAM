#!/usr/bin/env python3
"""Audit paired video inputs across checkpoints without loading a GPU model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from scripts.probe_robotwin_world_language import read, sha
from scripts.report_robotwin_world_language import verified_state
from scripts.run_robotwin_world_language_collection import write


def loss_inputs(rows, references, sigmas, seeds):
    expected = {(r, s, n) for r in references for s in sigmas for n in seeds}
    result = {}
    for row in rows:
        key = (row['reference'], row['sigma'], row['seed'])
        if key in result or key not in expected:
            raise ValueError('Unexpected or duplicate denoising cell')
        digest = row['noisy_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid noisy tensor digest')
        result[key] = digest
        right = row['reference']; wrong = 'target' if right == 'source' else 'source'
        for field, margin in [('mse', 'wrong_minus_correct_mse'),
                              ('object_roi_mse', 'wrong_minus_correct_roi_mse')]:
            errors = [row['errors'][lang][field] for lang in [right, wrong]]
            if not all(np.isfinite(e) and e >= 0 for e in errors):
                raise ValueError('Invalid denoising error')
            if not np.isclose(row[margin], errors[1] - errors[0], rtol=0, atol=1e-12):
                raise ValueError('Correct/wrong denoising margin sign or value changed')
    if set(result) != expected:
        raise ValueError('Missing denoising cells')
    for seed in seeds:
        if 1. in sigmas and len({result[r, 1., seed] for r in references}) != 1:
            raise ValueError('Pure noise differs across paired reference branches')
    return result


def decoded_initial(folder, seeds):
    hashes = []
    for language in ['source', 'target']:
        for seed in seeds:
            with Image.open(folder / f'video_{language}_{seed}' / '00.png') as image:
                pixels = np.asarray(image.convert('RGB'))
            if pixels.shape != (384, 320, 3):
                raise ValueError('Unexpected decoded initial image shape')
            hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
    if len(set(hashes)) != 1:
        raise ValueError('Decoded initial frame changed with language or seed')
    return hashes[0]


def audit(root, allow_partial=False):
    probe = root / 'probes'; plan = read(probe / 'plan.json')
    plan_hash = sha(probe / 'plan.json')
    coverage = {model: 0 for model in plan['models']}
    paired = []; individual = []; proof_hashes = {}
    for state in plan['states']:
        references = ['source', 'target'] if state['dual_reference_valid'] else [state['observation_branch']]
        model_inputs = {}
        for model in plan['models']:
            folder = probe / model / 'video' / state['id']
            proof = verified_state(folder, plan_hash)
            if proof is None:
                if allow_partial:
                    continue
                raise ValueError('Video coverage incomplete: ' + str(folder))
            if proof['id'] != state['id'] or proof['experiment'] != 'video':
                raise ValueError('Wrong video proof identity')
            losses = read(folder / 'video_losses.json')
            inputs = loss_inputs(losses['rows'], references, plan['sigmas'], plan['noise_seeds'])
            initial = decoded_initial(folder, plan['noise_seeds'])
            generations = read(folder / 'generations.json')['generations']
            if (len(generations) != 6 or any(g['frames'] != 9 for g in generations)
                    or {(g['language'], g['seed']) for g in generations}
                    != {(lang, seed) for lang in ['source', 'target'] for seed in plan['noise_seeds']}):
                raise ValueError('Wrong generated clip coverage')
            coverage[model] += 1
            proof_hashes[str(folder / 'complete.json')] = sha(folder / 'complete.json')
            model_inputs[model] = (inputs, initial)
            individual.append(dict(state_id=state['id'], model=model, denoising_cells=len(inputs),
                                   initial_decoded_rgb_sha256=initial))
        if len(model_inputs) == len(plan['models']):
            values = list(model_inputs.values())
            if any(value != values[0] for value in values[1:]):
                raise ValueError('Noisy tensors or initial decoded frame differ across models: ' + state['id'])
            paired.append(dict(state_id=state['id'], denoising_cells=len(values[0][0]),
                               initial_decoded_rgb_sha256=values[0][1]))
    result = dict(format='robotwin_world_language_video_input_audit_v1',
                  complete=len(paired) == len(plan['states']), plan_sha256=plan_hash,
                  expected_states=len(plan['states']), state_coverage=coverage,
                  paired_states=len(paired), paired=paired, individual=individual,
                  proof_sha256=proof_hashes,
                  checks=['Every completed state output rehashed against its proof.',
                          'Correct/wrong margins independently recomputed for both full image and task ROI.',
                          'Exact noisy tensor hashes match across models for every reference, sigma and seed.',
                          'Decoded initial RGB matches across languages, noise seeds and models.'],
                  limitation='Generation initial-noise tensors are not stored in frozen outputs. The generation seed and CPU RNG path require the separate runtime source audit; matching initial decoded frames alone does not prove matched future noise. This audit is not semantic scoring or whole-study completion.')
    write(root / ('video_input_audit_partial.json' if allow_partial else 'video_input_audit.json'), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = audit(args.root, args.allow_partial)
    print(json.dumps({k: result[k] for k in ['complete', 'expected_states', 'state_coverage', 'paired_states']}))
