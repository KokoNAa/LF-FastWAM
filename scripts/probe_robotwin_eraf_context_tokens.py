#!/usr/bin/env python3
"""Measure fixed-state denoising sensitivity to ERAF token presence/amplitude.

This is an offline mechanism diagnostic, not a closed-loop ablation or success
evaluation. Policy weights, observations, instructions and noise remain fixed.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


@contextmanager
def injection_mode(model, mode):
    scales = {'schedule_zero': 0., 'near_zero_tokens': 1e-6, 'full': 1.}
    module = model.policy_guard_modules['eraf_action_context_injector']
    original, enabled = module.forward, model.policy_guard_enabled
    measurements = []
    if mode != 'off' and mode not in scales:
        raise ValueError('Unknown token intervention.')
    def forward(**kwargs):
        result = original(**dict(kwargs, external_scale=scales[mode]))
        measurements.append({k: float(v.detach().cpu()) for k, v in result[2].items()})
        return result
    try:
        model.policy_guard_enabled = mode != 'off'
        if mode != 'off':
            module.forward = forward
        yield measurements
    finally:
        module.forward = original
        model.policy_guard_enabled = enabled


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'source-bank', 'checkpoint', 'output'):
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    import h5py
    import numpy as np
    import torch
    from experiments.robotwin.eraf_fg_bridge import load_policy, file_sha256
    from experiments.robotwin.eraf_fg_data import RawReplay
    from experiments.robotwin.eraf_fg_contract import CAMERAS
    from experiments.robotwin.image_io import decode_legacy_robotwin_rgb
    from experiments.robotwin.pgc_data import array_sha256
    manifest = json.loads(Path(args.manifest).read_text())
    selected = {}
    for row in manifest['states']:
        if row['replay_split'] != 'replay_holdout' or any(row.get(k)
                for k in ('native_retention', 'cf_retention', 'fg_correction')):
            continue
        if any(row.get(lang + '_frame_index', row.get('frame_index', 0)) != 0 for lang in ('source', 'target')):
            continue
        selected.setdefault((row['pair_id'], row['task_config']), row)
    if len(selected) != 10 or len({k[0] for k in selected}) != 5:
        raise ValueError('Require one existing initial holdout scene per task/domain.')
    policy = load_policy(args.checkpoint, manifest, seed=42)
    policy.model.requires_grad_(False)
    norm = policy.processor.normalizer.normalizers['action'][policy.processor.shape_meta['action'][0]['key']]
    raw = RawReplay(args.source_bank)
    records, arrays = [], {}
    for row in selected.values():
        for language in ('source', 'target'):
            path, frame = raw.locate(row, language)
            with h5py.File(path, 'r') as h:
                observation = {'joint_action': {'vector': h['joint_action/vector'][frame].astype(np.float32)},
                    'observation': {c: {'rgb': decode_legacy_robotwin_rgb(h[f'observation/{c}/rgb'][frame])}
                                    for c in CAMERAS}}
            instruction = row['source_instruction' if language == 'source' else 'counterfactual_instruction']
            predictions, metrics = {}, {}
            for mode in ('off', 'schedule_zero', 'near_zero_tokens', 'full'):
                policy.reset()
                with injection_mode(policy.model, mode) as measured, torch.no_grad():
                    actions = policy._infer_action_chunk(observation, instruction)
                normalized = norm.forward(torch.as_tensor(actions).float().unsqueeze(0)).detach().cpu().numpy()[0]
                if normalized.shape != (32, 14) or not np.isfinite(normalized).all():
                    raise ValueError('Invalid predicted action chunk.')
                predictions[mode] = normalized
                metrics[mode] = measured[-1] if measured else {}
                arrays[f'q{len(records):03d}_{mode}'] = normalized
            comparisons = {}
            for mode in ('schedule_zero', 'near_zero_tokens', 'full'):
                delta = predictions[mode] - predictions['off']
                comparisons[mode] = {'max_abs': float(np.abs(delta).max()),
                    **{f'rmse_first{n}': float(np.sqrt(np.mean(delta[:n] ** 2))) for n in (12, 24, 32)}}
            record = {'pair_id': row['pair_id'], 'task_config': row['task_config'],
                'scene_seed': row['scene_seed'], 'language': language, 'instruction': instruction,
                'raw_path': str(path), 'frame': frame,
                'state_sha256': array_sha256(observation['joint_action']['vector']),
                'rgb_sha256': {c: array_sha256(observation['observation'][c]['rgb']) for c in CAMERAS},
                'comparisons_to_eraf_off': comparisons, 'injection_metrics': metrics}
            records.append(record)
            print(f'[token-probe] queries={len(records)}/20 identity={comparisons["schedule_zero"]["max_abs"]:.6g} near_zero_rmse24={comparisons["near_zero_tokens"]["rmse_first24"]:.6g}', flush=True)
    np.savez_compressed(output / 'normalized_actions.npz', **arrays)
    identity = max(r['comparisons_to_eraf_off']['schedule_zero']['max_abs'] for r in records)
    result = {'complete': True, 'checkpoint': args.checkpoint, 'checkpoint_sha256': file_sha256(args.checkpoint),
        'queries': len(records), 'denoising_steps': 10, 'seed': 42,
        'schedule_zero_identity_max_abs': identity, 'schedule_zero_reproduces_off': identity <= 1e-5,
        'mean_rmse_first24': {mode: float(np.mean([r['comparisons_to_eraf_off'][mode]['rmse_first24'] for r in records]))
                             for mode in ('schedule_zero', 'near_zero_tokens', 'full')},
        'scope': 'Offline fixed-state action sensitivity. Near-zero amplitude retains valid appended tokens. No closed-loop success claim.',
        'records': records}
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
