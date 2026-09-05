#!/usr/bin/env python3
"""Evaluate a cold-loaded checkpoint on every audited multi-task replay state.

This measures deployment-sampled action alignment, not simulator goal success.
Every state's first noise-seed replay is checked against the normal policy.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]


def main():
    from scripts.probe_robotwin_no_eraf import read_json, write_json, sha256, load_probe_policy
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--checkpoint', required=True)
    kind = ap.add_mutually_exclusive_group(required=True)
    kind.add_argument('--step', type=int)
    kind.add_argument('--base', action='store_true')
    ap.add_argument('--output', required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = ap.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = read_json(args.manifest)
    if manifest.get('complete') is not True:
        raise ValueError('Replay bank is incomplete.')
    plan = read_json(Path(manifest['source_probe']) / 'plan.json')
    key = 'base' if args.base else f'step{args.step}'
    if args.base and Path(args.checkpoint).resolve() != Path(plan['base_checkpoint']).resolve():
        raise ValueError('Base probe must use the original released checkpoint.')
    plan['checkpoints'][key] = str(Path(args.checkpoint).resolve())
    plan['checkpoint_sha256'][key] = sha256(args.checkpoint)
    policy, audit = load_probe_policy(plan, key, args.gpu)
    import numpy as np
    import torch
    from experiments.robotwin.joint_adapter_repair import build_cache
    from experiments.robotwin.same_state_repair import move_cache, sample_cached_actions
    from experiments.robotwin.no_eraf_probe import CAMERAS, difference, reference_metrics
    model = policy.model
    norm = policy.processor.normalizer.normalizers['action'][policy.processor.shape_meta['action'][0]['key']]
    write_json(root / 'checkpoint_audit.json', audit)
    rows, replays = [], []
    for state in manifest['states']:
        if sha256(state['payload']) != state['payload_sha256'] or sha256(state['file']) != state['sha256']:
            raise ValueError('Audited state/payload changed.')
        payload = torch.load(state['payload'], map_location='cpu', weights_only=True)
        refs = {k: v[0].numpy() for k, v in payload['references'].items()}
        with np.load(state['file'], allow_pickle=False) as handle:
            obs = {'joint_action': {'vector': handle['state']},
                   'observation': {c: {'rgb': handle[c]} for c in CAMERAS}}
        with torch.no_grad():
            caches = {k: build_cache(model, move_cache(v, model.device)) for k, v in payload['captured'].items()}
            for seed in args.seeds:
                values = {k: sample_cached_actions(model, cache, seed, 10) for k, cache in caches.items()}
                np.savez_compressed(root / f"{state['id']}_seed{seed}.npz", **values,
                                    source_reference=refs['source'], target_reference=refs['target'])
                for horizon in (24, 32):
                    s, t, sr, tr = (v[:horizon] for v in (values['source'], values['target'], refs['source'], refs['target']))
                    result = reference_metrics(s, t, sr, 'native', sr, tr)['dual_reference']
                    rows.append({k: state[k] for k in ('id', 'pair_id', 'task_config', 'replay_split')} |
                                {'seed': seed, 'horizon': horizon, **result})
                if seed == args.seeds[0]:
                    for language, field in (('source', 'source_instruction'), ('target', 'counterfactual_instruction')):
                        policy.seed = seed
                        policy.policy_guard_state = None
                        raw = policy._infer_action_chunk(obs, state[field])
                        # Invert output denormalization without the reference
                        # normalizer's clamp: predictions may exceed +/-5.
                        normalized = (torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0) * norm.scale + norm.offset)[0].numpy()
                        error = difference(normalized, values[language])['max_abs']
                        if error > 1e-5:
                            raise ValueError('Production and bank inference differ.')
                        replays.append({'id': state['id'], 'language': language, 'max_abs': error})
        del caches, payload
        print(f"[evaluated] {state['id']}", flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row['pair_id'], row['task_config'], row['replay_split'], row['horizon']].append(row)
    summary = []
    for (pair, domain, split, horizon), values in sorted(groups.items()):
        eligible = [r for r in values if r['expert_references_distinguishable']]
        summary.append({'pair_id': pair, 'task_config': domain, 'replay_split': split, 'horizon': horizon,
            'rows': len(values), 'distinguishable_rows': len(eligible),
            'both_correct': sum(r['source_language_prefers_source_expert'] and r['target_language_prefers_target_expert'] for r in eligible),
            'source_rmse': float(np.mean([r['source_prediction_source_rmse'] for r in values])),
            'target_rmse': float(np.mean([r['target_prediction_target_rmse'] for r in values])),
            'delta_projection': float(np.mean([r['language_delta_projection_on_expert_delta'] for r in eligible])) if eligible else None})
    write_json(root / 'summary.json', {'complete': True, 'checkpoint': audit,
        'manifest_sha256': sha256(args.manifest), 'rows': rows, 'groups': summary,
        'production_replays': replays, 'max_production_error': max(r['max_abs'] for r in replays),
        'scope': 'Action alignment over multiple tasks/scenes/noise seeds. Replay holdout belongs to the original training split. Not closed-loop CF success.'})
    print('[complete]', root / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
