#!/usr/bin/env python3
"""Test deployed action sensitivity to synthetic goal-anchor interventions.

This diagnostic changes predicted semantics only. It does not supply oracle
labels, optimize weights, or measure task success.
"""
import argparse
import json
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'source-bank', 'checkpoint', 'output'):
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
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
        if row['replay_split'] != 'replay_holdout' or row['task_config'] != 'demo_clean': continue
        if any(row.get(k) for k in ('native_retention', 'cf_retention', 'fg_correction')): continue
        if row.get('target_frame_index', row.get('frame_index', 0)) != 0: continue
        task = row.get('source_task', '')
        if not task:
            from experiments.robotwin.pgc_data import ROBOTWIN_TEN_TASK_SPECS
            task = next(s.source_task for s in ROBOTWIN_TEN_TASK_SPECS if s.pair_id == row['pair_id'])
        if task in ('place_a2b_left', 'blocks_ranking_rgb'): selected.setdefault(task, row)
    if len(selected) != 2: raise ValueError('Require both target-task initial holdout observations.')
    policy = load_policy(args.checkpoint, manifest, seed=42)
    model = policy.model; model.requires_grad_(False)
    norm = policy.processor.normalizer.normalizers['action'][policy.processor.shape_meta['action'][0]['key']]
    bridge = model.policy_guard_modules['eraf_action_grounding_bridge']
    injector = model.policy_guard_modules['eraf_action_context_injector']
    raw = RawReplay(args.source_bank)
    routes = {name: {'nonzero': int(p.count_nonzero()), 'norm': float(p.float().norm())}
              for name, p in model.policy_guard_modules.named_parameters()
              if name.startswith(('entity_relation_affordance.query_delta_projection.',
                                  'eraf_action_grounding_bridge.query_delta_projection.'))}
    records = []
    for task, row in selected.items():
        path, frame = raw.locate(row, 'target')
        with h5py.File(path, 'r') as h:
            obs = {'joint_action': {'vector': h['joint_action/vector'][frame].astype(np.float32)},
                'observation': {c: {'rgb': decode_legacy_robotwin_rgb(h[f'observation/{c}/rgb'][frame])} for c in CAMERAS}}
        actions, queries = {}, {}
        for mode in ('base', 'repeat', 'goal_x_plus20cm'):
            captured = []
            def record(module, positional, kwargs):
                captured.append(kwargs['goal_queries'].detach().float().cpu().numpy().copy())
            def intervene(module, positional, kwargs):
                out = dict(kwargs['eraf_outputs']); out['goal_anchor'] = out['goal_anchor'].clone()
                out['goal_anchor'][..., 0] += .2
                return positional, dict(kwargs, eraf_outputs=out)
            hooks = [injector.register_forward_pre_hook(record, with_kwargs=True)]
            if mode == 'goal_x_plus20cm': hooks.append(bridge.register_forward_pre_hook(intervene, with_kwargs=True))
            try:
                policy.reset()
                with torch.no_grad(): value = policy._infer_action_chunk(obs, row['counterfactual_instruction'])
            finally:
                for hook in hooks: hook.remove()
            if not captured: raise ValueError('Production action context did not consume routed queries.')
            actions[mode] = norm.forward(torch.as_tensor(value).float().unsqueeze(0)).detach().cpu().numpy()[0]
            queries[mode] = np.stack(captured)
        if not np.array_equal(actions['base'], actions['repeat']):
            raise ValueError('Repeated fixed-seed inference differs; semantic intervention is not isolated.')
        delta = actions['goal_x_plus20cm'] - actions['base']
        qdelta = queries['goal_x_plus20cm'] - queries['base']
        record = {'task': task, 'pair_id': row['pair_id'], 'scene_seed': row['scene_seed'],
            'task_config': row['task_config'], 'raw_path': str(path), 'frame': frame,
            'state_sha256': array_sha256(obs['joint_action']['vector']),
            'rgb_sha256': {c: array_sha256(obs['observation'][c]['rgb']) for c in CAMERAS},
            'repeat_identical': True, 'query_max_abs': float(np.abs(qdelta).max()),
            'normalized_action_max_abs': float(np.abs(delta).max()),
            'normalized_action_rmse_first24': float(np.sqrt(np.mean(delta[:24] ** 2)))}
        if not all(np.isfinite(v).all() for v in (*actions.values(), *queries.values())):
            raise ValueError('Nonfinite diagnostic output.')
        records.append(record); print(json.dumps(record), flush=True)
    report = {'complete': True, 'checkpoint': args.checkpoint, 'checkpoint_sha256': file_sha256(args.checkpoint),
        'scope': 'Synthetic predicted goal-anchor intervention, fixed observation/instruction/noise, 10-step deployed sampler. Sensitivity is necessary, not evidence of useful semantics or CF success.',
        'optimizer_updates': 0, 'route_parameters': routes, 'records': records,
        'queries_respond': all(r['query_max_abs'] > 0 for r in records),
        'actions_respond': all(r['normalized_action_max_abs'] > 1e-5 for r in records)}
    (root / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__': main()
