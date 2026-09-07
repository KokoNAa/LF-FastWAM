#!/usr/bin/env python3
"""Freeze a success-only training pool while preserving partial collector status."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def initial_cf_relation(task, state):
    import numpy as np
    if task == 'place_a2b_left':
        a, b = state[:3], state[7:10]
        return bool(a[0] > b[0] and .08 < np.linalg.norm(a[:2] - b[:2]) < .2
                    and abs(a[1] - b[1]) < .05)
    if task == 'blocks_ranking_rgb':
        state = np.asarray(state, dtype=np.float32)
        red, green, blue = state[:3], state[7:10], state[14:17]
        eps = np.array([.13, .03], dtype=np.float32)
        return bool(blue[0] < green[0] < red[0]
                    and np.all(abs(red[:2] - green[:2]) < eps)
                    and np.all(abs(green[:2] - blue[:2]) < eps))
    raise ValueError('Only the two declared target tasks can be pooled')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections', type=Path, nargs='+', required=True)
    ap.add_argument('--parent-manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--task', choices=['place_a2b_left', 'blocks_ranking_rgb'], required=True)
    ap.add_argument('--scenes', type=int, default=12)
    args = ap.parse_args()
    if args.scenes < 10:
        ap.error('Retain at least ten training scenes per target task')
    import numpy as np
    from experiments.robotwin.eraf_fg_contract import scene_key
    from experiments.robotwin.eraf_fg_data import (
        file_metadata, verify_retention_capture, validate_retention_scene)
    from scripts.collect_robotwin_eraf_fg import historical_scene_keys
    parent = json.loads(args.parent_manifest.read_text())
    excluded = historical_scene_keys(parent['states'])
    groups, sources = {}, []
    for path in args.collections:
        before = file_metadata(path)
        data = json.loads(path.read_text())
        if file_metadata(path) != before:
            raise ValueError('Collector manifest changed during reading; retry after it stops')
        if data['source_task'] != args.task or data['condition'] != 'counterfactual':
            raise ValueError('Pool mixes tasks or conditions')
        current = defaultdict(list)
        for row in data['states']:
            current[scene_key(row)].append(row)
        if len(current) != data['successful_scenes']:
            raise ValueError('Collector scene count does not match its records')
        sources.append(dict(metadata=before, complete=data['complete'],
                            successful_scenes=data['successful_scenes'],
                            attempted_scenes=data['attempted_scenes']))
        for key, rows in current.items():
            if key in groups or key in excluded:
                raise ValueError('Duplicate scene or overlap with parent training/holdout data')
            if (key[0] != args.task or key[1] != 'demo_clean' or not 80200000 <= key[2] < 81000000
                    or any(row['replay_split'] != 'train' or not row.get('cf_retention') for row in rows)):
                raise ValueError('Only fresh successful target CF training scenes are eligible')
            if validate_retention_scene(rows) != 'target':
                raise ValueError('Retention language mismatch')
            groups[key] = rows
    if len(groups) < args.scenes:
        raise ValueError(f'Need {args.scenes} verified successful scenes, found {len(groups)}')
    selected = sorted(groups)[:args.scenes]
    rows, audit, teachers = [], [], set()
    for key in selected:
        scene = groups[key]
        verify_retention_capture(scene[0])
        teachers.add(json.dumps(scene[0]['teacher_checkpoint_metadata'], sort_keys=True))
        with np.load(scene[0]['capture_path'], allow_pickle=False) as arrays:
            state, actions = arrays['initial_physical_state'], arrays['executed_actions']
            if (not np.isfinite(state).all() or actions.ndim != 2 or actions.shape[1] != 14
                    or len(actions) < 32 or not np.isfinite(actions).all()):
                raise ValueError('Missing nontrivial actual policy execution')
            if initial_cf_relation(args.task, state):
                raise ValueError('An admitted training scene already had the CF relation initially')
            audit.append(dict(scene_seed=key[2], initial_cf_relation=False,
                              actual_executed_actions=len(actions), captured_states=len(scene)))
        rows.extend(scene)
    if len(teachers) != 1:
        raise ValueError('Selected scenes use different CF teachers')
    report = dict(format='robotwin_policy_retention_pool_v1', complete=True,
                  purpose='Curated successful training scenes, not a policy success-rate estimate',
                  source_task=args.task, condition='counterfactual', requested_scenes=args.scenes,
                  successful_scenes=len(selected), available_successful_scenes=len(groups),
                  selection_rule='First unique successful scene seeds in ascending order at freeze time',
                  source_collections=sources, original_collectors_all_complete=all(s['complete'] for s in sources),
                  parent_manifest=file_metadata(args.parent_manifest),
                  initial_goal_audit=audit, hash_scans=False, states=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'states', 'source_collections', 'initial_goal_audit'}}))


if __name__ == '__main__':
    main()
