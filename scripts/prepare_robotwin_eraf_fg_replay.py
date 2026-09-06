#!/usr/bin/env python3
"""Cache frozen inputs for verified FG windows and successful CF retention."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['worker', 'merge'])
    for key in ('manifest', 'checkpoint', 'output'):
        ap.add_argument('--' + key, required=True)
    ap.add_argument('--collections', nargs='+', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    args = ap.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(args.manifest).read_text())
    from experiments.robotwin.eraf_fg_contract import validate_correction, scene_key
    groups = []
    seen = set()
    for path in args.collections:
        collection = json.loads(Path(path).read_text())
        if collection.get('complete') is not True:
            raise ValueError(f'Incomplete collection: {path}')
        if 'records' in collection:
            items = [[validate_correction(r)] for r in collection['records']]
        else:
            scenes = defaultdict(list)
            for row in collection['states']:
                if not row.get('cf_retention') or not row.get('full_cf_episode_success'):
                    raise ValueError('Retention input is not an audited successful CF rollout.')
                scenes[scene_key(row)].append(row)
            items = list(scenes.values())
        for rows in items:
            key = scene_key(rows[0])
            if key in seen:
                raise ValueError(f'Duplicate scene across collection inputs: {key}')
            seen.add(key)
            groups.append(rows)
    if args.mode == 'merge':
        rows = []
        for shard in range(args.shards):
            report = json.loads((root / f'shard{shard}/complete.json').read_text())
            if not report['complete'] or report['collections'] != args.collections:
                raise ValueError('Cache shard provenance mismatch.')
            rows.extend(json.loads(line) for line in (root / f'shard{shard}/states.jsonl').read_text().splitlines())
        if {scene_key(r) for r in rows} != seen:
            raise ValueError('Missing prepared scenes.')
        train = {scene_key(r) for r in rows if r['replay_split'] == 'train'}
        holdout = {scene_key(r) for r in rows if r['replay_split'] == 'replay_holdout'}
        if train & holdout or len({r['id'] for r in base['states'] + rows}) != len(base['states']) + len(rows):
            raise ValueError('Split overlap or duplicate replay IDs.')
        base.update(states=base['states'] + rows, complete=True, parent_manifest=args.manifest,
                    correction_collections=args.collections, preparation_checkpoint=args.checkpoint)
        (root / 'manifest.json').write_text(json.dumps(base, indent=2))
        print(f'[merged] added={len(rows)} scenes={len(seen)}', flush=True)
        return
    if not 0 <= args.shard < args.shards:
        ap.error('Invalid shard.')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import numpy as np
    import torch
    from experiments.robotwin.eraf_fg_bridge import load_policy, capture_frozen_inputs, file_sha256
    from experiments.robotwin.eraf_fg_contract import CAMERAS, action_windows
    from experiments.robotwin.compact_replay import capture_delta
    from experiments.robotwin.pgc_data import array_sha256
    policy = load_policy(args.checkpoint, base)
    policy.model.requires_grad_(False)
    policy.num_inference_steps = 1  # Only frozen VAE/T5 inputs are retained.
    norm = policy.processor.normalizer.normalizers['action'][policy.processor.shape_meta['action'][0]['key']]
    shard = root / f'shard{args.shard}'
    (shard / 'payloads').mkdir(parents=True, exist_ok=False)
    journal = (shard / 'states.jsonl').open('x', buffering=1)
    count = 0
    for records in groups[args.shard::args.shards]:
        record = records[0]
        fg = not record.get('cf_retention')
        path = Path(record['frame_path'] if fg else record['capture_path'])
        if file_sha256(path) != record['frame_sha256' if fg else 'capture_sha256']:
            raise ValueError('Collection archive identity changed.')
        with np.load(path, allow_pickle=False) as arrays:
            actions = arrays['actions'] if fg else None
            if fg and array_sha256(actions) != record['correction_action_sha256']:
                raise ValueError('Correction actions changed.')
            windows = action_windows(actions) if fg else records
            parent = parent_path = None
            for window in windows:
                frame = window.start if fg else window['frame_index']
                raw = window.action if fg else arrays['reference_action_raw'][frame]
                if raw.shape != (32, 14) or not np.isfinite(raw).all():
                    raise ValueError('Expected finite 32x14 reference.')
                valid = window.valid if fg else np.ones(32, dtype=bool)
                observation = {'observation': {c: {'rgb': arrays[c][frame]} for c in CAMERAS},
                               'joint_action': {'vector': actions[frame] if fg else arrays['state'][frame]}}
                policy.reset()
                captured = capture_frozen_inputs(policy, observation, record['counterfactual_instruction'])
                # The clean capture explicitly carries no gold or teacher memory.
                if captured['policy_guard_state'] is not None:
                    raise ValueError('Frozen input capture carried a policy memory state.')
                references = {'target': norm.forward(torch.from_numpy(raw).unsqueeze(0)).cpu()}
                validity = {'target': torch.from_numpy(valid).unsqueeze(0)}
                ident = f'eraf_fg_{record["source_task"]}_{record["scene_seed"]}_{frame}' if fg else window['id']
                target = shard / 'payloads' / (ident + '.pt')
                body = {k: captured[k] for k in ('video_inputs', 'action_inputs')}
                extras = {k: captured[k] for k in ('proprio', 'policy_guard_state')}
                if parent is None:
                    parent, parent_path = body, target
                    payload = {'captured': {'target': captured}, 'references': references, 'valid': validity}
                else:
                    payload = {'format': 'robotwin_eraf_fg_compact_v1', 'parent_payload': str(parent_path),
                               'capture_deltas': {'target': capture_delta(body, parent)},
                               'capture_extras': {'target': extras}, 'references': references, 'valid': validity}
                torch.save(payload, target)
                row = (record if fg else window) | {'id': ident, 'payload': str(target),
                    'frame_index': frame, 'fg_correction': fg, 'reference_valid_actions': int(valid.sum()),
                    'frozen_input_protocol': 'robotwin_eraf_fg_pre_dit_v1',
                    'policy_memory': 'none', 'preparation_checkpoint': args.checkpoint}
                journal.write(json.dumps(row) + '\n')
                count += 1
            print(f'[cache] shard={args.shard} scene={record["scene_seed"]} windows={count}', flush=True)
    journal.close()
    (shard / 'complete.json').write_text(json.dumps({'complete': True, 'states': count,
        'collections': args.collections, 'shard': args.shard, 'shards': args.shards}))


if __name__ == '__main__':
    main()
