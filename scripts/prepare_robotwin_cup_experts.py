#!/usr/bin/env python3
"""Cache ordinary cup pairs and retain the declared FG target tasks."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['worker', 'merge'])
    for key in ('manifest', 'checkpoint', 'output'):
        ap.add_argument('--'+key, required=True)
    ap.add_argument('--collections', nargs='+', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=6)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--target-tasks', nargs='+', default=['place_a2b_left', 'place_empty_cup'])
    args = ap.parse_args()
    from experiments.robotwin.cup_expert_data import FORMAT, validate_pair, paired_windows
    from experiments.robotwin.eraf_fg_contract import CAMERAS, scene_key
    from scripts.collect_robotwin_eraf_fg import historical_scene_keys
    base = json.loads(Path(args.manifest).read_text())
    records = []
    for path in args.collections:
        collection = json.loads(Path(path).read_text())
        if collection.get('format') != FORMAT or collection.get('complete') is not True:
            raise ValueError('Incomplete or wrong ordinary expert collection')
        records.extend(validate_pair(row) for row in collection['records'])
    keys = {scene_key(r) for r in records}
    if not keys or len(keys) != len(records) or keys & historical_scene_keys(base['states']):
        raise ValueError('Empty, duplicated, or previously used ordinary cup scenes')
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    binding = {k: getattr(args, k) for k in ('manifest', 'checkpoint', 'collections', 'shards', 'target_tasks')}
    if args.mode == 'merge':
        rows = []
        for shard in range(args.shards):
            report = json.loads((root/f'shard{shard}/complete.json').read_text())
            if report.get('complete') is not True or report['binding'] != binding:
                raise ValueError('Ordinary cup cache shard mismatch')
            rows.extend(json.loads(line) for line in (root/f'shard{shard}/states.jsonl').read_text().splitlines())
        if {scene_key(r) for r in rows} != keys:
            raise ValueError('Missing ordinary expert cache scenes')
        retained = [r for r in base['states'] if not r.get('fg_correction')
                    or r['source_task'] in args.target_tasks]
        all_rows = retained + rows
        if len({r['id'] for r in all_rows}) != len(all_rows):
            raise ValueError('Duplicated ordinary expert cache IDs')
        if ({scene_key(r) for r in all_rows if r['replay_split'] == 'train'} &
                {scene_key(r) for r in all_rows if r['replay_split'] == 'replay_holdout'}):
            raise ValueError('Prepared train and holdout scenes overlap')
        base.update(states=all_rows, complete=True, parent_manifest=args.manifest,
                    cup_ordinary_collections=args.collections, target_tasks=args.target_tasks,
                    removed_non_target_fg_states=len(base['states'])-len(retained), hash_scans=False)
        (root/'manifest.json').write_text(json.dumps(base, indent=2)+'\n')
        print('[merged-cup-experts]', len(rows), 'states', len(keys), 'scenes', flush=True)
        return
    if not 0 <= args.shard < args.shards: ap.error('Invalid shard')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import numpy as np
    import torch
    from experiments.robotwin.eraf_fg_bridge import load_policy, capture_frozen_inputs
    from experiments.robotwin.compact_replay import capture_delta
    from experiments.robotwin.eraf_fg_training import same_observation
    policy = load_policy(args.checkpoint, base)
    policy.model.requires_grad_(False)
    policy.num_inference_steps = 1
    norm = policy.processor.normalizer.normalizers['action'][policy.processor.shape_meta['action'][0]['key']]
    shard = root/f'shard{args.shard}'
    (shard/'payloads').mkdir(parents=True, exist_ok=False)
    count = 0
    with (shard/'states.jsonl').open('x', buffering=1) as journal:
        for record in records[args.shard::args.shards]:
            arrays = {}
            for language in ('source', 'target'):
                with np.load(record[language]['frame_path'], allow_pickle=False) as data:
                    arrays[language] = {k: data[k] for k in ('actions', *CAMERAS)}
                values = arrays[language]
                if any(len(v) != record[language]['frames'] for v in values.values()):
                    raise ValueError('Ordinary expert frame/action count mismatch')
            parents = parent_path = None
            for index, windows in enumerate(paired_windows(arrays['source']['actions'], arrays['target']['actions'])):
                captured, references, valid, frame_indices = {}, {}, {}, {}
                for language, window in zip(('source', 'target'), windows, strict=True):
                    values, frame = arrays[language], window.start
                    observation = {'observation': {c: {'rgb': values[c][frame]} for c in CAMERAS},
                                   'joint_action': {'vector': values['actions'][frame]}}
                    policy.reset()
                    instruction = record['source_instruction' if language == 'source' else 'counterfactual_instruction']
                    captured[language] = capture_frozen_inputs(policy, observation, instruction)
                    if captured[language]['policy_guard_state'] is not None:
                        raise ValueError('Ordinary expert cache carried policy memory')
                    references[language] = norm.forward(torch.from_numpy(window.action).unsqueeze(0)).cpu()
                    valid[language] = torch.from_numpy(window.valid).unsqueeze(0)
                    frame_indices[language+'_frame_index'] = frame
                if index == 0 and not same_observation(captured):
                    raise ValueError('Frozen initial expert observations differ')
                ident = f'cup_ordinary_{record["task_config"]}_{record["scene_seed"]}_{index}'
                target = shard/'payloads'/(ident+'.pt')
                bodies = {language: {k: v[k] for k in ('video_inputs', 'action_inputs')}
                          for language, v in captured.items()}
                if parents is None:
                    parents, parent_path = bodies, target
                    payload = {'captured': captured, 'references': references, 'valid': valid}
                else:
                    payload = {'format': 'robotwin_eraf_fg_compact_v1', 'parent_payload': str(parent_path),
                        'capture_deltas': {k: capture_delta(bodies[k], parents[k]) for k in captured},
                        'capture_extras': {k: {name: v[name] for name in ('proprio', 'policy_guard_state')}
                                           for k, v in captured.items()},
                        'references': references, 'valid': valid}
                torch.save(payload, target)
                row = record | {'id': ident, 'payload': str(target), 'frame_index': index,
                    'initial_observations_exactly_equal': index == 0, **frame_indices,
                    'frozen_input_protocol': 'robotwin_eraf_fg_pre_dit_v1', 'policy_memory': 'none',
                    'preparation_checkpoint': args.checkpoint}
                journal.write(json.dumps(row)+'\n'); count += 1
            print('[cup-expert-cache]', args.shard, record['scene_seed'], count, flush=True)
    (shard/'complete.json').write_text(json.dumps({'complete': True, 'states': count, 'binding': binding})+'\n')


if __name__ == '__main__': main()
