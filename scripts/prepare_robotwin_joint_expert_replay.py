#!/usr/bin/env python3
"""Prepare scene-isolated joint-expert action and ERAF grounding supervision."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['plan', 'worker', 'merge'])
    for key in ('manifest', 'checkpoint', 'output'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--collections', type=Path, nargs='+', required=True)
    ap.add_argument('--train-per-task', type=int, default=12)
    ap.add_argument('--holdout-per-task', type=int, default=4)
    ap.add_argument('--stride', type=int, default=24)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    args = ap.parse_args()
    if args.stride < 1 or not 0 <= args.shard < args.shards:
        ap.error('Positive stride and valid shard required.')
    from experiments.robotwin.joint_expert_replay import read, collect_scenes, masked_window, scene_keys, validate_prepared_rows
    from experiments.robotwin.eraf_fg_data import file_metadata
    base = read(args.manifest)
    if not base.get('complete'):
        raise ValueError('Parent bank is incomplete.')
    scenes, metadata = collect_scenes(args.collections, base['states'],
        train_per_task=args.train_per_task, holdout_per_task=args.holdout_per_task)
    contract = dict(format='robotwin_joint_expert_replay_preparation_v1',
        parent_manifest=file_metadata(args.manifest), checkpoint=file_metadata(args.checkpoint),
        input_metadata=metadata, train_per_task=args.train_per_task, holdout_per_task=args.holdout_per_task,
        stride=args.stride, shards=args.shards,
        scene_split={f"{s['task_config']}/{s['pair_id']}/{s['scene_seed']}": s['replay_split'] for s in scenes},
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        semantics='Independent own-state expert branches; extra conditional difference only at identical initial observations. Full-goal corrections are not present in new data.')
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    plan_path = root / 'plan.json'
    if plan_path.exists():
        if read(plan_path) != contract:
            raise ValueError('Preparation inputs or code changed after the plan was frozen.')
    else:
        if args.mode != 'plan':
            raise ValueError('Create and verify the preparation plan before workers or merge.')
        plan_path.write_text(json.dumps(contract, indent=2) + '\n')
    if args.mode == 'plan':
        print(json.dumps(dict(scenes=len(scenes), tasks=sorted({s['source_task'] for s in scenes}),
                              contract=str(plan_path)), indent=2)); return
    if args.mode == 'merge':
        added = []
        for shard in range(args.shards):
            folder = root / f'shard{shard}'
            report = read(folder / 'complete.json')
            if not report['complete'] or report['contract'] != contract or report['shard'] != shard:
                raise ValueError('Incomplete cache shard or changed contract.')
            rows = [json.loads(line) for line in (folder / 'states.jsonl').read_text().splitlines()]
            if len(rows) != report['states'] or any(not Path(r['payload']).is_file() for r in rows):
                raise ValueError('Missing prepared expert payloads.')
            validate_prepared_rows(rows, scenes[shard::args.shards])
            added += rows
        all_rows = base['states'] + added
        if len({r['id'] for r in all_rows}) != len(all_rows):
            raise ValueError('Duplicate replay IDs.')
        if scene_keys(r for r in all_rows if r['replay_split'] == 'train') & scene_keys(
                r for r in all_rows if r['replay_split'] == 'replay_holdout'):
            raise ValueError('Training and holdout scenes overlap.')
        base = dict(base)
        if 'formal_data_audit' in base:
            base['parent_formal_data_audit'] = base.pop('formal_data_audit')
        base.update(states=all_rows, joint_expert_expansion=contract, added_joint_expert_states=len(added),
                    parent_manifest=str(args.manifest.resolve()), complete=True)
        target = root / 'manifest.json'
        with target.open('x') as handle:
            json.dump(base, handle, indent=2)
        print(f'[merged] scenes={len(scenes)} added_states={len(added)}'); return
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import h5py
    import numpy as np
    import torch
    from experiments.robotwin.eraf_fg_bridge import load_policy, capture_frozen_inputs
    from experiments.robotwin.eraf_fg_contract import CAMERAS
    from experiments.robotwin.compact_replay import capture_delta
    from experiments.robotwin.image_io import decode_legacy_robotwin_rgb
    from experiments.robotwin.pgc_data import array_sha256, validate_action_array
    from scripts.prepare_robotwin_trajectory_replay import dense_phase_frames
    policy = load_policy(args.checkpoint, base)
    policy.model.requires_grad_(False); policy.num_inference_steps = 1
    norm = policy.processor.normalizer.normalizers['action'][policy.processor.shape_meta['action'][0]['key']]
    folder = root / f'shard{args.shard}'; (folder / 'payloads').mkdir(parents=True, exist_ok=False)
    count = 0
    with (folder / 'states.jsonl').open('x', buffering=1) as log:
        for scene in scenes[args.shard::args.shards]:
            with h5py.File(scene['raw_paths']['native']) as source, h5py.File(scene['raw_paths']['counterfactual']) as target:
                actions = {k: validate_action_array(h['joint_action/vector'][:])
                           for k, h in [('source', source), ('target', target)]}
                for language, kind in [('source', 'native'), ('target', 'counterfactual')]:
                    row = scene['raw_records'][kind]
                    if len(actions[language]) != row['action_count'] or array_sha256(actions[language]) != row['action_sha256']:
                        raise ValueError('Expert actions differ from the replay-verified record.')
                frames = [(0, 0)] + dense_phase_frames(len(actions['source']), len(actions['target']), args.stride)
                parent = parent_path = None
                for source_frame, target_frame in frames:
                    captured, references, valid, observations = {}, {}, {}, {}
                    for language, handle, frame, field in [('source', source, source_frame, 'source_instruction'),
                            ('target', target, target_frame, 'counterfactual_instruction')]:
                        raw, mask = masked_window(actions[language], frame)
                        rgb = {c: decode_legacy_robotwin_rgb(handle[f'observation/{c}/rgb'][frame]) for c in CAMERAS}
                        observations[language] = rgb
                        observation = dict(joint_action={'vector': raw[0]}, observation={c: {'rgb': image} for c, image in rgb.items()})
                        policy.reset()
                        captured[language] = capture_frozen_inputs(policy, observation, scene[field])
                        if captured[language]['policy_guard_state'] is not None:
                            raise ValueError('Privileged memory entered an expert replay input.')
                        references[language] = norm.forward(torch.from_numpy(raw).unsqueeze(0)).cpu()
                        valid[language] = torch.from_numpy(mask).unsqueeze(0)
                    equal = source_frame == target_frame == 0 and np.array_equal(actions['source'][0], actions['target'][0]) and all(
                        np.array_equal(observations['source'][c], observations['target'][c]) for c in CAMERAS)
                    equal = bool(equal and all(bool(mask.all()) for mask in valid.values()))
                    ident = f"joint_expert_{scene['source_task']}_{scene['scene_seed']}_s{source_frame}_t{target_frame}"
                    path = folder / 'payloads' / (ident + '.pt')
                    body = {k: {g: v[g] for g in ('video_inputs', 'action_inputs')} for k, v in captured.items()}
                    if parent is None:
                        parent, parent_path = body, path
                        payload = dict(captured=captured, references=references, valid=valid)
                    else:
                        payload = dict(format='robotwin_eraf_fg_compact_v1', parent_payload=str(parent_path),
                            references=references, valid=valid,
                            capture_deltas={k: capture_delta(v, parent[k]) for k, v in body.items()},
                            capture_extras={k: {g: v[g] for g in ('proprio', 'policy_guard_state')} for k, v in captured.items()})
                    torch.save(payload, path)
                    row = {k: v for k, v in scene.items() if k != 'raw_records'}
                    row.update(id=ident, payload=str(path), frame_index=source_frame,
                        source_frame_index=source_frame, target_frame_index=target_frame,
                        initial_observations_exactly_equal=equal, frozen_input_protocol='robotwin_eraf_fg_pre_dit_v1',
                        preparation_checkpoint=str(args.checkpoint.resolve()), policy_memory='none',
                        reference_valid_actions={k: int(v.sum()) for k, v in valid.items()})
                    log.write(json.dumps(row) + '\n'); count += 1
                print(f"[prepared] {scene['source_task']} seed={scene['scene_seed']} split={scene['replay_split']} windows={len(frames)}", flush=True)
    if any(file_metadata(item['path']) != item for item in metadata + [contract['parent_manifest'], contract['checkpoint']]):
        raise ValueError('Preparation inputs changed while caching.')
    (folder / 'complete.json').write_text(json.dumps(dict(complete=True, shard=args.shard,
        states=count, contract=contract), indent=2) + '\n')


if __name__ == '__main__':
    main()
