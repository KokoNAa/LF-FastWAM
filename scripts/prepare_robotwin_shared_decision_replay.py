#!/usr/bin/env python3
"""Cache goal-verified expert pairs at identical post-grasp observations."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]
CAMERAS = ('head_camera', 'left_camera', 'right_camera')


def shared_decision_frames(a, b, images_equal, horizon=32, maximum=4, stride=4):
    import numpy as np
    n = min(len(a), len(b))
    different = np.flatnonzero(np.any(a[:n] != b[:n], axis=1))
    end = min(int(different[0]) - 1 if len(different) else n - 1, n - horizon)
    selected = []
    for frame in range(end, max(horizon - 1, end - horizon), -1):
        if selected and selected[-1]['frame'] - frame < stride:
            continue
        separation = float(np.mean((a[frame:frame+horizon] - b[frame:frame+horizon]) ** 2) ** .5)
        if separation <= .02 or not images_equal(frame):
            continue
        selected.append({'frame': frame, 'joint_action_rmse_32': separation})
        if len(selected) == maximum:
            break
    return list(reversed(selected))


def plan_states(collection):
    import h5py
    from experiments.robotwin.no_eraf_probe import require_pair
    from experiments.robotwin.decision_language_replay import bound_spatial_instruction_pairs
    states, skipped = [], []
    for domain in ('demo_clean', 'demo_randomized'):
        for pair_root in sorted((Path(collection) / domain / 'raw').iterdir()):
            records = {kind: [json.loads(line) for line in (pair_root / kind / 'meta/pgc_episodes.jsonl').read_text().splitlines()]
                       for kind in ('native', 'counterfactual')}
            assert len(records['native']) == len(records['counterfactual']) == 12
            for native, target in zip(records['native'], records['counterfactual'], strict=True):
                require_pair(native, target)
                assert native['episode_index'] == target['episode_index']
                paths = {kind: str(pair_root / kind / record['raw_hdf5'])
                         for kind, record in [('native', native), ('counterfactual', target)]}
                with h5py.File(paths['native'], 'r') as a, h5py.File(paths['counterfactual'], 'r') as b:
                    frames = shared_decision_frames(a['joint_action/vector'][:], b['joint_action/vector'][:],
                        lambda f: all(bytes(a[f'observation/{camera}/rgb'][f]) == bytes(b[f'observation/{camera}/rgb'][f])
                                      for camera in CAMERAS))
                if not frames:
                    skipped.append({'pair_id': native['pair_id'], 'task_config': domain, 'episode_index': native['episode_index']})
                    continue
                variants = (bound_spatial_instruction_pairs(REPO, native['pair_id'], native['scene_info']['info'])
                            if native['source_task'].startswith('place_a2b_') else [])
                source_instruction = variants[0]['source'] if variants else native['source_instruction']
                target_instruction = variants[0]['target'] if variants else native['counterfactual_instruction']
                for selected in frames:
                    frame = selected['frame']
                    ident = f'shared_grasp_{domain}_{native["pair_id"]}_ep{native["episode_index"]:04d}_f{frame:05d}'
                    states.append({'id': ident, 'pair_id': native['pair_id'], 'source_task': native['source_task'],
                        'task_config': domain, 'scene_seed': native['scene_seed'], 'episode_index': native['episode_index'],
                        'source_instruction': source_instruction, 'counterfactual_instruction': target_instruction,
                        'source_frame_index': frame, 'target_frame_index': frame, 'raw_paths': paths,
                        'shared_grasp_decision': True, 'validation_phase': 'post_grasp',
                        'initial_observations_exactly_equal': True, 'image_color_space': 'RGB',
                        'replay_split': 'replay_holdout' if native['episode_index'] == 11 else 'train',
                        'sampling_weight': 16, 'joint_action_rmse_32': selected['joint_action_rmse_32'],
                        'language_replay_key': f'shared_grasp_{domain}_{native["pair_id"]}_ep{native["episode_index"]:04d}',
                        'seen_instruction_pairs': variants})
    expected = {(task, domain) for task in ('place_a2b_left', 'place_a2b_right', 'place_burger_fries')
                for domain in ('demo_clean', 'demo_randomized')}
    for split in ('train', 'replay_holdout'):
        if {(r['source_task'], r['task_config']) for r in states if r['replay_split'] == split} != expected:
            raise ValueError(f'Missing shared decision coverage for {split}; skipped={skipped}')
    return states, skipped


def worker(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import h5py
    import numpy as np
    import torch
    from experiments.robotwin.image_io import decode_legacy_robotwin_rgb
    from experiments.robotwin.joint_adapter_repair import capture_inputs
    from scripts.train_robotwin_cf_decision_adapter import load_policy
    root = Path(args.output)
    manifest = json.loads(Path(args.manifest).read_text())
    rows = json.loads((root / 'plan.json').read_text())['states'][args.shard::args.shards]
    policy = load_policy(SimpleNamespace(checkpoint=args.checkpoint, seed=42), manifest)
    policy.model.requires_grad_(False)
    policy.num_inference_steps = 1
    normalizer = policy.processor.normalizer.normalizers['action'][policy.processor.shape_meta['action'][0]['key']]
    shard = root / f'shard{args.shard}'
    (shard / 'payloads').mkdir(parents=True, exist_ok=False)
    with (shard / 'states.jsonl').open('x', buffering=1) as journal:
        for index, row in enumerate(rows):
            frame = row['source_frame_index']
            with h5py.File(row['raw_paths']['native'], 'r') as a, h5py.File(row['raw_paths']['counterfactual'], 'r') as b:
                qa, qb = a['joint_action/vector'][frame:frame+32], b['joint_action/vector'][frame:frame+32]
                assert qa.shape == qb.shape == (32, 14) and np.array_equal(qa[0], qb[0])
                images = {}
                for camera in CAMERAS:
                    x, y = a[f'observation/{camera}/rgb'][frame], b[f'observation/{camera}/rgb'][frame]
                    assert bytes(x) == bytes(y), 'Shared RGB changed after planning.'
                    images[camera] = {'rgb': decode_legacy_robotwin_rgb(x)}
                observation = {'observation': images, 'joint_action': {'vector': qa[0]}}
            captured = {}
            for language, field in [('source', 'source_instruction'), ('target', 'counterfactual_instruction')]:
                _, captured[language], _ = capture_inputs(policy.model,
                    lambda: policy._infer_action_chunk(observation, row[field]))
            torch.testing.assert_close(captured['source']['video_inputs']['x'], captured['target']['video_inputs']['x'], rtol=0, atol=0)
            refs = {key: normalizer.forward(torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)).cpu()
                    for key, value in [('source', qa), ('target', qb)]}
            path = shard / 'payloads' / (row['id'] + '.pt')
            torch.save({'captured': captured, 'references': refs}, path)
            journal.write(json.dumps({**row, 'payload': str(path)}) + '\n')
            if (index+1) % 10 == 0:
                print(f'[shared] shard={args.shard} states={index+1}/{len(rows)}', flush=True)
    print(f'[complete] shard={args.shard} states={len(rows)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'worker'])
    for key in ('manifest', 'collection', 'checkpoint', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=list(range(6)))
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--shards', type=int)
    args = parser.parse_args()
    if args.mode == 'worker':
        return worker(args)
    for path in Path(args.collection).glob('*-plan.json'):
        plan = json.loads(path.read_text())
        assert json.loads(Path(plan['exit_file']).read_text())['exit_code'] == 0
    states, skipped = plan_states(args.collection)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'plan.json').write_text(json.dumps(vars(args) | {'states': states, 'skipped': skipped}, indent=2) + '\n')
    print(f'[plan] exact shared decisions={len(states)} skipped_scenes={len(skipped)}', flush=True)
    children = []
    for shard, gpu in enumerate(args.gpus):
        children.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker',
            '--manifest', args.manifest, '--collection', args.collection, '--checkpoint', args.checkpoint,
            '--output', str(root), '--gpu', str(gpu), '--shard', str(shard), '--shards', str(len(args.gpus))]))
    exits = [child.wait() for child in children]
    if any(exits):
        raise RuntimeError(f'Shared decision cache incomplete: {exits}')
    added = [json.loads(line) for shard in range(len(args.gpus))
             for line in (root / f'shard{shard}/states.jsonl').read_text().splitlines()]
    assert len(added) == len(states) and len({r['id'] for r in added}) == len(added)
    manifest = json.loads(Path(args.manifest).read_text())
    assert not {r['id'] for r in added} & {r['id'] for r in manifest['states']}
    manifest.update(states=manifest['states'] + added, parent_manifest=args.manifest, complete=True,
                    shared_grasp_decision_states=len(added))
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (root / 'complete.json').write_text(json.dumps({'complete': True, 'states': len(added),
        'splits': dict(Counter(r['replay_split'] for r in added))}, indent=2) + '\n')
    print(f'[complete] shared decisions={len(added)}', flush=True)


if __name__ == '__main__':
    main()
