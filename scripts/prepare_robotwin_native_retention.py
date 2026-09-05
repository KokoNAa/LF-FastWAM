#!/usr/bin/env python3
"""Capture train-only native policy states for source velocity distillation."""
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


def select_episodes(episodes, val_proportion, split_seed, excluded_seeds):
    from experiments.robotwin.no_eraf_probe import training_episode_ids
    training = training_episode_ids(len(episodes), val_proportion, split_seed)
    def scene(row):
        return row['source_task'], row['task_config'], row['scene_seed']
    heldout_scenes = {scene(r) for r in episodes if r['episode_index'] not in training}
    return [r for r in episodes if r['episode_index'] in training
            and scene(r) not in heldout_scenes and r['scene_seed'] not in excluded_seeds]


def worker(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('DIFFSYNTH_MODEL_BASE_PATH', '/root/gpufree-data/fastwam/FastWAM/checkpoints')
    import numpy as np
    import torch
    from experiments.robotwin.joint_adapter_repair import capture_inputs
    from scripts.train_robotwin_cf_decision_adapter import load_policy
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.output)
    plan = json.loads((root / 'plan.json').read_text())
    selected = plan['selected_episodes'][args.shard::args.shards]
    captures = {}
    for path in sorted((Path(args.native_root) / 'captures').rglob('*.json')):
        record = json.loads(path.read_text())
        if record.get('format') == 'pgc_robotwin_closed_loop_native_capture_v2':
            captures[record['capture_id']] = (path, record)
    policy = load_policy(SimpleNamespace(checkpoint=args.checkpoint, seed=42), manifest)
    policy.model.requires_grad_(False)
    normalizer = policy.processor.normalizer.normalizers['action'][policy.processor.shape_meta['action'][0]['key']]
    shard = root / f'shard{args.shard}'
    (shard / 'payloads').mkdir(parents=True, exist_ok=False)
    count = 0
    with (shard / 'states.jsonl').open('x', buffering=1) as journal:
        for episode in selected:
            path, record = captures[episode['capture_id']]
            assert record['rollout_policy'] == 'immutable_released_base'
            assert record['condition'] == 'correct' and record['instruction_goal'] == 'source'
            assert record['policy_instruction'] == record['source_instruction']
            assert record['scene_seed'] == episode['scene_seed']
            capture_path = path.parent / record['capture_file']
            with np.load(capture_path, allow_pickle=False) as arrays:
                for frame in (0, 4):
                    # Native NPZ is already camera RGB; only legacy JPEG needs a channel swap.
                    observation = {'observation': {camera + '_camera': {'rgb': arrays[key][frame]}
                        for camera, key in [('head', 'head_rgb'), ('left', 'left_rgb'), ('right', 'right_rgb')]},
                        'joint_action': {'vector': arrays['state'][frame]}}
                    actions, captured, _ = capture_inputs(policy.model, lambda: policy._infer_action_chunk(
                        observation, record['source_instruction']))
                    actions = np.asarray(actions, dtype=np.float32)
                    if actions.shape != (32, 14) or not np.isfinite(actions).all():
                        raise ValueError('Native teacher must generate an actual full action horizon.')
                    reference = normalizer.forward(torch.from_numpy(actions).unsqueeze(0)).cpu()
                    ident = f'native_{episode["capture_id"][:20]}_frame{frame}'
                    payload = shard / 'payloads' / (ident + '.pt')
                    torch.save({'captured': {'source': captured}, 'references': {'source': reference}}, payload)
                    row = {**episode, 'id': ident, 'pair_id': record['pair_id'],
                        'source_instruction': record['source_instruction'], 'frame_index': frame,
                        'native_retention': True, 'replay_split': 'train', 'payload': str(payload),
                        'image_color_space': 'RGB', 'capture_path': str(capture_path),
                        'reference_kind': 'teacher_sampled_32_actions', 'teacher_checkpoint': args.checkpoint}
                    journal.write(json.dumps(row) + '\n')
                    count += 1
            if count % 20 == 0:
                print(f'[native] shard={args.shard} states={count}/{len(selected)*2}', flush=True)
    print(f'[complete] shard={args.shard} states={count}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'worker'])
    for key in ('manifest', 'native-root', 'checkpoint', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=list(range(6)))
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--shards', type=int)
    parser.add_argument('--excluded-seeds', nargs='+', type=int, default=[4300000, 4300001, 4300002])
    args = parser.parse_args()
    if args.mode == 'worker':
        return worker(args)
    import yaml
    manifest = json.loads(Path(args.manifest).read_text())
    config = yaml.safe_load(Path(manifest['original_train_config']).read_text())['data']['train']
    episodes = [json.loads(line) for line in (
        Path(args.native_root) / 'lerobot/meta/pgc_episodes.jsonl').read_text().splitlines()]
    selected = select_episodes(episodes, float(config['val_set_proportion']), int(config.get('seed', 42)),
                               set(args.excluded_seeds))
    expected = {(r['pair_id'].split('::')[0], r['task_config']) for r in episodes}
    assert expected == {(r['pair_id'].split('::')[0], r['task_config']) for r in selected}
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = vars(args) | {'selected_episodes': selected, 'total_original_episodes': len(episodes),
        'excluded_episodes': len(episodes) - len(selected),
        'split': 'Original training episodes only; all scenes intersecting original holdout excluded.',
        'teacher_target': 'Full 32 actions freshly sampled; never padded from nine-frame captures.'}
    (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'[plan] selected={len(selected)}/{len(episodes)} native segments', flush=True)
    children = []
    for shard, gpu in enumerate(args.gpus):
        command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--manifest', args.manifest,
            '--native-root', args.native_root, '--checkpoint', args.checkpoint, '--output', str(root),
            '--gpu', str(gpu), '--shard', str(shard), '--shards', str(len(args.gpus))]
        children.append(subprocess.Popen(command))
    failures = [child.wait() for child in children]
    if any(failures):
        raise RuntimeError(f'Native replay incomplete: worker exits={failures}')
    native = [json.loads(line) for shard in range(len(args.gpus))
              for line in (root / f'shard{shard}/states.jsonl').read_text().splitlines()]
    assert len(native) == 2 * len(selected) and len({r['id'] for r in native}) == len(native)
    assert not set(r['id'] for r in native) & set(r['id'] for r in manifest['states'])
    manifest.update(states=manifest['states'] + native, native_retention_states=len(native),
        native_retention_teacher=args.checkpoint, parent_manifest=args.manifest, complete=True)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (root / 'complete.json').write_text(json.dumps({'complete': True, 'native_states': len(native),
        'task_domains': dict(Counter(r['pair_id'] + '/' + r['task_config'] for r in native))}, indent=2) + '\n')
    print(f'[complete] native states={len(native)} paired states={len(manifest["states"])-len(native)}', flush=True)


if __name__ == '__main__':
    main()
