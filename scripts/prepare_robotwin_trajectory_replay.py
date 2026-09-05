#!/usr/bin/env python3
"""Add branch-specific expert states after the first action chunk.

These are ordinary correct positives at each trajectory's OWN observation.
They are never treated as counterfactual labels at an identical later state.
Initial shared-state pairs keep extra sampling and conditional supervision.
"""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]


def phase_frames(source_length, target_length):
    last = min(source_length, target_length) - 32
    return sorted({min(24, last), min(72, last), round(last * .4), round(last * .75)} - {0}) if last > 0 else []


def worker(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import h5py
    import numpy as np
    import torch
    from PIL import Image
    from experiments.robotwin.no_eraf_probe import CAMERAS
    from experiments.robotwin.joint_adapter_repair import capture_inputs
    from scripts.train_robotwin_cf_decision_adapter import load_policy

    manifest = json.loads(Path(args.manifest).read_text())
    plan = json.loads((Path(args.source_bank) / args.domain / 'plan.json').read_text())
    templates = {p['pair_id']: p for p in plan['pairs']}
    raw = {}
    for pair_id, pair in templates.items():
        roots = {k: Path(pair[k]['hdf5']).parent.parent for k in ('native', 'counterfactual')}
        records = {k: {r['episode_index']: r for r in
                   map(json.loads, (path / 'meta/pgc_episodes.jsonl').read_text().splitlines())} for k, path in roots.items()}
        raw[pair_id] = roots, records
    root = Path(args.output) / args.domain
    (root / 'payloads').mkdir(parents=True, exist_ok=True)
    journal = root / 'states.jsonl'
    rows = [json.loads(l) for l in journal.read_text().splitlines()] if journal.exists() else []
    done = {r['id'] for r in rows}
    policy = load_policy(SimpleNamespace(checkpoint=args.checkpoint, seed=42), manifest)
    policy.num_inference_steps = 1
    policy.model.requires_grad_(False)
    norm = policy.processor.normalizer.normalizers['action'][policy.processor.shape_meta['action'][0]['key']]

    def capture(handle, frame, instruction):
        actions = np.asarray(handle['joint_action/vector'][frame:frame+32], dtype=np.float32)
        images = {}
        for camera in CAMERAS:
            blob = bytes(handle[f'observation/{camera}/rgb'][frame]).rstrip(b'\0')
            with Image.open(io.BytesIO(blob)) as im:
                images[camera] = np.asarray(im.convert('RGB'), dtype=np.uint8)
        obs = {'joint_action': {'vector': actions[0]}, 'observation': {c: {'rgb': images[c]} for c in CAMERAS}}
        _, captured, _ = capture_inputs(policy.model, lambda: policy._infer_action_chunk(obs, instruction))
        return captured, norm.forward(torch.as_tensor(actions).unsqueeze(0)).cpu()

    scenes = [r for r in manifest['states'] if r['replay_split'] == 'train' and r['task_config'] == args.domain]
    with journal.open('a', buffering=1) as log:
        for scene in scenes:
            roots, records = raw[scene['pair_id']]
            ep = scene['episode_index']
            paths = {k: roots[k] / records[k][ep]['raw_hdf5'] for k in roots}
            with h5py.File(paths['native'], 'r') as source, h5py.File(paths['counterfactual'], 'r') as target:
                frames = phase_frames(len(source['joint_action/vector']), len(target['joint_action/vector']))
                for frame in frames:
                    sid = scene['id'] + f'_trajectory_f{frame:05d}'
                    if sid in done:
                        continue
                    captured, refs = {}, {}
                    for language, handle, field in (('source', source, 'source_instruction'),
                                                     ('target', target, 'counterfactual_instruction')):
                        captured[language], refs[language] = capture(handle, frame, scene[field])
                    payload = root / 'payloads' / (sid + '.pt')
                    torch.save({'captured': captured, 'references': refs}, payload)
                    row = {k: scene[k] for k in ('pair_id', 'task_config', 'episode_index', 'scene_seed',
                                                 'source_instruction', 'counterfactual_instruction')}
                    row.update(id=sid, frame_index=frame, replay_split='train', payload=str(payload),
                               initial_observations_exactly_equal=False, sampling_weight=1,
                               supervision='Two own-state expert positives; no same-state conditional-difference weighting.')
                    log.write(json.dumps(row) + '\n')
                    rows.append(row)
                    if len(rows) % 25 == 0:
                        print(f'[prepared] {args.domain} trajectory_pairs={len(rows)}', flush=True)
    print(f'[complete] {args.domain} trajectory_pairs={len(rows)}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['run', 'worker'])
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--source-bank', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--gpus', type=int, nargs=2, default=[1, 3])
    ap.add_argument('--gpu', type=int)
    ap.add_argument('--domain', choices=['demo_clean', 'demo_randomized'])
    args = ap.parse_args()
    os.environ.setdefault('DIFFSYNTH_MODEL_BASE_PATH', '/root/gpufree-data/fastwam/FastWAM/checkpoints')
    if args.mode == 'worker':
        return worker(args)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'plan.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    children = []
    for domain, gpu in zip(('demo_clean', 'demo_randomized'), args.gpus):
        log = (root / (domain + '.log')).open('a')
        command = [sys.executable, '-u', __file__, 'worker', '--manifest', args.manifest,
                   '--source-bank', args.source_bank, '--checkpoint', args.checkpoint,
                   '--output', str(root), '--gpu', str(gpu), '--domain', domain]
        children.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
    statuses = []
    for child, log in children:
        statuses.append(child.wait())
        log.close()
    if any(statuses):
        raise RuntimeError('Trajectory cache worker failed; inspect its domain log.')
    manifest = json.loads(Path(args.manifest).read_text())
    initials = [r | {'sampling_weight': 4} for r in manifest['states']]
    phases = [json.loads(l) for d in ('demo_clean', 'demo_randomized')
              for l in (root / d / 'states.jsonl').read_text().splitlines()]
    result = manifest | {'format': 'robotwin_decision_trajectory_replay_v1', 'states': initials + phases,
        'trajectory_pairs': len(phases), 'initial_sampling_weight': 4,
        'scope': 'Preserve original replay holdout; later positives each use their own observation and plain endpoint MSE.'}
    (root / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'[complete] initial_pairs={len(initials)} trajectory_pairs={len(phases)}', flush=True)


if __name__ == '__main__':
    main()
