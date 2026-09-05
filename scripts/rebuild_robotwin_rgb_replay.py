#!/usr/bin/env python3
"""Re-encode legacy HDF5 images in deployment RGB; preserve labels and splits."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from collections import OrderedDict

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]


def corrected_payload(payload, original_path, latents):
    """Override only visual latents, retaining every state and action label."""
    if payload.get('format') == 'robotwin_compact_replay_v1':
        result = copy.deepcopy(payload)
    else:
        result = {'format': 'robotwin_compact_replay_v1', 'parent_payload': str(original_path),
                  'references': payload['references'], 'capture_deltas': {'source': {}, 'target': {}}}
    for language in ('source', 'target'):
        result['capture_deltas'].setdefault(language, {}).setdefault('video_inputs', {})['x'] = latents[language]
    result['image_color_space'] = 'RGB'
    return result


def worker(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import h5py
    import torch
    from experiments.robotwin.image_io import decode_legacy_robotwin_rgb
    from scripts.train_robotwin_cf_decision_adapter import load_policy
    manifest = json.loads(Path(args.manifest).read_text())
    paths = {}
    for domain in {r['task_config'] for r in manifest['states']}:
        plan = json.loads((Path(args.source_bank) / domain / 'plan.json').read_text())
        for pair in plan['pairs']:
            for language, kind in (('source', 'native'), ('target', 'counterfactual')):
                raw_root = Path(pair[kind]['hdf5']).parent.parent
                for record in map(json.loads, (raw_root / 'meta/pgc_episodes.jsonl').read_text().splitlines()):
                    paths[domain, pair['pair_id'], int(record['episode_index']), language] = raw_root / record['raw_hdf5']
    policy = load_policy(SimpleNamespace(checkpoint=args.checkpoint, seed=42), manifest)
    policy.model.requires_grad_(False)
    policy.num_inference_steps = 1
    rows = manifest['states'][args.shard::args.shards]
    root = Path(args.output) / f'shard{args.shard}'
    (root / 'payloads').mkdir(parents=True, exist_ok=False)
    handles = OrderedDict()
    cameras = ('head_camera', 'left_camera', 'right_camera')
    with (root / 'states.jsonl').open('x', buffering=1) as journal:
        for index, row in enumerate(rows):
            payload = torch.load(row['payload'], map_location='cpu', weights_only=True)
            latents = {}
            for language in ('source', 'target'):
                path = paths[row['task_config'], row['pair_id'], int(row['episode_index']), language]
                if path not in handles:
                    handles[path] = h5py.File(path, 'r')
                    if len(handles) > 8:
                        _, stale = handles.popitem(last=False)
                        stale.close()
                handles.move_to_end(path)
                handle = handles[path]
                frame = int(row.get(language + '_frame_index', row.get('frame_index', 0)))
                observation = {'observation': {camera: {'rgb': decode_legacy_robotwin_rgb(
                    handle[f'observation/{camera}/rgb'][frame])} for camera in cameras},
                    'joint_action': {'vector': handle['joint_action/vector'][frame]}}
                with torch.no_grad():
                    image = policy._build_robotwin_image_tensor(observation)
                    latents[language] = policy.model._encode_input_image_latents_tensor(image, tiled=False).cpu()
                if index == 0 and language == 'source':
                    from experiments.robotwin.joint_adapter_repair import capture_inputs
                    _, captured, _ = capture_inputs(policy.model, lambda: policy._infer_action_chunk(
                        observation, row['source_instruction']))
                    torch.testing.assert_close(captured['video_inputs']['x'], latents[language], rtol=0, atol=0)
                    print(f'[verified] shard={args.shard} RGB VAE matches production capture', flush=True)
            output_path = root / 'payloads' / (row['id'] + '.pt')
            torch.save(corrected_payload(payload, row['payload'], latents), output_path)
            clean = {k: v for k, v in row.items() if k not in (
                'file', 'sha256', 'observation_sha256', 'payload_sha256', 'production_replay_errors')}
            clean.update(payload=str(output_path), prior_payload=row['payload'], image_color_space='RGB')
            journal.write(json.dumps(clean) + '\n')
            if (index + 1) % 50 == 0:
                print(f'[rgb] shard={args.shard} pairs={index + 1}/{len(rows)}', flush=True)
    for handle in handles.values():
        handle.close()
    (root / 'complete.json').write_text(json.dumps({'complete': True, 'pairs': len(rows)}) + '\n')
    print(f'[complete] shard={args.shard} pairs={len(rows)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'worker'])
    for key in ('manifest', 'source-bank', 'checkpoint', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=list(range(6)))
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--shards', type=int)
    args = parser.parse_args()
    if args.mode == 'worker':
        return worker(args)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'plan.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    children = []
    for index, gpu in enumerate(args.gpus):
        command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--manifest', args.manifest,
                   '--source-bank', args.source_bank, '--checkpoint', args.checkpoint,
                   '--output', str(root), '--gpu', str(gpu), '--shard', str(index), '--shards', str(len(args.gpus))]
        children.append(subprocess.Popen(command))
    failures = [child.wait() for child in children]
    if any(failures):
        raise RuntimeError(f'RGB cache rebuild incomplete: worker exits={failures}')
    rows = {}
    for index in range(len(args.gpus)):
        for line in (root / f'shard{index}/states.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['id'] in rows:
                raise ValueError('Duplicate replay row.')
            rows[row['id']] = row
    manifest = json.loads(Path(args.manifest).read_text())
    assert set(rows) == {r['id'] for r in manifest['states']}
    manifest.update(states=[rows[r['id']] for r in manifest['states']], complete=True,
                    image_color_space='RGB', parent_manifest=args.manifest,
                    rgb_rebuild='Legacy OpenCV JPEG decoded to camera RGB; only frozen visual latents changed.')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (root / 'complete.json').write_text(json.dumps({'complete': True, 'pairs': len(rows)}) + '\n')
    print(f'[complete] RGB pairs={len(rows)}', flush=True)


if __name__ == '__main__':
    main()
