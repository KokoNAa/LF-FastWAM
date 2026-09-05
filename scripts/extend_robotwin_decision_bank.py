#!/usr/bin/env python3
"""Cache unused original-training scenes for paired decision supervision.

Keep the existing replay holdout. Read each additional pair's initial images
once, require a genuinely shared observation, and cache only frozen inputs.
There are no file hashes or repeated ten-step inference checks.
"""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]


def additional_scenes(manifest, bank):
    from experiments.robotwin.no_eraf_probe import training_episode_ids
    existing = {(r['task_config'], r['pair_id'], r['episode_index']) for r in manifest['states']}
    result = []
    for domain in ('demo_clean', 'demo_randomized'):
        plan = json.loads((bank / domain / 'plan.json').read_text())
        templates = {p['pair_id']: p for p in plan['pairs']}
        for pair_id, pair in sorted(templates.items()):
            roots = {k: Path(pair[k]['hdf5']).parent.parent for k in ('native', 'counterfactual')}
            records = {k: {int(r['episode_index']): r for r in
                          map(json.loads, (path / 'meta/pgc_episodes.jsonl').read_text().splitlines())}
                       for k, path in roots.items()}
            train = training_episode_ids(len(records['native']), plan['split']['validation_proportion'])
            for episode in sorted(train):
                if (domain, pair_id, episode) in existing:
                    continue
                source, target = (records[k][episode] for k in ('native', 'counterfactual'))
                result.append({'id': f'{domain}_historical_{pair_id}_ep{episode:04d}_native_f00000',
                    'task_config': domain, 'pair_id': pair_id, 'episode_index': episode,
                    'scene_seed': source['scene_seed'], 'frame_index': 0, 'replay_split': 'train',
                    'source_instruction': target['source_instruction'],
                    'counterfactual_instruction': target['counterfactual_instruction'],
                    'native_hdf5': str(roots['native'] / source['raw_hdf5']),
                    'target_hdf5': str(roots['counterfactual'] / target['raw_hdf5'])})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--gpu', type=int, default=5)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('DIFFSYNTH_MODEL_BASE_PATH', '/root/gpufree-data/fastwam/FastWAM/checkpoints')
    import h5py
    import numpy as np
    import torch
    from PIL import Image
    from experiments.robotwin.no_eraf_probe import CAMERAS
    from experiments.robotwin.joint_adapter_repair import capture_inputs
    from scripts.train_robotwin_cf_decision_adapter import load_policy

    bank = Path(args.manifest).resolve().parent
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.output).resolve()
    (root / 'payloads').mkdir(parents=True, exist_ok=True)
    (root / 'states').mkdir(exist_ok=True)
    journal = root / 'prepared.jsonl'
    prepared = [json.loads(l) for l in journal.read_text().splitlines()] if journal.exists() else []
    done = {r['id'] for r in prepared}
    pending = [r for r in additional_scenes(manifest, bank) if r['id'] not in done]
    source = json.loads((Path(manifest['source_probe']) / 'plan.json').read_text())
    policy = load_policy(SimpleNamespace(checkpoint=source['checkpoints']['step1000'], seed=42), manifest)
    policy.num_inference_steps = 1  # Frozen first-frame inputs do not depend on action denoising steps.
    policy.model.requires_grad_(False)
    norm = policy.processor.normalizer.normalizers['action'][policy.processor.shape_meta['action'][0]['key']]

    def read_initial(path):
        with h5py.File(path, 'r') as handle:
            actions = np.asarray(handle['joint_action/vector'][:32], dtype=np.float32)
            images = {}
            for camera in CAMERAS:
                blob = bytes(handle[f'observation/{camera}/rgb'][0]).rstrip(b'\0')
                with Image.open(io.BytesIO(blob)) as im:
                    images[camera] = np.asarray(im.convert('RGB'), dtype=np.uint8)
        return actions, images

    skipped = []
    with journal.open('a', buffering=1) as log:
        for row in pending:
            source_actions, source_images = read_initial(row['native_hdf5'])
            target_actions, target_images = read_initial(row['target_hdf5'])
            if source_actions.shape != (32, 14) or target_actions.shape != (32, 14):
                skipped.append({'id': row['id'], 'reason': 'incomplete action window'})
                continue
            if not np.array_equal(source_actions[0], target_actions[0]) or any(
                    not np.array_equal(source_images[c], target_images[c]) for c in CAMERAS):
                skipped.append({'id': row['id'], 'reason': 'different initial observation'})
                continue
            obs = {'joint_action': {'vector': source_actions[0]},
                   'observation': {c: {'rgb': source_images[c]} for c in CAMERAS}}
            refs = {k: norm.forward(torch.as_tensor(v).unsqueeze(0)).cpu()
                    for k, v in (('source', source_actions), ('target', target_actions))}
            captured = {}
            for language, field in (('source', 'source_instruction'), ('target', 'counterfactual_instruction')):
                _, captured[language], _ = capture_inputs(policy.model,
                    lambda: policy._infer_action_chunk(obs, row[field]))
            payload = root / 'payloads' / (row['id'] + '.pt')
            torch.save({'captured': captured, 'references': refs}, payload)
            state = root / 'states' / (row['id'] + '.npz')
            np.savez_compressed(state, state=source_actions[0], source_reference=source_actions,
                                target_reference=target_actions, **source_images)
            row.update(payload=str(payload), file=str(state), initial_observations_exactly_equal=True,
                       dual_reference_valid=True, own_reference_in_training_split=True,
                       expert_separation_rmse=float((refs['source']-refs['target']).square().mean().sqrt()))
            prepared.append(row)
            log.write(json.dumps(row) + '\n')
            if len(prepared) % 10 == 0:
                print(f'[prepared] additional_pairs={len(prepared)}', flush=True)
    result = manifest | {'format': 'robotwin_decision_replay_bank_v2',
        'states': manifest['states'] + prepared, 'additional_training_pairs': len(prepared),
        'skipped': skipped, 'capture_note': 'Frozen VAE/T5/proprio inputs; original replay holdout preserved; no file hashes.'}
    (root / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"[complete] train={sum(r['replay_split']=='train' for r in result['states'])} "
          f"holdout={sum(r['replay_split']=='replay_holdout' for r in result['states'])} skipped={len(skipped)}", flush=True)


if __name__ == '__main__':
    main()
