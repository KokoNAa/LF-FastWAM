#!/usr/bin/env python3
"""Read-only action-fit and deployed language-response diagnostic after expanded FG."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
TASKS = ('place_empty_cup', 'move_pillbottle_pad')
SUPPORTED_TASKS = (*TASKS, 'blocks_ranking_rgb', 'place_a2b_left')


def select_rows(rows, tasks=TASKS):
    """One train and two holdout scenes per task/kind; selection uses no outputs."""
    from experiments.robotwin.pgc_data import ROBOTWIN_TEN_TASK_SPECS
    names = {s.pair_id: s.source_task for s in ROBOTWIN_TEN_TASK_SPECS}
    if len(tasks) != 2 or len(set(tasks)) != 2 or not set(tasks) <= set(SUPPORTED_TASKS):
        raise ValueError('Select exactly two distinct supported tasks before inference.')
    buckets = defaultdict(lambda: defaultdict(list))
    splits = defaultdict(set)
    for row in rows:
        task = row.get('source_task') or names.get(row['pair_id'])
        if task not in tasks or row['task_config'] != 'demo_clean':
            continue
        split = row['replay_split']
        splits[task, row['scene_seed']].add(split)
        if row.get('native_retention') or row.get('cf_retention'):
            continue
        kind = 'fg' if row.get('fg_correction') else 'ordinary'
        frame = int(row.get('target_frame_index', row.get('frame_index', 0)))
        if kind == 'ordinary' and frame != 0:
            continue
        buckets[task, split, kind][row['scene_seed']].append(row)
    if any(len(s) != 1 for s in splits.values()):
        raise ValueError('A diagnostic scene crosses train/holdout splits.')
    result = []
    for task in tasks:
        for split, count in [('train', 1), ('replay_holdout', 2)]:
            for kind in ('ordinary', 'fg'):
                scenes = buckets[task, split, kind]
                if len(scenes) < count:
                    raise ValueError(f'Insufficient scenes: {task}/{split}/{kind}')
                for seed in sorted(scenes)[:count]:
                    candidates = sorted(scenes[seed], key=lambda r: (int(r.get('target_frame_index', r.get('frame_index', 0))), r['id']))
                    if int(candidates[0].get('target_frame_index', candidates[0].get('frame_index', 0))) != 0:
                        raise ValueError('Missing failure-start or initial observation.')
                    picks = [candidates[0]]
                    if kind == 'fg':
                        if len(candidates) < 2:
                            raise ValueError('FG diagnostic requires a later correction observation.')
                        picks.append(candidates[len(candidates)//2])
                    for row in picks:
                        result.append(dict(row, diagnostic_task=task, diagnostic_kind=kind))
    if len(result) != 18 or len({r['id'] for r in result}) != 18:
        raise ValueError('Expected 18 distinct diagnostic observations.')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'source-bank', 'checkpoint', 'output'):
        ap.add_argument('--' + key, required=True)
    ap.add_argument('--eraf', choices=['on', 'off'], required=True)
    ap.add_argument('--tasks', nargs=2, choices=SUPPORTED_TASKS, default=TASKS)
    args = ap.parse_args()
    import h5py
    import numpy as np
    import torch
    from experiments.robotwin.eraf_fg_bridge import load_policy, file_sha256, capture_frozen_inputs, predict, masked_mse
    from experiments.robotwin.eraf_fg_contract import CAMERAS
    from experiments.robotwin.eraf_fg_data import RawReplay
    from experiments.robotwin.compact_replay import ReplayPayloads
    from experiments.robotwin.image_io import decode_legacy_robotwin_rgb
    from experiments.robotwin.pgc_data import array_sha256
    from experiments.robotwin.same_state_repair import move_cache, noise_tensor

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    manifest_hash = file_sha256(args.manifest)
    checkpoint_hash = file_sha256(args.checkpoint)
    manifest = json.loads(Path(args.manifest).read_text())
    if not manifest['complete']:
        raise ValueError('Incomplete manifest.')
    rows = select_rows(manifest['states'], args.tasks)
    protocol = dict(checkpoint=args.checkpoint, checkpoint_sha256=checkpoint_hash,
                    manifest_sha256=manifest_hash, eraf=args.eraf, optimizer_updates=0,
                    selected_rows=rows, tasks=list(args.tasks), inference_steps=10, noise_seed=42,
                    scope='Small deterministic train/holdout action diagnostic, not CF success or independent test. Source-language outputs are compared to the CF expert only as a language-response diagnostic. Oracle-noised fits are not goal-selection tests.')
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    policy = load_policy(args.checkpoint, manifest, seed=42)
    model = policy.model
    model.requires_grad_(False); model.eval()
    model.policy_guard_enabled = args.eraf == 'on'
    norm = policy.processor.normalizer.normalizers['action'][policy.processor.shape_meta['action'][0]['key']]
    payloads = ReplayPayloads(rows, model.device)
    raw = RawReplay(args.source_bank)
    records = []
    with (root/'records.jsonl').open('x', buffering=1) as journal:
        for row in rows:
            frame = int(row.get('target_frame_index', row.get('frame_index', 0)))
            if row.get('fg_correction'):
                path = Path(row['frame_path'])
                if file_sha256(path) != row['frame_sha256']:
                    raise ValueError('FG observation archive changed.')
                with np.load(path, allow_pickle=False) as a:
                    obs = {'joint_action': {'vector': a['actions'][frame].astype(np.float32)},
                           'observation': {c: {'rgb': a[c][frame]} for c in CAMERAS}}
            else:
                path, frame = raw.locate(row, 'target')
                with h5py.File(path, 'r') as a:
                    obs = {'joint_action': {'vector': a['joint_action/vector'][frame].astype(np.float32)},
                           'observation': {c: {'rgb': decode_legacy_robotwin_rgb(a[f'observation/{c}/rgb'][frame])} for c in CAMERAS}}
            payload = payloads[row['id']]
            ref = payload['references']['target'].to(model.torch_dtype)
            valid = payload.get('valid', {}).get('target', torch.ones(ref.shape[:2], device=ref.device, dtype=torch.bool))
            policy.reset()
            captured = move_cache(capture_frozen_inputs(policy, obs, row['counterfactual_instruction']), model.device)
            if captured['policy_guard_state'] is not None:
                raise ValueError('Unexpected history in reset-state diagnostic.')
            # Fresh production pixels/proprio must reproduce the training observation.
            cached = payload['captured']['target']
            if not torch.equal(captured['video_inputs']['x'], cached['video_inputs']['x']):
                raise ValueError('Fresh production video differs from prepared target observation.')
            if 'proprio' in cached and not torch.equal(captured['proprio'], cached['proprio']):
                raise ValueError('Fresh proprio differs from prepared target observation.')
            actions = {}
            for language in ('target', 'source', 'repeat'):
                policy.reset()
                instruction = row['source_instruction'] if language == 'source' else row['counterfactual_instruction']
                with torch.no_grad():
                    value = policy._infer_action_chunk(obs, instruction)
                actions[language] = norm.forward(torch.as_tensor(value).float().unsqueeze(0)).to(ref.device)
            if not torch.equal(actions['target'], actions['repeat']):
                raise ValueError('Fixed-seed repeat inference differs.')
            fit = {}
            noise = noise_tensor((1, 32, 14), 42, model)
            scheduler = model.train_action_scheduler
            for sigma in (.25, .5, 1.):
                t = torch.tensor([sigma*scheduler.num_train_timesteps], device=model.device, dtype=model.torch_dtype)
                x = scheduler.add_noise(ref, noise, t)
                target = scheduler.training_target(ref, noise, t)
                estimate = predict(model, captured, x, t, eraf=args.eraf == 'on', checkpoint=False)
                fit[str(sigma)] = float(masked_mse(estimate, target, valid))
                if sigma == 1.:
                    if not torch.equal(x, noise):
                        raise ValueError('Expert action leaked into pure-noise endpoint.')
                    fit['endpoint_action_mse_first24'] = float(masked_mse((noise-estimate)[:, :24], ref[:, :24], valid[:, :24]))
            record = dict(id=row['id'], task=row['diagnostic_task'], kind=row['diagnostic_kind'],
                          split=row['replay_split'], scene_seed=row['scene_seed'], frame=frame,
                          raw_path=str(path), source_instruction=row['source_instruction'],
                          counterfactual_instruction=row['counterfactual_instruction'],
                          state_sha256=array_sha256(obs['joint_action']['vector']),
                          rgb_sha256={c: array_sha256(obs['observation'][c]['rgb']) for c in CAMERAS},
                          reference_sha256=array_sha256(ref.float().cpu().numpy()),
                          valid_sha256=array_sha256(valid.cpu().numpy()), repeat_identical=True,
                          flow_velocity_mse=fit,
                          deployed_cf_mse_first24=float(masked_mse(actions['target'][:, :24], ref[:, :24], valid[:, :24])),
                          deployed_source_to_cf_reference_mse_first24=float(masked_mse(actions['source'][:, :24], ref[:, :24], valid[:, :24])),
                          deployed_language_delta_mse_first24=float(masked_mse(actions['source'][:, :24], actions['target'][:, :24], valid[:, :24])))
            if not all(np.isfinite(v) for v in [*fit.values(), record['deployed_cf_mse_first24'], record['deployed_source_to_cf_reference_mse_first24'], record['deployed_language_delta_mse_first24']]):
                raise ValueError('Nonfinite diagnostic metric.')
            records.append(record); journal.write(json.dumps(record)+'\n')
            print(f'[probe] {len(records)}/18 {row["id"]}', flush=True)
    if file_sha256(args.checkpoint) != checkpoint_hash or file_sha256(args.manifest) != manifest_hash:
        raise ValueError('Inputs changed during read-only diagnostic.')
    (root/'summary.json').write_text(json.dumps(dict(complete=True, records=records, protocol=protocol,
                                                   input_hashes_stable=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
