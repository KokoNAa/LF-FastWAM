#!/usr/bin/env python3
"""Audit sampled windows with similar observations but different future actions.

These are sensitivity-tested distance proxies, not semantic equivalence labels.
Only complete windows at matching source/CF frame indices are compared. The
original saved sampler is reconstructed; no model or simulation is executed.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import io
import itertools
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]

# Require each of the three cameras to satisfy the RGB mean-error threshold.
THRESHOLDS = {
    'tight': (.01, 1., .1),
    'medium': (.05, 3., .1),
    'loose': (.1, 5., .1),
    'medium_action05': (.05, 3., .05),
    'medium_action25': (.05, 3., .25),
}


def distances(source, target, horizon=32):
    length = min(len(source), len(target))
    if length < horizon:
        raise ValueError('Missing complete action window.')
    square = np.mean((source[:length] - target[:length]) ** 2, axis=1)
    future = np.sqrt(np.maximum(0, np.convolve(square, np.ones(horizon) / horizon, mode='valid')))
    return np.sqrt(square[:len(future)]), future


def classify(point, future, rgb):
    return {key: (point <= q) & (rgb <= im) & (future >= a)
            for key, (q, im, a) in THRESHOLDS.items()}


def main():
    import h5py
    import torch
    from PIL import Image
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.utils.samplers import ResumableEpochSampler
    from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer, load_dataset_stats_from_json
    from experiments.robotwin.no_eraf_probe import CAMERAS, typed_hash, require_pair
    from scripts.probe_robotwin_no_eraf import read_json, write_json, sha256
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--world-size', type=int, required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(args.config)
    if (args.world_size < 1 or cfg.data.train.skip_padding_as_possible
            or int(cfg.data.train.global_sample_stride) != 1
            or cfg.data.train.processor.use_stepwise_action_norm
            or cfg.data.train.processor.norm_default_mode != 'z-score'
            or cfg.data.train.processor.norm_exception_mode is not None):
        raise ValueError('Unsupported sampler or action normalization.')
    torch.set_num_threads(1)
    full_stats = load_dataset_stats_from_json(cfg.data.train.pretrained_norm_stats)
    stats = full_stats['action']['default']
    normalizer = SingleFieldLinearNormalizer({k.removeprefix('global_'): v for k, v in stats.items() if k.startswith('global_')}, 'z-score')
    state_stats = full_stats['state']['default']
    state_normalizer = SingleFieldLinearNormalizer({k.removeprefix('global_'): v for k, v in state_stats.items() if k.startswith('global_')}, 'z-score')
    dataset = instantiate(cfg.data.train)
    underlying = dataset.lerobot_dataset.multi_dataset._datasets
    dirs = list(cfg.data.train.dataset_dirs) + list(cfg.data.train.pgc_counterfactual_dataset_dirs)
    global_batch = int(cfg.batch_size) * args.world_size * int(cfg.gradient_accumulation_steps)
    draw_count = int(cfg.max_steps) * global_batch
    if draw_count > len(dataset):
        raise ValueError('Expected epoch0 only.')
    sampler = ResumableEpochSampler(dataset, cfg.seed, cfg.batch_size, args.world_size, cfg.gradient_accumulation_steps)
    positions = np.fromiter(itertools.islice(iter(sampler), draw_count), dtype=np.int64, count=draw_count)
    draws = np.asarray(dataset._sample_indices, dtype=np.int64)[positions]
    metadata, inspected, rows, draw_rows = {}, {}, [], []
    cooccurrences = {k: defaultdict(set) for k in THRESHOLDS}
    offset = 0
    (root / 'episodes').mkdir()

    def inspect(raw_pair, episode):
        cache_key = (str(raw_pair), episode)
        if cache_key in inspected:
            return inspected[cache_key]
        if str(raw_pair) not in metadata:
            metadata[str(raw_pair)] = {kind: {int(r['episode_index']): r for r in (
                json.loads(line) for line in (raw_pair / kind / 'meta/pgc_episodes.jsonl').read_text().splitlines() if line.strip())}
                for kind in ('native', 'counterfactual')}
        records = {kind: metadata[str(raw_pair)][kind][episode] for kind in ('native', 'counterfactual')}
        require_pair(records['native'], records['counterfactual'])
        paths = {k: raw_pair / k / r['raw_hdf5'] for k, r in records.items()}
        with h5py.File(paths['native'], 'r') as source, h5py.File(paths['counterfactual'], 'r') as target:
            actions, raw_actions, states = [], [], []
            for kind, h in (('native', source), ('counterfactual', target)):
                raw = np.asarray(h['joint_action/vector'], dtype=np.float32)
                if typed_hash(raw) != records[kind]['action_sha256']:
                    raise ValueError('Raw action hash mismatch.')
                raw_actions.append(raw)
                actions.append(normalizer.forward(torch.as_tensor(raw)).numpy())
                states.append(state_normalizer.forward(torch.as_tensor(raw)).numpy())
            point, future = distances(*actions)
            point = np.sqrt(np.mean((states[0][:len(future)] - states[1][:len(future)]) ** 2, axis=1))
            rgb = np.full(len(point), np.inf)
            changed = np.full(len(point), np.nan)
            exact = np.zeros(len(point), dtype=bool)
            # This superset includes every frame that can satisfy any threshold.
            candidates = sorted({0, *np.flatnonzero((point <= .1) & (future >= .05)).tolist()})
            for frame in candidates:
                maes, changed_fractions, equals = [], [], []
                for camera in CAMERAS:
                    key = f'observation/{camera}/rgb'
                    a, b = [np.asarray(Image.open(io.BytesIO(bytes(h[key][frame]).rstrip(b'\0'))).convert('RGB')) for h in (source, target)]
                    if a.shape != b.shape:
                        raise ValueError('Camera shape mismatch.')
                    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
                    maes.append(float(delta.mean()))
                    changed_fractions.append(float((delta.max(axis=2) > 16).mean()))
                    equals.append(np.array_equal(a, b))
                rgb[frame] = max(maes)
                changed[frame] = max(changed_fractions)
                exact[frame] = np.array_equal(raw_actions[0][frame], raw_actions[1][frame]) and all(equals)
            if not exact[0]:
                raise ValueError('Paired initial observations do not match exactly.')
        masks = classify(point, future, rgb)
        identifier = '_'.join((raw_pair.parents[2].name, raw_pair.parents[1].name, raw_pair.name, f'ep{episode:04d}'))
        path = root / 'episodes' / (identifier + '.npz')
        if path.exists():
            raise ValueError('Duplicate episode artifact identifier.')
        np.savez_compressed(path, point_rmse=point, future32_rmse=future, max_camera_rgb_mae=rgb,
                            max_camera_changed_pixel_fraction=changed, exact_observation_equal=exact, **masks)
        info = {'id': identifier, 'pair_id': raw_pair.name, 'profile': raw_pair.parents[2].name,
                'task_config': raw_pair.parents[1].name, 'episode_index': episode,
                'complete_paired_windows': len(point), 'decoded_frames': len(candidates),
                'initial_future32_rmse': float(future[0]), 'artifact': str(path), 'artifact_sha256': sha256(path),
                'available': {k: int(v.sum()) for k, v in masks.items()}}
        write_json(path.with_suffix('.json'), info)
        inspected[cache_key] = (point, future, rgb, masks, info)
        if len(inspected) % 10 == 0:
            print(f'[episodes] {len(inspected)} latest={identifier}', flush=True)
        return inspected[cache_key]

    for di, (data, folder) in enumerate(zip(underlying, dirs, strict=True)):
        folder = Path(folder)
        role = dataset.pgc_entity_relation_indices[di]['artifact_role']
        episodes = np.asarray(data.hf_dataset['episode_index'], dtype=np.int64)
        frames = np.asarray(data.hf_dataset['frame_index'], dtype=np.int64)
        chosen = np.flatnonzero((draws >= offset) & (draws < offset + len(episodes)))
        local = draws[chosen] - offset
        row = {'dataset': str(folder), 'role': role, 'draws': len(local)}
        if role != 'closed_loop_native':
            raw_pair = folder.parents[2] / 'raw' / folder.parent.name
            counts, available = Counter(), Counter()
            valid_count = 0
            for episode in sorted(set(episodes.tolist())):
                point, future, rgb, masks, info = inspect(raw_pair, int(episode))
                available.update(info['available'])
                ix = np.flatnonzero(episodes[local] == episode)
                for i in ix:
                    frame = int(frames[local[i]])
                    valid = frame < len(point)
                    valid_count += valid
                    selected = {k: bool(valid and mask[frame]) for k, mask in masks.items()}
                    counts.update(k for k, ok in selected.items() if ok)
                    draw_position = int(chosen[i])
                    draw_rows.append({'position': draw_position, 'role': role, 'id': info['id'], 'frame': frame,
                                      'has_complete_paired_window': valid, **selected})
                    for k, ok in selected.items():
                        if ok:
                            cooccurrences[k][draw_position // global_batch, info['id'], frame].add(role)
            row.update(pair_id=raw_pair.name, task_config=raw_pair.parents[1].name,
                       valid_paired_draws=valid_count, sampled={k: counts[k] for k in THRESHOLDS}, available=dict(available))
        rows.append(row)
        offset += len(episodes)
        print(f'[dataset] {di+1}/{len(dirs)} {role} draws={len(local)}', flush=True)
    result = {'format': 'robotwin_choice_coverage_v1', 'complete': True, 'config': str(Path(args.config).resolve()),
        'config_sha256': sha256(args.config), 'world_size': args.world_size, 'global_batch': global_batch,
        'draw_count': draw_count, 'normalization': 'Released separate state/action z-score normalizers including clamp',
        'thresholds': {k: dict(qpos_rmse=v[0], max_camera_rgb_mae=v[1], future32_rmse=v[2]) for k,v in THRESHOLDS.items()},
        'datasets': rows, 'episodes': [v[-1] for v in inspected.values()], 'draws': sorted(draw_rows, key=lambda r:r['position']),
        'same_optimizer_same_frame_both_historical_positive_draws': {k: sum({'offline_native','historical_cf'} <= v for v in groups.values()) for k,groups in cooccurrences.items()},
        'scope': 'Reconstructed epoch0 sampler. Near-observation metrics are numerical proxies, NOT semantic equivalence or exact counterfactual labels. Same-index future actions belong to each expert own observation. Closed-loop native pool has no paired expert reference and is excluded from this classification.'}
    write_json(root / 'summary.json', result)
    print('[complete]', root / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
