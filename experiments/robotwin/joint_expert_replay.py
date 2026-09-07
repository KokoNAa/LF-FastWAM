"""Admit new joint-expert scenes without relabelling them as FG corrections."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.robotwin.pgc_data import array_sha256, pair_spec_from_source_task, validate_pair_record
from experiments.robotwin.eraf_fg_data import file_metadata

TRAIN_SEED_RANGE = (84_000_000, 85_000_000)
KINDS = ('native', 'counterfactual')


def read(path):
    return json.loads(Path(path).read_text())


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Raw expert record points outside its collection or to a missing file.')
    return path


def scene_keys(rows):
    return {(r['task_config'], r['pair_id'], int(r['scene_seed'])) for r in rows}


def collect_scenes(roots, parent_rows, *, train_per_task, holdout_per_task):
    """Split complete accepted scenes, then keep every frame in that split.

    Reserved 84M training seeds exclude all 90M/91M development/test catalogs.
    Expert feasibility alone selects scenes; no learned-policy results are read.
    """
    if min(train_per_task, holdout_per_task) < 1:
        raise ValueError('Both training and scene-holdout counts must be positive.')
    expected = train_per_task + holdout_per_task
    scenes, inputs, tasks = [], [], set()
    existing = scene_keys(parent_rows)
    for root in map(Path, roots):
        records, provenance = {}, {}
        for kind in KINDS:
            folder = root / kind
            p = folder / 'meta/pgc_provenance.json'
            provenance[kind] = contract = read(p)
            if (contract.get('collection_profile') != 'joint_expert'
                    or contract.get('artifact_role') != 'joint_expert_supervision'
                    or set(contract.get('allowed_training_stages', [])) != {'grounding', 'joint'}
                    or 'full_goal_correction' not in contract.get('forbidden_training_stages', [])
                    or contract.get('dataset_kind') != kind or not contract.get('state_aligned')
                    or contract.get('full_goal_usage') != 'not_present'
                    or contract.get('successful_episode_count') != expected):
                raise ValueError('Collection is incomplete or lacks explicit joint-expert provenance.')
            journal = folder / 'meta/pgc_episodes.jsonl'
            rows = [validate_pair_record(json.loads(line)) for line in journal.read_text().splitlines()]
            if len(rows) != expected or {r['episode_index'] for r in rows} != set(range(expected)):
                raise ValueError('Missing or duplicated expert episode indices.')
            records[kind] = {r['episode_index']: r for r in rows}
            inputs += [file_metadata(p), file_metadata(journal)]
        if provenance['native']['task_config'] != provenance['counterfactual']['task_config']:
            raise ValueError('Source and CF domains differ.')
        domain = provenance['native']['task_config']
        task = records['native'][0]['source_task']
        if task in tasks:
            raise ValueError('Declare one complete collection per task.')
        tasks.add(task)
        spec = pair_spec_from_source_task(task)
        for index in range(expected):
            pair = {kind: records[kind][index] for kind in KINDS}
            source = pair['native']
            seed = int(source['scene_seed'])
            key = domain, spec.pair_id, seed
            if not TRAIN_SEED_RANGE[0] <= seed < TRAIN_SEED_RANGE[1] or key in existing:
                raise ValueError('Expert scene overlaps a parent or leaves the reserved training seed range.')
            existing.add(key)
            states, paths = {}, {}
            for kind, row in pair.items():
                if (row['source_task'] != task or row['pair_id'] != spec.pair_id
                        or row['scene_seed'] != seed or row['dataset_kind'] != kind
                        or row.get('goal_verified') is not True
                        or row.get('verification_policy') != 'expert_goal_replay'
                        or row.get('capture_origin') != 'source_scene_expert_replay'
                        or row.get('full_goal_verified') is not False
                        or row.get('source_directed_failure_verified') is not False
                        or row.get('source_instruction') != spec.source_instruction
                        or row.get('counterfactual_instruction') != spec.counterfactual_instruction):
                    raise ValueError('Expert record goal, scene, instruction or training role changed.')
                selected = 'source_goal_verified' if kind == 'native' else 'counterfactual_goal_verified'
                opposite = 'counterfactual_goal_verified' if kind == 'native' else 'source_goal_verified'
                if row.get(selected) is not True or row.get(opposite) is not False:
                    raise ValueError('The actually executed expert goal was not verified.')
                folder = root / kind
                state_path = contained(folder, row['source_initial_state_catalog'])
                states[kind] = np.load(state_path, allow_pickle=False)
                if array_sha256(states[kind]) != row['initial_state_sha256']:
                    raise ValueError('Recorded expert initial state changed.')
                paths[kind] = str(contained(folder, row['raw_hdf5']))
                inputs += [file_metadata(state_path), file_metadata(paths[kind])]
            if not np.array_equal(states['native'], states['counterfactual']):
                raise ValueError('Expert source and CF do not share the same initial physical state.')
            scenes.append(dict(pair_id=spec.pair_id, source_task=task, task_config=domain,
                episode_index=index, scene_seed=seed, raw_paths=paths,
                source_instruction=spec.source_instruction, counterfactual_instruction=spec.counterfactual_instruction,
                replay_split='train' if index < train_per_task else 'replay_holdout',
                artifact_role='joint_expert_supervision', fg_correction=False,
                allowed_training_stages=['grounding', 'joint'], full_goal_usage='not_present',
                collection_root=str(root.resolve()), raw_records=pair))
    if not scenes:
        raise ValueError('No joint-expert collections provided.')
    return scenes, inputs


def validate_prepared_rows(rows, scenes):
    expected = {(s['task_config'], s['pair_id'], s['scene_seed']): s for s in scenes}
    if scene_keys(rows) != set(expected):
        raise ValueError('Cache shard does not cover its declared expert scenes.')
    for row in rows:
        source = expected[row['task_config'], row['pair_id'], row['scene_seed']]
        fields = ('replay_split', 'raw_paths', 'source_instruction', 'counterfactual_instruction',
                  'source_task', 'artifact_role', 'allowed_training_stages', 'full_goal_usage', 'fg_correction')
        if any(row.get(key) != source[key] for key in fields):
            raise ValueError('Prepared expert row changed its split, data source or supervision role.')


def masked_window(actions, frame, horizon=32):
    if (actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all()
            or not 0 <= frame < len(actions) or horizon < 1):
        raise ValueError('Invalid finite 14-D action trajectory or frame.')
    real = actions[frame:frame + horizon]
    value = np.concatenate((real, np.repeat(real[-1:], horizon - len(real), axis=0)))
    return value, np.arange(horizon) < len(real)
