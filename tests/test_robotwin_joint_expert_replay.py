import json
from pathlib import Path

import numpy as np
import pytest

from experiments.robotwin.joint_expert_replay import collect_scenes, masked_window, validate_prepared_rows
from experiments.robotwin.pgc_data import array_sha256, pair_spec_from_source_task


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_collection(root, start=84_000_000):
    spec = pair_spec_from_source_task('place_empty_cup')
    for kind in ('native', 'counterfactual'):
        folder = root / kind
        write(folder / 'meta/pgc_provenance.json', dict(collection_profile='joint_expert',
            artifact_role='joint_expert_supervision', allowed_training_stages=['grounding', 'joint'],
            forbidden_training_stages=['full_goal_correction'], dataset_kind=kind, state_aligned=True,
            full_goal_usage='not_present', successful_episode_count=3, task_config='demo_clean'))
        records = []
        for i in range(3):
            state = np.arange(28, dtype=np.float32) + i
            np.save(folder / f'meta/initial{i}.npy', state)
            raw = folder / f'episode{i}.hdf5'; raw.write_bytes(b'fixture')
            records.append(dict(episode_index=i, pair_id=spec.pair_id, source_task=spec.source_task,
                counterfactual_task=spec.counterfactual_task, source_variant=spec.source_variant,
                counterfactual_variant=spec.counterfactual_variant, dataset_kind=kind,
                executed_variant=spec.source_variant if kind == 'native' else spec.counterfactual_variant,
                scene_seed=start + i, initial_state_sha256=array_sha256(state), action_sha256='0' * 64,
                action_count=100, source_initial_state_catalog=f'meta/initial{i}.npy', raw_hdf5=raw.name,
                source_instruction=spec.source_instruction, counterfactual_instruction=spec.counterfactual_instruction,
                goal_verified=True, source_goal_verified=kind == 'native', counterfactual_goal_verified=kind == 'counterfactual',
                verification_policy='expert_goal_replay', capture_origin='source_scene_expert_replay',
                full_goal_verified=False, source_directed_failure_verified=False))
        (folder / 'meta/pgc_episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    return root


def test_new_tasks_have_complete_scene_splits_without_fg_relabelling(tmp_path):
    root = make_collection(tmp_path / 'expert')
    rows, metadata = collect_scenes([root], [], train_per_task=2, holdout_per_task=1)
    assert [r['replay_split'] for r in rows] == ['train', 'train', 'replay_holdout']
    assert all(not r['fg_correction'] and r['full_goal_usage'] == 'not_present' for r in rows)
    assert all(Path(m['path']).is_file() for m in metadata)
    validate_prepared_rows(rows, rows)
    with pytest.raises(ValueError, match='supervision role'):
        validate_prepared_rows([r | {'fg_correction': True} for r in rows], rows)
    with pytest.raises(ValueError, match='split'):
        validate_prepared_rows([r | {'replay_split': 'train'} for r in rows], rows)
    with pytest.raises(ValueError, match='overlaps'):
        collect_scenes([root], rows[:1], train_per_task=2, holdout_per_task=1)
    p = root / 'native/meta/pgc_provenance.json'
    d = json.loads(p.read_text()); d['allowed_training_stages'] = ['grounding']; write(p, d)
    with pytest.raises(ValueError, match='provenance'):
        collect_scenes([root], [], train_per_task=2, holdout_per_task=1)


def test_dev_scenes_and_changed_initial_states_are_rejected(tmp_path):
    root = make_collection(tmp_path / 'dev', start=91_300_000)
    with pytest.raises(ValueError, match='reserved training'):
        collect_scenes([root], [], train_per_task=2, holdout_per_task=1)
    root = make_collection(tmp_path / 'train')
    np.save(root / 'counterfactual/meta/initial0.npy', np.ones(28, dtype=np.float32))
    with pytest.raises(ValueError, match='initial state changed'):
        collect_scenes([root], [], train_per_task=2, holdout_per_task=1)


def test_tail_actions_are_padded_but_never_marked_as_real_supervision():
    actions = np.arange(35 * 14, dtype=np.float32).reshape(35, 14)
    first, first_valid = masked_window(actions, 0)
    tail, tail_valid = masked_window(actions, 32)
    assert np.array_equal(first, actions[:32]) and first_valid.all()
    assert tail.shape == (32, 14) and tail_valid.sum() == 3
    assert np.array_equal(tail[:3], actions[32:])
    assert np.array_equal(tail[3:], np.repeat(actions[-1:], 29, axis=0))
    with pytest.raises(ValueError, match='Invalid'):
        masked_window(actions, 35)


def test_prepare_merge_preserves_expert_branch_observations_and_raw_grounding_paths(tmp_path, monkeypatch):
    import h5py
    import io
    import sys
    import torch
    from PIL import Image
    from types import SimpleNamespace
    from experiments.robotwin import eraf_fg_bridge
    from experiments.robotwin.compact_replay import ReplayPayloads
    from scripts.prepare_robotwin_joint_expert_replay import main
    root = make_collection(tmp_path / 'expert')
    for kind in ('native', 'counterfactual'):
        journal = root / kind / 'meta/pgc_episodes.jsonl'
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        for row in rows:
            actions = np.zeros((35, 14), dtype=np.float32)
            actions[1:] = 1 if kind == 'native' else 2
            row['action_count'], row['action_sha256'] = len(actions), array_sha256(actions)
            buffer = io.BytesIO(); Image.new('RGB', (4, 4), (1, 2, 3)).save(buffer, format='JPEG')
            with h5py.File(root / kind / row['raw_hdf5'], 'w') as h:
                h['joint_action/vector'] = actions
                for camera in ('head_camera', 'left_camera', 'right_camera'):
                    h[f'observation/{camera}/rgb'] = np.asarray([buffer.getvalue()] * 35, dtype='S1000')
        journal.write_text('\n'.join(json.dumps(row) for row in rows))
    parent = tmp_path / 'parent.json'; write(parent, dict(complete=True, states=[]))
    checkpoint = tmp_path / 'checkpoint.pt'; checkpoint.write_bytes(b'fixture')
    policy = SimpleNamespace(model=SimpleNamespace(requires_grad_=lambda value: None), reset=lambda: None,
        processor=SimpleNamespace(normalizer=SimpleNamespace(normalizers={'action': {'default': SimpleNamespace(forward=lambda x: x)}}),
                                  shape_meta={'action': [{'key': 'default'}]}))
    monkeypatch.setattr(eraf_fg_bridge, 'load_policy', lambda *a, **kw: policy)
    def capture(policy, observation, instruction):
        state = torch.from_numpy(observation['joint_action']['vector'].copy()).unsqueeze(0)
        return dict(video_inputs={'x': torch.zeros(1, 2)}, action_inputs={'context': state.unsqueeze(1)},
                    proprio=state, policy_guard_state=None)
    monkeypatch.setattr(eraf_fg_bridge, 'capture_frozen_inputs', capture)
    output = tmp_path / 'prepared'
    for mode in ('plan', 'worker', 'merge'):
        monkeypatch.setattr(sys, 'argv', ['prepare', mode, '--manifest', str(parent), '--checkpoint', str(checkpoint),
            '--output', str(output), '--collections', str(root), '--train-per-task', '2', '--holdout-per-task', '1'])
        main()
    result = json.loads((output / 'manifest.json').read_text())
    assert result['complete'] and result['added_joint_expert_states'] == 9
    rows = result['states']; replay = ReplayPayloads(rows, 'cpu')
    assert sum(r['initial_observations_exactly_equal'] for r in rows) == 3
    tail_row = next(r for r in rows if r['source_frame_index'] == 34)
    tail = replay[tail_row['id']]
    assert int(tail['valid']['target'].sum()) == 1
    assert torch.all(tail['captured']['source']['proprio'] == 1)
    assert torch.all(tail['captured']['target']['proprio'] == 2)
    assert set(tail_row['raw_paths']) == {'native', 'counterfactual'}
    assert {r['scene_seed'] for r in rows if r['replay_split'] == 'train'}.isdisjoint(
        {r['scene_seed'] for r in rows if r['replay_split'] == 'replay_holdout'})
