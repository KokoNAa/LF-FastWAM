import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.robotwin.eraf_fg_data import (
    file_metadata, validate_cf_retention_coverage, validate_retention_scene, verify_retention_capture)
from scripts import collect_robotwin_cf_retention as collector


@pytest.mark.parametrize('task', ['place_a2b_left', 'blocks_ranking_rgb'])
def test_original_target_cf_collection_admits_only_training_seeds(task):
    args = SimpleNamespace(task=task, condition='counterfactual', start_seed=80800000,
                           scenes=10, max_attempts=80, max_seconds=3600)
    collector.validate_request(args)
    for start in (4300001, 90000000, 91000000, 93000000):
        with pytest.raises(ValueError, match='training seeds'):
            collector.validate_request(SimpleNamespace(**(vars(args) | {'start_seed': start})))


def test_new_retention_metadata_keeps_teacher_capture_and_condition_together(tmp_path):
    capture, teacher = tmp_path / 'capture.npz', tmp_path / 'teacher.pt'
    capture.write_bytes(b'original capture')
    teacher.write_bytes(b'teacher')
    row = dict(source_task='blocks_ranking_rgb', task_config='demo_clean', scene_seed=80810000,
               cf_retention=True, full_cf_episode_success=True, retention_condition='counterfactual',
               retention_format='robotwin_policy_retention_v2', capture_path=str(capture),
               capture_metadata=file_metadata(capture), teacher_checkpoint=str(teacher),
               teacher_checkpoint_metadata=file_metadata(teacher),
               source_instruction='RGB', counterfactual_instruction='BGR', frame_index=0)
    assert validate_retention_scene([row]) == 'target'
    verify_retention_capture(row)
    for change in ({'retention_condition': 'correct'}, {'full_cf_episode_success': False},
                   {'capture_metadata': {}}, {'teacher_checkpoint': '/different.pt'}):
        with pytest.raises(ValueError):
            validate_retention_scene([row | change])
    capture.write_bytes(b'changed capture bytes and length')
    with pytest.raises(ValueError, match='metadata changed'):
        verify_retention_capture(row)


def test_five_task_retention_counts_training_scenes_and_keeps_previous_tasks():
    rows = [dict(source_task=task, task_config='demo_clean', scene_seed=80800000 + scene,
                 replay_split='train', cf_retention=True, full_cf_episode_success=True)
            for task in collector.TASKS for scene in range(10)]
    assert set(validate_cf_retention_coverage(rows, collector.TASKS).values()) == {10}
    with pytest.raises(ValueError, match='declared coverage'):
        validate_cf_retention_coverage(rows, collector.TASKS[2:])
    with pytest.raises(ValueError, match='previous tasks'):
        validate_cf_retention_coverage(rows[:20], collector.TASKS[:2])
    rows[0]['replay_split'] = 'replay_holdout'
    with pytest.raises(ValueError, match='10 CF-retention training scenes'):
        validate_cf_retention_coverage(rows, collector.TASKS)


@pytest.mark.parametrize('task_name', ['place_a2b_left', 'blocks_ranking_rgb'])
def test_collector_saves_only_selected_goal_success_and_real_action_trace(tmp_path, monkeypatch, task_name):
    from experiments.robotwin import eraf_fg_collection, pgc_task_variants
    from scripts import collect_pgc_robotwin_pairs as pairs, collect_robotwin_eraf_fg as fg
    from scripts import train_robotwin_cf_decision_adapter as training
    from experiments.robotwin.eraf_fg_contract import CAMERAS
    manifest, checkpoint, output = tmp_path / 'base.json', tmp_path / 'teacher.pt', tmp_path / 'out'
    manifest.write_text(json.dumps({'states': []}))
    checkpoint.write_bytes(b'teacher')
    task = SimpleNamespace(take_action_cnt=0)
    task.setup_demo = lambda **kwargs: setattr(task, 'scene_seed', kwargs['seed'])
    policy = SimpleNamespace(_infer_action_chunk=lambda observation, instruction: np.ones((32, 14)))
    original = policy._infer_action_chunk
    monkeypatch.setattr(training, 'load_policy', lambda *args: policy)
    monkeypatch.setattr(pairs, '_load_robotwin_args', lambda **kwargs: (task, {}))
    monkeypatch.setattr(pairs, '_capture_data_type', lambda options: {})
    monkeypatch.setattr(pairs, '_close', lambda task: None)
    monkeypatch.setattr(pgc_task_variants, 'install_pgc_task_contract', lambda *args: None)
    monkeypatch.setattr(fg, 'instructions', lambda *args: {'source': 'source goal', 'target': 'CF goal'})
    calls = []

    def rollout(task, policy, spec, instruction, *, selected_goal):
        calls.append((instruction, selected_goal))
        policy._infer_action_chunk({'joint_action': {'vector': np.zeros(14)},
            'observation': {c: {'rgb': np.zeros((3, 4, 3), dtype=np.uint8)} for c in CAMERAS}}, instruction)
        return {'initial': np.arange(50), 'actions': np.full((24, 14), 7),
                'audit': {'source': True, 'target': task.scene_seed == 80800001}}

    monkeypatch.setattr(eraf_fg_collection, 'run_failure_rollout', rollout)
    monkeypatch.setattr(eraf_fg_collection, 'full_goal',
                        lambda task, spec, selected_goal: task.scene_seed == 80800001)
    monkeypatch.setattr(sys, 'argv', ['collect', '--manifest', str(manifest), '--checkpoint', str(checkpoint),
        '--output', str(output), '--robotwin-root', str(tmp_path), '--task', task_name,
        '--start-seed', '80800000', '--max-attempts', '2', '--scenes', '1'])
    collector.main()
    result = json.loads((output / 'manifest.json').read_text())
    assert result['complete'] and result['successful_scenes'] == 1 and result['attempted_scenes'] == 2
    assert {r['scene_seed'] for r in result['states']} == {80800001}
    assert calls == [('CF goal', 'target')] * 2
    assert policy._infer_action_chunk is original
    row = result['states'][0]
    assert validate_retention_scene(result['states']) == 'target'
    verify_retention_capture(row)
    assert 'capture_sha256' not in row and 'teacher_checkpoint_sha256' not in row
    with np.load(row['capture_path']) as arrays:
        # Deployed reference chunks and actually executed controls are distinct.
        assert arrays['reference_action_raw'].shape == (1, 32, 14)
        np.testing.assert_array_equal(arrays['executed_actions'], np.full((24, 14), 7))
        assert arrays['capture_action_steps'].tolist() == [0]
