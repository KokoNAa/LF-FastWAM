from copy import deepcopy
import numpy as np
import pytest
from experiments.robotwin.cup_expert_data import FORMAT, validate_pair, paired_windows
from experiments.robotwin.cup_counterfactual import SOURCE_INSTRUCTION, FRONT_INSTRUCTION


def record():
    endpoint = {'full_goal_success': True, 'replay_success': True,
                'replay_state_max_abs': 0., 'frames': 100, 'frame_path': '/frames.npz',
                'control_path': '/controls.pkl'}
    return {'format': FORMAT, 'source_task': 'place_empty_cup', 'pair_id': 'place_empty_cup_on_to_front',
            'task_config': 'demo_clean', 'scene_seed': 84000000, 'replay_split': 'train',
            'source_instruction': SOURCE_INSTRUCTION, 'counterfactual_instruction': FRONT_INSTRUCTION,
            'initial_observations_exactly_equal': True, 'source': deepcopy(endpoint), 'target': deepcopy(endpoint)}


@pytest.mark.parametrize('kind', ['fg_correction', 'native_retention', 'cf_retention'])
def test_initial_expert_cannot_masquerade_as_correction_or_successful_policy(kind):
    row = record(); row[kind] = True
    with pytest.raises(ValueError): validate_pair(row)


def test_reject_failed_goal_changed_initial_scene_and_development_seed():
    for mutate in (lambda r: r['target'].update(full_goal_success=False),
                   lambda r: r.update(initial_observations_exactly_equal=False),
                   lambda r: r.update(scene_seed=93000000),
                   lambda r: r['target'].update(replay_state_max_abs=.01)):
        row = record(); mutate(row)
        with pytest.raises(ValueError): validate_pair(row)
    assert validate_pair(record())['scene_seed'] == 84000000


def test_progress_pairs_keep_real_separate_frames_and_final_action_masks():
    source = np.arange(100*14, dtype=np.float32).reshape(100, 14)
    target = np.arange(150*14, dtype=np.float32).reshape(150, 14) + 10000
    pairs = paired_windows(source, target)
    assert len(pairs) == 12
    assert pairs[0][0].start == pairs[0][1].start == 0
    assert pairs[4][0].start != pairs[4][1].start
    for a, b in pairs:
        np.testing.assert_array_equal(a.action[a.valid], source[a.start:a.start+32])
        np.testing.assert_array_equal(b.action[b.valid], target[b.start:b.start+32])
    assert pairs[-1][0].valid.sum() == pairs[-1][1].valid.sum() == 1


def test_control_replay_reports_largest_observed_error(monkeypatch):
    from types import SimpleNamespace
    from experiments.robotwin import eraf_fg_collection as collection
    task = SimpleNamespace(state=np.zeros(1))
    task.take_dense_action = lambda seq, save_freq: setattr(task, 'state', np.asarray(seq))
    monkeypatch.setattr(collection, 'physical_state', lambda current: current.state)
    controls = [{'control_seq': [1.], 'save_freq': -1, 'state_after': np.array([1.+5e-8])},
                {'control_seq': [2.], 'save_freq': -1, 'state_after': np.array([2.+2e-8])}]
    assert collection.replay_continuation(task, controls) == pytest.approx(5e-8)
    controls[-1]['state_after'] = np.array([3.])
    with pytest.raises(ValueError): collection.replay_continuation(task, controls)
