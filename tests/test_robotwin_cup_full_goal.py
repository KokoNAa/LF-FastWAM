import pytest
from experiments.robotwin.cup_full_goal import validate_cup_correction
from experiments.robotwin.cup_counterfactual import SOURCE_INSTRUCTION,FRONT_INSTRUCTION


def record():
    return dict(source_task='place_empty_cup',pair_id='place_empty_cup_on_to_front',
        verification_binding='direct_physics_replay',fg_correction=True,full_goal_verified=True,
        counterfactual_goal_final_success=True,both_grippers_open_final=True,
        counterfactual_goal_ever_success=False,source_goal_ever_success=True,
        capture_action_index=192,prefix_action_count=192,verified_replay_count=2,
        recorded_action_count=100,replay_state_max_abs=0.,state_atol=1e-7,
        replay_split='train',scene_seed=82000000,source_instruction=SOURCE_INSTRUCTION,
        counterfactual_instruction=FRONT_INSTRUCTION,frame_path='/run/frames.npz',
        control_path='/run/controls.pkl',rollout_path='/run/rollout.npz',rollout_checkpoint='/run/model.pt')


def test_direct_replay_record_needs_no_file_hashes():
    r=validate_cup_correction(record())
    assert r['format']=='robotwin_cup_full_goal_direct_replay_v1'
    assert not any('sha256' in key for key in r)


@pytest.mark.parametrize('change',[
    dict(capture_action_index=0,prefix_action_count=0),dict(counterfactual_goal_ever_success=True),
    dict(full_goal_verified=False),dict(replay_state_max_abs=1e-4),dict(scene_seed=93000000),
    dict(recorded_action_count=8),dict(verified_replay_count=1)])
def test_rejects_fresh_successful_partial_or_leaking_corrections(change):
    with pytest.raises(ValueError):validate_cup_correction(record()|change)


def test_held_goal_branch_keeps_target_without_fabricating_failed_source_pair():
    from copy import deepcopy
    from experiments.robotwin.cup_goal_branches import branch_record, FORMAT
    parent = record() | {'task_config': 'demo_clean'}
    endpoint = dict(full_goal_success=True, plan_success=True, initial_observations_equal=True,
                    initial_goal=False, verified_replays=2, frames=60, replay_state_max_abs=0.,
                    cup_height_above_initial=.07, frame_path='/target.npz', control_path='/controls.pkl')
    result = dict(format='robotwin_cup_held_goal_branch_probe_v1', complete=True, scene_seed=82000000,
                  task_config='demo_clean', replay_split='train', prefix_steps=120,
                  conditions={'source': deepcopy(endpoint), 'target': deepcopy(endpoint)},
                  action_difference_rmse24=.15)
    row = branch_record(parent, result, '/results.json')
    assert row['format'] == FORMAT and row['paired_goal_branch']
    assert row['capture_action_index'] == row['prefix_action_count'] == 120
    result['conditions']['source']['full_goal_success'] = False
    row = branch_record(parent, result, '/results.json')
    assert not row['paired_goal_branch'] and row['source'] is None
    result['conditions']['target']['verified_replays'] = 1
    with pytest.raises(ValueError): branch_record(parent, result, '/results.json')
