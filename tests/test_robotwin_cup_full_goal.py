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
