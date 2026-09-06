"""Full-goal continuations and optional paired goals at real held-cup states."""
from copy import deepcopy
from experiments.robotwin.cup_full_goal import validate_cup_correction

FORMAT = 'robotwin_cup_held_goal_branches_v1'


def branch_record(parent, result, result_path):
    parent = validate_cup_correction(parent)
    if (result.get('format') != 'robotwin_cup_held_goal_branch_probe_v1' or
            result.get('complete') is not True or result['scene_seed'] != parent['scene_seed'] or
            result['task_config'] != parent['task_config'] or result['replay_split'] != parent['replay_split']):
        raise ValueError('Incomplete or mismatched goal branch experiment')
    target = result['conditions']['target']
    if not verified(target): raise ValueError('Unverified full CF branch')
    source = result['conditions'].get('source', {})
    paired = verified(source)
    row = deepcopy(parent)
    row.update(format=FORMAT, paired_goal_branch=paired, branch_result_path=str(result_path),
        capture_action_index=int(result['prefix_steps']), prefix_action_count=int(result['prefix_steps']),
        frame_path=target['frame_path'], control_path=target['control_path'],
        recorded_action_count=target['frames'], verified_replay_count=target['verified_replays'],
        replay_state_max_abs=max(target['replay_state_max_abs'], source['replay_state_max_abs'] if paired else 0.),
        target=deepcopy(target), source=deepcopy(source) if paired else None,
        action_difference_rmse24=result.get('action_difference_rmse24'))
    return validate_branch(row)


def verified(condition):
    return (condition.get('full_goal_success') is True and condition.get('plan_success') is True
            and condition.get('initial_observations_equal') is True
            and condition.get('initial_goal') is False
            and condition.get('verified_replays', 0) >= 2 and condition.get('frames', 0) >= 32
            and 0 <= condition.get('replay_state_max_abs', float('inf')) <= 1e-7
            and condition.get('cup_height_above_initial', 0) >= .03
            and all(isinstance(condition.get(k), str) and condition[k] for k in ('frame_path', 'control_path')))


def validate_branch(record):
    if record.get('format') != FORMAT: raise ValueError('Unknown held-cup branch protocol')
    row = validate_cup_correction(record)
    row['format'] = FORMAT
    if not verified(row['target']): raise ValueError('Invalid held-cup target continuation')
    if bool(row.get('paired_goal_branch')) != bool(row.get('source')):
        raise ValueError('Paired branch flag contradicts its source continuation')
    if row.get('source') and not verified(row['source']):
        raise ValueError('Invalid source continuation')
    return row
