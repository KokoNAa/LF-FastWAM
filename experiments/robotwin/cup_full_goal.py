"""Admission rules for cup corrections verified by physical control replay."""
import math
from experiments.robotwin.cup_counterfactual import SOURCE_INSTRUCTION, FRONT_INSTRUCTION

FORMAT = 'robotwin_cup_full_goal_direct_replay_v1'


def validate_cup_correction(record):
    row = dict(record)
    if (row.get('source_task') != 'place_empty_cup' or row.get('pair_id') != 'place_empty_cup_on_to_front'
            or row.get('verification_binding') != 'direct_physics_replay'):
        raise ValueError('Unknown cup correction task or verification protocol')
    for key in ('fg_correction', 'full_goal_verified', 'counterfactual_goal_final_success', 'both_grippers_open_final'):
        if row.get(key) is not True: raise ValueError(f'Incomplete cup full goal: {key}')
    if row.get('counterfactual_goal_ever_success') is not False or not isinstance(row.get('source_goal_ever_success'),bool):
        raise ValueError('FG must originate from an audited unsuccessful CF rollout')
    step = int(row.get('capture_action_index', 0))
    if step <= 0 or int(row.get('prefix_action_count', -1)) != step:
        raise ValueError('FG must include a nonempty, complete policy prefix')
    if int(row.get('verified_replay_count',0)) < 2 or int(row.get('recorded_action_count',0)) < 12:
        raise ValueError('Insufficient complete replays or corrective actions')
    error,atol = float(row.get('replay_state_max_abs',math.inf)),float(row.get('state_atol',math.inf))
    if not math.isfinite(error) or not 0 <= error <= atol <= 1e-7:
        raise ValueError('Physical replay mismatch')
    lower = {'train':82000000,'replay_holdout':83000000}.get(row.get('replay_split'))
    if lower is None or not lower <= int(row['scene_seed']) < lower+1000000:
        raise ValueError('Cup correction has an invalid data split')
    if (row.get('source_instruction') != SOURCE_INSTRUCTION
            or row.get('counterfactual_instruction') != FRONT_INSTRUCTION):
        raise ValueError('Instruction and verified goal differ')
    for key in ('frame_path','control_path','rollout_path','rollout_checkpoint'):
        if not isinstance(row.get(key),str) or not row[key]:raise ValueError(f'Missing {key}')
    row['format']=FORMAT
    return row
