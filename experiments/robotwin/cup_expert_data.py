"""Ordinary cup expert pairs, kept distinct from failed-state corrections."""
import math
import numpy as np
from experiments.robotwin.cup_counterfactual import SOURCE_INSTRUCTION, FRONT_INSTRUCTION
from experiments.robotwin.eraf_fg_contract import action_windows

FORMAT = 'robotwin_cup_ordinary_expert_pairs_v1'


def validate_pair(record):
    row = dict(record)
    if row.get('format') != FORMAT or row.get('source_task') != 'place_empty_cup':
        raise ValueError('Unknown ordinary cup pair')
    if row.get('pair_id') != 'place_empty_cup_on_to_front':
        raise ValueError('Unknown cup relation')
    if any(row.get(k) for k in ('fg_correction', 'native_retention', 'cf_retention')):
        raise ValueError('Ordinary initial experts cannot carry failure or policy-retention labels')
    if row.get('initial_observations_exactly_equal') is not True:
        raise ValueError('Expert pair must share exact initial RGB and proprio')
    lower = {'train': 84000000, 'replay_holdout': 85000000}.get(row.get('replay_split'))
    if lower is None or not lower <= int(row['scene_seed']) < lower + 1000000:
        raise ValueError('Ordinary expert scene is outside its reserved split')
    if (row.get('source_instruction') != SOURCE_INSTRUCTION or
            row.get('counterfactual_instruction') != FRONT_INSTRUCTION):
        raise ValueError('Ordinary expert instructions differ from the evaluated goals')
    for language in ('source', 'target'):
        item = row[language]
        error = float(item.get('replay_state_max_abs', math.inf))
        if (item.get('full_goal_success') is not True or item.get('replay_success') is not True
                or not math.isfinite(error) or not 0 <= error <= 1e-7
                or int(item.get('frames', 0)) < 32):
            raise ValueError('Incomplete ordinary expert or physical replay')
        for key in ('frame_path', 'control_path'):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ValueError('Missing ordinary expert artifact')
    return row


def paired_windows(source, target, count=12):
    """Pair relative progress, retaining separate observations after frame zero."""
    if count < 2:
        raise ValueError('Need initial and later ordinary expert observations')
    windows = [action_windows(a, stride=1) for a in (source, target)]
    indices = [np.unique(np.rint(np.linspace(0, len(w)-1, min(count, len(w)))).astype(int))
               for w in windows]
    if len(indices[0]) != len(indices[1]):
        raise ValueError('Expert trajectories too short for matched progress samples')
    return [(windows[0][a], windows[1][b]) for a, b in zip(*indices, strict=True)]
