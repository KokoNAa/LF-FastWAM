"""Explicit ordinary-initial sampling and matched task-specific correction weights."""
import math


def source_task(row):
    if row.get('source_task'):
        return row['source_task']
    from experiments.robotwin.pgc_data import ROBOTWIN_TEN_TASK_SPECS
    return next(s.source_task for s in ROBOTWIN_TEN_TASK_SPECS if s.pair_id == row['pair_id'])


def validate_weights(weights):
    from experiments.robotwin.pgc_data import ROBOTWIN_TEN_TASK_NAMES
    if not isinstance(weights, dict) or any(k not in ROBOTWIN_TEN_TASK_NAMES for k in weights):
        raise ValueError('Correction weights require a mapping of known task names.')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError('Task correction weights must be finite and positive.')
    return weights


def effective_weight(row, base, weights):
    if not (row.get('fg_correction') or row.get('ordinary_cf_control')):
        return 1.
    if not weights:
        return base
    return base * weights.get(source_task(row), 1.)


def initial_rows(rows, tasks):
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError('Initial anchor tasks must be nonempty and distinct.')
    selected = [r for r in rows if r['replay_split'] == 'train'
                and not any(r.get(k) for k in ('fg_correction', 'native_retention', 'cf_retention'))
                and source_task(r) in tasks
                and int(r.get('source_frame_index', r.get('frame_index', 0))) == 0
                and int(r.get('target_frame_index', r.get('frame_index', 0))) == 0
                and r.get('initial_observations_exactly_equal') is True]
    if {source_task(r) for r in selected} != set(tasks):
        raise ValueError('Every anchor task requires an audited same-state ordinary initial training pair.')
    return selected
