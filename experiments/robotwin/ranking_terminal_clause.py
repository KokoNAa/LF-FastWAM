"""Optional expert-only labels giving every ranking block an execution clause.

The added goal uses the recorded expert endpoint in XY and the existing placement
height. It is a training label, never an observation or deployment input. Raw
trajectories and the benchmark's ordered-row success rule remain unchanged.
"""
from __future__ import annotations

import numpy as np

RANKING_PAIRS = {'blocks_ranking_rgb_to_bgr', 'blocks_ranking_size_large_to_small_to_small_to_large'}
LABEL_SCHEMA = 'ranking_terminal_expert_endpoint_v1'


def terminal_clause_arrays(handle, prefix):
    if prefix not in {'source', 'target'}:
        raise ValueError('Ranking labels require source or target language.')
    order = (0, 1, 2) if prefix == 'source' else (2, 1, 0)
    root = 'pgc_entity_state/'
    positions = np.asarray(handle[root + 'entity_positions'][:], dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (4, 3) or not len(positions) or not np.isfinite(positions).all():
        raise ValueError('Invalid complete ranking position history.')
    fields = ['subject_indices', 'reference_indices', 'predicate_ids', 'goal_positions', 'predicate_truth', 'clause_valid']
    values = {name: np.asarray(handle[root + prefix + '_' + name][:]).copy() for name in fields}
    for name, value in values.items():
        expected = (len(positions), 4, 3) if name == 'goal_positions' else (len(positions), 4)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError('Invalid ranking clause array: ' + name)
    if (not np.all(values['clause_valid'] == [True, True, False, False])
            or not np.all(values['subject_indices'][:, :2] == order[:2])
            or not np.all(values['reference_indices'][:, :2] == order[1:])
            or not np.all(values['predicate_ids'][:, :2] == 3)):
        raise ValueError('Expected the unchanged two-clause ranking schema.')
    if not np.all(values['predicate_truth'][-1, :2] >= .5):
        raise ValueError('Terminal goal labels require a successful full expert endpoint.')
    final = positions[-1, list(order)]
    deltas = np.diff(final[:, :2], axis=0)
    if not np.all((deltas[:, 0] > 0) & (deltas[:, 0] < .13) & (np.abs(deltas[:, 1]) < .03)):
        raise ValueError('Recorded endpoint contradicts the selected ranking order.')
    subject, reference = order[-1], order[-2]
    values['subject_indices'][:, 2] = subject
    values['reference_indices'][:, 2] = reference
    values['predicate_ids'][:, 2] = 4  # Existing right predicate; no vocabulary change.
    goal = positions[-1, subject].copy()
    goal[2] = values['goal_positions'][0, 1, 2]
    values['goal_positions'][:, 2] = goal
    delta = positions[:, subject, :2] - positions[:, reference, :2]
    values['predicate_truth'][:, 2] = (delta[:, 0] > 0) & (delta[:, 0] < .13) & (np.abs(delta[:, 1]) < .03)
    values['clause_valid'][:, 2] = True
    return {root + prefix + '_' + name: value for name, value in values.items()}
