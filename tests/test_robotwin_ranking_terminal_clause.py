import numpy as np
import pytest

from experiments.robotwin.ranking_terminal_clause import terminal_clause_arrays


def fixture(prefix):
    order = (0, 1, 2) if prefix == 'source' else (2, 1, 0)
    positions = np.zeros((3, 4, 3), dtype=np.float32)
    positions[:, :3, 1:] = [-.14, .765]
    positions[:, list(order), 0] = [-.08, 0, .083]
    positions[0, order[-1], 0] = -.2
    root = 'pgc_entity_state/'
    clauses = root + prefix + '_'
    data = {root + 'entity_positions': positions,
        clauses + 'subject_indices': np.tile([*order[:2], 0, 0], (3, 1)),
        clauses + 'reference_indices': np.tile([*order[1:], 0, 0], (3, 1)),
        clauses + 'predicate_ids': np.tile([3, 3, 0, 0], (3, 1)),
        clauses + 'clause_valid': np.tile([True, True, False, False], (3, 1)),
        clauses + 'predicate_truth': np.tile([1., 1., 0., 0.], (3, 1)),
        clauses + 'goal_positions': np.zeros((3, 4, 3), dtype=np.float32)}
    data[clauses + 'goal_positions'][:, :2, 2] = .74
    return data, clauses, order


@pytest.mark.parametrize('prefix', ['source', 'target'])
def test_terminal_actor_gets_own_goal_without_changing_existing_clauses(prefix):
    data, key, order = fixture(prefix)
    original = {k: v.copy() for k, v in data.items()}
    out = terminal_clause_arrays(data, prefix)
    assert np.all(out[key + 'subject_indices'][:, 2] == order[-1])
    assert np.all(out[key + 'reference_indices'][:, 2] == order[-2])
    assert np.all(out[key + 'predicate_ids'][:, 2] == 4)
    assert out[key + 'predicate_truth'][:, 2].tolist() == [0., 1., 1.]
    np.testing.assert_allclose(out[key + 'goal_positions'][:, 2], np.tile([.083, -.14, .74], (3, 1)))
    for k in out:
        np.testing.assert_array_equal(out[k][:, :2], original[k][:, :2])
        np.testing.assert_array_equal(out[k][:, 3], original[k][:, 3])
    for k in data:
        np.testing.assert_array_equal(data[k], original[k])


@pytest.mark.parametrize('broken', ['invalid_order', 'already_three', 'failed_endpoint', 'wrong_end_positions'])
def test_rejects_ambiguous_or_failed_expert_labels(broken):
    data, key, _ = fixture('target')
    if broken == 'invalid_order': data[key + 'subject_indices'][0, 0] = 0
    if broken == 'already_three': data[key + 'clause_valid'][:, 2] = True
    if broken == 'failed_endpoint': data[key + 'predicate_truth'][-1, 0] = 0
    if broken == 'wrong_end_positions': data['pgc_entity_state/entity_positions'][-1, 0, 0] = -.3
    with pytest.raises(ValueError): terminal_clause_arrays(data, 'target')
