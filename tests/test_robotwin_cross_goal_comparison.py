from copy import deepcopy
import pytest
from scripts.compare_robotwin_cross_goal_semantics import compare


def report():
    rows = []
    for language, truth in [('source', True), ('target', False)]:
        rows.append(dict(pair_id='cup', task_config='clean', scene_seed=1,
            observation_language='source', instruction_language=language, raw_path='native.h5',
            frame=17, instruction=language, state_sha256='state', rgb_sha256={'head': 'rgb'},
            crossed=language != 'source', clauses=[dict(clause=0, truth_label=truth,
                truth_probability=.9 if truth else .1, goal_error_cm=1.)]))
    return dict(complete=True, action_quality_evaluated=False, records=rows,
                manifest_sha256='manifest', checkpoint_sha256='model')


def run(reports):
    return compare(reports, expected_queries=2, expected_scenes=1)


def test_scores_derive_from_records_and_accept_different_record_order():
    a, b = report(), report()
    b['records'].reverse()
    assert run(dict(a=a, b=b))['models']['b']['balanced_clause_accuracy'] == 1
    for row in b['records']:
        row['clauses'][0]['truth_probability'] = 0
    result = run(dict(a=a, b=b))['models']['b']
    assert result['balanced_clause_accuracy'] == .5
    assert result['own']['true_positives'] == result['crossed']['false_positives'] == 0


@pytest.mark.parametrize('field,value', [('instruction', 'changed'), ('state_sha256', 'other'),
                                       ('raw_path', 'other.h5'), ('rgb_sha256', {'head': 'other'})])
def test_reject_actual_input_mismatches(field, value):
    a, b = report(), report()
    b['records'][0][field] = value
    with pytest.raises(ValueError, match='differ'):
        run(dict(a=a, b=b))


def test_reject_missing_queries_changed_labels_and_nonfinite_scores():
    a = report()
    b = deepcopy(a)
    b['records'][1] = deepcopy(b['records'][0])
    with pytest.raises(ValueError, match='duplicate'):
        run(dict(b=b))
    b = deepcopy(a)
    b['records'][0]['clauses'][0]['truth_label'] = False
    with pytest.raises(ValueError, match='differ'):
        run(dict(a=a, b=b))
    b = deepcopy(a)
    b['records'][0]['clauses'][0]['truth_probability'] = float('nan')
    with pytest.raises(ValueError, match='Invalid'):
        run(dict(b=b))
