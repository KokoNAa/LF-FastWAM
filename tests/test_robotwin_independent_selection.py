from copy import deepcopy
import pytest

from scripts.freeze_robotwin_independent_selection import REQUIRED,select_methods,freeze
from scripts.compare_robotwin_ten_task_methods import GROUPS


def report():
    cells={task:dict(episodes=n,successes=0) for n,tasks in GROUPS.values() for task in tasks}
    methods={name:dict(cells=deepcopy(cells)) for name in REQUIRED}
    methods['eraf_fg']['cells']['place_empty_cup']['successes']=1
    return dict(complete=True,target='eraf_fg',methods=methods)


def test_selection_recomputes_scores_and_keeps_all_required_controls():
    r=report();names,scores=select_methods(r)
    assert set(names)==set(REQUIRED) and scores['eraf_fg']==pytest.approx(1/30)
    r['methods']['eraf_fg']['ten_task_macro_cf']=.99
    assert select_methods(r)==(names,scores)


@pytest.mark.parametrize('change',['tie','loss','missing_task','missing_control','wrong_budget','incomplete'])
def test_no_freeze_from_ties_losses_missing_or_partial_dev_evidence(change):
    r=report()
    if change=='tie':r['methods']['no_eraf']['cells']['place_empty_cup']['successes']=1
    if change=='loss':r['methods']['no_eraf']['cells']['place_empty_cup']['successes']=2
    if change=='missing_task':del r['methods']['fg_only']['cells']['place_empty_cup']
    if change=='missing_control':del r['methods']['historical_best_fg_only']
    if change=='wrong_budget':r['methods']['eraf_only']['cells']['place_empty_cup']['episodes']=2
    if change=='incomplete':r['complete']=False
    with pytest.raises(ValueError):select_methods(r)


def test_unlisted_historical_best_control_is_added_instead_of_ignored():
    r=report();r['methods']['eraf_fg']['cells']['place_empty_cup']['successes']=3
    extra=deepcopy(r['methods']['no_eraf']);extra['cells']['place_empty_cup']['successes']=2
    r['methods']['another_control']=extra
    names,_=select_methods(r)
    assert 'another_control' in names


def test_existing_test_directory_is_rejected_before_reading_any_test_or_source_data(tmp_path):
    test=tmp_path/'test';test.mkdir()
    with pytest.raises(ValueError,match='before creating'):freeze(tmp_path/'missing_source',test)
