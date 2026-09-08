from copy import deepcopy
import pytest
from scripts.probe_robotwin_balanced_postjoint_actions import validate_completed, compare_actions
from tests.test_robotwin_deployed_postjoint_actions import reports


def completed():
    return dict(complete=True,terminal=True,stage='complete',jobs={'eval':{'exit_code':0}}),dict(complete=True,controller_gone=True,all_jobs_exit_zero_and_gone=True,episodes=180,comparison_methods=29)


def test_completed_trial_is_eligible():
    validate_completed(*completed())


@pytest.mark.parametrize('which,key,value',[(0,'complete',False),(0,'terminal',False),(0,'stage','evaluating'),(0,'jobs',{}),(0,'jobs',{'eval':{'exit_code':1}}),(1,'complete',False),(1,'controller_gone',False),(1,'all_jobs_exit_zero_and_gone',False),(1,'episodes',179),(1,'comparison_methods',28)])
def test_incomplete_or_failed_trial_is_rejected(which,key,value):
    inputs=completed();inputs[which][key]=value
    with pytest.raises(ValueError):validate_completed(*inputs)


def test_matched_inputs_compare_and_changed_rgb_rejected():
    old=reports();new=deepcopy(old)
    assert compare_actions(old,new)['actual_inputs_equal']
    new['records'][0]['rgb_sha256']='different'
    with pytest.raises(ValueError):compare_actions(old,new)
