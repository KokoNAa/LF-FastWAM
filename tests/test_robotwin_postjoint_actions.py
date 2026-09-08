from copy import deepcopy
import pytest
from scripts.probe_robotwin_postjoint_actions import eligible


def fixture():
    return dict(stage='joint_training',terminal=False,jobs={a:dict(exit_code=0 if a.startswith('eraf') else None)
        for a in ('no_eraf','fg_only','eraf_only','eraf_fg')}),dict(no_eraf=125,fg_only=124)


def test_finished_eraf_arms_and_slower_remaining_training_allow_bounded_probe():
    eligible(*fixture())


@pytest.mark.parametrize('change',['next_stage','terminal','eraf_running','eraf_failed','off_finished','off_too_close','no_updates'])
def test_probe_cannot_compete_with_primary_stage_or_use_unfinished_models(change):
    driver,steps=fixture()
    if change=='next_stage':driver['stage']='auditing_joint'
    if change=='terminal':driver['terminal']=True
    if change=='eraf_running':driver['jobs']['eraf_fg']['exit_code']=None
    if change=='eraf_failed':driver['jobs']['eraf_only']['exit_code']=1
    if change=='off_finished':driver['jobs']['no_eraf']['exit_code']=0
    if change=='off_too_close':steps['fg_only']=171
    if change=='no_updates':steps['fg_only']=0
    with pytest.raises(ValueError):eligible(driver,steps)
