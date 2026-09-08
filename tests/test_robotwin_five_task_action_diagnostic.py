import copy
import pytest
from scripts.run_robotwin_five_task_action_diagnostic import eligible


def state():
    return dict(status='training',complete=False,jobs={
        'train_no_eraf':dict(exit_code=None),'train_eraf_only':dict(exit_code=0),'train_eraf_fg':dict(exit_code=0)})


def test_only_uses_gap_after_both_eraf_jobs_exit_with_enough_control_work_left():
    eligible(state(),110)
    eligible(state(),160)
    for step in (0,161,200):
        with pytest.raises(ValueError):eligible(state(),step)


@pytest.mark.parametrize('change',['primary_finished','eraf_failed','eraf_running','control_finished'])
def test_cannot_compete_with_primary_next_stage(change):
    s=copy.deepcopy(state())
    if change=='primary_finished':s['status']='screening_dev_scenes'
    elif change=='eraf_failed':s['jobs']['train_eraf_fg']['exit_code']=1
    elif change=='eraf_running':s['jobs']['train_eraf_fg']['exit_code']=None
    else:s['jobs']['train_no_eraf']['exit_code']=0
    with pytest.raises(ValueError):eligible(s,110)
