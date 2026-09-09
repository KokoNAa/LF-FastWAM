from pathlib import Path
from scripts.run_robotwin_five_task_repair import training_command, coverage
from experiments.robotwin.five_task_action import TASKS, FG_TASKS, OBJECTIVE
from tests.test_robotwin_five_task_action import rows


def test_three_gpu_schedule_preserves_global_batch_and_rejects_invalid_devices():
    import pytest
    from scripts.run_robotwin_five_task_repair import validated_gpus
    assert validated_gpus([0,1,2], {0,1,2}) == [0,1,2]
    assert validated_gpus([1,3,5], {1,3,5}) == [1,3,5]
    assert 12 // len(validated_gpus([0,1,2])) == 4
    for bad in ([],[0,0],[0,-1],[0,1,2,3,4],[False,1]):
        with pytest.raises(ValueError): validated_gpus(bad)
    with pytest.raises(ValueError,match='unavailable'):validated_gpus([0,1,2],{0,1})


def test_stage_commands_keep_recipe_but_fg_continues_completed_eraf():
    plan = dict(steps=200,manifest='/manifest',source_bank='/bank',strongest_checkpoint='/R',arms={
        'no_eraf':dict(eraf='off',fg='off',parent='/R',gpus=[0]),
        'eraf_only':dict(eraf='on',fg='off',parent='/Sordinary',gpus=[1,2],identity_audit='/ordinary.json'),
        'eraf_fg':dict(eraf='on',fg='full',parent='/completed_eraf',gpus=[3,4],parent_sha256='a'*64)})
    for arm,spec in plan['arms'].items():
        cmd=training_command(plan,Path('/output'),arm)
        for flag,value in [('--steps','200'),('--save-every','200'),('--cf-weight','1'),('--correct-count','0'),
                           ('--cf-count','4'),('--action-objective',OBJECTIVE),('--fg',spec['fg']),('--eraf',spec['eraf'])]:
            assert cmd[cmd.index(flag)+1]==value
        assert cmd[cmd.index('--target-tasks')+1:cmd.index('--cf-retention-tasks')]==list(FG_TASKS)
        start=cmd.index('--cf-retention-tasks')+1
        assert cmd[start:start+5]==list(TASKS)
        assert ('--warm-policy' in cmd)==(arm=='no_eraf')
        assert ('--zero-context-joint' in cmd)==(arm=='eraf_only')
        assert ('--continue-eraf-with-fg' in cmd)==(arm=='eraf_fg')


def test_preflight_counts_actual_sampler_and_balances_late_trajectory_exposure():
    report=coverage(rows(),200)
    assert report['expected_examples']==2400
    for task in TASKS:
        assert report['counts']['expert|'+task]==160
        assert report['counts']['teacher|'+task]==160
        assert report['temporal_thirds']['expert|'+task+'|2']==53
    for task in FG_TASKS:
        assert report['counts']['fg_or_ordinary|'+task]==400
        assert report['temporal_thirds']['fg_or_ordinary|'+task+'|2']==133
