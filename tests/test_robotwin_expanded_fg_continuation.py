import json
from pathlib import Path

import pytest

from scripts.continue_robotwin_expanded_fg_actions import validate_source
from scripts.run_robotwin_expanded_fg_trial import ARMS,action_command


def fixture(tmp_path):
    root=tmp_path/'source';root.mkdir();proc=tmp_path/'proc';proc.mkdir()
    jobs={k:dict(pid=100+i,exit_code=0) for i,k in enumerate(
        ('ordinary','fg','semantic_audit','identity_ordinary','identity_fg','terminal_ordinary','terminal_fg'))}
    driver=dict(complete=False,terminal=True,stage='stopped',jobs=jobs,
        error="ValueError('Actual observations, instructions, manifest, or labels differ.')")
    (root/'driver.json').write_text(json.dumps(driver))
    root.with_name('source-launch.json').write_text(json.dumps(dict(pid=99)))
    (root/'protocol.json').write_text(json.dumps(dict(format='robotwin_expanded_fg_matched_trial_v1')))
    return root,proc,driver


def test_only_terminal_diagnostic_failure_with_no_action_updates_can_continue(tmp_path):
    root,proc,_=fixture(tmp_path)
    assert validate_source(root,proc_root=proc)['format']=='robotwin_expanded_fg_matched_trial_v1'


@pytest.mark.parametrize('corruption',['live_controller','live_child','action_started','missing_probe','failed_probe','other_error','nonterminal'])
def test_reject_duplicate_work_and_unverified_source(tmp_path,corruption):
    root,proc,driver=fixture(tmp_path)
    if corruption in ('live_controller','live_child'):
        p=proc/('99' if corruption=='live_controller' else '100');p.mkdir();(p/'stat').write_text('99 (python) S')
    if corruption=='action_started':(root/'eraf_fg/joint').mkdir(parents=True)
    if corruption=='missing_probe':del driver['jobs']['identity_fg']
    if corruption=='failed_probe':driver['jobs']['identity_fg']['exit_code']=1
    if corruption=='other_error':driver['error']='Something else failed'
    if corruption=='nonterminal':driver['terminal']=False
    (root/'driver.json').write_text(json.dumps(driver))
    with pytest.raises(ValueError):validate_source(root,proc_root=proc)


def test_continuation_binds_existing_identity_and_new_training_output():
    plan=dict(manifest='/manifest',source_bank='/bank',correct_teacher='/correct',strongest_checkpoint='/best',
        identity_root='/completed/identity',arms={a:dict(s,parent='/completed/'+a) for a,s in ARMS.items()})
    cmd=action_command(plan,Path('/new'),'eraf_fg')
    assert cmd[cmd.index('--identity-audit')+1]=='/completed/identity/fg/summary.json'
    assert cmd[cmd.index('--checkpoint')+1]=='/completed/eraf_fg'
    assert cmd[cmd.index('--output')+1]=='/new/eraf_fg/joint'
    assert cmd[cmd.index('--steps')+1]=='200'
