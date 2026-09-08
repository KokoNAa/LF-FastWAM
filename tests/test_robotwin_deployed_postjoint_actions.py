from copy import deepcopy
import pytest
from scripts.probe_robotwin_deployed_postjoint_actions import eligible


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


def reports():
    keys=('id','task','kind','split','scene_seed','frame','raw_path','source_instruction',
          'counterfactual_instruction','state_sha256','rgb_sha256','reference_sha256','valid_sha256')
    rows=[]
    for i in range(18):
        row={k:str(i) for k in keys}
        row.update(repeat_identical=True,task='cup',kind='ordinary' if i%2 else 'fg',
                   split='train' if i<6 else 'replay_holdout',
                   deployed_cf_mse_first24=2.,flow_velocity_mse={'endpoint_action_mse_first24':3.})
        rows.append(row)
    return {'complete':True,'records':rows,'input_hashes_stable':True,'protocol':dict(inference_steps=10,noise_seed=42,optimizer_updates=0,manifest_sha256='same')}

def test_action_comparison_uses_actual_matched_observations_and_all_rows():
    from scripts.probe_robotwin_deployed_postjoint_actions import compare_actions
    previous=reports();current=deepcopy(previous)
    for r in current['records']:r['deployed_cf_mse_first24']=1.
    result=compare_actions(previous,current)
    assert result['actual_inputs_equal']
    assert sum(v['current']['observations'] for v in result['groups'].values())==18
    assert all(v['current']['mean_deployed_cf_mse_first24']==1 for v in result['groups'].values())

@pytest.mark.parametrize('change',['incomplete','missing','rgb','reference','instruction'])
def test_action_comparison_rejects_incomplete_or_changed_input(change):
    from scripts.probe_robotwin_deployed_postjoint_actions import compare_actions
    previous=reports();current=deepcopy(previous)
    if change=='incomplete':current['complete']=False
    elif change=='missing':current['records'].pop()
    else:
        key={'rgb':'rgb_sha256','reference':'reference_sha256','instruction':'counterfactual_instruction'}[change]
        current['records'][0][key]='changed'
    with pytest.raises(ValueError):compare_actions(previous,current)

@pytest.mark.parametrize('key,value',[('inference_steps',20),('noise_seed',43),('optimizer_updates',1),('manifest_sha256','other')])
def test_action_comparison_rejects_protocol_changes(key,value):
    from scripts.probe_robotwin_deployed_postjoint_actions import compare_actions
    previous=reports();current=deepcopy(previous);current['protocol'][key]=value
    with pytest.raises(ValueError):compare_actions(previous,current)
