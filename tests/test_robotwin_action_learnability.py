from copy import deepcopy
import pytest
import torch
from scripts.probe_robotwin_action_learnability import training_weights,error_metrics,REGIMES,TASKS,STEPS,EVAL_SEEDS
from scripts.run_robotwin_action_learnability import compare

def training():
    return [dict(id=str(i),replay_split='train',diagnostic_task='cup',diagnostic_kind='ordinary' if i==0 else 'fg') for i in range(3)]

@pytest.mark.parametrize('regime,expected', [('initial_only',[1.,0.,0.]),('correction_only',[0.,.5,.5]),('balanced',[.5,.25,.25])])
def test_regimes_preserve_kind_mass(regime,expected):
    w=training_weights(training(),regime)
    assert list(w.values())==expected
    assert sum(w.values())==1.

@pytest.mark.parametrize('change',['holdout','task','duplicate','missing','kind'])
def test_training_rejects_invalid_or_leaked_rows(change):
    r=training()
    if change=='holdout':r[0]['replay_split']='replay_holdout'
    elif change=='task':r[0]['diagnostic_task']='pill'
    elif change=='duplicate':r[0]['id']='1'
    elif change=='missing':r.pop()
    else:r[1]['diagnostic_kind']='ordinary'
    with pytest.raises(ValueError):training_weights(r,'balanced')

def test_error_uses_executed_valid_horizon_and_reports_each_dimension():
    ref=torch.zeros(1,32,14);pred=ref.clone();valid=torch.ones(1,32,dtype=torch.bool)
    pred[:,0,6]=2;pred[:,1,13]=3;pred[:,24:]=999;pred[:,2]=999;valid[:,2]=False
    m=error_metrics(pred,ref,valid)
    assert m['mse_first24']==pytest.approx(13/(23*14))
    assert m['mse_by_action_dimension'][6]==pytest.approx(4/23)
    assert m['mse_by_action_dimension'][13]==pytest.approx(9/23)
    assert m['valid_positions']==23

@pytest.mark.parametrize('change',['nan','empty','shape'])
def test_invalid_error_inputs_rejected(change):
    p=torch.zeros(1,32,14);r=p.clone();v=torch.ones(1,32,dtype=torch.bool)
    if change=='nan':p[0,0,0]=float('nan')
    elif change=='empty':v[:]=False
    else:p=p[:,:,:13]
    with pytest.raises(ValueError):error_metrics(p,r,v)

def reports():
    result={}
    for task in TASKS:
        for regime in REGIMES:
            rows=[dict(id=str(i),replay_split='train' if i<3 else 'replay_holdout',diagnostic_kind='ordinary' if i%3==0 else 'fg') for i in range(9)]
            records=[dict(step=s,id=r['id'],noise_seed=n,split=r['replay_split'],kind=r['diagnostic_kind'],mse_first24=1. if s==0 else .5) for s in (0,16,STEPS) for r in rows for n in EVAL_SEEDS]
            result[task+'__'+regime]=dict(complete=True,optimizer_steps=STEPS,input_hashes_stable=True,optimizer_master_binding=True,
                model_files_written=False,unexpected_changes=[],changed_parameters=['mot.a'],records=records,
                protocol=dict(task=task,regime=regime,input_sha256={'checkpoint':'same'},selected_rows=rows,actual_payload_hashes={str(i):str(i) for i in range(9)}))
    return result

def test_compare_reports_fit_without_claiming_cf_or_selection():
    r=compare(reports());assert r['complete'] and not r['goal_achievement_claim'] and not r['checkpoint_selection']
    assert all(v['train__ordinary']['final_relative_change']==-.5 for v in r['pilots'].values())

@pytest.mark.parametrize('change',['pilot','row','baseline','payload','step','weight','frozen','save','binding'])
def test_compare_rejects_missing_or_invalid_evidence(change):
    d=reports();key=TASKS[0]+'__balanced';r=d[key]
    if change=='pilot':del d[key]
    elif change=='row':r['records'].pop()
    elif change=='baseline':r['records'][0]['mse_first24']=2.
    elif change=='payload':r['protocol']['actual_payload_hashes']['0']='different'
    elif change=='step':r['optimizer_steps']=16
    elif change=='weight':r['changed_parameters']=[]
    elif change=='frozen':r['unexpected_changes']=['backbone']
    elif change=='save':r['model_files_written']=True
    else:r['optimizer_master_binding']=False
    with pytest.raises(ValueError):compare(d)
