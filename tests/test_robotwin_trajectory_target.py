from collections import Counter
from copy import deepcopy
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.robotwin.trajectory_target import TASKS,cycling_task_stream,mixture_stream,pools,target_payload,backward_example,OBJECTIVE
from scripts.run_robotwin_trajectory_target_trial import coverage,require_coverage,training_command
from tests.test_robotwin_balanced_target import rows,payload


def test_task_cycles_visit_every_row_before_repeating_even_with_unequal_sizes():
    data=pools(rows())[2];data=[r for r in data if not (r['source_task']==TASKS[0] and r['scene_seed']==20)]
    seen={t:[] for t in TASKS}
    for r in islice(cycling_task_stream(data,42),24):seen[r['source_task']].append(r['id'])
    for t,ids in seen.items():
        expected={r['id'] for r in data if r['source_task']==t};n=len(expected)
        for start in range(0,len(ids)-n+1,n):assert set(ids[start:start+n])==expected
    assert len(seen[TASKS[0]])==len(seen[TASKS[1]])


def test_complete_trajectory_schedule_matches_off_controls_and_never_uses_holdout():
    data=rows();original=deepcopy(data);full=mixture_stream(data,42);off=mixture_stream(data,42,'off');ids=[]
    for _ in range(800):
        a,b=next(full),next(off)
        assert len(a)==12
        assert sum(bool(r.get('ordinary_target_trajectory')) for r in a)==4
        assert sum(bool(r.get('fg_correction')) for r in a)==4
        for x,y in zip(a,b):
            assert x['replay_split']==y['replay_split']=='train'
            if x.get('fg_correction'):
                assert y.get('ordinary_cf_control') and not y.get('fg_correction')
                assert (x['pair_id'],x['task_config'])==(y['pair_id'],y['task_config'])
            else:assert x==y
        ids.extend(r['id'] for r in a)
    report=coverage(data,ids);require_coverage(report)
    assert all(v['unique_sampled']==v['rows'] for v in report.values())
    assert sum(v['exposures'] for v in report.values())==9600
    assert data==original


def test_schedule_is_independent_of_manifest_order_and_replayable():
    data=rows();a=mixture_stream(data,42);b=mixture_stream(data,42)
    assert list(islice(a,8))==list(islice(b,8))
    fg=pools(data)[2]
    assert list(islice(cycling_task_stream(fg,42),20))==list(islice(cycling_task_stream(list(reversed(fg)),42),20))

@pytest.mark.parametrize('change',['holdout','duplicate','empty'])
def test_cycles_reject_invalid_inputs(change):
    data=pools(rows())[2]
    if change=='holdout':data[0]['replay_split']='replay_holdout'
    if change=='duplicate':data.append(data[0])
    if change=='empty':data=[]
    with pytest.raises(ValueError):next(cycling_task_stream(data,42))


def test_missing_middle_trajectory_frames_are_rejected():
    data=[r for r in rows() if r.get('fg_correction') or r.get('cf_retention') or r['frame_index']==0]
    with pytest.raises(ValueError):pools(data)


def test_actual_coverage_rejects_omitted_training_states():
    data=rows();report=coverage(data,[])
    with pytest.raises(ValueError):require_coverage(report)


def test_real_ordinary_trajectory_gradient_is_target_only(monkeypatch):
    import experiments.robotwin.deployed_action_objective as deployed
    model=SimpleNamespace(torch_dtype=torch.float32,x=torch.nn.Parameter(torch.tensor(0.)))
    calls=[]
    def sample(m,c,n,**kw):calls.append(c['name']);return m.x.expand_as(n)
    monkeypatch.setattr(deployed,'sample',sample)
    result=backward_example(model,{'ordinary_target_trajectory':True},payload(),torch.zeros(1,32,14),None,teachers={})
    assert calls==['target'] and model.x.grad.item()==pytest.approx(-6.)
    assert result['action_objective']==OBJECTIVE and result['deployed_objective']==9
    assert result['supervised_languages']==['target'] and result['conditional_difference'] is False


def test_teacher_still_retains_same_noise_old_cf(monkeypatch):
    import experiments.robotwin.deployed_action_objective as deployed
    model=SimpleNamespace(torch_dtype=torch.float32,x=torch.nn.Parameter(torch.tensor(0.)))
    noise=torch.zeros(1,32,14)
    def teacher(t,c,n):assert n is noise;return torch.ones_like(n)
    monkeypatch.setattr(deployed,'teacher_action',teacher);monkeypatch.setattr(deployed,'sample',lambda m,c,n,**kw:m.x.expand_as(n))
    result=backward_example(model,{'cf_retention':True},payload(),noise,None,teachers={'cf':object()})
    assert model.x.grad.item()==pytest.approx(-8.) and result['deployed_objective']==4


def test_candidate_command_preserves_parent_lr_identity_and_records800six_ranks():
    cmd=['python','-m','torch.distributed.run','--standalone','--nproc_per_node','2','/old/train.py','--steps','200','--save-every','200','--action-objective','balanced_target_rollout_v1','--output','/old','--checkpoint','/S','--learning-rate','0.000003','--identity-audit','/identity']
    got=training_command(cmd,Path('/new'))
    for flag,v in [('--steps','800'),('--save-every','800'),('--nproc_per_node','6'),('--checkpoint','/S'),('--learning-rate','0.000003'),('--identity-audit','/identity'),('--action-objective',OBJECTIVE)]:assert got[got.index(flag)+1]==v
    assert cmd[cmd.index('--steps')+1]=='200'

@pytest.mark.parametrize('r',[{}, {'native_retention':True},{'initial_expert_anchor':True}])
def test_new_protocol_does_not_silently_accept_old_strata(r):
    with pytest.raises(ValueError):target_payload(r,payload())
