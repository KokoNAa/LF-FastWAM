from copy import deepcopy
from itertools import islice
from types import SimpleNamespace
import pytest
import torch
from experiments.robotwin.balanced_target import TASKS,OLD_TASKS,OBJECTIVE,mixture_counts,mixture_stream,target_payload,backward_example,validate_recipe

def rows():
    result=[]
    def row(task,kind,seed,frame=0,split='train'):
        r=dict(id=f'{task}_{kind}_{seed}_{frame}',source_task=task,pair_id=task,task_config='demo_clean',scene_seed=seed,replay_split=split,
            source_frame_index=frame,target_frame_index=frame,frame_index=frame,initial_observations_exactly_equal=frame==0,payload='never_read_in_sampler')
        if kind=='cf':r['cf_retention']=True
        if kind=='fg':r['fg_correction']=True
        return r
    for task in OLD_TASKS:result.append(row(task,'cf',1))
    for task in TASKS:
        for seed in (10,20):
            for kind in ('ordinary','fg'):
                for frame in (0,16):result.append(row(task,kind,seed,frame))
        result.append(row(task,'ordinary',99,split='replay_holdout'))
    return result

def kind(row):
    return 'cf' if row.get('cf_retention') else 'initial' if row.get('initial_expert_anchor') else 'correction'

def test_200_steps_have_exact_exposure_and_matched_controls_without_holdout():
    data=rows();a=mixture_stream(data,42,'full');b=mixture_stream(data,42,'off');counts={t:0 for t in TASKS};correction={t:0 for t in TASKS};old={t:0 for t in OLD_TASKS}
    for _ in range(200):
        full=next(a);off=next(b)
        assert [kind(r) for r in full].count('cf')==4
        assert [kind(r) for r in full].count('initial')==4
        assert [kind(r) for r in full].count('correction')==4
        for x,y in zip(full,off):
            assert x['replay_split']==y['replay_split']=='train'
            if x.get('fg_correction'):
                assert y.get('ordinary_cf_control') and not y.get('fg_correction')
                assert (x['pair_id'],x['task_config'])==(y['pair_id'],y['task_config'])
                correction[x['source_task']]+=1
            else:assert x==y
            if x.get('initial_expert_anchor'):
                assert x['source_frame_index']==x['target_frame_index']==0
                counts[x['source_task']]+=1
            if x.get('cf_retention'):old[x['source_task']]+=1
    assert counts==dict.fromkeys(TASKS,400) and correction==dict.fromkeys(TASKS,400)
    assert old==dict.fromkeys(OLD_TASKS,160)

@pytest.mark.parametrize('correct,cf',[(2,4),(0,3),(False,4),(0,True)])
def test_reject_wrong_declared_counts(correct,cf):
    with pytest.raises(ValueError):mixture_counts(correct,cf)

@pytest.mark.parametrize('change',['missing_old','missing_fg','missing_initial','unbalanced','wrong_tasks','local'])
def test_sampler_rejects_missing_data_or_protocol(change):
    data=rows();kwargs={};fg='full'
    if change=='missing_old':data=[r for r in data if r['source_task']!=OLD_TASKS[0]]
    elif change=='missing_fg':data=[r for r in data if not (r['source_task']==TASKS[0] and r.get('fg_correction'))]
    elif change=='missing_initial':data=[r for r in data if not (r['source_task']==TASKS[0] and not r.get('fg_correction'))]
    elif change=='unbalanced':kwargs['task_balanced']=False
    elif change=='wrong_tasks':kwargs['initial_expert_tasks']=[TASKS[0]]
    else:fg='local'
    with pytest.raises(ValueError):next(mixture_stream(data,42,fg,**kwargs))

def payload():
    return dict(captured={'source':{'name':'source'},'target':{'name':'target'}},references={'source':torch.full((1,32,14),99.),'target':torch.full((1,32,14),3.)},valid={'source':torch.ones(1,32,dtype=torch.bool),'target':torch.ones(1,32,dtype=torch.bool)})

def test_real_gradient_uses_target_only_and_unit_expert_weight(monkeypatch):
    import experiments.robotwin.deployed_action_objective as deployed
    model=SimpleNamespace(torch_dtype=torch.float32,x=torch.nn.Parameter(torch.tensor(0.)))
    calls=[]
    def sample(m,c,n,**kwargs):calls.append(c['name']);return m.x.expand_as(n)
    monkeypatch.setattr(deployed,'sample',sample)
    report=backward_example(model,{'initial_expert_anchor':True},payload(),torch.zeros(1,32,14),None,teachers={})
    assert calls==['target'] and model.x.grad.item()==pytest.approx(-6.)
    assert report['deployed_objective']==9 and report['action_objective']==OBJECTIVE
    assert report['supervised_languages']==['target'] and report['conditional_difference'] is False


def test_cf_uses_same_noise_teacher_instead_of_expert_reference(monkeypatch):
    import experiments.robotwin.deployed_action_objective as deployed
    model=SimpleNamespace(torch_dtype=torch.float32,x=torch.nn.Parameter(torch.tensor(0.)))
    noise=torch.zeros(1,32,14);calls=[]
    def teacher(t,c,n):assert n is noise;calls.append(c['name']);return torch.ones_like(n)
    monkeypatch.setattr(deployed,'teacher_action',teacher)
    monkeypatch.setattr(deployed,'sample',lambda m,c,n,**kw:m.x.expand_as(n))
    report=backward_example(model,{'cf_retention':True},payload(),noise,None,teachers={'cf':object()},cf_weight=4.)
    assert calls==['target'] and report['deployed_objective']==4 and model.x.grad.item()==pytest.approx(-8.)

@pytest.mark.parametrize('row',[{'native_retention':True},{}])
def test_target_protocol_rejects_undeclared_strata(row):
    with pytest.raises(ValueError):target_payload(row,payload())

def test_source_and_valid_are_removed_without_mutating_payload():
    p=payload();result=target_payload({'fg_correction':True},p)
    assert all(set(result[k])=={'target'} for k in ('captured','references','valid'))
    assert all(set(p[k])=={'source','target'} for k in ('captured','references','valid'))

def recipe():
    return dict(stage='joint',policy_scope='action',interface_scope='all',correct_count=0,cf_count=4,task_balanced=True,disable_seen_language_augmentation=True,correction_weight=1.,correction_task_weights={},fg_gradient_route='joint',learning_rate=3e-6,interface_learning_rate=3e-5,cf_weight=4.,target_tasks=list(TASKS),initial_expert_tasks=list(TASKS),cf_retention_tasks=list(OLD_TASKS),fg='full')

@pytest.mark.parametrize('field,value',[('correct_count',2),('target_tasks',[TASKS[0]]),('fg','local'),('stage','interface'),('disable_seen_language_augmentation',False)])
def test_recipe_rejects_implicit_changes(field,value):
    p=recipe();validate_recipe(p);p[field]=value
    with pytest.raises(ValueError):validate_recipe(p)

def test_all_four_commands_bind_balanced_protocol():
    from pathlib import Path
    from scripts.run_robotwin_expanded_fg_trial import ARMS,action_command
    p=dict(manifest='/manifest',source_bank='/bank',correct_teacher='/unused',strongest_checkpoint='/R',arms={a:dict(s,parent='/S/'+a if s['eraf']=='on' else '/R') for a,s in ARMS.items()},action_objective=OBJECTIVE,initial_expert_tasks=list(TASKS),target_tasks=list(TASKS),correction_task_weights={})
    for arm in ARMS:
        cmd=action_command(p,Path('/trial'),arm)
        for flag,value in [('--correct-count','0'),('--cf-count','4'),('--steps','200'),('--save-every','200'),('--cf-weight','4'),('--learning-rate','0.000003'),('--interface-learning-rate','0.00003'),('--action-objective',OBJECTIVE)]:
            assert cmd[cmd.index(flag)+1]==value
        assert cmd[cmd.index('--target-tasks')+1:cmd.index('--target-tasks')+3]==list(TASKS)


def evidence(root):
    import json
    from scripts.run_robotwin_expanded_fg_trial import ARMS
    def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    data=rows();manifest=root/'manifest.json';write(manifest,{'states':data})
    write(root/'protocol.json',dict(manifest=str(manifest),arms=ARMS,initial_expert_tasks=list(TASKS),correction_task_weights={},action_objective=OBJECTIVE))
    for arm,spec in ARMS.items():
        p=recipe()|dict(fg=spec['fg'],steps=200,start_optimizer_step=0,global_batch=12,seed=42,world_size=len(spec['gpus']),action_objective=OBJECTIVE)
        dest=root/arm/'joint';write(dest/'plan.json',p);write(dest/'freeze_audit.json',dict(complete=True,optimizer_checkpoint_and_contract_match=True,unexpected_changes=[]))
        logs=[[] for _ in spec['gpus']];stream=mixture_stream(data,42,spec['fg'])
        for step in range(1,201):
            batch=next(stream)
            for rank in range(p['world_size']):
                logs[rank].append(dict(step=step,grad_norm=1.,examples=[dict(id=r['id'],ordinary_cf_control=bool(r.get('ordinary_cf_control')),initial_expert_anchor=bool(r.get('initial_expert_anchor')),effective_correction_weight=1.,seen_variant=None,action_objective=OBJECTIVE,denoising_steps=10,executed_horizon=24,gradient_horizon='all_ten_steps',deployed_objective=.1,supervised_languages=['target'],conditional_difference=False) for r in batch[rank::p['world_size']]]))
        for rank,log in enumerate(logs):(dest/f'rank{rank}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in log))

@pytest.mark.parametrize('corruption',[None,'sample','language','objective','anchor','step'])
def test_cross_arm_audit_accepts_only_complete_actual_balanced_schedule(tmp_path,corruption):
    import json
    from scripts.audit_robotwin_expanded_fg_trial import audit_action_pairing
    evidence(tmp_path);path=tmp_path/'eraf_fg/joint/rank0.jsonl';logs=[json.loads(x) for x in path.read_text().splitlines()]
    e=logs[0]['examples'][0]
    if corruption=='sample':e['id']='bad'
    elif corruption=='language':e['supervised_languages']=['source','target']
    elif corruption=='objective':e['action_objective']='deployed_rollout_v1'
    elif corruption=='anchor':e['initial_expert_anchor']=not e['initial_expert_anchor']
    elif corruption=='step':logs.pop()
    path.write_text(''.join(json.dumps(x)+'\n' for x in logs))
    if corruption:
        with pytest.raises(AssertionError):audit_action_pairing(tmp_path)
    else:
        r=audit_action_pairing(tmp_path)
        assert r['complete'] and r['common_examples_per_arm']==1600 and r['matched_fg_or_ordinary_slots_per_arm']==800
