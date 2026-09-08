import json
from pathlib import Path
import pytest

from experiments.robotwin.trajectory_controls import (
    SPECS, command, common_command, audit_arm, audit_pairing, comparison_config)
from experiments.robotwin.trajectory_target import mixture_stream
from tests.test_robotwin_balanced_target import rows


def journals(root, data, fg, steps=800):
    root.mkdir(parents=True)
    plan=dict(steps=800,world_size=6,global_batch=12,seed=42,start_optimizer_step=0,
              fg=fg,correct_count=0,cf_count=4,action_objective='trajectory_target_rollout_v1')
    (root/'plan.json').write_text(json.dumps(plan))
    files=[(root/f'rank{i}.jsonl').open('w') for i in range(6)]
    stream=mixture_stream(data,42,fg)
    try:
        for i in range(steps):
            batch=next(stream)
            for rank,f in enumerate(files):
                examples=[]
                for row in batch[rank::6]:
                    examples.append(dict(id=row['id'],seen_variant=None,supervised_languages=['target'],conditional_difference=False,
                        action_objective='trajectory_target_rollout_v1',denoising_steps=10,executed_horizon=24,
                        gradient_horizon='all_ten_steps',deployed_objective=.01,
                        **{flag:bool(row.get(flag)) for flag in ['ordinary_target_trajectory','ordinary_cf_control','initial_expert_anchor']}))
                f.write(json.dumps(dict(step=i+1,grad_norm=.1,examples=examples))+'\n')
    finally:
        for f in files:f.close()


def test_actual_four_arm_audit_covers800steps_and_matched_replacements(tmp_path):
    data=rows();joints={a:tmp_path/a for a in [*SPECS,'eraf_fg']}
    for a,p in joints.items():journals(p,data,'full' if a in ('fg_only','eraf_fg') else 'off')
    r=audit_pairing(joints,data)
    assert r['actual_examples_all_arms']==38400 and r['common_examples_per_arm']==6400
    assert r['matched_fg_or_ordinary_slots_per_arm']==3200
    assert r['arms']['fg_only']['counts']=={'common':6400,'fg':3200}
    assert r['arms']['no_eraf']['counts']=={'common':6400,'replacement':3200}
    assert all(v['exposures']==0 for k,v in r['arms']['no_eraf']['coverage'].items() if k.endswith('|fg'))


@pytest.mark.parametrize('mutation',['id','stratum','source_language','nonfinite','missing_step','world'])
def test_audit_rejects_actual_runtime_divergence(tmp_path,mutation):
    data=rows();joint=tmp_path/'joint';journals(joint,data,'off')
    p=joint/'rank0.jsonl';items=[json.loads(x) for x in p.read_text().splitlines()]
    if mutation=='id':items[0]['examples'][0]['id']='wrong'
    if mutation=='stratum':items[0]['examples'][0]['ordinary_cf_control']=not items[0]['examples'][0]['ordinary_cf_control']
    if mutation=='source_language':items[0]['examples'][0]['supervised_languages']=['source','target']
    if mutation=='nonfinite':items[0]['grad_norm']=float('nan')
    if mutation=='missing_step':items.pop()
    if mutation=='world':
        q=joint/'plan.json';plan=json.loads(q.read_text());plan['world_size']=2;q.write_text(json.dumps(plan))
    p.write_text('\n'.join(json.dumps(x) for x in items)+'\n')
    with pytest.raises(ValueError):audit_arm(joint,data,'off')


def test_first_step_audit_accepts_live_prefix_but_not_missing_rank(tmp_path):
    data=rows();joint=tmp_path/'joint';journals(joint,data,'full',steps=2)
    assert audit_arm(joint,data,'full',1)['actual_examples']==12
    (joint/'rank4.jsonl').write_text('')
    with pytest.raises(ValueError):audit_arm(joint,data,'full',1)


@pytest.mark.parametrize('arm',list(SPECS))
def test_control_command_preserves_same_optimizer_recipe_and_own_lineage(arm):
    spec=SPECS[arm]
    old=['python','-m','torch.distributed.run','--standalone','--nproc_per_node','1','/old/train.py',
         '--checkpoint','/parent/'+arm,'--output','/old','--eraf',spec['eraf'],'--fg',spec['fg'],
         '--steps','200','--save-every','200','--learning-rate','0.000003',
         '--interface-learning-rate','0.00003','--action-objective','balanced_target_rollout_v1']
    c=command(old,Path('/new'),arm,Path('/repo'))
    for flag,value in [('--nproc_per_node','6'),('--steps','800'),('--save-every','800'),('--checkpoint','/parent/'+arm),
                       ('--output','/new/'+arm+'/joint'),('--learning-rate','0.000003')]:
        assert c[c.index(flag)+1]==value
    other=list(c);other[other.index('--learning-rate')+1]='0.00003'
    assert common_command(c)!=common_command(other)
    assert old[old.index('--steps')+1]=='200'


def test_comparison_keeps_full_candidate_and_every_historical_control():
    old={'methods':{a:{'original_five':'/'+a} for a in [*SPECS,'eraf_fg','historical_strongest_no_eraf']}}
    new=comparison_config(old,Path('/new'),['original_five','additional_five'])
    assert new['methods']['eraf_fg']==old['methods']['eraf_fg']
    assert len(new['methods'])==len(old['methods'])+3
    for a in SPECS:
        assert new['methods']['pre_trajectory_'+a]==old['methods'][a]
        assert new['methods'][a]['additional_five']=='/new/eval_additional_five/'+a+'/dev'
    with pytest.raises(ValueError):comparison_config(new,Path('/next'),['original_five'])
