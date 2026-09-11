import numpy as np
import pytest
from types import SimpleNamespace
from experiments.robotwin.language_wm_causal import layer_groups, stage_active, select_states, generation_conditions


def test_layers_and_sampling_stages_cover_once():
    groups=layer_groups()
    assert groups['early']+groups['middle']+groups['late']==list(range(30))
    assert all(sum(stage_active(s,i) for s in ['early','middle','late'])==1 for i in range(20))
    assert [i for i in range(20) if stage_active('middle',i)]==list(range(6,13))
    conditions=generation_conditions()
    assert len(conditions)==27 and len({x['name'] for x in conditions})==27
    with pytest.raises(ValueError): layer_groups(29)
    with pytest.raises(ValueError): stage_active('early',20)


def test_selection_uses_catalog_and_common_decision_not_goal_outcome():
    old=dict(tasks=['place_a2b_left','stack_blocks_two'],scenes=[],states=[])
    for task in old['tasks']:
        for seed in [7,4,9]:
            old['scenes'].append(dict(task=task,scene_seed=seed))
            for phase in ['initial','shared_decision','target_late']:
                old['states'].append(dict(id=f'{task}-{seed}-{phase}',task=task,scene_seed=seed,
                                          phase=phase,dual_reference_valid=phase!='target_late'))
    rows=select_states(old,2)
    assert [r['scene_seed'] for r in rows]==[7,4,7,4]
    assert [r['phase'] for r in rows]==['shared_decision']*2+['initial']*2
    rows[0]['dual_reference_valid']=False
    with pytest.raises(ValueError): select_states(old,2)


def test_torch_residual_interchange_and_text_mask(monkeypatch):
    torch=pytest.importorskip('torch')
    import experiments.robotwin.language_wm_causal as m
    class Cross(torch.nn.Module):
        def forward(self,x,ctx,ctx_mask=None):
            weights=torch.softmax((x@ctx.transpose(-1,-2)).masked_fill(~ctx_mask[:,None],-1e9),-1)
            return weights@ctx
    blocks=[SimpleNamespace(cross_attn=Cross()) for _ in range(3)]
    model=SimpleNamespace(video_expert=SimpleNamespace(blocks=blocks))
    def tiny(model,latent,t,ctx,mask):
        x=latent.clone()
        for block in model.video_expert.blocks:x=torch.tanh(x)+block.cross_attn(torch.tanh(x),ctx,ctx_mask=mask)
        return x
    monkeypatch.setattr(m,'video_velocity',tiny)
    x=torch.tensor([[[.1,.2],[.3,-.2]]]); t=torch.tensor([1.])
    source=torch.tensor([[[1.,0.],[0.,1.],[.1,.4]]]); target=source.clone();target[:,0]=torch.tensor([-1.,.5])
    mask=torch.ones(1,3,dtype=torch.bool)
    a,act_a=m.causal_velocity(model,x,t,source,mask,capture=True)
    b,act_b=m.causal_velocity(model,x,t,target,mask,capture=True)
    assert not torch.equal(a,b)
    sham,_=m.causal_velocity(model,x,t,source,mask,mode='patch',layers=range(3),donor=act_a)
    full,_=m.causal_velocity(model,x,t,source,mask,mode='patch',layers=range(3),donor=act_b)
    assert torch.equal(sham,a) and torch.equal(full,b)
    no_a,_=m.causal_velocity(model,x,t,source,mask,mode='mask_text',layers=range(3))
    no_b,_=m.causal_velocity(model,x,t,target,mask,mode='mask_text',layers=range(3))
    assert torch.equal(no_a,no_b) and not torch.equal(no_a,a)
    assert mask.all() and all('forward' not in vars(b.cross_attn) for b in blocks)
    # Different proprio remains observable even when all text positions are masked.
    changed=source.clone();changed[:,-1]*=2
    no_c,_=m.causal_velocity(model,x,t,changed,mask,mode='mask_text',layers=range(3))
    assert not torch.equal(no_a,no_c)
    wrong={0:torch.zeros(1,1,2)}
    with pytest.raises(ValueError):m.causal_velocity(model,x,t,source,mask,mode='patch',layers=[0],donor=wrong)
    assert all('forward' not in vars(b.cross_attn) for b in blocks)
