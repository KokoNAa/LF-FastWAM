from copy import deepcopy
import pytest
import torch
from test_robotwin_eraf_fg_bridge import case
from experiments.robotwin.eraf_fg_bridge import predict, trainable_parameters, MasterAdamW
from experiments.robotwin.eraf_action_protocol import validate_action_parent
from experiments.robotwin.context_residual import initialize, ZEROED
import experiments.robotwin.eraf_fg_bridge as bridge


def configure(model):
    module=model.policy_guard_modules['eraf_action_context_injector']
    module.injection_mode='context_residual_v1'
    with torch.no_grad():
        module.projection[2].weight.zero_();module.projection[2].bias.zero_()
    return module


def test_zero_residual_has_exact_production_off_identity_and_learnable_gradient(case):
    model,captured,noisy,time=case;module=configure(model)
    selected=trainable_parameters(model,'interface')
    frozen={k:p.detach().clone() for k,p in model.named_parameters() if not p.requires_grad}
    with torch.no_grad():
        baseline=predict(model,captured,noisy,time,eraf=False,checkpoint=False)
        actual=predict(model,captured,noisy,time,eraf=True,checkpoint=False)
    assert torch.equal(baseline,actual)
    optimizer=MasterAdamW(selected.values(),lr=.01);optimizer.zero_grad()
    predict(model,captured,noisy,time,eraf=True,checkpoint=False).square().mean().backward()
    assert module.projection[2].weight.grad.abs().sum()>0
    optimizer.step()
    with torch.no_grad():
        changed=predict(model,captured,noisy,time,eraf=True,checkpoint=False)
        assert not torch.equal(changed,baseline)
        assert torch.equal(predict(model,captured,noisy,time,eraf=False,checkpoint=False),baseline)
    assert all(torch.equal(p,frozen[k]) for k,p in model.named_parameters() if k in frozen)


def test_residual_preserves_length_padding_and_final_state_token(case):
    module=configure(case[0]);d=module.text_dim
    context=torch.randn(1,5,d);mask=torch.tensor([[True,False,True,False,True]])
    queries=torch.randn(1,4,module.goal_dim)
    with torch.no_grad():module.projection[2].weight.normal_(0,.2)
    output,outmask,metrics=module(context=context,context_mask=mask,goal_queries=queries)
    assert output.shape==context.shape and torch.equal(outmask,mask)
    assert torch.equal(output[:,[1,3,4]],context[:,[1,3,4]])
    assert not torch.equal(output[:,0],context[:,0])
    assert metrics['pgc_v925_action_context_token_count']==0
    value,identity_mask,_=module(context=context,context_mask=mask,goal_queries=queries,external_scale=0)
    assert torch.equal(value,context) and torch.equal(identity_mask,mask)


def test_only_explicit_zero_step_residual_bootstrap_is_accepted():
    parent=dict(stage='bootstrap',optimizer_steps=0,context_injection_mode='context_residual_v1',
                provenance={'context_residual_initialization':True})
    validate_action_parent(parent,stage='interface',eraf='on',fg='full')
    for patch in ({'optimizer_steps':1},{'context_injection_mode':'append_v1'},{'provenance':{}}):
        with pytest.raises(ValueError,match='semantic checkpoint'):
            validate_action_parent(parent|patch,stage='interface',eraf='on',fg='full')


def payload_for(model):
    return dict(format=bridge.CHECKPOINT_FORMAT, protocol=bridge.PROTOCOL,
        stage='grounding', optimizer_steps=100, fg_supervision='full',
        base_checkpoint='/server/release.pt', parent_checkpoint='/server/parent.pt',
        guard_config=bridge.eraf_guard_config(),
        lora_config={'enabled': True, 'experts': ['video', 'action']},
        geometry=dict(action_dim=14, proprio_dim=14, camera_count=3,
            camera_layout='robotwin_mosaic', action_horizon=32, replan_steps=24, inference_steps=10),
        mot_trainable={'action.lora_A': torch.tensor([.4])},
        policy_guard=deepcopy(model.policy_guard_modules.state_dict()),
        provenance={'geometry_replay': 'fg'})


def test_bootstrap_changes_only_declared_outputs_and_preserves_lineage(case):
    payload=payload_for(case[0]);before=deepcopy(payload)
    result=initialize(payload,parent_path='/server/semantic100.pt',parent_sha256='a'*64)
    assert result['stage']=='bootstrap' and result['optimizer_steps']==0
    assert result['fg_supervision']=='full'
    assert result['parent_checkpoint']=='/server/semantic100.pt'
    assert result['parent_sha256']=='a'*64
    assert result['provenance']['source_optimizer_steps']==100
    assert result['provenance']['source_provenance']==before['provenance']
    for section in ('policy_guard','mot_trainable'):
        for name,tensor in before[section].items():
            assert torch.equal(payload[section][name],tensor)
            if section=='policy_guard' and name in ZEROED:
                assert not result[section][name].count_nonzero()
            else:
                assert torch.equal(result[section][name],tensor)
    with pytest.raises(ValueError,match='append-mode'):
        initialize(result,parent_path='/server/bootstrap.pt',parent_sha256='b'*64)


def test_checkpoint_loader_applies_explicit_mode_and_resets_legacy_mode(case,monkeypatch):
    model=case[0];payload=payload_for(model)
    monkeypatch.setattr(bridge,'validate_model_geometry',lambda model: None)
    monkeypatch.setattr(bridge,'restore_policy_adapter',lambda model,payload: None)
    monkeypatch.setattr(model,'load_checkpoint',lambda path: None)
    result=initialize(payload,parent_path='/server/semantic.pt',parent_sha256='a'*64)
    bridge.load_repair_checkpoint(model,'/server/bootstrap.pt',payload=result)
    module=model.policy_guard_modules['eraf_action_context_injector']
    assert module.injection_mode=='context_residual_v1'
    assert not module.projection[2].weight.count_nonzero()
    bridge.load_repair_checkpoint(model,'/server/legacy.pt',payload=payload)
    assert module.injection_mode=='append_v1'
    with pytest.raises(ValueError,match='injection mode'):
        bridge.validate_payload(payload|{'context_injection_mode':'unknown'})
