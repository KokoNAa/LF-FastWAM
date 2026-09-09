from copy import deepcopy
import pytest
import torch

from experiments.robotwin.current_noeraf import branch_parent, current_baseline
from experiments.robotwin.context_residual import ZEROED, MODE
from experiments.robotwin.eraf_action_protocol import validate_action_parent
from experiments.robotwin.eraf_fg_bridge import CHECKPOINT_FORMAT, PROTOCOL, eraf_guard_config


def components(fg='off'):
    policy = dict(format=CHECKPOINT_FORMAT, protocol=PROTOCOL,
        stage='joint', optimizer_steps=200, fg_supervision='off',
        base_checkpoint='/server/released.pt', parent_checkpoint='/server/old_R.pt',
        guard_config=eraf_guard_config(), lora_config={'enabled':True,'experts':['video','action']},
        geometry=dict(action_dim=14, proprio_dim=14, camera_count=3,
            camera_layout='robotwin_mosaic', action_horizon=32, replan_steps=24, inference_steps=10),
        mot_trainable={'mixtures.action.a.lora_A':torch.tensor([9.]),
                       'mixtures.video.a.lora_A':torch.tensor([2.])},
        policy_guard={**{k:torch.zeros(2) for k in ZEROED},'semantic':torch.tensor([0.])},
        context_injection_mode=MODE, provenance={'eraf':'off','policy_scope':'action'})
    semantic = deepcopy(policy)
    semantic.update(stage='grounding', optimizer_steps=1500, fg_supervision=fg,
                    provenance={'paired_cross_goals':True})
    semantic['mot_trainable']['mixtures.action.a.lora_A'].fill_(-1.)
    semantic['policy_guard']['semantic'].fill_(3.)
    return policy, semantic


def build(policy, semantic, fg='off'):
    return branch_parent(policy, semantic, policy_path='/server/current/no_eraf/step_000200.pt',
        policy_sha256='a'*64, semantic_path='/server/semantic1500.pt', semantic_sha256='b'*64, fg=fg)


@pytest.mark.parametrize('fg', ['off','full'])
def test_branch_uses_current_policy_not_donor_policy_and_keeps_training_history(fg):
    policy, semantic = components(fg)
    result = build(policy, semantic, fg)
    assert result['parent_checkpoint']=='/server/current/no_eraf/step_000200.pt'
    assert result['parent_sha256']=='a'*64 and result['optimizer_steps']==0
    assert result['provenance']['semantic_donor']['optimizer_steps']==1500
    assert result['provenance']['action_parent']['optimizer_steps']==200
    for key, value in policy['mot_trainable'].items():
        assert torch.equal(result['mot_trainable'][key], value)
    for key, value in semantic['policy_guard'].items():
        assert torch.equal(result['policy_guard'][key], value)
    validate_action_parent(result,stage='joint',eraf='on',fg=fg,zero_context_joint=True)
    result['mot_trainable']['mixtures.action.a.lora_A'].add_(1)
    result['policy_guard']['semantic'].add_(1)
    assert policy['mot_trainable']['mixtures.action.a.lora_A'].item()==9
    assert semantic['policy_guard']['semantic'].item()==3


def test_reject_wrong_control_nonidentity_donor_and_changed_visual_features():
    p, s = components()
    with pytest.raises(ValueError, match='no-ERAF'):
        build(p | {'provenance':{'eraf':'on'}},s)
    with pytest.raises(ValueError, match='matching'):
        build(p,s,'full')
    with pytest.raises(ValueError, match='matching'):
        build(p,s | {'stage':'joint'})
    s['policy_guard'][ZEROED[0]].fill_(1)
    with pytest.raises(ValueError, match='matching'):
        build(p,s)
    p,s=components()
    s['mot_trainable']['mixtures.video.a.lora_A'].fill_(4)
    with pytest.raises(ValueError, match='video'):
        build(p,s)


def test_control_binding_rejects_old_parent_even_with_matching_hash():
    path='/server/trial/no_eraf/joint/step_000200.pt'
    models={'no_eraf':dict(eraf='off',fg='off',parent='/server/old_R.pt',
                          final_checkpoint=path,final_sha256='a'*64)}
    audit=dict(complete=True,episodes=180,checkpoint_sha256={'no_eraf':'a'*64})
    assert current_baseline('/server/trial',models,audit)==(path,'a'*64)
    models['no_eraf']['final_checkpoint']='/server/old_R.pt'
    with pytest.raises(ValueError,match='final200'):
        current_baseline('/server/trial',models,audit)


def test_branch_commands_use_current_control_as_teacher_and_no_optimizer_resume():
    from pathlib import Path
    from scripts.run_robotwin_five_task_repair import training_command
    plan=dict(manifest='/m',source_bank='/b',strongest_checkpoint='/current/no_eraf.pt',steps=200,
        arms={a:dict(eraf='on',fg=fg,parent='/new/initialization/'+a+'.pt',gpus=[1,2],
                     identity_audit='/new/identity/'+a+'/summary.json')
              for a,fg in [('eraf_only','off'),('eraf_fg','full')]})
    for arm in plan['arms']:
        cmd=training_command(plan,Path('/new'),arm)
        for option in ('--correct-teacher','--cf-teacher'):
            assert cmd[cmd.index(option)+1]=='/current/no_eraf.pt'
        assert cmd[cmd.index('--checkpoint')+1]==plan['arms'][arm]['parent']
        assert '--zero-context-joint' in cmd and '--resume-state' not in cmd


def test_preparation_roundtrip_preserves_current_control_and_binds_both_parents(tmp_path):
    import json
    from scripts.prepare_robotwin_current_noeraf import prepare
    from scripts.run_robotwin_formal_five40 import sha
    source=tmp_path/'source'; path=source/'no_eraf/joint/step_000200.pt'
    path.parent.mkdir(parents=True)
    policy,_=components();torch.save(policy,path);digest=sha(path)
    write=lambda name,value:(source/name).write_text(json.dumps(value))
    write('final_models.json',{'no_eraf':dict(eraf='off',fg='off',final_checkpoint=str(path),final_sha256=digest)})
    write('completion_audit.json',dict(complete=True,episodes=180,checkpoint_sha256={'no_eraf':digest}))
    protocol=dict(arms={},input_sha256={})
    for arm,fg in [('eraf_only','off'),('eraf_fg','full')]:
        _,donor=components(fg);donor_path=source/(arm+'.pt');torch.save(donor,donor_path)
        protocol['arms'][arm]={'parent':str(donor_path)}
        protocol['input_sha256'][str(donor_path)]=sha(donor_path)
    write('protocol.json',protocol)
    output=tmp_path/'new/initialization'
    receipt=prepare(source,output,storage_root=tmp_path)
    assert sha(path)==digest and receipt['optimizer_updates']==0
    assert not receipt['baseline_retrained'] and not receipt['deployed_identity_verified']
    for arm in ('eraf_only','eraf_fg'):
        loaded=torch.load(receipt['arms'][arm]['path'],weights_only=False)
        assert loaded['parent_checkpoint']==str(path) and loaded['parent_sha256']==digest
        assert loaded['mot_trainable']['mixtures.action.a.lora_A'].item()==9
        assert receipt['arms'][arm]['semantic_pretraining_steps']==1500
    with pytest.raises(FileExistsError):
        prepare(source,output,storage_root=tmp_path)
    with pytest.raises(ValueError,match='server data disk'):
        prepare(source,tmp_path/'outside',storage_root=tmp_path/'allowed')
