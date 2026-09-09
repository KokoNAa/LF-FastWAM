import json
from pathlib import Path
import pytest
import torch

from experiments.robotwin.eraf_fg_bridge import CHECKPOINT_FORMAT, PROTOCOL, eraf_guard_config
from experiments.robotwin.eraf_fg_training import mixture_stream
from scripts.audit_robotwin_eraf_fg_action_stage import audit


def metadata(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def test_metadata_audit_binds_actual_weights_and_cf_priority_sample_schedule(tmp_path, monkeypatch):
    from experiments.robotwin import eraf_fg_bridge
    monkeypatch.setattr(eraf_fg_bridge, 'file_sha256', lambda path: pytest.fail('Unexpected hash scan'))
    base = dict(format=CHECKPOINT_FORMAT, protocol=PROTOCOL, stage='joint', optimizer_steps=200,
        fg_supervision='off', guard_config=eraf_guard_config(),
        geometry=dict(action_dim=14, proprio_dim=14, camera_count=3, camera_layout='robotwin_mosaic',
                      action_horizon=32, replan_steps=24, inference_steps=10),
        lora_config=dict(enabled=True, rank=16, experts=['video', 'action']),
        mot_trainable={'fixed': torch.tensor([1.])}, policy_guard={'interface': torch.tensor([0.])},
        provenance={'eraf': 'off'})
    parent = tmp_path / 'parent.pt'; torch.save(base, parent)
    root = tmp_path / 'stage'; root.mkdir()
    rows = [dict(id=f'{kind}_{task}', pair_id=task, task_config='clean', scene_seed=1,
                 frame_index=0, replay_split='train', **{kind: True})
            for kind in ('native_retention', 'cf_retention', 'pair', 'fg_correction')
            for task in ('left', 'rank')]
    manifest = tmp_path / 'manifest.json'; manifest.write_text(json.dumps({'states': rows}))
    contract = dict(manifest_identity=metadata(manifest))
    plan = dict(checkpoint=str(parent), steps=1, stage='interface', fg='full', eraf='on', skip_file_hashes=True,
        trainable_parameters=['guard.interface'], optimization_contract=contract,
        manifest=str(manifest), world_size=2, start_optimizer_step=0, global_batch=12,
        seed=42, task_balanced=True, correct_count=2, cf_count=4)
    (root / 'plan.json').write_text(json.dumps(plan))
    (root / 'complete.json').write_text(json.dumps({'complete': True}))
    candidate = base | dict(stage='interface', optimizer_steps=1, fg_supervision='full',
        parent_checkpoint=str(parent), provenance={'eraf': 'on'}, policy_guard={'interface': torch.tensor([.2])})
    checkpoint = root / 'step_000001.pt'; torch.save(candidate, checkpoint)
    companion = dict(checkpoint_identity=metadata(checkpoint), step=1,
        parameter_names=['guard.interface'], optimization_contract=contract,
        optimizer={'master': [torch.tensor([.2])]})
    torch.save(companion, root / 'optimizer_last.pt')
    batch = next(mixture_stream(rows, 42, 'full', task_balanced=True, correct_count=2, cf_count=4))
    for rank in range(2):
        (root / f'rank{rank}.jsonl').write_text(json.dumps(dict(step=1, grad_norm=.2,
            examples=[{'id': r['id']} for r in batch[rank::2]])) + '\n')
    report = audit(root)
    assert report['complete'] and not report['hash_scans']
    assert report['changed_lora_tensors'] == 0 and report['changed_guard_tensors'] == 1
    companion['optimizer']['master'][0] += 1
    torch.save(companion, root / 'optimizer_last.pt')
    with pytest.raises(ValueError, match='companion'):
        audit(root)
    candidate['mot_trainable'] = {'fixed': torch.tensor([2.])}
    torch.save(candidate, checkpoint)
    with pytest.raises(ValueError, match='Unexpected'):
        audit(root)


def test_fg_stage_audit_requires_completed_eraf_hash_and_unreset_initialization(tmp_path):
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    parent_payload=dict(format=CHECKPOINT_FORMAT,protocol=PROTOCOL,stage='joint',optimizer_steps=200,
        fg_supervision='off',guard_config=eraf_guard_config(),
        geometry=dict(action_dim=14,proprio_dim=14,camera_count=3,camera_layout='robotwin_mosaic',
                      action_horizon=32,replan_steps=24,inference_steps=10),
        lora_config=dict(enabled=True,rank=16,experts=['video','action']),
        mot_trainable={'fixed':torch.tensor([1.])},policy_guard={'interface':torch.tensor([.5])},
        provenance={'eraf':'on'})
    parent=tmp_path/'eraf200.pt';torch.save(parent_payload,parent);parent_sha=file_sha256(parent)
    root=tmp_path/'fg';root.mkdir()
    rows=[dict(id=f'{kind}_{task}',pair_id=task,task_config='clean',scene_seed=1,
               frame_index=0,replay_split='train',**{kind:True})
          for kind in ('native_retention','cf_retention','pair','fg_correction') for task in ('left','rank')]
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'states':rows}))
    contract=dict(manifest_identity=metadata(manifest))
    plan=dict(checkpoint=str(parent),steps=1,stage='joint',fg='full',eraf='on',skip_file_hashes=True,
        continue_eraf_with_fg=True,continuation_parent_sha256=parent_sha,
        trainable_parameters=['mot.fixed','guard.interface'],optimization_contract=contract,
        manifest=str(manifest),world_size=1,start_optimizer_step=0,global_batch=12,
        seed=42,task_balanced=True,correct_count=2,cf_count=4)
    (root/'plan.json').write_text(json.dumps(plan));(root/'complete.json').write_text('{"complete":true}')
    candidate=parent_payload|dict(optimizer_steps=1,fg_supervision='full',parent_checkpoint=str(parent),
        mot_trainable={'fixed':torch.tensor([1.1])},policy_guard={'interface':torch.tensor([.6])},
        provenance=dict(eraf='on',continue_eraf_with_fg=True,continuation_parent_sha256=parent_sha,
                        continuation_parent_optimizer_steps=200))
    checkpoint=root/'step_000001.pt';torch.save(candidate,checkpoint)
    torch.save(dict(checkpoint_identity=metadata(checkpoint),step=1,
        parameter_names=plan['trainable_parameters'],optimization_contract=contract,
        optimizer={'master':[torch.tensor([1.1]),torch.tensor([.6])]}),root/'optimizer_last.pt')
    batch=next(mixture_stream(rows,42,'full',task_balanced=True,correct_count=2,cf_count=4))
    (root/'rank0.jsonl').write_text(json.dumps(dict(step=1,grad_norm=.2,examples=[{'id':r['id']} for r in batch]))+'\n')
    initial=dict(complete=True,parent_checkpoint=str(parent),parent_sha256=parent_sha,
        parent_optimizer_steps=200,all_policy_and_eraf_tensors_equal=True,optimizer_updates=0,
        eraf_reinitialized=False,optimizer_restarted_for_fg=True)
    proof=root/'continuation_initialization.json';proof.write_text(json.dumps(initial))
    assert audit(root)['fg_continuation_parent_and_initialization_verified']
    for patch in ({'parent_sha256':'0'*64},{'eraf_reinitialized':True},{'all_policy_and_eraf_tensors_equal':False}):
        proof.write_text(json.dumps(initial|patch))
        with pytest.raises(ValueError,match='initialization evidence'):
            audit(root)
