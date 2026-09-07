from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_robotwin_expanded_fg_trial import ARMS, TARGETS, action_command, comparison_config, semantic_command, source_process


def test_commands_keep_common_policy_and_equal_corrective_coefficient():
    plan=dict(manifest='/new/manifest.json',source_bank='/source',correct_teacher='/correct',
        strongest_checkpoint='/strongest',masks='/new/masks/complete.json',label_cache='/labels',
        semantic_parents={m:'/old/'+m for m in ('ordinary','fg')},
        arms={a:dict(s,parent='/strongest' if s['eraf']=='off' else '/semantic/'+a) for a,s in ARMS.items()})
    root=Path('/trial')
    for arm,spec in plan['arms'].items():
        cmd=action_command(plan,root,arm)
        def value(flag):return cmd[cmd.index(flag)+1]
        assert value('--checkpoint')==spec['parent']
        assert value('--manifest')==plan['manifest']
        assert value('--correction-weight')=='1'
        assert value('--steps')=='200' and value('--cf-count')=='4'
        assert value('--nproc_per_node')==str(len(spec['gpus']))
        assert cmd[cmd.index('--target-tasks')+1:][:len(TARGETS)]==list(TARGETS)
        assert ('--warm-policy' in cmd)==(spec['eraf']=='off')
        assert ('--zero-context-joint' in cmd)==(spec['eraf']=='on')
        assert ('--identity-audit' in cmd)==(spec['eraf']=='on')
    for mode in ('ordinary','fg'):
        cmd=semantic_command(plan,root,mode)
        assert cmd[cmd.index('--checkpoint')+1]==plan['semantic_parents'][mode]
        assert cmd[cmd.index('--step-offset')+1]=='1000'
        assert cmd[cmd.index('--steps')+1]=='500'
        assert cmd[cmd.index('--geometry-replay')+1]==mode
        assert '--paired-cross-goals' in cmd


def test_comparison_retains_every_actual_prior_control_and_rejects_alias_collision():
    previous=dict(methods={a:{'original_five':a+'/old','additional_five':a+'/new'} for a in ARMS})
    previous['methods']['strongest']={'original_five':'best/old','additional_five':'best/new'}
    saved=deepcopy(previous)
    result=comparison_config(previous,Path('/trial'),('original_five','additional_five'))
    assert previous==saved and len(result['methods'])==9
    for arm in ARMS:assert result['methods']['pre_expanded_fg_'+arm]==previous['methods'][arm]
    assert result['methods']['strongest']==previous['methods']['strongest']
    with pytest.raises(ValueError):comparison_config(result,Path('/another'),('original_five','additional_five'))


def test_wait_uses_actual_process_identity(tmp_path):
    proc=tmp_path/'123';proc.mkdir()
    fields=['S']+['0']*18+['6789']
    (proc/'stat').write_text('123 (python) '+' '.join(fields))
    (proc/'cmdline').write_bytes(b'python\0/repo/run_robotwin_expanded_fg_preparation.py\0--output\0/runs/source\0')
    assert source_process(123,Path('/runs/source'),proc_root=tmp_path)=='6789'
    with pytest.raises(ValueError):source_process(123,Path('/runs/other'),proc_root=tmp_path)
    fields[0]='Z';(proc/'stat').write_text('123 (python) '+' '.join(fields))
    assert source_process(123,Path('/runs/source'),proc_root=tmp_path) is None
    assert source_process(999,Path('/runs/source'),proc_root=tmp_path) is None


def action_evidence(root):
    from experiments.robotwin.eraf_fg_training import mixture_stream
    def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj))
    rows=[]
    for task in ('left','rgb','cup','pill'):
        for kind in ('native_retention','cf_retention','expert','fg_correction'):
            rows.append(dict(id=task+'_'+kind,pair_id=task,task_config='clean',scene_seed=1,
                replay_split='train',frame_index=0,**{kind:True}))
    manifest=root/'manifest.json';write(manifest,dict(states=rows))
    write(root/'protocol.json',dict(manifest=str(manifest),arms=ARMS))
    for arm,spec in ARMS.items():
        dest=root/arm/'joint';world=len(spec['gpus'])
        write(dest/'plan.json',dict(steps=200,start_optimizer_step=0,global_batch=12,seed=42,
            correction_weight=1,task_balanced=True,correct_count=2,cf_count=4,
            disable_seen_language_augmentation=True,world_size=world,fg=spec['fg']))
        write(dest/'freeze_audit.json',dict(complete=True,optimizer_checkpoint_and_contract_match=True,unexpected_changes=[]))
        logs=[[] for _ in range(world)]
        stream=mixture_stream(rows,42,spec['fg'],task_balanced=True,correct_count=2,cf_count=4)
        for step in range(1,201):
            batch=next(stream)
            for rank in range(world):
                logs[rank].append(dict(step=step,grad_norm=1.,examples=[dict(id=r['id'],
                    ordinary_cf_control=bool(r.get('ordinary_cf_control')),seen_variant=None,
                    flow_target=.2,endpoint_objective=.3) for r in batch[rank::world]]))
        for rank,log in enumerate(logs):(dest/f'rank{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in log))


@pytest.mark.parametrize('corruption',[None,'sample','norm','missing_step','control','augmentation','nonfinite'])
def test_actual_audit_rejects_corrupted_samples_steps_and_losses(tmp_path,corruption):
    from scripts.audit_robotwin_expanded_fg_trial import audit_action_pairing
    action_evidence(tmp_path)
    path=tmp_path/'eraf_fg/joint/rank0.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    if corruption=='sample':rows[0]['examples'][0]['id']='wrong'
    if corruption=='norm':rows[0]['grad_norm']=2.
    if corruption=='missing_step':rows.pop()
    if corruption=='control':rows[0]['examples'][0]['ordinary_cf_control']=True
    if corruption=='augmentation':rows[0]['examples'][0]['seen_variant']='changed'
    if corruption=='nonfinite':rows[0]['examples'][0]['flow_target']=float('nan')
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if corruption:
        with pytest.raises(AssertionError):audit_action_pairing(tmp_path)
    else:
        result=audit_action_pairing(tmp_path)
        assert result['complete'] and result['common_examples_per_arm']==1800
        assert result['matched_fg_or_ordinary_slots_per_arm']==600
