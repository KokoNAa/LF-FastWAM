from collections import Counter
import pytest
from experiments.robotwin.initial_anchor import initial_rows, effective_weight, validate_weights
from experiments.robotwin.eraf_fg_training import mixture_stream

TASKS=['place_empty_cup','move_pillbottle_pad']


def rows():
    result=[]
    for task in TASKS:
        for kind in ('native_retention','cf_retention','expert','fg_correction'):
            for split,seed in [('train',1),('replay_holdout',2)]:
                for frame in (0,16):
                    result.append(dict(id=f'{task}/{kind}/{split}/{frame}',pair_id=task,source_task=task,
                        scene_seed=seed,task_config='demo_clean',replay_split=split,frame_index=frame,
                        initial_observations_exactly_equal=frame==0,**{kind:True}))
    return result


def test_initial_stratum_preserves_all_retention_and_fg_draws_and_ablation_pairing():
    data=rows();kwargs=dict(task_balanced=True,correct_count=2,cf_count=4)
    old=mixture_stream(data,42,**kwargs)
    new=mixture_stream(data,42,initial_expert_tasks=TASKS,**kwargs)
    off=mixture_stream(data,42,'off',initial_expert_tasks=TASKS,**kwargs)
    count=Counter()
    for _ in range(200):
        a,b,c=next(old),next(new),next(off)
        assert len(b)==12
        anchors=[r for r in b if r.get('initial_expert_anchor')]
        assert len(anchors)==1
        r=anchors[0];assert r['frame_index']==0 and r['replay_split']=='train' and r['expert']
        count[r['source_task']]+=1
        for before,after,replacement in zip(a,b,c,strict=True):
            if not after.get('initial_expert_anchor'):assert before==after
            if after.get('fg_correction'):
                assert replacement.get('ordinary_cf_control') and not replacement.get('fg_correction')
                assert after['source_task']==replacement['source_task']
            else:assert after==replacement
    assert count==dict.fromkeys(TASKS,100)


@pytest.mark.parametrize('change',['holdout','fg','mismatch','later'])
def test_initial_selection_rejects_missing_proven_initial_pair(change):
    data=rows()
    for row in data:
        if row.get('expert') and row['source_task']==TASKS[0] and row['frame_index']==0:
            if change=='holdout':row['replay_split']='replay_holdout'
            if change=='fg':row['fg_correction']=True
            if change=='mismatch':row['initial_observations_exactly_equal']=False
            if change=='later':row['target_frame_index']=16
    with pytest.raises(ValueError,match='audited same-state'):initial_rows(data,TASKS)


@pytest.mark.parametrize('weights',[{'typo':1},{'place_empty_cup':0},{'place_empty_cup':float('nan')},{'place_empty_cup':True},[]])
def test_weight_contract_rejects_silent_invalid_configuration(weights):
    with pytest.raises(ValueError):validate_weights(weights)


def test_old_task_weights_apply_equally_to_corrective_and_matched_ordinary_only():
    weights=validate_weights({'place_a2b_left':.2,'blocks_ranking_rgb':.2})
    for task in ['place_a2b_left','blocks_ranking_rgb','place_empty_cup','move_pillbottle_pad']:
        for flag in ['fg_correction','ordinary_cf_control']:
            assert effective_weight(dict(source_task=task,**{flag:True}),1,weights)==(.2 if task in weights else 1.)
        assert effective_weight(dict(source_task=task,native_retention=True),1,weights)==1.


@pytest.mark.parametrize('corruption',[None,'weight','anchor'])
def test_actual_cross_arm_audit_checks_executed_weights_and_anchor_flags(tmp_path,corruption):
    import json
    from scripts.run_robotwin_expanded_fg_trial import ARMS
    from scripts.audit_robotwin_expanded_fg_trial import audit_action_pairing
    weights={'place_a2b_left':.2,'blocks_ranking_rgb':.2}
    def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    data=rows();manifest=tmp_path/'manifest.json';write(manifest,dict(states=data))
    write(tmp_path/'protocol.json',dict(manifest=str(manifest),arms=ARMS,initial_expert_tasks=TASKS,correction_task_weights=weights))
    for arm,spec in ARMS.items():
        dest=tmp_path/arm/'joint';world=len(spec['gpus'])
        write(dest/'plan.json',dict(steps=200,start_optimizer_step=0,global_batch=12,seed=42,
            correction_weight=1,task_balanced=True,correct_count=2,cf_count=4,
            disable_seen_language_augmentation=True,world_size=world,fg=spec['fg'],
            initial_expert_tasks=TASKS,correction_task_weights=weights))
        write(dest/'freeze_audit.json',dict(complete=True,optimizer_checkpoint_and_contract_match=True,unexpected_changes=[]))
        stream=mixture_stream(data,42,spec['fg'],task_balanced=True,correct_count=2,cf_count=4,initial_expert_tasks=TASKS)
        logs=[[] for _ in range(world)]
        for step in range(1,201):
            batch=next(stream)
            for rank in range(world):
                examples=[dict(id=r['id'],ordinary_cf_control=bool(r.get('ordinary_cf_control')),
                    initial_expert_anchor=bool(r.get('initial_expert_anchor')),
                    effective_correction_weight=effective_weight(r,1,weights),seen_variant=None,flow_target=.2,endpoint_objective=.3)
                    for r in batch[rank::world]]
                if arm=='eraf_fg' and step==1 and rank==0:
                    if corruption=='weight':examples[0]['effective_correction_weight']=7
                    if corruption=='anchor':examples[0]['initial_expert_anchor']=not examples[0]['initial_expert_anchor']
                logs[rank].append(dict(step=step,grad_norm=1.,examples=examples))
        for rank,log in enumerate(logs):(dest/f'rank{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in log))
    if corruption:
        with pytest.raises(AssertionError):audit_action_pairing(tmp_path)
    else:
        result=audit_action_pairing(tmp_path)
        assert result['complete'] and result['counts']['eraf_fg']['initial_expert_anchor|place_empty_cup']==100
