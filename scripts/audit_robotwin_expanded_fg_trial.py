"""Cross-arm actual sample audit for the expanded FG four-arm action trial."""
from collections import Counter
import json
import math
from pathlib import Path


def audit_action_pairing(root):
    from experiments.robotwin.eraf_fg_training import mixture_stream
    from experiments.robotwin.initial_anchor import effective_weight
    root=Path(root);read=lambda p:json.loads(Path(p).read_text())
    protocol=read(root/'protocol.json');rows=read(protocol['manifest'])['states']
    plans={a:read(root/a/'joint/plan.json') for a in protocol['arms']}
    balanced=protocol.get('action_objective')=='balanced_target_rollout_v1'
    if balanced:
        from experiments.robotwin.balanced_target import mixture_stream, validate_recipe
        for p in plans.values():validate_recipe(p)
    streams={};journals={};counts={a:Counter() for a in plans}
    for a,p in plans.items():
        assert (p['steps']==200 and p['start_optimizer_step']==0 and p['global_batch']==12
                and p['seed']==42 and p['correction_weight']==1 and p['task_balanced']
                and p['correct_count']==(0 if balanced else 2) and p['cf_count']==4 and p['disable_seen_language_augmentation'])
        report=read(root/a/'joint/freeze_audit.json')
        assert report['complete'] and report['optimizer_checkpoint_and_contract_match'] and not report['unexpected_changes']
        journals[a]=[[read_line for read_line in map(json.loads,(root/a/'joint'/f'rank{rank}.jsonl').read_text().splitlines())]
                     for rank in range(p['world_size'])]
        assert all([r['step'] for r in j]==list(range(1,201)) for j in journals[a])
        assert p.get('initial_expert_tasks',[])==protocol.get('initial_expert_tasks',[])
        assert p.get('correction_task_weights',{})==protocol.get('correction_task_weights',{})
        assert p.get('action_objective','flow_endpoint_v1')==protocol.get('action_objective','flow_endpoint_v1')
        streams[a]=mixture_stream(rows,42,p['fg'],task_balanced=True,correct_count=0 if balanced else 2,cf_count=4,
                                 initial_expert_tasks=p.get('initial_expert_tasks',[]))
    shared=replaced=0
    for i in range(200):
        batches={a:next(s) for a,s in streams.items()}
        for a,batch in batches.items():
            p=plans[a];assert len({j[i]['grad_norm'] for j in journals[a]})==1
            assert all(math.isfinite(j[i]['grad_norm']) and j[i]['grad_norm']>0 for j in journals[a])
            for rank,j in enumerate(journals[a]):
                examples=j[i]['examples'];expected=batch[rank::p['world_size']]
                assert [e['id'] for e in examples]==[r['id'] for r in expected]
                assert [e['ordinary_cf_control'] for e in examples]==[bool(r.get('ordinary_cf_control')) for r in expected]
                assert all(e['seen_variant'] is None for e in examples)
                if p.get('initial_expert_tasks'):
                    assert [e['initial_expert_anchor'] for e in examples]==[bool(r.get('initial_expert_anchor')) for r in expected]
                    assert [e['effective_correction_weight'] for e in examples]==[effective_weight(r,p['correction_weight'],p['correction_task_weights']) for r in expected]
                assert all(math.isfinite(v) for e in examples for k,v in e.items() if k.startswith('flow_') or k=='endpoint_objective')
                if p.get('action_objective') in ('deployed_rollout_v1','balanced_target_rollout_v1'):
                    assert all(e.get('action_objective')==p['action_objective']
                               and e.get('denoising_steps')==10 and e.get('executed_horizon')==24
                               and e.get('gradient_horizon')=='all_ten_steps'
                               and math.isfinite(e['deployed_objective']) for e in examples)
                    assert all(math.isfinite(v) for e in examples for k,v in e.items() if k.startswith('deployed_'))
                    if balanced:assert all(e.get('supervised_languages')==['target'] and e.get('conditional_difference') is False and 'deployed_source_mse_first24' not in e for e in examples)
            for r in batch:
                assert r['replay_split']=='train'
                kind='fg' if r.get('fg_correction') else 'ordinary_cf_control' if r.get('ordinary_cf_control') else 'common'
                counts[a][kind+'|'+r['pair_id']]+=1
                if r.get('initial_expert_anchor'):counts[a]['initial_expert_anchor|'+r['pair_id']]+=1
        assert batches['eraf_fg']==batches['fg_only']
        assert batches['eraf_only']==batches['no_eraf']
        for full,ordinary in zip(batches['eraf_fg'],batches['eraf_only'],strict=True):
            if full.get('fg_correction'):
                assert ordinary.get('ordinary_cf_control') and not ordinary.get('fg_correction')
                assert (full['pair_id'],full['task_config'])==(ordinary['pair_id'],ordinary['task_config'])
                replaced+=1
            else:assert full==ordinary;shared+=1
    assert shared==(1600 if balanced else 1800) and replaced==(800 if balanced else 600)
    return dict(complete=True,actual_optimizer_steps_per_arm=200,actual_examples_per_arm=2400,
        common_examples_per_arm=shared,matched_fg_or_ordinary_slots_per_arm=replaced,
        actual_sample_ids_and_rank_norms_verified=True,all_sampled_rows_train=True,
        identical_batches_within_fg_setting=True,counts={a:dict(c) for a,c in counts.items()})
