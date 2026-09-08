"""Explicit target-only 4CF/4initial/4correction training protocol."""
from __future__ import annotations
OBJECTIVE='balanced_target_rollout_v1'
TASKS=('place_empty_cup','move_pillbottle_pad')
OLD_TASKS=('place_a2b_left','blocks_ranking_rgb','place_a2b_right','place_burger_fries','stack_blocks_two')

def mixture_counts(correct=0,cf=4):
    if type(correct) is not int or type(cf) is not int or (correct,cf)!=(0,4):raise ValueError('Balanced target protocol requires0 Correct and4 CF slots.')
    return dict(correct=0,cf=4,pair=4,fg=4)

def filter_fg_rows(rows):
    from experiments.robotwin.initial_anchor import source_task
    return [r for r in rows if not r.get('fg_correction') or source_task(r) in TASKS]

def validate_recipe(plan):
    expected=dict(stage='joint',policy_scope='action',interface_scope='all',correct_count=0,cf_count=4,
        task_balanced=True,disable_seen_language_augmentation=True,correction_weight=1.,correction_task_weights={},fg_gradient_route='joint',learning_rate=3e-6,interface_learning_rate=3e-5,cf_weight=4.)
    if any(plan.get(k)!=v for k,v in expected.items()):raise ValueError('Balanced target protocol settings differ.')
    if (set(plan['target_tasks'])!=set(TASKS) or set(plan['initial_expert_tasks'])!=set(TASKS)
            or set(plan['cf_retention_tasks'])!=set(OLD_TASKS) or plan['fg'] not in ('full','off')):
        raise ValueError('Balanced target protocol requires exact target and retention tasks.')

def mixture_stream(rows,seed,fg='full',*,task_balanced=True,correct_count=0,cf_count=4,initial_expert_tasks=TASKS):
    from experiments.robotwin.initial_anchor import initial_rows,source_task
    from experiments.robotwin.eraf_fg_training import balanced_group_stream
    import random
    mixture_counts(correct_count,cf_count)
    if fg not in ('full','off') or not task_balanced or set(initial_expert_tasks)!=set(TASKS):raise ValueError('Invalid balanced stream settings.')
    rows=filter_fg_rows(rows)
    train=[r for r in rows if r['replay_split']=='train']
    cf=[r for r in train if r.get('cf_retention')]
    corrections=[r for r in train if r.get('fg_correction')]
    if {source_task(r) for r in cf}!=set(OLD_TASKS) or {source_task(r) for r in corrections}!=set(TASKS):raise ValueError('Missing or unexpected CF/FG task coverage.')
    streams=[balanced_group_stream(bucket,seed+i*1009,task_balanced=True) for i,bucket in enumerate([cf,initial_rows(rows,TASKS),corrections])]
    replacements={}
    if fg=='off':
        for i,key in enumerate(sorted({(r['pair_id'],r['task_config']) for r in corrections})):
            candidates=[r for r in train if (r['pair_id'],r['task_config'])==key and not any(r.get(k) for k in ('fg_correction','native_retention','cf_retention'))]
            replacements[key]=balanced_group_stream(candidates,seed+40009+i*1009,task_balanced=True)
    rng=random.Random(seed)
    while True:
        batch=[dict(next(streams[0])) for _ in range(4)]+[dict(next(streams[1]),initial_expert_anchor=True) for _ in range(4)]+[dict(next(streams[2])) for _ in range(4)]
        rng.shuffle(batch)
        if fg=='off':batch=[dict(next(replacements[r['pair_id'],r['task_config']]),ordinary_cf_control=True) if r.get('fg_correction') else r for r in batch]
        yield batch

def target_payload(row,payload):
    if row.get('native_retention'):raise ValueError('Correct retention is absent from this protocol.')
    if not any(row.get(k) for k in ('cf_retention','initial_expert_anchor','fg_correction','ordinary_cf_control')):raise ValueError('Unknown balanced supervision stratum.')
    if any('target' not in payload[k] for k in ('captured','references')):raise ValueError('Actual target observation/reference required.')
    result=dict(payload,captured={'target':payload['captured']['target']},references={'target':payload['references']['target']})
    if 'valid' in payload:result['valid']={k:v for k,v in payload['valid'].items() if k=='target'}
    return result

def backward_example(model,row,payload,noise,time,*,teachers,coefficient=1.,eraf=True,fg='full',correct_weight=1.,cf_weight=4.,correction_weight=1.,fg_gradient_route='joint'):
    from experiments.robotwin.deployed_action_objective import backward_deployed_example
    if fg not in ('full','off') or fg_gradient_route!='joint' or correction_weight!=1.:raise ValueError('Unexpected balanced objective settings.')
    result=backward_deployed_example(model,row,target_payload(row,payload),noise,time,teachers=teachers,coefficient=coefficient,
        eraf=eraf,fg=fg,correct_weight=correct_weight,cf_weight=cf_weight,target_weight=1.,conditional_gain=1.,correction_weight=1.,fg_gradient_route='joint')
    result.update(action_objective=OBJECTIVE,supervised_languages=['target'],conditional_difference=False)
    return result
