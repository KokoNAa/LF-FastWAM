"""Target trajectory supervision with deterministic complete per-task sampling cycles."""
from __future__ import annotations
import random
from collections import defaultdict
from experiments.robotwin.balanced_target import TASKS, OLD_TASKS, mixture_counts, filter_fg_rows, validate_recipe
OBJECTIVE='trajectory_target_rollout_v1'


def cycling_task_stream(rows,seed):
    """Alternate tasks; visit every row once before reshuffling that task."""
    from experiments.robotwin.initial_anchor import source_task
    buckets=defaultdict(list)
    for row in rows:
        if row['replay_split']!='train':raise ValueError('Sampling accepts only training rows.')
        buckets[source_task(row)].append(row)
    if not buckets or len({r['id'] for r in rows})!=len(rows):raise ValueError('Nonempty distinct rows required.')
    rng=random.Random(seed);tasks=sorted(buckets);queues={}
    while True:
        for task in tasks:
            if not queues.get(task):
                queues[task]=sorted(buckets[task],key=lambda r:r['id'])
                rng.shuffle(queues[task])
            yield queues[task].pop()


def pools(rows):
    from experiments.robotwin.initial_anchor import source_task
    train=[r for r in filter_fg_rows(rows) if r['replay_split']=='train']
    cf=[r for r in train if r.get('cf_retention')]
    ordinary=[r for r in train if source_task(r) in TASKS and not any(r.get(k) for k in ('fg_correction','native_retention','cf_retention'))]
    fg=[r for r in train if r.get('fg_correction')]
    if {source_task(r) for r in cf}!=set(OLD_TASKS) or any({source_task(r) for r in pool}!=set(TASKS) for pool in (ordinary,fg)):
        raise ValueError('Missing exact task coverage.')
    for task in TASKS:
        selected=[r for r in ordinary if source_task(r)==task]
        if not any(int(r.get('target_frame_index',r.get('frame_index',0)))>0 for r in selected):
            raise ValueError('Ordinary target trajectory requires noninitial frames.')
    return cf,ordinary,fg


def mixture_stream(rows,seed,fg='full',*,task_balanced=True,correct_count=0,cf_count=4,initial_expert_tasks=TASKS):
    from experiments.robotwin.eraf_fg_training import balanced_group_stream
    mixture_counts(correct_count,cf_count)
    if fg not in ('full','off') or not task_balanced or set(initial_expert_tasks)!=set(TASKS):raise ValueError('Invalid trajectory settings.')
    cf,ordinary,corrections=pools(rows)
    streams=[balanced_group_stream(cf,seed,task_balanced=True),cycling_task_stream(ordinary,seed+1009),cycling_task_stream(corrections,seed+2018)]
    replacements={}
    if fg=='off':
        for i,key in enumerate(sorted({(r['pair_id'],r['task_config']) for r in corrections})):
            replacements[key]=cycling_task_stream([r for r in ordinary if (r['pair_id'],r['task_config'])==key],seed+40009+i*1009)
    rng=random.Random(seed)
    while True:
        batch=[dict(next(streams[0])) for _ in range(4)]+[dict(next(streams[1]),ordinary_target_trajectory=True) for _ in range(4)]+[dict(next(streams[2])) for _ in range(4)]
        rng.shuffle(batch)
        if fg=='off':batch=[dict(next(replacements[r['pair_id'],r['task_config']]),ordinary_cf_control=True) if r.get('fg_correction') else r for r in batch]
        yield batch


def target_payload(row,payload):
    if row.get('native_retention'):raise ValueError('Correct retention is absent.')
    if not any(row.get(k) for k in ('cf_retention','ordinary_target_trajectory','fg_correction','ordinary_cf_control')):raise ValueError('Unknown trajectory stratum.')
    if any('target' not in payload[k] for k in ('captured','references')):raise ValueError('Target observation and reference required.')
    result=dict(payload,captured={'target':payload['captured']['target']},references={'target':payload['references']['target']})
    if 'valid' in payload:result['valid']={k:v for k,v in payload['valid'].items() if k=='target'}
    return result


def backward_example(model,row,payload,noise,time,*,teachers,coefficient=1.,eraf=True,fg='full',correct_weight=1.,cf_weight=4.,correction_weight=1.,fg_gradient_route='joint'):
    from experiments.robotwin.deployed_action_objective import backward_deployed_example
    if fg not in ('full','off') or fg_gradient_route!='joint' or correction_weight!=1.:raise ValueError('Unexpected trajectory objective settings.')
    result=backward_deployed_example(model,row,target_payload(row,payload),noise,time,teachers=teachers,coefficient=coefficient,eraf=eraf,fg=fg,correct_weight=correct_weight,cf_weight=cf_weight,target_weight=1.,conditional_gain=1.,correction_weight=1.,fg_gradient_route='joint')
    result.update(action_objective=OBJECTIVE,supervised_languages=['target'],conditional_difference=False)
    return result
