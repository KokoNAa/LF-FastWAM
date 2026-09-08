"""Learn the evaluated five tasks from expert actions, with matched FG controls.

Temporal thirds are sampling strata, not predicted physical phases. Teacher
retention stays a separate, unit-weight anchor; it does not replace expert data.
"""
from __future__ import annotations

from collections import defaultdict
import random

from experiments.robotwin.initial_anchor import source_task

OBJECTIVE = 'five_task_expert_rollout_v1'
TASKS = ('blocks_ranking_rgb', 'stack_blocks_two', 'place_a2b_left',
         'place_a2b_right', 'place_burger_fries')
FG_TASKS = ('blocks_ranking_rgb', 'place_a2b_left')


def mixture_counts(correct=0, cf=4):
    if type(correct) is not int or type(cf) is not int or (correct, cf) != (0, 4):
        raise ValueError('Five-task mixture requires 0 Correct, 4 CF, 4 expert, 4 matched corrective slots.')
    return dict(correct=0, cf=4, pair=4, fg=4)


def filter_fg_rows(rows):
    return [r for r in rows if source_task(r) in TASKS and not r.get('native_retention')
            and (not r.get('fg_correction') or source_task(r) in FG_TASKS)]


def validate_recipe(plan):
    expected = dict(stage='joint', policy_scope='action', interface_scope='all',
                    correct_count=0, cf_count=4, task_balanced=True,
                    disable_seen_language_augmentation=True, correction_weight=1.,
                    correction_task_weights={}, fg_gradient_route='joint',
                    learning_rate=3e-6, interface_learning_rate=3e-5, cf_weight=1.,
                    initial_expert_tasks=[])
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError('Five-task expert recipe settings differ from the declared protocol.')
    if (set(plan['target_tasks']) != set(FG_TASKS)
            or set(plan['cf_retention_tasks']) != set(TASKS) or plan['fg'] not in ('off', 'full')):
        raise ValueError('Five evaluated tasks and two available FG tasks must remain explicit.')


def temporal_rows(rows):
    """Assign relative recorded-trajectory thirds without reading payloads."""
    scenes = defaultdict(list)
    for row in rows:
        if row['replay_split'] != 'train':
            raise ValueError('Training sampler received a held-out row.')
        scenes[row['pair_id'], row['task_config'], row['scene_seed']].append(row)
    result = []
    for values in scenes.values():
        frame = lambda r: int(r.get('target_frame_index', r.get('frame_index', 0)))
        frames = sorted({frame(r) for r in values})
        bins = {f: min(2, 3 * i // len(frames)) for i, f in enumerate(frames)}
        result.extend(dict(r, replay_trajectory_third=bins[frame(r)]) for r in values)
    return result


def stratified_stream(rows, seed):
    """Cycle tasks and temporal thirds; sample domains, scenes, then frames."""
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        groups[source_task(row)][row['task_config'], row['replay_trajectory_third']][row['scene_seed']].append(row)
    if not groups:
        raise ValueError('Empty expert/corrective sampling pool.')
    rng = random.Random(seed)
    counters = defaultdict(int)
    while True:
        for task in sorted(groups):
            phases = sorted({phase for _, phase in groups[task]})
            phase = phases[counters[task] % len(phases)]
            counters[task] += 1
            keys = sorted(k for k in groups[task] if k[1] == phase)
            scenes = groups[task][rng.choice(keys)]
            yield rng.choice(scenes[rng.choice(sorted(scenes))])


def pools(rows):
    selected = filter_fg_rows(rows)
    if len({r['id'] for r in selected}) != len(selected):
        raise ValueError('Replay row IDs must be unique.')
    keys = lambda split: {(source_task(r), r['task_config'], r['scene_seed'])
                          for r in selected if r['replay_split'] == split}
    if keys('train') & keys('replay_holdout'):
        raise ValueError('A physical scene occurs in both training and replay holdout.')
    train = [r for r in selected if r['replay_split'] == 'train']
    cf = [r for r in train if r.get('cf_retention')]
    ordinary = [r for r in train if not r.get('cf_retention') and not r.get('fg_correction')]
    corrections = [r for r in train if r.get('fg_correction')]
    for bucket, tasks in ((cf, TASKS), (ordinary, TASKS), (corrections, FG_TASKS)):
        if {source_task(r) for r in bucket} != set(tasks):
            raise ValueError('Missing required five-task expert/retention or two-task corrective coverage.')
    ordinary, corrections = temporal_rows(ordinary), temporal_rows(corrections)
    for task in TASKS:
        if {r['replay_trajectory_third'] for r in ordinary if source_task(r) == task} != {0, 1, 2}:
            raise ValueError('Each evaluated task needs real noninitial expert trajectory coverage.')
    return cf, ordinary, corrections


def mixture_stream(rows, seed, fg='full', *, task_balanced=True, correct_count=0,
                   cf_count=4, initial_expert_tasks=()):
    from experiments.robotwin.eraf_fg_training import balanced_group_stream
    mixture_counts(correct_count, cf_count)
    if fg not in ('off', 'full') or not task_balanced or initial_expert_tasks:
        raise ValueError('Five-task sampling requires full/off FG and explicit temporal balancing.')
    cf, ordinary, corrections = pools(rows)
    streams = [balanced_group_stream(cf, seed, task_balanced=True),
               stratified_stream(ordinary, seed + 1009), stratified_stream(corrections, seed + 2018)]
    replacements = {}
    if fg == 'off':
        keys = sorted({(r['pair_id'], r['task_config'], r['replay_trajectory_third']) for r in corrections})
        for i, key in enumerate(keys):
            candidates = [r for r in ordinary if (r['pair_id'], r['task_config'], r['replay_trajectory_third']) == key]
            replacements[key] = stratified_stream(candidates, seed + 40009 + i * 1009)
            if not candidates:
                raise ValueError('FG-off needs same task/domain/temporal-third ordinary expert data.')
    rng = random.Random(seed)
    while True:
        batch = ([dict(next(streams[0])) for _ in range(4)]
                 + [dict(next(streams[1]), ordinary_target_trajectory=True) for _ in range(4)]
                 + [dict(next(streams[2])) for _ in range(4)])
        rng.shuffle(batch)
        if fg == 'off':
            batch = [dict(next(replacements[r['pair_id'], r['task_config'], r['replay_trajectory_third']]),
                          ordinary_cf_control=True) if r.get('fg_correction') else r for r in batch]
        yield batch


def target_payload(row, payload):
    if row.get('native_retention') or not any(row.get(k) for k in
            ('cf_retention', 'ordinary_target_trajectory', 'fg_correction', 'ordinary_cf_control')):
        raise ValueError('Unknown five-task target supervision stratum.')
    if any('target' not in payload[k] for k in ('captured', 'references')):
        raise ValueError('An actual target observation and target action reference are required.')
    result = dict(payload, captured={'target': payload['captured']['target']},
                  references={'target': payload['references']['target']})
    if 'valid' in payload:
        result['valid'] = {'target': payload['valid']['target']}
    return result


def backward_example(model, row, payload, noise, time, *, teachers, coefficient=1.,
                     eraf=True, fg='full', correct_weight=1., cf_weight=1.,
                     correction_weight=1., fg_gradient_route='joint'):
    from experiments.robotwin.deployed_action_objective import backward_deployed_example
    if cf_weight != 1. or correction_weight != 1. or fg_gradient_route != 'joint':
        raise ValueError('Unit retention/expert/correction weights and joint gradients are fixed.')
    result = backward_deployed_example(model, row, target_payload(row, payload), noise, time,
        teachers=teachers, coefficient=coefficient, eraf=eraf, fg=fg, correct_weight=correct_weight,
        cf_weight=1., target_weight=1., conditional_gain=1., correction_weight=1., fg_gradient_route='joint')
    result.update(action_objective=OBJECTIVE, supervised_languages=['target'], conditional_difference=False,
                  supervision_target='teacher_action' if row.get('cf_retention') else 'expert_action')
    return result
