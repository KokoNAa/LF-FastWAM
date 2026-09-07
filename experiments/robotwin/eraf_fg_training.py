"""Masked flow, endpoint, same-observation contrast and two-teacher retention."""
from __future__ import annotations

FG_OFF_PROTOCOL = 'task_domain_matched_target_only_v1'


def mixture_counts(correct=4, cf=2):
    """Keep expert/FG exposure fixed while reallocating the six retention slots."""
    if isinstance(correct, bool) or isinstance(cf, bool) or not isinstance(correct, int) or not isinstance(cf, int):
        raise ValueError('Retention counts must be integers.')
    if min(correct, cf) < 1 or correct + cf != 6:
        raise ValueError('Positive Correct/CF counts must sum to six retention slots.')
    return {'correct': correct, 'cf': cf, 'pair': 3, 'fg': 3}


def supervision_payload(row, payload):
    """The ordinary-CF replacement has the same target-only loss as FG."""
    if not row.get('ordinary_cf_control'):
        return payload
    if any(row.get(k) for k in ('fg_correction', 'native_retention', 'cf_retention')):
        raise ValueError('An ordinary-CF control cannot carry corrective or retention labels.')
    if any('target' not in payload[k] for k in ('captured', 'references')):
        raise ValueError('Ordinary-CF control requires an actual target observation and reference.')
    result = dict(payload, captured={'target': payload['captured']['target']},
                  references={'target': payload['references']['target']})
    if 'valid' in payload:
        result['valid'] = {k: v for k, v in payload['valid'].items() if k == 'target'}
    return result


def same_observation(captured):
    import torch
    a, b = captured['source'], captured['target']
    return torch.equal(a['video_inputs']['x'], b['video_inputs']['x']) and torch.equal(a['proprio'], b['proprio'])


def backward_example(model, row, payload, noise, time, *, teachers, coefficient=1.,
                     eraf=True, fg='full', correct_weight=2., cf_weight=1.,
                     target_weight=2., endpoint_weight=1., conditional_gain=4.,
                     gradient_observer=None, correction_weight=1.):
    import torch
    import math
    if not math.isfinite(correction_weight) or correction_weight <= 0:
        raise ValueError('Correction/control weight must be positive and finite.')
    if row.get('fg_correction') or row.get('ordinary_cf_control'):
        coefficient *= correction_weight
    from experiments.robotwin.eraf_fg_bridge import predict, masked_mse
    from experiments.robotwin.same_state_repair import paired_velocity_losses
    scheduler = model.train_action_scheduler
    end = torch.tensor([scheduler.num_train_timesteps], device=model.device, dtype=model.torch_dtype)
    captured = payload['captured']
    refs = {k: v.to(model.torch_dtype) for k, v in payload['references'].items()}
    valid = {k: payload.get('valid', {}).get(k, torch.ones(v.shape[:2], device=v.device, dtype=torch.bool))
             for k, v in refs.items()}
    if row.get('fg_correction') and fg == 'local':
        if row['frame_index'] != 0:
            raise ValueError('Local control may only use the same failure-start observation.')
        valid['target'] = valid['target'] & (torch.arange(refs['target'].shape[1], device=model.device) < 12)
        # Masking the loss alone still exposes later expert actions through
        # the flow's noisy input and cross-token attention. Remove those
        # targets before constructing any noisy input or velocity target.
        refs['target'] = refs['target'].masked_fill(~valid['target'].unsqueeze(-1), 0.)
    retention = 'correct' if row.get('native_retention') else 'cf' if row.get('cf_retention') else None
    languages = list(refs)
    paired = set(languages) == {'source', 'target'}
    gain = conditional_gain if paired and row.get('initial_observations_exactly_equal', True) else 1.
    if gain != 1 and not same_observation(captured):
        raise ValueError('Conditional-difference loss requires exact same observation and proprio.')
    # Every teacher call ends before the first student graph is created.
    targets = {}
    for language in languages:
        x = scheduler.add_noise(refs[language], noise, time)
        if retention:
            teacher = teachers[retention]
            targets[language] = (teacher.predict(captured[language], x, time),
                                 teacher.predict(captured[language], noise, end))
        else:
            targets[language] = (scheduler.training_target(refs[language], noise, time),
                                 scheduler.training_target(refs[language], noise, end))
    metrics = {}

    def backward(loss, component, *, retain_graph=False):
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite action objective.')
        if gradient_observer is None:
            (coefficient * loss).backward()
        else:
            # Diagnostics collect individual gradients without accumulating .grad
            # or stepping an optimizer. The default training path stays unchanged.
            gradient_observer(coefficient * loss, component, retain_graph=retain_graph)

    def weight(language):
        if retention:
            return correct_weight if retention == 'correct' else cf_weight
        return (target_weight if language == 'target' else 1.) / len(languages)

    for language in languages:
        x = scheduler.add_noise(refs[language], noise, time)
        prediction = predict(model, captured[language], x, time, eraf=eraf)
        loss = masked_mse(prediction, targets[language][0], valid[language])
        backward(float(scheduler.training_weight(time).item()) * weight(language) * loss,
                 'flow_' + language)
        metrics['flow_' + language] = float(loss.detach())
    predictions = {}
    for language in languages:
        if not torch.equal(scheduler.add_noise(refs[language], noise, end), noise):
            raise ValueError('Expert actions leaked into the endpoint input.')
        predictions[language] = predict(model, captured[language], noise, end, eraf=eraf)
    objective = sum(weight(k) * masked_mse(predictions[k], targets[k][1], valid[k]) for k in languages)
    endpoint_fit = objective
    conditional = None
    if gain != 1:
        if not all(bool(v.all()) for v in valid.values()):
            raise ValueError('Paired difference currently requires full real action horizons.')
        parts = paired_velocity_losses(predictions, {k: targets[k][1] for k in languages})
        conditional = (gain - 1.) * parts['conditional_mse']
        objective = objective + conditional
        metrics['conditional_mse'] = float(parts['conditional_mse'].detach())
    metrics['endpoint_objective'] = float(objective.detach())
    if gradient_observer is None:
        backward(endpoint_weight * objective, 'endpoint_objective')
    else:
        backward(endpoint_weight * endpoint_fit, 'endpoint_fit', retain_graph=conditional is not None)
        if conditional is not None:
            backward(endpoint_weight * conditional, 'conditional_difference')
    return metrics


def balanced_group_stream(rows, seed, *, task_balanced=False):
    """Sample task/domain, then scene, then frame to avoid long-clip dominance."""
    from collections import defaultdict
    import random
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row['pair_id'], row['task_config']][row['scene_seed']].append(row)
    if not groups:
        raise ValueError('Empty training mixture bucket.')
    rng = random.Random(seed)
    while True:
        keys = sorted(groups)
        if task_balanced:
            # A task with two domains must not receive twice the weight of
            # a newly collected task with only one domain.
            tasks = sorted({key[0] for key in keys})
            rng.shuffle(tasks)
            keys = [rng.choice([key for key in keys if key[0] == task]) for task in tasks]
        else:
            rng.shuffle(keys)
        for key in keys:
            scenes = groups[key]
            yield rng.choice(scenes[rng.choice(sorted(scenes))])


def mixture_stream(rows, seed, fg='full', *, task_balanced=False, correct_count=4, cf_count=2):
    counts = mixture_counts(correct_count, cf_count)
    if fg == 'off':
        # Reuse only the full arm's metadata schedule (task/domain and batch
        # position). Never return a failed-state row or read its payload.
        groups = {(r['pair_id'], r['task_config']) for r in rows
                  if r['replay_split'] == 'train' and r.get('fg_correction')}
        if not groups:
            raise ValueError('A matched FG-off control needs the declared target task/domain schedule.')
        ordinary = {key: [] for key in groups}
        for row in rows:
            key = row['pair_id'], row['task_config']
            if (row['replay_split'] == 'train' and key in ordinary
                    and not any(row.get(k) for k in ('fg_correction', 'native_retention', 'cf_retention'))):
                ordinary[key].append(row)
        if any(not group for group in ordinary.values()):
            raise ValueError('Missing ordinary CF data for an FG target task/domain.')
        replacements = {key: balanced_group_stream(ordinary[key], seed + 40009 + index * 1009)
                        for index, key in enumerate(sorted(groups))}
        for batch in mixture_stream(rows, seed, 'full', task_balanced=task_balanced,
                                   correct_count=correct_count, cf_count=cf_count):
            yield [(next(replacements[row['pair_id'], row['task_config']]) | {'ordinary_cf_control': True})
                   if row.get('fg_correction') else row for row in batch]
        return
    buckets = {'correct': [], 'cf': [], 'pair': [], 'fg': []}
    for row in rows:
        if row['replay_split'] != 'train':
            continue
        kind = ('correct' if row.get('native_retention') else 'cf' if row.get('cf_retention')
                else 'fg' if row.get('fg_correction') else 'pair')
        buckets[kind].append(row)
    streams = {k: balanced_group_stream(buckets[k], seed + i * 1009, task_balanced=task_balanced)
               for i, k in enumerate(counts) if counts[k]}
    import random
    rng = random.Random(seed)
    first = {(r['pair_id'], r['task_config'], r['scene_seed']): r
             for r in buckets['fg'] if r['frame_index'] == 0}
    while True:
        batch = [next(streams[k]) for k, n in counts.items() for _ in range(n)]
        if fg == 'local':
            # Draw the identical full-arm schedule, then expose only each
            # scene's failure-start observation and its first12 actions.
            batch = [first[r['pair_id'], r['task_config'], r['scene_seed']]
                     if r.get('fg_correction') else r for r in batch]
        rng.shuffle(batch)
        yield batch
