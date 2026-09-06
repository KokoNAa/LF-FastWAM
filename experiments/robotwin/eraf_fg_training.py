"""Masked flow, endpoint, same-observation contrast and two-teacher retention."""
from __future__ import annotations


def same_observation(captured):
    import torch
    a, b = captured['source'], captured['target']
    return torch.equal(a['video_inputs']['x'], b['video_inputs']['x']) and torch.equal(a['proprio'], b['proprio'])


def backward_example(model, row, payload, noise, time, *, teachers, coefficient=1.,
                     eraf=True, fg='full', correct_weight=2., cf_weight=1.,
                     target_weight=2., endpoint_weight=1., conditional_gain=4.):
    import torch
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

    def backward(loss):
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite action objective.')
        (coefficient * loss).backward()

    def weight(language):
        if retention:
            return correct_weight if retention == 'correct' else cf_weight
        return (target_weight if language == 'target' else 1.) / len(languages)

    for language in languages:
        x = scheduler.add_noise(refs[language], noise, time)
        prediction = predict(model, captured[language], x, time, eraf=eraf)
        loss = masked_mse(prediction, targets[language][0], valid[language])
        backward(float(scheduler.training_weight(time).item()) * weight(language) * loss)
        metrics['flow_' + language] = float(loss.detach())
    predictions = {}
    for language in languages:
        if not torch.equal(scheduler.add_noise(refs[language], noise, end), noise):
            raise ValueError('Expert actions leaked into the endpoint input.')
        predictions[language] = predict(model, captured[language], noise, end, eraf=eraf)
    objective = sum(weight(k) * masked_mse(predictions[k], targets[k][1], valid[k]) for k in languages)
    if gain != 1:
        if not all(bool(v.all()) for v in valid.values()):
            raise ValueError('Paired difference currently requires full real action horizons.')
        parts = paired_velocity_losses(predictions, {k: targets[k][1] for k in languages})
        objective = objective + (gain - 1.) * parts['conditional_mse']
        metrics['conditional_mse'] = float(parts['conditional_mse'].detach())
    metrics['endpoint_objective'] = float(objective.detach())
    backward(endpoint_weight * objective)
    return metrics


def balanced_group_stream(rows, seed):
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
        rng.shuffle(keys)
        for key in keys:
            scenes = groups[key]
            yield rng.choice(scenes[rng.choice(sorted(scenes))])


def mixture_stream(rows, seed, fg='full'):
    buckets = {'correct': [], 'cf': [], 'pair': [], 'fg': []}
    for row in rows:
        if row['replay_split'] != 'train':
            continue
        kind = ('correct' if row.get('native_retention') else 'cf' if row.get('cf_retention')
                else 'fg' if row.get('fg_correction') else 'pair')
        if kind == 'fg' and fg == 'off':
            continue
        buckets[kind].append(row)
    counts = {'correct': 4, 'cf': 2, 'pair': 6 if fg == 'off' else 3, 'fg': 0 if fg == 'off' else 3}
    streams = {k: balanced_group_stream(buckets[k], seed + i * 1009)
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
