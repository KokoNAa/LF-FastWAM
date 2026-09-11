"""Language interventions inside the video expert, without action inference."""
from __future__ import annotations
from contextlib import ExitStack
from experiments.robotwin.world_language_probe import replace_method, video_velocity


def layer_groups(n=30):
    if n < 3 or n % 3:
        raise ValueError('Expected three equal layer bands')
    k = n // 3
    return {'early': list(range(k)), 'middle': list(range(k, 2*k)),
            'late': list(range(2*k, n)), 'all': list(range(n))}


def stage_active(stage, step, total=20):
    if not 0 <= step < total:
        raise ValueError('Step outside sampling schedule')
    if stage == 'all': return True
    groups = {'early': (0, total//3), 'middle': (total//3, 2*total//3),
              'late': (2*total//3, total)}
    if stage not in groups: raise ValueError('Unknown denoising stage')
    a, b = groups[stage]
    return a <= step < b


def select_states(old, per_task=2):
    """Freeze first catalog scenes; no policy/generation outcomes enter selection."""
    if per_task < 1: raise ValueError('Need a positive number of scenes')
    out = []
    for task in old['tasks']:
        scenes = [r for r in old['scenes'] if r['task'] == task][:per_task]
        if len(scenes) != per_task: raise ValueError('Insufficient scenes')
        phase = 'shared_decision' if task.startswith('place_') else 'initial'
        for scene in scenes:
            found = [r for r in old['states'] if r['task'] == task
                     and r['scene_seed'] == scene['scene_seed'] and r['phase'] == phase]
            if len(found) != 1 or not found[0]['dual_reference_valid']:
                raise ValueError('Need one common observation with paired expert futures')
            out.append(found[0])
    return out


def causal_velocity(model, latent, timestep, context, mask, *, mode='baseline',
                    layers=(), donor=None, capture=False):
    """Patch cross-attention residuals or mask text while preserving proprio.

    Donor activations must be recomputed at the SAME current noisy latent/time.
    They may include effects propagated through earlier donor layers. Patching is
    a distributed residual intervention, not localization to individual words.
    """
    if mode not in {'baseline', 'mask_text', 'patch'}: raise ValueError(mode)
    blocks = list(model.video_expert.blocks)
    chosen = set(layers)
    if not chosen <= set(range(len(blocks))): raise ValueError('Invalid layer')
    if mask.ndim != 2 or mask.shape[-1] < 2 or not bool(mask[:, -1].all()):
        raise ValueError('Expected text followed by one unmasked proprio token')
    if mode == 'patch' and (donor is None or not chosen <= set(donor)):
        raise ValueError('Missing donor activations')
    observed, calls = {}, {}
    with ExitStack() as stack:
        for i, block in enumerate(blocks):
            original = block.cross_attn.forward
            def forward(x, ctx, ctx_mask=None, *, index=i, fn=original):
                calls[index] = calls.get(index, 0) + 1
                if calls[index] != 1: raise ValueError('Unexpected repeated layer call')
                if index in chosen and mode == 'patch':
                    y = donor[index]
                    if y.shape != x.shape: raise ValueError('Donor shape mismatch')
                    y = y.to(x)
                else:
                    if index in chosen and mode == 'mask_text':
                        if ctx_mask is None: raise ValueError('Missing context mask')
                        ctx_mask = ctx_mask.clone()
                        ctx_mask[..., :-1] = False
                        if not bool(ctx_mask[..., -1].all()):
                            raise ValueError('Proprio was masked')
                    y = fn(x, ctx, ctx_mask=ctx_mask)
                if capture: observed[index] = y.detach().clone()
                return y
            stack.enter_context(replace_method(block.cross_attn, 'forward', forward))
        prediction = video_velocity(model, latent, timestep, context, mask)
    if len(calls) != len(blocks): raise ValueError('Video layer coverage incomplete')
    return prediction, observed


def generation_conditions():
    rows = [dict(name=n, base=n, mode='baseline', layer='all', stage='all')
            for n in ['source', 'target', 'source_paraphrase', 'target_paraphrase', 'empty']]
    for base in ['source', 'target']:
        for band in ['early', 'middle', 'late', 'all']:
            rows.append(dict(name=f'{base}_mask_layer_{band}', base=base,
                             mode='mask_text', layer=band, stage='all'))
        other = 'target' if base == 'source' else 'source'
        for band in ['early', 'middle', 'late', 'all']:
            rows.append(dict(name=f'{base}_patch_{other}_layer_{band}', base=base,
                             donor=other, mode='patch', layer=band, stage='all'))
        for stage in ['early', 'middle', 'late']:
            rows.append(dict(name=f'{base}_mask_time_{stage}', base=base,
                             mode='mask_text', layer='all', stage=stage))
    return rows
