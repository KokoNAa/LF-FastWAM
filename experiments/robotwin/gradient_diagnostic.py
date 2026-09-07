"""Collect objective gradients without optimizer updates or parameter .grad writes."""
from __future__ import annotations
import math


def bucket(row):
    for flag, label in (('native_retention', 'correct_retention'),
                        ('cf_retention', 'cf_retention'), ('fg_correction', 'fg'),
                        ('ordinary_cf_control', 'ordinary_cf_control')):
        if row.get(flag):
            return label
    return 'expert_pair'


def scope(name):
    if name.startswith('guard.'):
        return 'eraf'
    return 'video' if '.video.' in name else 'action' if '.action.' in name else 'other'


def diagnostic_recipe(plan=None):
    """Recover the executed action recipe; retain historical probe defaults."""
    from experiments.robotwin.eraf_fg_training import mixture_counts, validate_fg_gradient_route
    recipe = dict(stage='joint', fg='full', eraf='off', seed=42, policy_scope='all',
                  correct_count=4, cf_count=2, correct_weight=4., cf_weight=2.,
                  correction_weight=1., task_balanced=False,
                  disable_seen_language_augmentation=False, fg_gradient_route='joint')
    if plan is not None:
        required = {'stage', 'fg', 'eraf', 'seed', 'correct_weight', 'cf_weight', 'global_batch'}
        if required - plan.keys():
            raise ValueError(f'Incomplete training plan: {sorted(required - plan.keys())}')
        if plan['global_batch'] != 12:
            raise ValueError('Diagnostic expects the executed global batch of 12.')
        recipe.update({k: plan[k] for k in recipe if k in plan})
        for key, value in recipe.items():
            if key in plan.get('optimization_contract', {}) and plan['optimization_contract'][key] != value:
                raise ValueError(f'Training plan disagrees with optimization contract: {key}')
    if recipe['stage'] not in {'joint', 'interface'} or recipe['fg'] not in {'off', 'local', 'full'}:
        raise ValueError('Require a supported action-training recipe.')
    if recipe['eraf'] not in {'off', 'on'} or recipe['policy_scope'] not in {'all', 'action'}:
        raise ValueError('Invalid ERAF or policy scope.')
    if recipe['stage'] == 'interface' and recipe['eraf'] == 'off':
        raise ValueError('ERAF-off has no trainable interface.')
    validate_fg_gradient_route(recipe['fg_gradient_route'], eraf=recipe['eraf'] == 'on', fg=recipe['fg'])
    for key in ('correct_weight', 'cf_weight', 'correction_weight'):
        if not math.isfinite(recipe[key]) or recipe[key] <= 0:
            raise ValueError(f'Invalid loss weight: {key}')
    counts = mixture_counts(recipe['correct_count'], recipe['cf_count'])
    if plan is not None and 'mixture' in plan:
        expected = dict(correct_retention=counts['correct'], cf_retention=counts['cf'],
                        expert_pairs=3, fg=0 if recipe['fg'] == 'off' else 3,
                        ordinary_cf_control=3 if recipe['fg'] == 'off' else 0)
        if plan['mixture'] != expected:
            raise ValueError('Training plan mixture disagrees with recovered recipe.')
    return recipe


class GradientCollector:
    def __init__(self, parameters):
        self.parameters = dict(parameters)
        self.groups = {}
        self.terms = []

    def observer(self, group, example_id):
        def collect(loss, component, *, retain_graph=False, allowed_parameter_prefix=None):
            import torch
            active = {k: v for k, v in self.parameters.items()
                      if allowed_parameter_prefix is None or k.startswith(allowed_parameter_prefix)}
            if not active:
                raise ValueError('No diagnostic parameters match the declared gradient route.')
            gradients = torch.autograd.grad(loss, tuple(active.values()),
                                            allow_unused=True, retain_graph=retain_graph)
            total = self.groups.setdefault(group, {})
            squares = {}
            used = 0
            for (name, _), gradient in zip(active.items(), gradients):
                if gradient is None:
                    continue
                gradient = gradient.detach().float().cpu()
                if not bool(torch.isfinite(gradient).all()):
                    raise ValueError('Nonfinite diagnostic gradient.')
                used += 1
                category = scope(name)
                squares[category] = squares.get(category, 0.) + float(gradient.square().sum())
                if name in total:
                    total[name].add_(gradient)
                else:
                    total[name] = gradient.clone()
            self.terms.append({'id': example_id, 'group': group, 'component': component,
                               'weighted_loss': float(loss.detach()), 'used_tensors': used,
                               'norm_by_scope': {k: math.sqrt(v) for k, v in squares.items()}})
        return collect

    def summary(self):
        names = sorted(self.groups)
        result = {}
        for category in ('all', 'video', 'action', 'eraf', 'other'):
            def dot(a, b):
                return sum(float((v * b[k]).sum()) for k, v in a.items()
                           if k in b and (category == 'all' or scope(k) == category))
            dots = {a: {c: dot(self.groups[a], self.groups[c]) for c in names} for a in names}
            norms = {a: math.sqrt(max(0., dots[a][a])) for a in names}
            cosines = {a: {c: dots[a][c] / (norms[a] * norms[c])
                          if norms[a] and norms[c] else None for c in names} for a in names}
            total_squared = sum(sum(v.values()) for v in dots.values())
            result[category] = {'norms': norms, 'dot_products': dots, 'cosines': cosines,
                                'combined_norm': math.sqrt(max(0., total_squared)),
                                'dot_with_combined': {a: sum(dots[a].values()) for a in names}}
        return {'scopes': result, 'terms': self.terms}
