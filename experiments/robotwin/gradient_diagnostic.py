"""Collect objective gradients without optimizer updates or parameter .grad writes."""
from __future__ import annotations
import math


def bucket(row):
    for flag, label in (('native_retention', 'correct_retention'),
                        ('cf_retention', 'cf_retention'), ('fg_correction', 'fg')):
        if row.get(flag):
            return label
    return 'expert_pair'


def scope(name):
    return 'video' if '.video.' in name else 'action' if '.action.' in name else 'other'


class GradientCollector:
    def __init__(self, parameters):
        self.parameters = dict(parameters)
        self.groups = {}
        self.terms = []

    def observer(self, group, example_id):
        def collect(loss, component, *, retain_graph=False):
            import torch
            gradients = torch.autograd.grad(loss, tuple(self.parameters.values()),
                                            allow_unused=True, retain_graph=retain_graph)
            total = self.groups.setdefault(group, {})
            squares = {}
            used = 0
            for (name, _), gradient in zip(self.parameters.items(), gradients):
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
        for category in ('all', 'video', 'action', 'other'):
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
