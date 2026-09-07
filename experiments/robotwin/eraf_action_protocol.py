"""Explicit parent and learning-rate contracts for warm policy experiments."""
import math


def validate_action_parent(parent, *, stage, eraf, fg, resume=False, warm_policy=False):
    if warm_policy and resume:
        raise ValueError('Warm policy initialization and optimizer resume are different operations.')
    if resume:
        if (parent['stage'] != stage or parent['fg_supervision'] != fg
                or parent['provenance'].get('eraf') != eraf):
            raise ValueError('Resumed checkpoint must have the same stage and ablation settings.')
        return
    if warm_policy:
        if (parent['stage'] != 'joint' or parent['fg_supervision'] != 'off'
                or parent['provenance'].get('eraf') != 'off'):
            raise ValueError('A common warm policy must be a trained ERAF-off, FG-off joint checkpoint.')
        if eraf == 'on' and stage != 'interface':
            raise ValueError('Warm ERAF joint training requires a matching interface warmup first.')
        return
    if stage == 'interface':
        if parent['stage'] != 'grounding':
            raise ValueError('Interface training needs a semantic checkpoint or explicit warm policy.')
    elif eraf == 'on':
        if parent['stage'] != 'interface' or parent['fg_supervision'] != fg:
            raise ValueError('Joint ERAF training needs its own matching interface arm.')
    elif parent['stage'] != 'grounding':
        raise ValueError('ERAF-off controls need a common semantic checkpoint or explicit warm policy.')


def parameter_learning_rates(names, policy_lr, interface_lr=None):
    if any(not math.isfinite(value) or value <= 0 for value in
           [policy_lr] + ([] if interface_lr is None else [interface_lr])):
        raise ValueError('Learning rates must be positive and finite.')
    return [interface_lr if name.startswith('guard.') and interface_lr is not None else policy_lr
            for name in names]
