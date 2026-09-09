"""Explicit parent and learning-rate contracts for warm policy experiments."""
import math


def is_zero_context_parent(parent):
    """Accept explicit bootstraps or semantically trained zero-output residuals.

    Semantic steps remain recorded as grounding steps; they are never relabeled
    as zero training. The caller must still bind a deployed identity audit.
    """
    from experiments.robotwin.context_residual import MODE, ZEROED
    provenance = parent.get('provenance', {})
    bootstrap = (parent.get('stage') == 'bootstrap' and parent.get('optimizer_steps') == 0
                 and provenance.get('context_residual_initialization') is True)
    calibrated = (parent.get('stage') == 'grounding' and parent.get('optimizer_steps', 0) > 0
                  and provenance.get('paired_cross_goals') is True)
    return bool((bootstrap or calibrated) and parent.get('context_injection_mode') == MODE
        and all(k in parent.get('policy_guard', {}) and not parent['policy_guard'][k].count_nonzero()
                for k in ZEROED))


def validate_action_parent(parent, *, stage, eraf, fg, resume=False, warm_policy=False,
                           zero_context_joint=False, continue_eraf_with_fg=False):
    if continue_eraf_with_fg:
        if (resume or warm_policy or zero_context_joint or stage != 'joint' or eraf != 'on' or fg != 'full'
                or parent.get('stage') != 'joint' or parent.get('optimizer_steps', 0) <= 0
                or parent.get('fg_supervision') != 'off' or parent.get('provenance', {}).get('eraf') != 'on'):
            raise ValueError('FG continuation requires a trained ERAF-on, FG-off joint parent and a fresh FG optimizer.')
        return
    if zero_context_joint:
        if (resume or warm_policy or stage != 'joint' or eraf != 'on'
                or parent.get('fg_supervision') != fg or not is_zero_context_parent(parent)):
            raise ValueError('Zero-context joint start requires its own untouched residual bootstrap or paired semantic zero-output checkpoint.')
        return
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
        residual_bootstrap = (parent['stage'] == 'bootstrap'
            and parent.get('context_injection_mode') == 'context_residual_v1'
            and parent.get('provenance', {}).get('context_residual_initialization') is True
            and parent['optimizer_steps'] == 0)
        if parent['stage'] != 'grounding' and not residual_bootstrap:
            raise ValueError('Interface training needs a semantic checkpoint or explicit warm policy.')
    elif eraf == 'on':
        if parent['stage'] != 'interface' or parent['fg_supervision'] != fg:
            raise ValueError('Joint ERAF training needs its own matching interface arm.')
    elif parent['stage'] != 'grounding':
        raise ValueError('ERAF-off controls need a common semantic checkpoint or explicit warm policy.')


def verify_continuation_weights(parent, policy_adapters, guard):
    """Check the actual loaded weights before FG can perform its first update."""
    import torch
    for label, expected, actual in [('policy', parent['mot_trainable'], policy_adapters),
                                    ('ERAF', parent['policy_guard'], guard)]:
        if expected.keys() != actual.keys() or any(
                v.dtype != actual[k].dtype or not torch.equal(v.cpu(), actual[k].detach().cpu())
                for k, v in expected.items()):
            raise ValueError('FG continuation changed or reset initial ' + label + ' weights.')


def validate_joint_identity_audit(audit, *, checkpoint_sha256, manifest_sha256):
    """The opt-in joint start must bind actual full-denoising ten-task evidence."""
    from experiments.robotwin.pgc_data import ROBOTWIN_TEN_TASK_SPECS
    if (audit.get('complete') is not True or audit.get('full_eraf_equals_off_exactly') is not True
            or audit.get('include_expanded_tasks') is not True or audit.get('task_count') != 10
            or audit.get('checkpoint_sha256') != checkpoint_sha256
            or audit.get('manifest_sha256') != manifest_sha256
            or audit.get('denoising_steps') != 10 or audit.get('seed') != 42):
        raise ValueError('Zero-context joint start needs matching exact ten-task identity evidence.')
    records = audit.get('records', [])
    groups = {}
    for row in records:
        key = row['pair_id'], row['task_config']
        languages = groups.setdefault(key, set())
        if row['language'] in languages or row['language'] not in ('source', 'target'):
            raise ValueError('Identity evidence has duplicated or invalid instruction branches.')
        languages.add(row['language'])
        comparisons = row.get('comparisons_to_eraf_off', {})
        if set(comparisons) != {'full', 'repeat_full'} or any(
                c.get('raw_actions_exact') is not True or c.get('normalized_actions_exact') is not True
                or c.get('max_abs') != 0 for c in comparisons.values()):
            raise ValueError('Identity evidence contains a changed deployed action.')
    if ({pair for pair, domain in groups} != {s.pair_id for s in ROBOTWIN_TEN_TASK_SPECS}
            or any(languages != {'source', 'target'} for languages in groups.values())
            or len(groups) != audit.get('task_domain_count')
            or len(records) != audit.get('queries') or len(records) != 2 * len(groups)):
        raise ValueError('Identity evidence does not cover every declared task/domain branch.')


def parameter_learning_rates(names, policy_lr, interface_lr=None):
    if any(not math.isfinite(value) or value <= 0 for value in
           [policy_lr] + ([] if interface_lr is None else [interface_lr])):
        raise ValueError('Learning rates must be positive and finite.')
    return [interface_lr if name.startswith('guard.') and interface_lr is not None else policy_lr
            for name in names]
