"""Bind the FG retention target to the entire completed ERAF parent."""


def validate_teacher_binding(plan):
    """This opt-in experiment changes only the frozen CF teacher, not student scope."""
    from pathlib import Path
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    mode = plan.get('cf_teacher_mode', 'native_off')
    if mode == 'native_off':
        return None
    if mode != 'parent_eraf':
        raise ValueError('Unknown CF teacher mode.')
    if (plan.get('action_objective') != 'five_task_expert_rollout_v1'
            or not plan.get('continue_eraf_with_fg') or plan.get('resume_state')
            or plan.get('stage') != 'joint' or plan.get('eraf') != 'on'
            or plan.get('fg') != 'full' or plan.get('policy_scope') != 'action'
            or plan.get('interface_scope') != 'all' or plan.get('fg_gradient_route') != 'joint'):
        raise ValueError('Parent ERAF teacher requires fresh joint five-task FG continuation.')
    if Path(plan['cf_teacher']).resolve() != Path(plan['checkpoint']).resolve():
        raise ValueError('CF teacher must be the actual completed ERAF continuation parent.')
    digest = file_sha256(plan['cf_teacher'])
    if not digest or digest != plan.get('continuation_parent_sha256'):
        raise ValueError('CF teacher does not match the completed ERAF parent hash.')
    return digest
