"""Initialize ERAF on the current no-ERAF policy; FG later inherits trained ERAF."""
from copy import deepcopy


def branch_parent(policy, semantic, *, policy_path, policy_sha256,
                  semantic_path, semantic_sha256, fg):
    import torch
    from experiments.robotwin.eraf_fg_bridge import validate_payload
    from experiments.robotwin.eraf_action_protocol import is_zero_context_parent
    from experiments.robotwin.context_residual import MODE
    validate_payload(policy)
    validate_payload(semantic)
    if (policy['stage'] != 'joint' or policy['optimizer_steps'] <= 0
            or policy['fg_supervision'] != 'off' or policy['provenance'].get('eraf') != 'off'):
        raise ValueError('The action parent must be a completed no-ERAF, no-FG policy.')
    if (fg != 'off' or semantic['fg_supervision'] != fg
            or semantic['stage'] != 'grounding' or not is_zero_context_parent(semantic)):
        raise ValueError('Only the matching ordinary zero-output donor initializes ERAF; FG must continue trained ERAF.')
    for key in ('base_checkpoint', 'geometry', 'guard_config', 'lora_config'):
        if policy[key] != semantic[key]:
            raise ValueError('Incompatible component architecture: ' + key)
    if policy['mot_trainable'].keys() != semantic['mot_trainable'].keys():
        raise ValueError('Policy adapter tensor sets differ.')
    for key, value in policy['mot_trainable'].items():
        other = semantic['mot_trainable'][key]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError('Policy adapter geometry differs: ' + key)
        # Semantics depend on frozen visual features. Never silently rebase those.
        if (key.startswith('mixtures.video.') or '.mixtures.video.' in key) and not torch.equal(value, other):
            raise ValueError('Frozen video adapters differ: ' + key)
    if policy['policy_guard'].keys() != semantic['policy_guard'].keys():
        raise ValueError('Guard tensor sets differ.')
    for key, value in policy['policy_guard'].items():
        other = semantic['policy_guard'][key]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError('Guard tensor geometry differs: ' + key)
    # The policy is the primary parent. Only the separately identified guard is donated.
    result = deepcopy(policy)
    result.update(policy_guard=deepcopy(semantic['policy_guard']),
        stage='bootstrap', optimizer_steps=0, fg_supervision=fg, context_injection_mode=MODE,
        parent_checkpoint=str(policy_path), parent_sha256=policy_sha256,
        provenance=dict(context_residual_initialization=True, current_no_eraf_initialization=True,
            optimizer_updates=0, action_parent=dict(path=str(policy_path), sha256=policy_sha256,
                optimizer_steps=policy['optimizer_steps'], provenance=deepcopy(policy['provenance'])),
            semantic_donor=dict(path=str(semantic_path), sha256=semantic_sha256,
                optimizer_steps=semantic['optimizer_steps'], provenance=deepcopy(semantic['provenance'])),
            all_policy_adapters_from_current_no_eraf=True, semantic_donor_policy_adapters_used=False,
            initialization_identity_requires_deployed_action_audit=True))
    validate_payload(result)
    if not is_zero_context_parent(result):
        raise ValueError('Rebased branch is not an identity-initialized ERAF parent.')
    return result


def current_baseline(source, models, audit):
    """Bind to the completed source's FINAL control, never its training parent."""
    from pathlib import Path
    path = str(Path(source).resolve() / 'no_eraf/joint/step_000200.pt')
    spec = models['no_eraf']
    if (not audit.get('complete') or audit.get('episodes') != 180
            or spec.get('eraf') != 'off' or spec.get('fg') != 'off'
            or spec.get('final_checkpoint') != path
            or audit.get('checkpoint_sha256', {}).get('no_eraf') != spec.get('final_sha256')):
        raise ValueError('Current no-ERAF must be the audited final200 control, not an older parent.')
    return path, spec['final_sha256']
