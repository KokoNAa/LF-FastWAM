"""Explicit zero-output initialization for length-preserving ERAF context."""
from copy import deepcopy

MODE = 'context_residual_v1'
ZEROED = ('eraf_action_context_injector.projection.2.weight',
          'eraf_action_context_injector.projection.2.bias')


def initialize(payload, *, parent_path, parent_sha256):
    import torch
    from experiments.robotwin.eraf_fg_bridge import validate_payload
    validate_payload(payload)
    if payload.get('context_injection_mode', 'append_v1') != 'append_v1':
        raise ValueError('Residual initialization must declare an append-mode source.')
    result = deepcopy(payload)
    for key in ZEROED:
        if key not in result['policy_guard']: raise ValueError('Missing context output tensor: ' + key)
        result['policy_guard'][key] = torch.zeros_like(result['policy_guard'][key])
    result.update(stage='bootstrap', optimizer_steps=0, context_injection_mode=MODE,
        parent_checkpoint=str(parent_path), parent_sha256=parent_sha256,
        provenance={'context_residual_initialization': True, 'optimizer_updates': 0,
                    'source_stage': payload['stage'], 'source_optimizer_steps': payload['optimizer_steps'],
                    'source_provenance': deepcopy(payload.get('provenance', {})),
                    'zeroed_parameters': list(ZEROED),
                    'policy_and_other_guard_tensors_preserved': True,
                    'initialization_identity_requires_deployed_action_audit': True})
    validate_payload(result)
    return result
