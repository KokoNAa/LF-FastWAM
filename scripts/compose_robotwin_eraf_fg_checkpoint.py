#!/usr/bin/env python3
"""Compose verified action-only FG weights with a separately trained ERAF interface."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def compose(policy, interface, selected):
    import torch
    from experiments.robotwin.eraf_fg_bridge import validate_payload
    validate_payload(policy)
    validate_payload(interface)
    if (policy['stage'] != 'joint' or policy['fg_supervision'] != 'full'
            or policy['provenance'].get('eraf') != 'off' or policy['provenance'].get('policy_scope') != 'action'
            or interface['stage'] != 'interface' or interface['fg_supervision'] != 'full'):
        raise ValueError('Expected an action-only FG policy and its independent full-FG interface')
    for key in ('parent_checkpoint', 'base_checkpoint', 'guard_config', 'lora_config', 'geometry'):
        if policy[key] != interface[key]:
            raise ValueError('Component initialization or architecture differs: ' + key)
    if set(policy['policy_guard']) != set(interface['policy_guard']):
        raise ValueError('Guard geometry differs')
    changed = []
    for key, value in policy['policy_guard'].items():
        other = interface['policy_guard'][key]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError('Guard tensor geometry differs: ' + key)
        if not torch.equal(value, other):
            if 'guard.' + key not in selected:
                raise ValueError('A frozen semantic parameter differs: ' + key)
            changed.append(key)
    if not changed:
        raise ValueError('No trained interface updates to compose')
    result = dict(policy, policy_guard=dict(interface['policy_guard']), provenance=dict(policy['provenance']))
    result['provenance'].update(eraf='on', composed_for_inference_only=True,
        composition_jointly_optimized=False, policy_optimizer_steps=policy['optimizer_steps'],
        separate_interface_optimizer_steps=interface['optimizer_steps'],
        aggregate_component_optimizer_steps=policy['optimizer_steps'] + interface['optimizer_steps'])
    # This operation makes no optimizer update. The original policy step count
    # and parent remain the action policy's actual training history.
    validate_payload(result)
    return result, changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('policy', 'interface', 'interface-plan', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    import torch
    from experiments.robotwin.eraf_fg_data import file_metadata
    policy = torch.load(args.policy, map_location='cpu', weights_only=False)
    interface = torch.load(args.interface, map_location='cpu', weights_only=False)
    plan = json.loads(args.interface_plan.read_text())
    parent = torch.load(policy['parent_checkpoint'], map_location='cpu', weights_only=False)
    if set(parent['mot_trainable']) != set(interface['mot_trainable']) or any(
            not torch.equal(value, interface['mot_trainable'][key]) for key, value in parent['mot_trainable'].items()):
        raise ValueError('Interface training changed the common base policy')
    result, changed = compose(policy, interface, set(plan['trainable_parameters']))
    if not all(torch.equal(value, result['mot_trainable'][key]) for key, value in policy['mot_trainable'].items()):
        raise ValueError('Composition changed the FG action policy')
    receipt = dict(complete=True, optimizer_updates_this_operation=0, policy_adapters_equal=True,
        only_interface_tensors_changed=len(changed), frozen_semantics_equal=True,
        policy_source=file_metadata(args.policy), interface_source=file_metadata(args.interface),
        policy_optimizer_steps=policy['optimizer_steps'], separate_interface_optimizer_steps=interface['optimizer_steps'],
        jointly_optimized=False, inference_only=True, hash_scans=False)
    result['provenance']['component_sources'] = {key: receipt[key] for key in ('policy_source', 'interface_source')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    temporary = args.output.with_suffix('.tmp')
    with temporary.open('xb') as stream:
        torch.save(result, stream)
    temporary.replace(args.output)
    args.output.with_suffix('.composition.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
