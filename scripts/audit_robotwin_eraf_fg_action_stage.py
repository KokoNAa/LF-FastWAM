#!/usr/bin/env python3
"""Audit completed action-stage lineage, frozen tensors and sampled updates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def audit(root):
    import torch
    from experiments.robotwin.eraf_fg_bridge import file_sha256, validate_payload
    from experiments.robotwin.eraf_fg_training import mixture_stream, FG_OFF_PROTOCOL
    from experiments.robotwin.initial_anchor import effective_weight
    root = Path(root)
    read = lambda path: json.loads(Path(path).read_text())
    plan = read(root / 'plan.json')
    if read(root / 'complete.json').get('complete') is not True:
        raise ValueError('Training stage is incomplete.')
    if plan['fg'] == 'off' and plan['optimization_contract'].get('fg_off_protocol') != FG_OFF_PROTOCOL:
        raise ValueError('Legacy FG-off sampling is not the declared matched control protocol.')
    checkpoint = root / f"step_{plan['steps']:06d}.pt"
    candidate = validate_payload(torch.load(checkpoint, map_location='cpu', weights_only=False))
    parent = validate_payload(torch.load(plan['checkpoint'], map_location='cpu', weights_only=False))
    skip_hashes = plan.get('skip_file_hashes', False)
    metadata = lambda path: {'path': str(Path(path).resolve()), 'bytes': Path(path).stat().st_size,
                             'mtime_ns': Path(path).stat().st_mtime_ns}
    parent_matches = (Path(candidate['parent_checkpoint']).resolve() == Path(plan['checkpoint']).resolve()
                      if skip_hashes else candidate['parent_sha256'] == file_sha256(plan['checkpoint']))
    if (not parent_matches or candidate['optimizer_steps'] != plan['steps'] or candidate['stage'] != plan['stage']
            or candidate['fg_supervision'] != plan['fg'] or candidate['provenance']['eraf'] != plan['eraf']):
        raise ValueError('Checkpoint lineage/stage does not match the run plan.')
    allowed = set(plan['trainable_parameters'])
    changed, frozen = [], []
    counts = {}
    for section, prefix in [('mot_trainable', 'mot.'), ('policy_guard', 'guard.')]:
        if parent[section].keys() != candidate[section].keys():
            raise ValueError(f'Tensor set changed in {section}.')
        count = 0
        for name, tensor in parent[section].items():
            full = prefix + name
            if not torch.equal(tensor, candidate[section][name]):
                changed.append(full); count += 1
            elif full not in allowed:
                frozen.append(full)
        counts[section] = count
    unexpected = sorted(set(changed) - allowed)
    if unexpected or not changed:
        raise ValueError(f'Unexpected or absent parameter changes: {unexpected}')
    if (plan['stage'] == 'interface' and counts['mot_trainable']) or (plan['eraf'] == 'off' and counts['policy_guard']):
        raise ValueError('Ablation parameter isolation failed.')
    digest = None if skip_hashes else file_sha256(checkpoint)
    optimizer = torch.load(root / 'optimizer_last.pt', map_location='cpu', weights_only=False)
    if skip_hashes:
        saved = {'mot.' + k: v for k, v in candidate['mot_trainable'].items()}
        saved.update({'guard.' + k: v for k, v in candidate['policy_guard'].items()})
        masters = optimizer['optimizer']['master']
        bound = (optimizer.get('checkpoint_identity') == metadata(checkpoint)
                 and len(masters) == len(plan['trainable_parameters'])
                 and all(torch.equal(value.to(saved[name]), saved[name]) for name, value in
                         zip(plan['trainable_parameters'], masters, strict=True)))
        manifest_matches = plan['optimization_contract'].get('manifest_identity') == metadata(plan['manifest'])
    else:
        bound = optimizer['checkpoint_sha256'] == digest
        manifest_matches = plan['optimization_contract']['manifest_sha256'] == file_sha256(plan['manifest'])
    if (not bound or optimizer['step'] != plan['steps']
            or optimizer['parameter_names'] != plan['trainable_parameters']
            or optimizer['optimization_contract'] != plan['optimization_contract']
            or not manifest_matches):
        raise ValueError('Optimizer companion or immutable data contract mismatch.')
    del parent, candidate, optimizer
    world, start = plan['world_size'], plan['start_optimizer_step']
    journals = [[json.loads(line) for line in (root / f'rank{rank}.jsonl').read_text().splitlines()]
                for rank in range(world)]
    expected_steps = list(range(start + 1, plan['steps'] + 1))
    if any([r['step'] for r in journal] != expected_steps for journal in journals):
        raise ValueError('Missing, duplicated or discontinuous optimizer steps.')
    stream = mixture_stream(read(plan['manifest'])['states'], plan['seed'], plan['fg'],
                            task_balanced=plan.get('task_balanced', False),
                            correct_count=plan.get('correct_count', 4), cf_count=plan.get('cf_count', 2),
                            initial_expert_tasks=plan.get('initial_expert_tasks', []))
    for _ in range(start):
        next(stream)
    for index, step in enumerate(expected_steps):
        batch = next(stream)
        if len({journal[index]['grad_norm'] for journal in journals}) != 1:
            raise ValueError(f'Rank gradient norms differ at step{step}.')
        for rank, journal in enumerate(journals):
            if [r['id'] for r in journal[index]['examples']] != [r['id'] for r in batch[rank::world]]:
                raise ValueError(f'Actual sampled states differ at step{step}, rank{rank}.')
            if ([bool(r.get('ordinary_cf_control')) for r in journal[index]['examples']]
                    != [bool(r.get('ordinary_cf_control')) for r in batch[rank::world]]):
                raise ValueError('Ordinary CF replacement supervision differs from the matched schedule.')
            if plan.get('initial_expert_tasks') or plan.get('correction_task_weights'):
                for example,row in zip(journal[index]['examples'],batch[rank::world],strict=True):
                    if (example.get('initial_expert_anchor') != bool(row.get('initial_expert_anchor'))
                        or example.get('effective_correction_weight') != effective_weight(row,plan['correction_weight'],plan['correction_task_weights'])):
                        raise ValueError('Actual initial anchors or task weights differ from the immutable contract.')
    return {'complete': True, 'checkpoint_sha256': digest, 'checkpoint_identity': metadata(checkpoint),
        'hash_scans': not skip_hashes, 'parent_checkpoint': plan['checkpoint'],
        'changed_lora_tensors': counts['mot_trainable'], 'changed_guard_tensors': counts['policy_guard'],
        'unchanged_frozen_tensors': len(frozen), 'unexpected_changes': unexpected,
        'optimizer_checkpoint_and_contract_match': True, 'all_rank_gradient_norms_equal': True,
        'all_sampled_state_ids_match': True, 'world_size': world, 'global_batch': plan['global_batch'],
        'first_step': start + 1, 'final_step': plan['steps'],
        'note': 'Gradient norm equality is checked; individual gradient tensors are not archived.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, help='Completed action training stage directory.')
    args = ap.parse_args()
    report = audit(args.output)
    target = Path(args.output) / 'freeze_audit.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
