#!/usr/bin/env python3
"""Audit actual paired semantic updates, immutable policy and matched FG slots."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def audit(root):
    import torch
    from experiments.robotwin.cross_goal_semantics import paired_cross_goal_batch
    from experiments.robotwin.fg_geometry_replay import geometry_mixture
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    from experiments.robotwin.context_residual import ZEROED
    root = Path(root)
    read = lambda p: json.loads(Path(p).read_text())
    plans = {mode: read(root / mode / 'plan.json') for mode in ('ordinary', 'fg')}
    journals, arms, batches = {}, {}, {}
    steps = plans['ordinary']['steps']
    offset = plans['ordinary']['step_offset']
    exposure, crossed_exposure = Counter(), Counter()
    for mode, plan in plans.items():
        assert plan['paired_cross_goals'] and plan['world_size'] == 3 and plan['global_batch'] == 12
        assert plan['geometry_replay'] == mode and plan['geometry_replay_slots'] == 3 and plan['task_balanced']
        assert plan['steps'] == steps and plan['step_offset'] == offset
        assert read(root / mode / 'complete.json')['complete']
        assert plan['manifest_sha256'] == file_sha256(plan['manifest'])
        rows = read(plan['manifest'])['states']
        checkpoint = root / mode / f'step_{offset + steps:06d}.pt'
        parent = torch.load(plan['checkpoint'], map_location='cpu', weights_only=False)
        candidate = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert candidate['stage'] == 'grounding' and candidate['optimizer_steps'] == offset + steps
        assert Path(candidate['parent_checkpoint']).resolve() == Path(plan['checkpoint']).resolve()
        assert candidate['provenance']['paired_cross_goals'] is True
        assert parent['mot_trainable'].keys() == candidate['mot_trainable'].keys()
        assert all(torch.equal(value, candidate['mot_trainable'][key]) for key, value in parent['mot_trainable'].items())
        assert parent['policy_guard'].keys() == candidate['policy_guard'].keys()
        allowed = set(plan['trainable_parameters'])
        changed = [key for key, value in parent['policy_guard'].items() if not torch.equal(value, candidate['policy_guard'][key])]
        assert changed and all('guard.' + key in allowed for key in changed)
        assert all(candidate['policy_guard'][key].count_nonzero() == 0 for key in ZEROED)
        arms[mode] = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
            parent_checkpoint=plan['checkpoint'], policy_tensors_identical=True,
            only_declared_semantics_changed=True, changed_guard_tensors=changed, residual_output_still_zero=True)
        del parent, candidate
        journals[mode] = [[json.loads(s) for s in (root / mode / f'rank{rank}.jsonl').read_text().splitlines()] for rank in range(3)]
        assert all([r['step'] for r in journal] == list(range(offset + 1, offset + steps + 1)) for journal in journals[mode])
        stream = geometry_mixture(rows, plan['seed'], batch_size=12, slots=3, mode=mode, task_balanced=True, world_size=3)
        batches[mode] = [paired_cross_goal_batch(next(stream), 3) for _ in range(steps)]
        for i, batch in enumerate(batches[mode]):
            norms = [journal[i]['grad_norm'] for journal in journals[mode]]
            assert len(set(norms)) == 1 and math.isfinite(norms[0]) and norms[0] > 0
            for rank, journal in enumerate(journals[mode]):
                record, selected = journal[i], batch[rank::3]
                assert math.isfinite(record['mean_loss'])
                assert record['samples'] == [[r['id'], observation] for r, observation, partial, instruction in selected]
                assert record['partial_geometry_slots'] == [partial for r, observation, partial, instruction in selected]
                assert record['cross_goal_slots'] == [observation != instruction for r, observation, partial, instruction in selected]
                assert record['instruction_languages'] == [instruction for r, observation, partial, instruction in selected]
                assert sum(record['cross_goal_slots']) == 1
                assert all(r['replay_split'] == 'train' for r, observation, partial, instruction in selected)
    assert plans['ordinary']['manifest_sha256'] == plans['fg']['manifest_sha256']
    for left, right in zip(batches['ordinary'], batches['fg'], strict=True):
        for (a, obs_a, partial_a, instr_a), (b, obs_b, partial_b, instr_b) in zip(left, right, strict=True):
            assert (obs_a, partial_a, instr_a) == (obs_b, partial_b, instr_b)
            if partial_a:
                assert (a['pair_id'], a['task_config']) == (b['pair_id'], b['task_config'])
                assert not a.get('fg_correction') and b['fg_correction']
            else:
                assert a == b
                exposure[a['pair_id']] += 1
                if obs_a != instr_a:
                    crossed_exposure[a['pair_id']] += 1
    assert sum(exposure.values()) == steps * 9 and sum(crossed_exposure.values()) == steps * 3
    return dict(complete=True, format='robotwin_paired_cross_goal_training_audit_v1',
        local_optimizer_steps=steps, step_offset=offset, samples_per_arm=steps * 12,
        common_semantic_samples_per_arm=steps * 9, cross_goal_samples_per_arm=steps * 3,
        matched_partial_samples_per_arm=steps * 3, same_observation_goal_pairs_per_arm=steps * 3,
        actual_samples_and_instruction_branches_verified=True, all_rank_norms_equal_and_finite=True,
        common_expert_samples=dict(exposure), cross_goal_expert_samples=dict(crossed_exposure), arms=arms,
        action_quality_evaluated=False, phase_history_labels_on_cross_goal_views='invalid; no invented history')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    target = args.output / 'training_audit.json'
    if target.exists():
        raise FileExistsError(target)
    report = audit(args.output)
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
