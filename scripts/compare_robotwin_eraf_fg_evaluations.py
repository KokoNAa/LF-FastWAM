#!/usr/bin/env python3
"""Audit and compare two complete, physically matched repair evaluations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def evaluation(root):
    root = Path(root)
    summary = read(root / 'summary.json')
    if summary.get('complete') is not True:
        raise ValueError(f'Incomplete evaluation: {root}')
    cells = {}
    for cell in summary['cells']:
        key = (cell['source_task'], cell['condition'])
        if key in cells:
            raise ValueError(f'Duplicate evaluation cell: {key}')
        folder = root / key[0] / 'demo_clean' / key[1]
        complete = read(folder / 'complete.json')
        episodes = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
        initial = read(folder / 'initial_states.json')
        if (complete.get('complete') is not True or len(episodes) != cell['episodes']
                or len(initial) != len(episodes)
                or cell['selected_goal_successes'] != sum(x['selected_goal_success'] for x in episodes)
                or complete['checkpoint'] != summary['checkpoint']):
            raise ValueError(f'Incomplete or inconsistent evaluation cell: {folder}')
        cells[key] = (episodes, initial, complete)
    if sum(len(c[0]) for c in cells.values()) != summary['episodes']:
        raise ValueError('Aggregate episode count differs from audited cells.')
    if len({c[2]['checkpoint_sha256'] for c in cells.values()}) != 1:
        raise ValueError('One evaluation contains multiple checkpoint hashes.')
    return summary, cells


def compare(baseline_root, candidate_root, tasks=None):
    baseline, a = evaluation(baseline_root)
    candidate, b = evaluation(candidate_root)
    if tasks is not None:
        requested = set(tasks)
        if not requested or any(not requested.issubset({k[0] for k in source}) for source in (a, b)):
            raise ValueError('Every explicitly requested task must exist in both evaluations.')
        a = {k: v for k, v in a.items() if k[0] in requested}
        b = {k: v for k, v in b.items() if k[0] in requested}
    if not a or a.keys() != b.keys():
        raise ValueError('The two evaluation matrices differ or are empty.')
    cells, pairs = [], []
    identity = ('scene_seed', 'episode_index', 'source_instruction', 'counterfactual_instruction',
                'policy_instruction', 'instruction_goal', 'selected_goal', 'condition', 'source_task')
    for task, condition in sorted(a):
        left, initial_a, meta_a = a[(task, condition)]
        right, initial_b, meta_b = b[(task, condition)]
        if (len(left) != len(right) or initial_a != initial_b
                or meta_a['canonical_sha256'] != meta_b['canonical_sha256']
                or meta_a['deployment'] != meta_b['deployment']):
            raise ValueError(f'Physical state, catalog, or deployment mismatch: {task}/{condition}')
        gained, lost = [], []
        for x, y in zip(left, right, strict=True):
            if any(x[k] != y[k] for k in identity):
                raise ValueError(f'Episode identity or instruction mismatch: {task}/{condition}')
            before, after = bool(x['selected_goal_success']), bool(y['selected_goal_success'])
            if after and not before:
                gained.append(x['scene_seed'])
            if before and not after:
                lost.append(x['scene_seed'])
            pairs.append({'task': task, 'condition': condition, 'scene_seed': x['scene_seed'],
                'baseline_success': before, 'candidate_success': after,
                'baseline_steps': x['steps'], 'candidate_steps': y['steps'],
                'source_instruction': x['source_instruction'],
                'counterfactual_instruction': x['counterfactual_instruction']})
        successes_a = sum(x['selected_goal_success'] for x in left)
        successes_b = sum(x['selected_goal_success'] for x in right)
        cells.append({'task': task, 'condition': condition, 'episodes': len(left),
            'baseline_successes': successes_a, 'candidate_successes': successes_b,
            'delta_successes': successes_b - successes_a,
            'gained_scene_seeds': gained, 'lost_scene_seeds': lost,
            'baseline_eraf': meta_a['eraf'], 'candidate_eraf': meta_b['eraf'],
            'baseline_policy_kind': meta_a['policy_kind'], 'candidate_policy_kind': meta_b['policy_kind']})
    return {'complete': True, 'baseline': baseline['checkpoint'], 'candidate': candidate['checkpoint'],
            'matched_episodes': len(pairs), 'cells': cells,
            'explicit_task_subset': sorted(set(tasks)) if tasks is not None else None,
            'interpretation': 'Paired descriptive results; no component attribution from one comparison.'}, pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'output'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--tasks', nargs='+', help='Explicitly declared diagnostic subset; both inputs must be complete.')
    args = ap.parse_args()
    report, pairs = compare(args.baseline, args.candidate, args.tasks)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'comparison.json').write_text(json.dumps(report, indent=2))
    with (output / 'paired_episodes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
