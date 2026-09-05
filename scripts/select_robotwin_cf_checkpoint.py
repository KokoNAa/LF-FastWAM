#!/usr/bin/env python3
"""Rank completed paired evaluations by CF success under a native-task floor.

This selects a feasible checkpoint; it does not declare the research goal met.
Run on episode records already produced by the ordinary CIS evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.summarize_robotwin_cis import load_job_output

TASKS = ('blocks_ranking_rgb', 'place_a2b_left', 'place_a2b_right',
         'place_burger_fries', 'stack_blocks_two')
CONDITIONS = ('correct', 'counterfactual')


def read_evaluation(root):
    records = {}
    for task in TASKS:
        for condition in CONDITIONS:
            _, rows = load_job_output(Path(root) / task / 'demo_clean' / condition,
                                      expected_episodes=3, expected_source_task=task,
                                      expected_task_config='demo_clean', expected_condition=condition)
            for row in rows:
                key = task, condition, row['scene_seed']
                if key in records:
                    raise ValueError(f'Duplicate scene: {key}')
                records[key] = row
    if len({row['checkpoint'] for row in records.values()}) != 1:
        raise ValueError('Evaluation mixes multiple checkpoints.')
    return records


def summarize(records, baseline, native_minimum):
    fields = ('policy_instruction', 'source_instruction', 'counterfactual_instruction',
              'instruction_type', 'step_limit')
    matched = set(records) == set(baseline) and all(
        all(row[field] == baseline[key][field] for field in fields) for key, row in records.items())
    counts = {condition: sum(row['selected_goal_success'] for key, row in records.items()
                            if key[1] == condition) for condition in CONDITIONS}
    by_task = {task: {condition: sum(row['selected_goal_success'] for key, row in records.items()
                                   if key[:2] == (task, condition)) for condition in CONDITIONS}
               for task in TASKS}
    cf_keys = {key for key, row in records.items() if key[1] == 'counterfactual' and row['selected_goal_success']}
    gained = [list(key) for key in sorted(cf_keys) if key in baseline and not baseline[key]['selected_goal_success']]
    cf_failures = [row for key, row in records.items() if key[1] == 'counterfactual' and not row['selected_goal_success']]
    return {'complete': True, 'metadata_matched_to_base': matched,
            'correct_successes': counts['correct'], 'cf_successes': counts['counterfactual'],
            'episodes_per_condition': 15, 'native_minimum': native_minimum,
            'eligible': matched and counts['correct'] >= native_minimum,
            'cf_successful_tasks': sorted({key[0] for key in cf_keys}),
            'cf_gained_scenes': gained, 'by_task': by_task,
            'failed_cf_completed_source_goal': sum(row['source_goal_ever_success'] for row in cf_failures),
            'failed_cf_completed_neither_goal': sum(not row['source_goal_ever_success'] for row in cf_failures),
            'checkpoint': next(iter(records.values()))['checkpoint']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', action='append', required=True, metavar='NAME=ROOT')
    parser.add_argument('--native-minimum', type=int, default=10)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    baseline = read_evaluation(args.baseline)
    models = {}
    for item in args.candidate:
        name, path = item.split('=', 1)
        try:
            models[name] = summarize(read_evaluation(path), baseline, args.native_minimum)
        except (OSError, ValueError, KeyError) as error:
            models[name] = {'complete': False, 'eligible': False, 'error': str(error)}
    eligible = [name for name, result in models.items() if result['eligible']]
    # Preserve caller order on exact ties, allowing an earlier checkpoint first.
    best = max(eligible, key=lambda name: (models[name]['cf_successes'], models[name]['correct_successes']), default=None)
    result = {'format': 'robotwin_cf_native_constrained_selection_v1', 'models': models,
              'best_feasible_candidate': best,
              'scope': 'Feasibility and observed counts only. A small gain does not establish substantial or general improvement.'}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'best_feasible_candidate': best, 'models': {
        name: {key: row.get(key) for key in ('eligible', 'correct_successes', 'cf_successes', 'cf_successful_tasks', 'error')}
        for name, row in models.items()}}, indent=2))


if __name__ == '__main__':
    main()
