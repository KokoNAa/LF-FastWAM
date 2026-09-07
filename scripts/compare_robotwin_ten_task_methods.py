#!/usr/bin/env python3
"""Compare complete ten-task development matrices using paired episode evidence."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from scripts.compare_robotwin_eraf_fg_evaluations import compare, evaluation

GROUPS = {
    'original_five': (6, {'blocks_ranking_rgb', 'stack_blocks_two', 'place_a2b_left', 'place_a2b_right', 'place_burger_fries'}),
    'additional_five': (3, {'blocks_ranking_size', 'place_empty_cup', 'place_mouse_pad', 'move_stapler_pad', 'move_pillbottle_pad'}),
}


def score_cells(cells):
    """Equal task weights despite unequal scene counts; reject partial matrices."""
    expected = {task: count for count, tasks in GROUPS.values() for task in tasks}
    if set(cells) != set(expected):
        raise ValueError('Exactly the ten declared tasks are required.')
    for task, cell in cells.items():
        if cell['episodes'] != expected[task] or not 0 <= cell['successes'] <= cell['episodes']:
            raise ValueError('Incomplete or invalid task count: ' + task)
    # An exact tie must stay a tie regardless of completion/dictionary order.
    # Floating accumulation previously made equal30% scores differ by1e-17.
    return float(sum((Fraction(c['successes'], c['episodes']) for c in cells.values()),
                     Fraction()) / len(expected))


def build_report(config):
    target = config['target']
    methods = config['methods']
    if target not in methods or len(methods) < 2:
        raise ValueError('A target and actual competing methods are required.')
    scores = {}
    bindings = {}
    for name, groups in methods.items():
        if set(groups) != set(GROUPS):
            raise ValueError('Every method needs both complete task groups.')
        cells = {}
        checkpoints = set()
        for group, path in groups.items():
            summary, evidence = evaluation(path)
            count, tasks = GROUPS[group]
            if set(evidence) != {(task, 'counterfactual') for task in tasks}:
                raise ValueError(f'{name}/{group} does not contain the declared CF matrix.')
            checkpoints.add(summary['checkpoint'])
            for (task, _), (episodes, _initial, _complete) in evidence.items():
                if len(episodes) != count:
                    raise ValueError('Wrong episode budget.')
                cells[task] = dict(episodes=len(episodes), successes=sum(bool(r['selected_goal_success']) for r in episodes))
        if len(checkpoints) != 1:
            raise ValueError('A method uses different checkpoints across task groups.')
        scores[name] = dict(ten_task_macro_cf=score_cells(cells), cells=cells,
                           pooled_successes=sum(c['successes'] for c in cells.values()), pooled_episodes=45)
        bindings[name] = checkpoints.pop()
    paired = {}
    for name, groups in methods.items():
        if name == target:
            continue
        result = {}
        for group in GROUPS:
            report, episodes = compare(groups[group], methods[target][group])
            result[group] = dict(report=report, episodes=episodes)
        paired[name] = dict(groups=result,
            gains=sum(len(c['gained_scene_seeds']) for r in result.values() for c in r['report']['cells']),
            losses=sum(len(c['lost_scene_seeds']) for r in result.values() for c in r['report']['cells']),
            macro_delta=scores[target]['ten_task_macro_cf'] - scores[name]['ten_task_macro_cf'])
    # Supplement the historical burger predicate consistently for every method.
    slots = {}
    for name, groups in methods.items():
        path = Path(groups['original_five']) / 'place_burger_fries/demo_clean/counterfactual/episodes.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        strict = simultaneous = 0
        for row in rows:
            original = {a['actor']: a for a in row['final_source_goal']['assignments']}
            cf = {a['actor']: a for a in row['final_counterfactual_goal']['assignments']}
            if original.keys() != cf.keys() or len(cf) != 2:
                raise ValueError('Incomplete burger slot geometry.')
            nearest = all(a['planar_distance'] < original[actor]['planar_distance'] for actor, a in cf.items())
            strict += bool(row['counterfactual_goal_final_success'] and nearest)
            simultaneous += bool(row['source_goal_final_success'] and row['counterfactual_goal_final_success'])
        slots[name] = dict(episodes=len(rows), strict_final_slot_successes=strict, simultaneous_final_goals=simultaneous)
    return dict(complete=True, format='robotwin_ten_task_method_comparison_v1', target=target,
        methods=scores, checkpoint_paths=bindings, paired_target_vs_controls=paired,
        target_strictly_exceeds_all_listed_dev_controls=all(r['macro_delta'] > 0 for r in paired.values()),
        matched_episodes_per_comparison=45, supplementary_final_slots=slots,
        independent_test=False, goal_achievement_claim=False,
        macro_arithmetic='Exact rational task fractions, converted once to float; task order cannot break ties.',
        scope='Complete development CF matrices only; equal-task macro is primary. Paired gains/losses and slot ownership are descriptive. A development lead alone does not establish independent superiority or complete the user goal.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = build_report(json.loads(args.config.read_text()))
    report['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: r['ten_task_macro_cf'] for name, r in report['methods'].items()}, indent=2))


if __name__ == '__main__':
    main()
