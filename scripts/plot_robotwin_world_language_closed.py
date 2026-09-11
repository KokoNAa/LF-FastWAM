#!/usr/bin/env python3
"""Plot the complete, audited 1,200-episode language intervention study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

TASKS = ['blocks_ranking_rgb', 'stack_blocks_two', 'place_a2b_left',
         'place_a2b_right', 'place_burger_fries']
LABELS = ['Ranking', 'Stacking', 'Place left', 'Place right', 'Burger / fries']
MODELS = ['released', 'no_eraf']
CELLS = [('source', 'source'), ('target', 'source'),
         ('source', 'target'), ('target', 'target')]
CELL_LABELS = ['S / S', 'CF / S', 'S / CF', 'CF / CF']
CATEGORIES = ['source', 'counterfactual', 'neither', 'ambiguous']
EFFECTS = ['video_at_source_action', 'video_at_target_action',
           'action_at_source_video', 'action_at_target_video']
EFFECT_LABELS = ['Video | action S', 'Video | action CF',
                 'Action | video S', 'Action | video CF']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(q, audit):
    """Reject partial coverage and cross-check plotted totals against raw-data audit."""
    require(audit.get('complete') is True and audit.get('allow_partial') is False
            and audit.get('validated_episodes') == 1200 and not audit.get('pending_arms'),
            'Final closed-data audit of all 1200 episodes is required')
    require(q.get('compute_coverage_complete') is True
            and q.get('actual_closed_loop_episodes') == 1200,
            'Complete quantitative coverage is required')
    names = {f'{m}-closed-v_{v}-a_{a}-seed{s}' for m in MODELS
             for v, a in CELLS for s in [42, 43, 44]}
    require(set(q['closed_loop']) == set(audit['completed_arms']) == names,
            'Expected exact 24-arm coverage')
    for name in names:
        arm = audit['completed_arms'][name]
        require(arm['process_exit']['status'] == 'exited'
                and arm['process_exit']['exit_code'] == 0 and arm['episodes'] == 50,
                'Nonterminal or incomplete arm: ' + name)
        for field in ['episodes', 'source_success', 'cf_success']:
            require(q['closed_loop'][name][field] == arm[field],
                    'Quantitative/audit arm mismatch: ' + name)
        require(len(arm['tasks']) == 5 and {r['task'] for r in arm['tasks']} == set(TASKS)
                and all(r['episodes'] == 10 for r in arm['tasks']),
                'Task coverage mismatch: ' + name)
    indexed = {}
    for r in q['statistics']:
        if not r['phase'].startswith('closed_'):
            continue
        key = (r['model'], r['task'], r['phase'], r['metric'])
        require(key not in indexed, 'Duplicate statistic: ' + str(key))
        require(r['scenes'] == 10 and r['repeated_observations'] == 30,
                'Expected 10 scenes with 3 repeated noise seeds')
        require(len(r['ci95']) == 2 and all(math.isfinite(x) for x in [r['mean'], *r['ci95']])
                and r['ci95'][0] <= r['ci95'][1], 'Invalid statistic interval')
        indexed[key] = r

    def get(m, t, phase, metric):
        key = (m, t, phase, metric)
        require(key in indexed, 'Missing statistic: ' + str(key))
        return indexed[key]

    mapping = {'source_goal_ever_success': 'source_success',
               'counterfactual_goal_ever_success': 'cf_success',
               'any_correct_object_lifted': 'any_correct_lift',
               'all_instruction_objects_lifted': 'all_instruction_lift',
               'full_goal_after_any_lift': 'full_goal_after_lift',
               'correct_placement_after_lift': 'strict_placement_after_lift'}
    for m in MODELS:
        for t in TASKS:
            for v, a in CELLS:
                phase = f'closed_v_{v}_a_{a}'
                task_rows = [next(r for r in audit['completed_arms'][
                    f'{m}-closed-v_{v}-a_{a}-seed{s}']['tasks'] if r['task'] == t)
                    for s in [42, 43, 44]]
                for metric, field in mapping.items():
                    r = get(m, t, phase, metric)
                    require(abs(r['mean'] * 30 - sum(x[field] for x in task_rows)) < 1e-8,
                            'Plotted metric disagrees with audited counts: ' + str((m, t, phase, metric)))
                first = [get(m, t, phase, 'first_goal_' + c)['mean'] for c in CATEGORIES]
                require(all(0 <= x <= 1 for x in first) and abs(sum(first) - 1) < 1e-8,
                        'First-goal categories do not partition episodes')
            for metric in ['source_goal_ever_success', 'counterfactual_goal_ever_success']:
                values = [get(m, t, f'closed_v_{v}_a_{a}', metric)['mean'] for v, a in CELLS]
                ss, ts, st, tt = values
                for effect, expected in zip(EFFECTS, [ts - ss, tt - st, st - ss, tt - ts]):
                    r = get(m, t, 'closed_paired', effect + '_' + metric)
                    require(abs(r['mean'] - expected) < 1e-8,
                            'Paired-effect sign or cell mismatch')
    return indexed


def render(q_path, audit_path, output):
    q = json.loads(q_path.read_text())
    audit = json.loads(audit_path.read_text())
    indexed = validate_inputs(q, audit)
    # Matplotlib is needed only after the final-data guards have passed.
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    require(not output.exists() or not any(output.iterdir()),
            'Use a new output directory to preserve existing figure evidence')
    output.mkdir(parents=True, exist_ok=True)
    used = {}
    files = []

    def stat(m, t, phase, metric):
        key = (m, t, phase, metric)
        used[key] = indexed[key]
        return indexed[key]

    def finish(fig, name, footer):
        fig.text(.5, .018, footer, ha='center', va='bottom', fontsize=9)
        fig.tight_layout(rect=[0, .12, 1, .90])
        for ext in ['png', 'pdf']:
            p = output / f'{name}.{ext}'
            fig.savefig(p, dpi=160)
            files.append({'path': str(p), 'sha256': sha(p)})
        plt.close(fig)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 5, figsize=(17, 8), sharey=True)
    colors = ['#3478ad', '#df8830', '#d5d8db', '#965a9e']
    for mi, m in enumerate(MODELS):
        for ti, t in enumerate(TASKS):
            ax = axes[mi, ti]
            bottom = np.zeros(4)
            for c, color in zip(CATEGORIES, colors):
                values = np.array([stat(m, t, f'closed_v_{v}_a_{a}', 'first_goal_' + c)['mean']
                                   for v, a in CELLS]) * 100
                ax.bar(range(4), values, bottom=bottom, color=color, label=c)
                bottom += values
            ax.set_title(LABELS[ti] + ' / ' + m.replace('_', '-'))
            ax.set_xticks(range(4), CELL_LABELS, rotation=25)
            ax.set_ylim(0, 100)
            if ti == 0:
                ax.set_ylabel('First-goal category (%)')
    fig.suptitle('Which goal was reached first?', y=.99)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .955), ncol=4, frameon=False)
    finish(fig, 'closed_first_goal',
           'Cell labels: video language / action language; S = source, CF = counterfactual. Each bar: 30 runs on 10 scenes (3 noise seeds).\n'
           'Action language also selects the termination goal. First-goal order uses the reviewed immediate-termination invariant; ambiguous cases stay separate.')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for mi, m in enumerate(MODELS):
        for ci, (metric, title) in enumerate([
                ('all_instruction_objects_lifted', 'All instruction objects correctly lifted'),
                ('correct_placement_after_lift', 'Strict placement after lift')]):
            ax = axes[mi, ci]
            values = np.array([[stat(m, t, f'closed_v_{v}_a_{a}', metric)['mean']
                                for v, a in CELLS] for t in TASKS])
            ax.imshow(values, vmin=0, vmax=1, cmap='Blues', aspect='auto')
            for i in range(5):
                for j in range(4):
                    ax.text(j, i, f'{round(values[i, j] * 30)}/30', ha='center', va='center',
                            color='white' if values[i, j] > .55 else '#17293a')
            ax.set_xticks(range(4), CELL_LABELS)
            ax.set_yticks(range(5), LABELS)
            ax.set_title(m.replace('_', '-') + ': ' + title)
    fig.suptitle('Physical execution under language interventions', y=.98)
    finish(fig, 'closed_manipulation',
           'Cell labels: video / action language. Counts use 10 matched scenes with 3 noise seeds; instruction objects and targets follow action language.\n'
           'Strict placement requires the recorded grasp/lift and placement conditions; it is distinct from benchmark goal success. No extra settling steps.')

    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True, sharey=True)
    for mi, m in enumerate(MODELS):
        for ti, t in enumerate(TASKS):
            ax = axes[mi, ti]
            for gi, (metric, label, color) in enumerate([
                    ('source_goal_ever_success', 'Source goal', '#3478ad'),
                    ('counterfactual_goal_ever_success', 'CF goal', '#df8830')]):
                rows = [stat(m, t, 'closed_paired', e + '_' + metric) for e in EFFECTS]
                means = np.array([r['mean'] for r in rows]) * 100
                limits = np.array([r['ci95'] for r in rows]) * 100
                y = np.arange(4) + (gi - .5) * .18
                # Drawing interval lines directly also handles a percentile interval excluding its point estimate.
                ax.hlines(y, limits[:, 0], limits[:, 1], color=color, linewidth=1.5)
                ax.plot(means, y, 'o', color=color, ms=4, label=label)
            ax.axvline(0, color='#777777', lw=.8)
            ax.set_xlim(-105, 105)
            ax.set_xticks([-100, 0, 100])
            ax.set_yticks(range(4), EFFECT_LABELS)
            ax.set_title(LABELS[ti] + ' / ' + m.replace('_', '-'))
            ax.grid(axis='x', alpha=.15)
            if mi == 1:
                ax.set_xlabel('Change (percentage points)')
    axes[0, 0].invert_yaxis()
    fig.suptitle('Paired effect of switching source language to CF language', y=.99)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .955), ncol=2, frameon=False)
    finish(fig, 'closed_paired_language_effects',
           'Other language held fixed as labelled; action-language contrasts also change the selected termination goal. 10 scenes/task, 3 repeated seeds.\n'
           'Points: within-scene paired mean differences. Lines: 95% scene-percentile bootstrap intervals, no multiplicity correction.\n'
           'Intervals may collapse for constant scene outcomes; this does not establish population certainty or equivalence.')
    data = {'sources': {str(p): sha(p) for p in [q_path, audit_path]},
            'plan_sha256': audit['plan_sha256'], 'statistics': list(used.values()),
            'episodes': 1200, 'unique_scenes_per_task': 10, 'noise_seeds': [42, 43, 44]}
    dp = output / 'figure_data.json'
    dp.write_text(json.dumps(data, indent=2) + '\n')
    manifest = {'complete': True, 'figures': 3, 'files': files,
                'figure_data_sha256': sha(dp), 'visual_review_complete': False,
                'scope': 'Final closed-loop figures only. Rendering is not visual review or whole-study completion.'}
    (output / 'figures.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'figures': 3, 'statistics': len(used), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quantitative', required=True, type=Path)
    parser.add_argument('--closed-audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    render(args.quantitative, args.closed_audit, args.output)
