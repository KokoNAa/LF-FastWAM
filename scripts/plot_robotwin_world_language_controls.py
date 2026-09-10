#!/usr/bin/env python3
"""Plot paraphrase controls and action direction at distinct-reference states."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from scripts.probe_robotwin_world_language import read, sha, TASKS
from scripts.run_robotwin_world_language_collection import write

MODELS = ['released', 'no_eraf']
LABELS = ['Ranking', 'Stacking', 'Place\nleft', 'Place\nright', 'Burger /\nfries']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    root = ap.parse_args().root
    plan_path = root / 'probes/plan.json'; plan = read(plan_path)
    q = read(root / 'report/quantitative_report.json')
    audit_path = root / 'expert_reference_window_audit.json'; audit = read(audit_path)
    assert audit['complete'] and audit['plan_sha256'] == sha(plan_path)
    assert all(q['state_coverage'][m]['features_cache'] == len(plan['states']) for m in MODELS)
    phases = {task: 'initial' if i < 2 else 'shared_decision' for i, task in enumerate(TASKS)}
    for task in TASKS:
        rows = [r for r in audit['rows'] if r['task'] == task and r['phase'] == phases[task]]
        assert len(rows) == 10 and all(not r['identical_action_window'] for r in rows)
    used = {}
    def stat(model, task, phase, metric):
        rows = [r for r in q['statistics'] if (r['model'], r['task'], r['phase'], r['metric']) == (model, task, phase, metric)]
        assert len(rows) == 1 and rows[0]['scenes'] == 10
        used[(model, task, phase, metric)] = rows[0]
        return rows[0]
    out = root / 'report/language_control_figures'; out.mkdir(exist_ok=True)
    files = []
    def save(fig, name):
        for extension in ['png', 'pdf']:
            path = out / f'{name}.{extension}'
            fig.savefig(path, dpi=160)
            files.append(dict(path=str(path), sha256=sha(path)))
        plt.close(fig)
    def bars(ax, rows, positions, width, **kwargs):
        y = np.array([r['mean'] for r in rows]); ci = np.array([r['ci95'] for r in rows])
        ax.bar(positions, y, width=width, yerr=np.stack([y - ci[:, 0], ci[:, 1] - y]),
               error_kw=dict(capsize=2, elinewidth=.8), **kwargs)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    contrasts = [('target', 'Change goal', '#2467a5'),
                 ('source_paraphrase', 'Rephrase source', '#dd8534'),
                 ('target_paraphrase', 'Rephrase target', '#34916b')]
    for i, model in enumerate(MODELS):
        ax = axes[i]
        for j, (comparison, label, color) in enumerate(contrasts):
            rows = [stat(model, task, 'initial', comparison + '_last_hidden_relative_l2') for task in TASKS]
            bars(ax, rows, np.arange(5) + (j - 1) * .25, .23, color=color, label=label)
        ax.set_title(model.replace('_', '-')); ax.set_xticks(range(5), LABELS)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylabel('Last video hidden: relative L2 difference')
    fig.suptitle('Initial-state language controls: goal change versus same-goal rephrasing', y=.995)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .945), ncol=3, frameon=False)
    fig.text(.5, .025, '10 scenes per task; 95% scene-bootstrap intervals. Each paraphrase is compared with its own original instruction.\nOne checked paraphrase per goal; sensitivity differences do not establish general language invariance or goal success.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .10, 1, .89]); save(fig, 'goal_vs_paraphrase')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for i, (branch, held, title) in enumerate([
            ('video', 'action', 'Change video language; hold action text'),
            ('action', 'video', 'Change action text; hold video KV')]):
        ax = axes[i]
        for j, (model, condition) in enumerate([(m, c) for m in MODELS for c in ['source', 'target']]):
            rows = [stat(model, task, phases[task], f'{branch}_at_{condition}_{held}_target_axis_effect') for task in TASKS]
            bars(ax, rows, np.arange(5) + (j - 1.5) * .19, .18,
                 color='#2467a5' if model == 'released' else '#dd8534',
                 hatch='' if condition == 'source' else '///', edgecolor='white', linewidth=.5,
                 label=model.replace('_', '-') + ', ' + condition + ' held')
        ax.axhline(0, color='#444444', lw=.8); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.set_title(title); ax.set_xticks(range(5), LABELS)
    axes[0].set_ylabel('Action shift along source-to-target expert direction')
    fig.suptitle('Offline action direction: states with distinct expert futures', y=.995)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .945), ncol=4, frameon=False, fontsize=9)
    fig.text(.5, .025, 'Ranking/stacking: initial state. Placement tasks: shared-decision state; their initial expert futures are identical.\n10 scenes per task, 3 noise draws; 95% scene-bootstrap intervals. +1 is one expert-separation vector component, not 100% success.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .10, 1, .87]); save(fig, 'expert_direction_interventions')
    data = out / 'figure_data.json'
    write(data, dict(plan_sha256=sha(plan_path), reference_window_audit_sha256=sha(audit_path),
                     action_direction_phases=phases, statistics=list(used.values())))
    write(out / 'figures.json', dict(complete=True, files=files, data_sha256=sha(data),
          scope='Completed feature/cache experiment controls only; generated-video semantics and closed-loop outcomes remain separate.'))
    print(json.dumps(dict(figures=2, files=len(files), output=str(out))))


if __name__ == '__main__':
    main()
