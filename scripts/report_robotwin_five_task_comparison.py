#!/usr/bin/env python3
"""Paired scene uncertainty for a complete, unchanged five-task comparison."""
from __future__ import annotations

import argparse
from fractions import Fraction
from math import comb
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from scripts.run_robotwin_formal_five40 import read, records, sha, write
from scripts.report_robotwin_formal_five40 import ARM_NAMES, TASK_NAMES, summarize_cell, frac

ARMS = tuple(ARM_NAMES)
TASKS = tuple(TASK_NAMES)
PAIRS = (('no_eraf', 'eraf_only'), ('eraf_only', 'eraf_fg'), ('no_eraf', 'eraf_fg'))


def exact_paired_p(gains, losses):
    """Two-sided conditional binomial test over discordant scene pairs."""
    n = gains + losses
    return min(1., 2 * sum(comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n)


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def paired_summary(outcomes, *, draws=10000, seed=42):
    """Resample paired three-arm outcomes within each of the five fixed tasks."""
    if set(outcomes) != set(TASKS) or draws < 1000:
        raise ValueError('All five fixed tasks and at least 1000 bootstrap draws required.')
    for values in outcomes.values():
        if not values or any(len(v) != 3 or any(type(x) is not bool for x in v) for v in values):
            raise ValueError('Every scene needs exactly three Boolean outcomes.')
    exact_macro = {a: sum(Fraction(sum(row[i] for row in outcomes[t]), len(outcomes[t])) for t in TASKS) / 5
                   for i, a in enumerate(ARMS)}
    macro = {a: float(value) for a, value in exact_macro.items()}
    samples = [[] for _ in PAIRS]
    rng = random.Random(seed)
    for _ in range(draws):
        rates = [0., 0., 0.]
        for task in TASKS:
            values = outcomes[task]
            for _ in values:
                row = values[rng.randrange(len(values))]
                for i in range(3): rates[i] += row[i] / (5 * len(values))
        for j, (a, b) in enumerate(PAIRS):
            samples[j].append(rates[ARMS.index(b)] - rates[ARMS.index(a)])
    comparisons = []
    for j, (a, b) in enumerate(PAIRS):
        ia, ib = ARMS.index(a), ARMS.index(b)
        cells = []
        for task in TASKS:
            values = outcomes[task]
            gains = sum(not row[ia] and row[ib] for row in values)
            losses = sum(row[ia] and not row[ib] for row in values)
            cells.append(dict(task=task, episodes=len(values), gains=gains, losses=losses,
                              net=gains-losses, difference=(gains-losses)/len(values)))
        gains, losses = sum(x['gains'] for x in cells), sum(x['losses'] for x in cells)
        comparisons.append(dict(reference=a, candidate=b, by_task=cells, gains=gains, losses=losses,
            net=gains-losses, macro_difference=float(exact_macro[b]-exact_macro[a]),
            paired_stratified_bootstrap_ci95=[quantile(samples[j], .025), quantile(samples[j], .975)],
            exact_paired_p_two_sided=exact_paired_p(gains, losses)))
    # Holm adjustment across all three reported pairwise tests.
    previous = 0.
    for rank, item in enumerate(sorted(comparisons, key=lambda c: c['exact_paired_p_two_sided'])):
        previous = max(previous, min(1., (3-rank)*item['exact_paired_p_two_sided']))
        item['holm_p_three_comparisons'] = previous
    return dict(macro_cf=macro, comparisons=comparisons,
        desired_order_observed=exact_macro['eraf_fg'] > exact_macro['eraf_only'] > exact_macro['no_eraf'],
        bootstrap=dict(draws=draws, seed=seed, unit='paired scene within task',
                       scope='Scene sampling uncertainty for these five fixed tasks; not training-seed or unseen-task uncertainty.',
                       intervals='Marginal percentile 95% intervals; not simultaneous confidence intervals.'),
        goal_achieved=False)


def report(root, *, draws=10000):
    root = Path(root)
    state, terminal = read(root/'status.json'), read(root/'terminal_verification.json')
    if not state.get('complete') or state.get('status') != 'complete' or not terminal.get('complete'):
        raise ValueError('All cells and terminal verification must be complete.')
    formal = (root/'frozen_protocol.json').exists()
    plan = read(root/('frozen_protocol.json' if formal else 'protocol.json'))
    models = plan['models'] if formal else state['arms']
    n = 40 if formal else plan['dev_episodes']
    if plan.get('smoke_only') or set(plan['tasks']) != set(TASKS) or set(models) != set(ARMS):
        raise ValueError('Only the complete three-arm five-task matrix is accepted.')
    if terminal['episodes'] != 15*n: raise ValueError('Unexpected episode total.')
    outcomes, cells, sources = {}, [], {}
    for task in TASKS:
        catalog = root/'catalog'/task/'demo_clean/correct/episodes.jsonl'
        canonical = records(catalog)
        if len(canonical) != n or len({c['scene_seed'] for c in canonical}) != n:
            raise ValueError('Canonical scenes must be complete and unique.')
        signatures = [dict(scene_seed=c['scene_seed'], sha256=c['initial_physical_state_sha256']) for c in canonical]
        sources[str(catalog.relative_to(root))] = sha(catalog)
        values = []
        for arm in ARMS:
            folder = root/'evaluation'/arm/task/'demo_clean/counterfactual'
            path = folder/'episodes.jsonl'
            episodes, complete = records(path), read(folder/'complete.json')
            model_sha = models[arm]['sha256' if formal else 'final_sha256']
            if len(episodes) != n or not complete['complete'] or read(folder/'initial_states.json') != signatures:
                raise ValueError('Incomplete cell or unpaired physical states.')
            if complete['checkpoint_sha256'] != model_sha or complete['canonical_sha256'] != sha(catalog):
                raise ValueError('Model/catalog binding changed.')
            if (complete['eraf'] != models[arm]['eraf'] or complete['memory_mode'] != 'carry'
                or complete['policy_kind'] != 'repair' or not complete['manipulation_metrics']
                or complete['deployment'] != dict(action_horizon=32, replan_steps=24, inference_steps=10)):
                raise ValueError('Inference protocol changed.')
            for e, c in zip(episodes, canonical):
                if any(e[k] != c[k] for k in ('scene_seed', 'episode_index', 'source_instruction', 'counterfactual_instruction')):
                    raise ValueError('Scene/instruction ordering changed.')
                if (e['source_task'] != task or any(e[k] != 'counterfactual' for k in ('condition', 'selected_goal', 'instruction_goal'))
                    or type(e['counterfactual_goal_ever_success']) is not bool):
                    raise ValueError('Wrong task, goal or outcome type.')
            values.append([e['counterfactual_goal_ever_success'] for e in episodes])
            cells.append(dict(task=task, arm=arm, **summarize_cell(episodes)))
            for name in ('episodes.jsonl', 'initial_states.json', 'complete.json'):
                sources[str((folder/name).relative_to(root))] = sha(folder/name)
        outcomes[task] = list(zip(*values))
    return dict(format='robotwin_five_task_paired_comparison_v1', complete=True, episodes=15*n,
        episodes_per_cell=n, cells=cells, source_sha256=sources,
        independent_test=bool(plan.get('independent_test', False)),
        checkpoint_sha256={a: m['sha256' if formal else 'final_sha256'] for a, m in models.items()},
        correct_evaluated=False, **paired_summary(outcomes, draws=draws))


def markdown(result):
    n = result['episodes_per_cell']
    scope = '独立正式测试' if result['independent_test'] else '开发/历史诊断比较'
    text = [f'# 五任务三组配对比较：{scope}', '',
        f'每任务每模型 {n} 个配对场景，总计 {15*n} 回合；五任务等权。', '',
        '| 任务 | no-eraf | ERAF | ERAF+FG |', '|---|---:|---:|---:|']
    for t in TASKS:
        text.append('| '+TASK_NAMES[t]+' | '+' | '.join(frac(next(c['cf'] for c in result['cells'] if c['task']==t and c['arm']==a), n) for a in ARMS)+' |')
    text.extend(['| 宏平均 | '+' | '.join(f'{100*result["macro_cf"][a]:.1f}%' for a in ARMS)+' |', '',
        '| 配对比较 | 新增/丢失成功 | 宏平均变化 | 95%区间 | Holm校正p |', '|---|---:|---:|---:|---:|'])
    for c in result['comparisons']:
        low, high = c['paired_stratified_bootstrap_ci95']
        text.append(f'| {ARM_NAMES[c["candidate"]]} 相对 {ARM_NAMES[c["reference"]]} | {c["gains"]}/{c["losses"]} | {100*c["macro_difference"]:+.1f}个百分点 | [{100*low:+.1f}, {100*high:+.1f}] | {c["holm_p_three_comparisons"]:.4f} |')
    text.extend(['', '区间按任务内配对场景重采样，保留三组在同一场景上的关联；它们是单项95%区间。p值为不一致回合的双侧精确配对检验，并对三项比较作Holm校正。',
        '不确定性只覆盖固定五任务中的场景抽样，不覆盖重新训练、更多任务或资产变化。观察到均值排序不自动等于显著优势。', '',
        '| 模型 | 正确夹起 | 夹起后任务目标 | 严格脱手放置 |', '|---|---:|---:|---:|'])
    for a in ARMS:
        cells = [c for c in result['cells'] if c['arm']==a]
        lift = sum(c['lift'] for c in cells)
        text.append(f'| {ARM_NAMES[a]} | {frac(lift, 5*n)} | {frac(sum(c["cf_after_lift"] for c in cells), lift)} | {frac(sum(c["strict_release_placement"] for c in cells), lift)} |')
    text.extend(['', '夹起指至少一个指令物体被验证夹起；严格脱手未确认不能单独判定放置失败。完整逐任务物理指标保存在JSON。未评测Correct。', ''])
    return '\n'.join(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = report(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output/'PAIRED_COMPARISON.json', result)
    (args.output/'PAIRED_COMPARISON.md').write_text(markdown(result))


if __name__ == '__main__': main()
