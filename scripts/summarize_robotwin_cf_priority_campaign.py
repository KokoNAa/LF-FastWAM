#!/usr/bin/env python3
"""Report every selected arm's ten-task CF results, including lost scenes."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_robotwin_cf_priority_campaign import OLD_TASKS, NEW_TASKS


def read(path): return json.loads(Path(path).read_text())


def paired_cell(root, model, task, episodes, reference=None):
    folder = root / model / 'dev' / task / 'demo_clean/counterfactual'
    complete = read(folder / 'complete.json')
    if not complete['complete']:
        raise ValueError(f'Incomplete CF evaluation: {folder}')
    rows = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
    states = read(folder / 'initial_states.json')
    if len(rows) != episodes or len({r['scene_seed'] for r in rows}) != episodes:
        raise ValueError('Missing or duplicated CF episodes.')
    for row in rows:
        if (row['selected_goal'] != 'counterfactual' or row['policy_instruction'] != row['counterfactual_instruction']
                or row['initial_source_goal_success'] or row['initial_counterfactual_goal_success']):
            raise ValueError('CF goal or instruction protocol changed.')
    signature = [{k: r[k] for k in ('scene_seed', 'episode_index', 'source_instruction',
                                   'counterfactual_instruction', 'step_limit')} for r in rows]
    result = dict(successes=sum(r['selected_goal_success'] for r in rows), episodes=episodes,
                  rate=sum(r['selected_goal_success'] for r in rows) / episodes,
                  rows=rows, signature=signature, states=states, checkpoint=complete['checkpoint'])
    if reference is not None:
        if signature != reference['signature'] or states != reference['states']:
            raise ValueError('Candidate and baseline do not share exact CF scenes and instructions.')
        pairs = list(zip(reference['rows'], rows, strict=True))
        result['gained_seeds'] = [a['scene_seed'] for a, b in pairs if not a['selected_goal_success'] and b['selected_goal_success']]
        result['lost_seeds'] = [a['scene_seed'] for a, b in pairs if a['selected_goal_success'] and not b['selected_goal_success']]
    return result


def summarize(root):
    root = Path(root)
    if not read(root / 'driver.json')['complete']:
        raise ValueError('Campaign is incomplete; do not summarize a partial comparison.')
    plan, selection = read(root / 'protocol.json'), read(root / 'selection.json')
    matrices = [root / 'eval_original_five', root / 'eval_additional_five', root / 'eval_eraf_bypass']
    for matrix in matrices:
        driver = read(matrix / 'driver.json')
        if not driver['complete'] or not driver['matched_initial_states']:
            raise ValueError('An evaluation matrix is incomplete or unmatched.')
    names = ['all5_off200'] + [selection[arm]['name'] for arm in plan['arms']]
    screen_plan = read(root / 'eval_original_five_plan.json')
    screen = {}
    for name, model in screen_plan['models'].items():
        candidate_cells = {}
        for task in OLD_TASKS:
            baseline = paired_cell(matrices[0], 'all5_off200', task, 6)
            cell = paired_cell(matrices[0], name, task, 6, baseline)
            if cell['checkpoint'] != model['checkpoint']:
                raise ValueError('Original-five checkpoint does not match the declared screen.')
            candidate_cells[task] = {k: v for k, v in cell.items() if k not in ('rows', 'signature', 'states')}
        screen[name] = dict(cells=candidate_cells,
            successes=sum(c['successes'] for c in candidate_cells.values()), episodes=30)
    cells, checkpoints = {}, {}
    for task in OLD_TASKS + NEW_TASKS:
        matrix, n = (matrices[0], 6) if task in OLD_TASKS else (matrices[1], 3)
        baseline = paired_cell(matrix, 'all5_off200', task, n)
        cells[task] = {}
        for name in names:
            cell = paired_cell(matrix, name, task, n, baseline)
            expected = checkpoints.setdefault(name, cell['checkpoint'])
            if cell['checkpoint'] != expected:
                raise ValueError('A model name refers to different checkpoints across tasks.')
            cells[task][name] = {k: v for k, v in cell.items() if k not in ('rows', 'signature', 'states')}
    aggregates = {name: dict(
        macro_cf=sum(cells[t][name]['rate'] for t in cells) / len(cells),
        successes=sum(cells[t][name]['successes'] for t in cells),
        episodes=sum(cells[t][name]['episodes'] for t in cells),
        gained_scenes=sum(len(cells[t][name]['gained_seeds']) for t in cells),
        lost_scenes=sum(len(cells[t][name]['lost_seeds']) for t in cells),
        regressed_tasks=[t for t in cells if cells[t][name]['rate'] < cells[t]['all5_off200']['rate']])
        for name in names}
    eraf_name = selection['eraf_fg']['name']
    bypass = {}
    for task in OLD_TASKS:
        on = paired_cell(matrices[0], eraf_name, task, 6)
        off = paired_cell(matrices[2], 'selected_eraf_fg_bypass', task, 6, on)
        if off['checkpoint'] != on['checkpoint']:
            raise ValueError('ERAF bypass changed the policy checkpoint.')
        bypass[task] = dict(on_successes=on['successes'], off_successes=off['successes'],
                            gained_by_bypass=off['gained_seeds'], lost_by_bypass=off['lost_seeds'])
    return dict(format='robotwin_cf_priority_campaign_results_v1', complete=True,
        code_commit=plan['code_commit'], selection=selection, cells=cells, aggregates=aggregates,
        all_original_five_candidates=screen,
        training_budget=dict(joint_steps=plan['joint_steps'], eraf_interface_steps=plan['interface_steps']),
        cf_ranking=sorted(names, key=lambda n: aggregates[n]['macro_cf'], reverse=True),
        eraf_fg_strictly_best=aggregates[eraf_name]['macro_cf'] > max(aggregates[n]['macro_cf'] for n in names if n != eraf_name),
        same_checkpoint_eraf_bypass=bypass, independent_test=False, correct_evaluated=False,
        scope='Small fixed development catalog; checkpoint selection uses original-five CF. Additional five tasks were not trained. No statistical-significance or independent-test claim.')


def markdown(report):
    names = list(report['aggregates'])
    lines = ['# ERAF + FG：CF优先实验结果', '',
             '各格为CF成功数/回合数。原五任务每项6回合，新增五任务每项3回合。', '',
             '|任务|' + '|'.join(names) + '|', '|---|' + '---:|' * len(names)]
    for task, cells in report['cells'].items():
        lines.append('|' + task + '|' + '|'.join(f"{cells[n]['successes']}/{cells[n]['episodes']}" for n in names) + '|')
    lines += ['', '|模型|十任务宏平均CF|相对父模型新成功|丢失成功|', '|---|---:|---:|---:|']
    for name in report['cf_ranking']:
        a = report['aggregates'][name]
        lines.append(f"|{name}|{a['macro_cf']:.1%}|{a['gained_scenes']}|{a['lost_scenes']}|")
    lines += ['', '原五任务全部候选筛选结果（包含未入选checkpoint）：', '',
              '|模型|CF成功数|', '|---|---:|']
    for name, result in report['all_original_five_candidates'].items():
        lines.append(f"|{name}|{result['successes']}/{result['episodes']}|")
    lines += ['', 'ERAF+FG是否严格超过本轮所有对照：' + ('是' if report['eraf_fg_strictly_best'] else '否') + '。',
              '同权重ERAF开/关结果与逐场景得失见同目录JSON。三组均从all5_off200出发；'
              f"各训练{report['training_budget']['joint_steps']}步，ERAF组另有{report['training_budget']['eraf_interface_steps']}步接口预热，单独计费/计步。",
              '这是开发筛选结果；独立测试未打开，额外五任务尚未加入训练。Correct本轮未评，不将缺失数据记作零。', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args(); result = summarize(args.root)
    (args.root / 'comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (args.root / 'RESULTS.md').write_text(markdown(result))
    print(json.dumps({k: result[k] for k in ('complete', 'cf_ranking', 'eraf_fg_strictly_best', 'aggregates')}, indent=2))


if __name__ == '__main__': main()
