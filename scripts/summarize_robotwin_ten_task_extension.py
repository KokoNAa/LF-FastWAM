#!/usr/bin/env python3
"""Compare the same five fixed models across the original and additional tasks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

OLD_TASKS = ['place_a2b_left', 'place_a2b_right', 'place_burger_fries', 'stack_blocks_two', 'blocks_ranking_rgb']
NEW_TASKS = ['blocks_ranking_size', 'place_empty_cup', 'place_mouse_pad', 'move_stapler_pad', 'move_pillbottle_pad']
MODEL_DIRS = {'fastwam_release': 'eval_raw_fastwam_combined', 'shared400': 'eval_baseline',
              'all5_off200': 'eval_all5_off200', 'all5_full200': 'eval_all5_full200',
              'full200_eraf_on': 'eval_full200_eraf_on'}


def read(path):
    return json.loads(path.read_text())


def summarize(previous, extension):
    driver = read(extension / 'eval/driver.json')
    if not driver.get('complete') or not driver.get('matched_initial_states'):
        raise ValueError('Additional task evaluation is incomplete')
    cells, aggregates, paired = {}, {}, {}
    all_records = {}
    for task in OLD_TASKS + NEW_TASKS:
        n = 6 if task in OLD_TASKS else 3
        cells[task] = {'episodes_per_condition': n, 'models': {}}
        initial, canonical = None, None
        for model, directory in MODEL_DIRS.items():
            base = previous / directory if task in OLD_TASKS else extension / 'eval'
            root = base / model / 'dev' / task / 'demo_clean'
            counts = {}
            for condition in ('correct', 'counterfactual'):
                path = root / condition
                if not read(path / 'complete.json')['complete']:
                    raise ValueError(f'Incomplete cell: {path}')
                records = [json.loads(line) for line in (path / 'episodes.jsonl').read_text().splitlines()]
                if len(records) != n or len({r['scene_seed'] for r in records}) != n:
                    raise ValueError(f'Missing or duplicate scene: {path}')
                states = read(path / 'initial_states.json')
                signature = [{k: r[k] for k in ('scene_seed', 'episode_index', 'source_instruction',
                             'counterfactual_instruction', 'step_limit')} for r in records]
                if initial is None:
                    initial, canonical = states, signature
                if states != initial or signature != canonical:
                    raise ValueError(f'Unmatched scenes or prompts: {path}')
                expected_goal = 'source' if condition == 'correct' else 'counterfactual'
                for row in records:
                    expected_prompt = row['source_instruction'] if condition == 'correct' else row['counterfactual_instruction']
                    if row['selected_goal'] != expected_goal or row['policy_instruction'] != expected_prompt:
                        raise ValueError(f'Wrong selected goal or instruction: {path}')
                    if row['initial_source_goal_success'] or row['initial_counterfactual_goal_success']:
                        raise ValueError(f'Already satisfied goal: {path}')
                counts[condition] = sum(r['selected_goal_success'] for r in records)
                all_records[task, model, condition] = records
            cells[task]['models'][model] = counts
        paired[task] = {}
        for model in MODEL_DIRS:
            paired[task][model] = {}
            for condition in ('correct', 'counterfactual'):
                reference = all_records[task, 'fastwam_release', condition]
                candidate = all_records[task, model, condition]
                pairs = list(zip(reference, candidate, strict=True))
                paired[task][model][condition] = {
                    'gained_seeds': [a['scene_seed'] for a,b in pairs if not a['selected_goal_success'] and b['selected_goal_success']],
                    'lost_seeds': [a['scene_seed'] for a,b in pairs if a['selected_goal_success'] and not b['selected_goal_success']]}
    for model in MODEL_DIRS:
        aggregates[model] = {}
        for label, tasks in [('original_five', OLD_TASKS), ('additional_five', NEW_TASKS), ('all_ten', OLD_TASKS + NEW_TASKS)]:
            aggregates[model][label] = {condition: {
                'successes': sum(cells[t]['models'][model][condition] for t in tasks),
                'episodes': sum(cells[t]['episodes_per_condition'] for t in tasks),
                'task_mean_rate': sum(cells[t]['models'][model][condition] / cells[t]['episodes_per_condition'] for t in tasks) / len(tasks)}
                for condition in ('correct', 'counterfactual')}
    order = sorted(MODEL_DIRS, key=lambda m: aggregates[m]['all_ten']['counterfactual']['task_mean_rate'], reverse=True)
    return dict(format='robotwin_ten_task_comparison_v1', complete=True, matched_initial_states_and_instructions=True,
                priority='CF task mean; report Correct costs without an exclusion threshold',
                additional_task_training=False, independent_test=False, gate_recovery_measured=False,
                cells=cells, aggregates=aggregates, paired_vs_raw_fastwam=paired, cf_ranking=order)


def markdown(report):
    models = list(MODEL_DIRS)
    lines = ['# RoboTwin 十任务开发评测', '',
        '每格为 Correct / CF 成功数。原五项每条件 6 场景，新增五项每条件 3 场景。',
        '新增五项未参与本轮训练；这是小样本开发集迁移评测，非独立最终测试。', '',
        '| 任务 | 每条件场景数 | ' + ' | '.join(models) + ' |',
        '|---|---:|' + '---:|' * len(models)]
    for task, cell in report['cells'].items():
        values = [str(cell['models'][m]['correct']) + ' / ' + str(cell['models'][m]['counterfactual']) for m in models]
        lines.append('| ' + task + ' | ' + str(cell['episodes_per_condition']) + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '按十任务等权平均排序；原五项与新增五项的每条件样本数不同。', '',
              '| 模型 | CF 任务平均 | Correct 任务平均 | 相对原始 FastWAM 的 Correct 变化 |',
              '|---|---:|---:|---:|']
    base = report['aggregates']['fastwam_release']['all_ten']['correct']['task_mean_rate']
    for model in report['cf_ranking']:
        row = report['aggregates'][model]['all_ten']
        cf, correct = row['counterfactual']['task_mean_rate'], row['correct']['task_mean_rate']
        lines.append(f'| {model} | {cf:.1%} | {correct:.1%} | {(correct-base)*100:+.1f} 个百分点 |')
    lines += ['', 'CF 优先，Correct 退步完整保留。当前结果没有验证部署 gate 的补偿效果。',
              'Full200+ERAF 为固定 Full200 策略与单独训练 10 步的接口组合，未联合微调。',
              'Full Goal 对照比较整个纠正数据方案，未单独隔离监督长度。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-study', type=Path, required=True)
    parser.add_argument('--extension', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.previous_study, args.extension)
    (args.extension / 'ten_task_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.extension / 'TEN_TASK_COMPARISON.md').write_text(markdown(report))
    print(json.dumps({'complete': True, 'cf_ranking': report['cf_ranking'], 'aggregates': report['aggregates']}))


if __name__ == '__main__':
    main()

