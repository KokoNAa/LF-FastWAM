import pytest

from scripts.report_robotwin_five_task_comparison import (
    ARMS, TASKS, exact_paired_p, paired_summary, report, markdown,
)
from scripts.run_robotwin_formal_five40 import read, write, records
from test_robotwin_formal_five40 import matrix


def test_exact_paired_test_uses_only_discordant_pairs():
    assert exact_paired_p(0, 0) == 1
    assert exact_paired_p(5, 0) == .0625
    assert exact_paired_p(0, 5) == .0625
    assert exact_paired_p(3, 3) == 1


def test_macro_averages_tasks_and_resamples_paired_outcomes():
    # Unequal cell sizes must not turn a fixed-task macro into a pooled rate.
    outcomes = {t: [(False, True, True)] * (2 if i == 0 else 8) for i, t in enumerate(TASKS)}
    outcomes[TASKS[0]] = [(True, False, False)] * 2
    r = paired_summary(outcomes, draws=1000)
    assert r['macro_cf'] == dict(no_eraf=.2, eraf_only=.8, eraf_fg=.8)
    first = r['comparisons'][0]
    assert first['gains'] == 32 and first['losses'] == 2
    assert first['macro_difference'] == pytest.approx(.6)
    assert first['paired_stratified_bootstrap_ci95'] == pytest.approx([.6, .6])
    same = r['comparisons'][1]
    assert same['paired_stratified_bootstrap_ci95'] == [0., 0.]
    assert same['exact_paired_p_two_sided'] == 1
    assert not r['desired_order_observed'] and not r['goal_achieved']


def test_missing_task_and_numeric_instead_of_boolean_outcomes_are_rejected():
    with pytest.raises(ValueError): paired_summary({TASKS[0]: [(True, False, False)]}, draws=1000)
    with pytest.raises(ValueError): paired_summary({t: [(1, False, False)] for t in TASKS}, draws=1000)


def test_equal_success_counts_cannot_be_ordered_by_float_roundoff():
    # Both sum to 5/60. Summing their per-task float rates gives different floats.
    first, second = [0, 0, 0, 1, 4], [0, 0, 0, 0, 5]
    assert sum(n/12 for n in first) != sum(n/12 for n in second)
    outcomes = {t:[(i<first[j], i<second[j], i<second[j]) for i in range(12)] for j,t in enumerate(TASKS)}
    result = paired_summary(outcomes, draws=1000)
    assert len(set(result['macro_cf'].values())) == 1
    assert all(c['macro_difference'] == 0 for c in result['comparisons'])
    assert not result['desired_order_observed']


def test_complete_report_checks_actual_model_goal_and_physical_bindings(matrix):
    root, plan = matrix
    plan.update(tasks=list(TASKS), independent_test=True)
    write(root/'frozen_protocol.json', plan)
    write(root/'status.json', dict(complete=True, status='complete'))
    write(root/'terminal_verification.json', dict(complete=True, episodes=600))
    for arm in ARMS:
        for task in TASKS:
            base = root/'evaluation'/arm/task/'demo_clean/counterfactual'
            complete = read(base/'complete.json')
            complete.update(policy_kind='repair', memory_mode='carry',
                            deployment=dict(action_horizon=32, replan_steps=24, inference_steps=10))
            write(base/'complete.json', complete)
            data = records(base/'episodes.jsonl')
            for row in data:
                row.update(source_task=task, condition='counterfactual', selected_goal='counterfactual', instruction_goal='counterfactual')
            (base/'episodes.jsonl').write_text(''.join(__import__('json').dumps(r)+'\n' for r in data))
    result = report(root, draws=1000)
    assert result['episodes'] == 600 and result['independent_test']
    assert all(p['gains'] == p['losses'] == 0 for p in result['comparisons'])
    assert '独立正式测试' in markdown(result)
    assert not result['goal_achieved']
    base = root/'evaluation'/ARMS[0]/TASKS[0]/'demo_clean/counterfactual'
    complete = read(base/'complete.json')
    complete['checkpoint_sha256'] = 'changed'
    write(base/'complete.json', complete)
    with pytest.raises(ValueError, match='binding'): report(root, draws=1000)
    write(root/'status.json', dict(complete=False, status='evaluating_dev'))
    with pytest.raises(ValueError, match='complete'): report(root, draws=1000)
