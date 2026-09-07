from pathlib import Path
from types import SimpleNamespace
import pytest

from scripts.run_robotwin_cf_priority_campaign import (
    build_protocol, cf_summary_result, select_arm, training_command, OLD_TASKS)


def plan():
    args = SimpleNamespace(checkpoint=Path('/data/all5_off200.pt'), manifest=Path('/data/manifest.json'),
        source_bank=Path('/data/raw'), correct_teacher=Path('/data/dense600.pt'),
        old_catalog=Path('/data/old'), new_catalog=Path('/data/new'),
        deadline='2026-09-07T19:00:00+08:00', steps=400, interface_steps=100)
    return build_protocol(args, 'test_commit')


def test_arms_share_the_actual_best_policy_and_record_interface_budget():
    p = plan()
    assert p['cf_teacher'] == p['checkpoint']
    assert p['checkpoints'] == [200, 400]
    for arm in p['arms']:
        stage = 'interface' if arm == 'eraf_fg' else 'joint'
        cmd = training_command(p, Path('/out'), arm, stage, p['checkpoint'], warm=True)
        assert cmd[cmd.index('--checkpoint') + 1] == p['checkpoint']
        assert cmd[cmd.index('--cf-teacher') + 1] == p['checkpoint']
        assert '--warm-policy' in cmd
        assert '--disable-seen-language-augmentation' in cmd
    cmd = training_command(p, Path('/out'), 'eraf_fg', 'joint', '/out/eraf_fg/interface/step_000100.pt', warm=False)
    assert '--warm-policy' not in cmd
    assert float(cmd[cmd.index('--learning-rate') + 1]) < float(cmd[cmd.index('--interface-learning-rate') + 1])


def test_cf_selection_cannot_use_incomplete_or_correct_only_results():
    cells = [dict(source_task=t, condition='counterfactual', total_episodes=6,
                  selected_goal_successes=4) for t in OLD_TASKS]
    summary = dict(complete=True, episodes=30, cells=cells)
    row = cf_summary_result(summary, name='arm200', step=200)
    assert row['successes'] == 20
    assert select_arm([row | {'step': 400}, row])['step'] == 200
    with pytest.raises(ValueError):
        cf_summary_result(summary | {'cells': cells[:-1]}, name='partial', step=200)
    cells[0]['condition'] = 'correct'
    with pytest.raises(ValueError):
        cf_summary_result(summary, name='wrong_metric', step=200)
