from pathlib import Path
from types import SimpleNamespace

from scripts.run_robotwin_cf_priority_campaign import build_protocol
from scripts.run_robotwin_fg_routing_trial import command, protocol


def test_routing_comparison_changes_only_declared_gradient_route():
    args = SimpleNamespace(checkpoint=Path('/data/all5.pt'), manifest=Path('/data/bank.json'),
        source_bank=Path('/data/raw'), correct_teacher=Path('/data/dense600.pt'),
        old_catalog=Path('/data/old'), new_catalog=Path('/data/new'),
        deadline='2026-09-07T19:15:00+08:00', steps=400, interface_steps=100)
    original = build_protocol(args, 'primary')
    plan = protocol(original, Path('/primary'), Path('/trial'), args.deadline, 'routing')
    standard = command(plan, Path('/trial'), 'eraf_fg')
    routed = command(plan, Path('/trial'), 'eraf_fg_routed')
    routed[routed.index('--output') + 1] = standard[standard.index('--output') + 1]
    routed[routed.index('--fg-gradient-route') + 1] = 'joint'
    assert routed == standard
    assert '--warm-policy' not in standard
    assert standard[standard.index('--checkpoint') + 1] == '/primary/eraf_fg/interface/step_000100.pt'
    for arm in ['ordinary_cf', 'fg_only']:
        control = command(plan, Path('/trial'), arm)
        assert control[control.index('--checkpoint') + 1] == str(args.checkpoint)
        assert '--warm-policy' in control
        assert '--nproc_per_node=1' in control
        for flag in ['--manifest', '--correct-teacher', '--cf-teacher', '--steps',
                     '--learning-rate', '--correct-weight', '--cf-weight', '--correction-weight',
                     '--correct-count', '--cf-count', '--seed', '--policy-scope']:
            assert control[control.index(flag) + 1] == standard[standard.index(flag) + 1]
    assert original['correction_weight'] == 1.0
    assert plan['correction_weight'] == 0.1
    assert plan['strongest_comparator']['ten_task_macro_cf'] == 0.35
