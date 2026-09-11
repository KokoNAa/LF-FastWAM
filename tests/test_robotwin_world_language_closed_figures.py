"""Coverage and sign checks for final closed-loop plots (synthetic inputs only)."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'closed_figures', Path(__file__).resolve().parents[1] / 'scripts/plot_robotwin_world_language_closed.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture():
    q = dict(compute_coverage_complete=True, actual_closed_loop_episodes=1200,
             closed_loop={}, statistics=[])
    audit = dict(complete=True, allow_partial=False, validated_episodes=1200,
                 pending_arms=[], completed_arms={})
    for m in mod.MODELS:
        for v, a in mod.CELLS:
            source = 8 if v == 'source' else 6
            cf = 1 if v == 'source' else 3
            for seed in [42, 43, 44]:
                name = f'{m}-closed-v_{v}-a_{a}-seed{seed}'
                counts = dict(episodes=50, source_success=source * 5, cf_success=cf * 5)
                q['closed_loop'][name] = counts.copy()
                audit['completed_arms'][name] = dict(
                    **counts, process_exit=dict(status='exited', exit_code=0), tasks=[
                        dict(task=t, episodes=10, source_success=source, cf_success=cf,
                             any_correct_lift=10, all_instruction_lift=9,
                             full_goal_after_lift=source if a == 'source' else cf,
                             strict_placement_after_lift=source if a == 'source' else cf)
                        for t in mod.TASKS])
            for t in mod.TASKS:
                metrics = dict(source_goal_ever_success=source / 10,
                               counterfactual_goal_ever_success=cf / 10,
                               any_correct_object_lifted=1, all_instruction_objects_lifted=.9,
                               full_goal_after_any_lift=(source if a == 'source' else cf) / 10,
                               correct_placement_after_lift=(source if a == 'source' else cf) / 10,
                               first_goal_source=source / 10, first_goal_counterfactual=cf / 10,
                               first_goal_neither=.1, first_goal_ambiguous=0)
                for metric, mean in metrics.items():
                    q['statistics'].append(dict(model=m, task=t, phase=f'closed_v_{v}_a_{a}',
                                                metric=metric, mean=mean, ci95=[mean, mean],
                                                scenes=10, repeated_observations=30))
        for t in mod.TASKS:
            for effect in mod.EFFECTS:
                for goal in ['source', 'counterfactual']:
                    mean = (1 if goal == 'counterfactual' else -1) * .2 if effect.startswith('video') else 0
                    q['statistics'].append(dict(model=m, task=t, phase='closed_paired',
                        metric=effect + '_' + goal + '_goal_ever_success', mean=mean,
                        ci95=[mean - .1, mean + .1], scenes=10, repeated_observations=30))
    return q, audit


def test_complete_synthetic_coverage():
    q, audit = fixture()
    assert len(mod.validate_inputs(q, audit)) == 480


def test_reject_partial_even_if_episode_total_is_forged():
    q, audit = fixture()
    audit['allow_partial'] = True
    with pytest.raises(ValueError, match='Final closed-data audit'):
        mod.validate_inputs(q, audit)


def test_reject_missing_arm():
    q, audit = fixture()
    del q['closed_loop'][next(iter(q['closed_loop']))]
    with pytest.raises(ValueError, match='exact 24-arm'):
        mod.validate_inputs(q, audit)


def test_reject_rate_not_equal_to_audited_count():
    q, audit = fixture()
    q['statistics'][0]['mean'] -= .1
    with pytest.raises(ValueError, match='audited counts'):
        mod.validate_inputs(q, audit)


def test_reject_noise_seeds_as_independent_scenes():
    q, audit = fixture()
    q['statistics'][0]['scenes'] = 30
    with pytest.raises(ValueError, match='10 scenes'):
        mod.validate_inputs(q, audit)


def test_reject_reversed_effect_sign():
    q, audit = fixture()
    row = next(r for r in q['statistics'] if r['phase'] == 'closed_paired')
    row['mean'] *= -1
    with pytest.raises(ValueError, match='sign or cell mismatch'):
        mod.validate_inputs(q, audit)


def test_reject_duplicate_statistics():
    q, audit = fixture()
    q['statistics'].append(deepcopy(q['statistics'][0]))
    with pytest.raises(ValueError, match='Duplicate statistic'):
        mod.validate_inputs(q, audit)
