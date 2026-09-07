import copy
from types import SimpleNamespace
import pytest
import torch

from test_robotwin_eraf_fg_bridge import case
from experiments.robotwin.eraf_fg_training import backward_example
from experiments.robotwin.eraf_fg_bridge import trainable_parameters
from experiments.robotwin.gradient_diagnostic import GradientCollector, bucket, diagnostic_recipe


def test_saved_plan_recovers_current_cf_priority_recipe_and_rejects_drift():
    plan = dict(stage='joint', fg='full', eraf='on', seed=42, global_batch=12,
                correct_weight=1., cf_weight=4., correction_weight=1., policy_scope='action',
                correct_count=2, cf_count=4, task_balanced=True,
                disable_seen_language_augmentation=True,
                mixture=dict(correct_retention=2, cf_retention=4, expert_pairs=3,
                             fg=3, ordinary_cf_control=0))
    plan['optimization_contract'] = {k: v for k, v in plan.items() if k != 'mixture'}
    recovered = diagnostic_recipe(plan)
    assert recovered['eraf'] == 'on' and recovered['policy_scope'] == 'action'
    assert recovered['correct_count'] == 2 and recovered['cf_count'] == 4
    assert recovered['correct_weight'] == 1. and recovered['cf_weight'] == 4.
    assert recovered['task_balanced'] and recovered['disable_seen_language_augmentation']
    with pytest.raises(ValueError, match='optimization contract'):
        diagnostic_recipe(plan | {'cf_weight': 2.})
    with pytest.raises(ValueError, match='mixture'):
        diagnostic_recipe(plan | {'mixture': plan['mixture'] | {'cf_retention': 2}})
    with pytest.raises(ValueError, match='Incomplete'):
        diagnostic_recipe({})
    legacy = diagnostic_recipe()
    assert legacy['policy_scope'] == 'all' and legacy['eraf'] == 'off'
    assert legacy['correct_count'] == 4 and legacy['cf_count'] == 2
    assert legacy['correct_weight'] == 4. and legacy['cf_weight'] == 2.


def test_control_gradient_and_eraf_interface_are_reported_separately():
    action = torch.nn.Parameter(torch.ones(2))
    interface = torch.nn.Parameter(torch.ones(2))
    collector = GradientCollector({'mot.mixtures.action.p': action, 'guard.goal_graph.p': interface})
    group = bucket({'ordinary_cf_control': True})
    assert group == 'ordinary_cf_control'
    collector.observer(group, 'control')((action + 2 * interface).sum(), 'endpoint')
    report = collector.summary()['scopes']
    assert report['action']['norms'][group] == pytest.approx(2 ** .5)
    assert report['eraf']['norms'][group] == pytest.approx(8 ** .5)
    assert report['other']['norms'][group] == 0
    assert action.grad is None and interface.grad is None


def test_cosines_detect_opposition_and_zero_without_mutation():
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    collector = GradientCollector({'mot.mixtures.action.p': parameter})
    collector.observer('a', 'a')(parameter.sum(), 'flow')
    collector.observer('b', 'b')(-2 * parameter.sum(), 'flow')
    collector.observer('zero', 'z')(0 * parameter.sum(), 'flow')
    result = collector.summary()['scopes']['all']
    assert result['cosines']['a']['b'] == pytest.approx(-1.)
    assert result['cosines']['a']['zero'] is None
    assert result['dot_with_combined']['a'] == pytest.approx(-2.)
    assert parameter.grad is None and torch.equal(parameter, torch.tensor([1., 2.]))


def test_split_paired_objective_gradients_sum_to_actual_training(case, monkeypatch):
    from experiments.robotwin import eraf_fg_bridge
    production, _, noise, time = case
    model = SimpleNamespace(train_action_scheduler=production.train_action_scheduler,
                            device='cpu', torch_dtype=torch.float32)
    parameter = torch.nn.Parameter(torch.tensor(.2))
    monkeypatch.setattr(eraf_fg_bridge, 'predict',
                        lambda model, captured, noisy, time, **kw: parameter * noisy * captured['scale'])
    captured = {k: {'scale': v, 'video_inputs': {'x': torch.zeros(1)}, 'proprio': torch.zeros(1)}
                for k, v in [('source', 1.), ('target', 2.)]}
    payload = {'captured': captured, 'references': {'source': noise * .3, 'target': noise * -.7}}
    row = {'initial_observations_exactly_equal': True}
    expected_report = backward_example(model, row, payload, noise, time, teachers={}, coefficient=1/12)
    expected = parameter.grad.clone()
    parameter.grad = None
    collector = GradientCollector({'mot.mixtures.action.p': parameter})
    actual_report = backward_example(model, row, payload, noise, time, teachers={}, coefficient=1/12,
                                     gradient_observer=collector.observer('pair', 'one'))
    assert actual_report == expected_report
    assert torch.allclose(collector.groups['pair']['mot.mixtures.action.p'], expected, atol=1e-6)
    conditional = [v for v in collector.terms if v['component'] == 'conditional_difference']
    assert len(conditional) == 1 and conditional[0]['norm_by_scope']['action'] > 0
    assert parameter.grad is None and parameter.item() == pytest.approx(.2)


def test_observer_matches_real_checkpointed_model_gradients_with_tail_mask(case):
    model, captured, noise, time = case
    selected = trainable_parameters(model, 'interface')
    before = {n: p.detach().clone() for n, p in selected.items()}
    payload = {'captured': {'target': captured}, 'references': {'target': torch.zeros_like(noise)},
               'valid': {'target': torch.tensor([[True, True, False, False]])}}
    row = {'fg_correction': True, 'frame_index': 0}
    backward_example(model, row, payload, noise, time, teachers={})
    expected = {n: p.grad.clone() for n, p in selected.items() if p.grad is not None}
    for p in selected.values():
        p.grad = None
    collector = GradientCollector(selected)
    backward_example(model, row, payload, noise, time, teachers={},
                     gradient_observer=collector.observer('fg', 'one'))
    assert expected.keys() == collector.groups['fg'].keys()
    for n, gradient in expected.items():
        assert torch.allclose(gradient, collector.groups['fg'][n], rtol=2e-4, atol=2e-6), n
    assert all(p.grad is None and torch.equal(p, before[n]) for n, p in selected.items())
