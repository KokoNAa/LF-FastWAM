import copy
from types import SimpleNamespace
import pytest
import torch

from test_robotwin_eraf_fg_bridge import case
from experiments.robotwin.eraf_fg_training import backward_example
from experiments.robotwin.eraf_fg_bridge import trainable_parameters
from experiments.robotwin.gradient_diagnostic import GradientCollector


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
