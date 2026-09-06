from types import SimpleNamespace
import pytest
import torch
from fastwam.models.wan22.policy_guard import ERAFActionContextInjector
from scripts.probe_robotwin_eraf_context_tokens import injection_mode


def fixture():
    module = ERAFActionContextInjector(goal_dim=4, text_dim=6, hidden_dim=8)
    model = SimpleNamespace(policy_guard_enabled=True,
        policy_guard_modules={'eraf_action_context_injector': module})
    inputs = dict(context=torch.randn(1, 5, 6), context_mask=torch.ones(1, 5, dtype=torch.bool),
                  goal_queries=torch.randn(1, 2, 4), external_scale=1.)
    return model, module, inputs


def test_zero_schedule_removes_tokens_but_small_amplitude_keeps_them_valid():
    model, module, inputs = fixture()
    with injection_mode(model, 'schedule_zero'):
        context, mask, _ = module(**inputs)
        assert torch.equal(context, inputs['context'])
        assert torch.equal(mask, inputs['context_mask'])
    with injection_mode(model, 'near_zero_tokens') as measurements:
        context, mask, _ = module(**inputs)
        assert context.shape == (1, 7, 6)
        assert torch.equal(context[:, :5], inputs['context'])
        assert mask[:, 5:].all()
        assert context[:, 5:].abs().max() <= 1e-6
        assert measurements[-1]['pgc_v925_action_context_token_count'] == 2


def test_intervention_restores_forward_and_enabled_state_on_error():
    model, module, inputs = fixture()
    original = module.forward
    with pytest.raises(RuntimeError):
        with injection_mode(model, 'off'):
            assert not model.policy_guard_enabled
            raise RuntimeError('inference failed')
    assert model.policy_guard_enabled
    assert module.forward == original
    with pytest.raises(RuntimeError):
        with injection_mode(model, 'schedule_zero'):
            module(**inputs)
            raise RuntimeError('inference failed')
    assert module.forward == original
    assert model.policy_guard_enabled
