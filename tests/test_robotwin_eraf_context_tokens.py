from types import SimpleNamespace
import pytest
import torch
from fastwam.models.wan22.policy_guard import ERAFActionContextInjector
from scripts.probe_robotwin_eraf_context_tokens import injection_mode


def test_expanded_identity_covers_new_tasks_without_changing_historical_scope():
    from experiments.robotwin.pgc_data import ROBOTWIN_ERAF_PAIR_IDS, ROBOTWIN_TEN_TASK_SPECS
    from scripts.probe_robotwin_eraf_context_tokens import identity_rows
    old = [dict(pair_id=pair, task_config=domain, replay_split='replay_holdout', frame_index=0)
           for pair in ROBOTWIN_ERAF_PAIR_IDS for domain in ('demo_clean', 'demo_randomized')]
    assert len(identity_rows(old)) == 10
    extra = [dict(pair_id=s.pair_id, task_config='demo_clean', replay_split='replay_holdout', frame_index=0)
             for s in ROBOTWIN_TEN_TASK_SPECS if s.pair_id not in ROBOTWIN_ERAF_PAIR_IDS]
    with pytest.raises(ValueError, match='declared task set'):
        identity_rows(old + extra)
    selected = identity_rows(old + extra + extra, include_expanded_tasks=True)
    assert len(selected) == 15
    assert len({pair for pair, domain in selected}) == 10
    missing_initial = [dict(row, target_frame_index=1) if row['pair_id'] == extra[-1]['pair_id'] else row
                       for row in old + extra]
    with pytest.raises(ValueError, match='initial held-out scene'):
        identity_rows(missing_initial, include_expanded_tasks=True)
    with pytest.raises(ValueError, match='declared task set'):
        identity_rows(old + extra[:-1], include_expanded_tasks=True)


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
