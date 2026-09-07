"""A classifier prediction must not bypass the predicate completion condition."""
import pytest
import torch
from fastwam.models.wan22.entity_relation_affordance import PhaseSafeClauseMemory


def run_memory(*, truth, phase, previous=None, active=True):
    memory = PhaseSafeClauseMemory(hidden_dim=8, adapter_hidden_dim=8,
        num_heads=2, max_clauses=1).eval()
    with torch.no_grad():
        memory.state_output.bias[memory.COMPLETED] = 20
    phase_logits = torch.full((1, 1, 3), -10.)
    phase_logits[0, 0, phase] = 10
    inputs = dict(clause_hidden=torch.zeros(1, 1, 8), subject_tokens=torch.zeros(1, 1, 8),
        reference_tokens=torch.zeros(1, 1, 8), active_logits=torch.full((1, 1), 10. if active else -10.),
        predicate_truth_logits=torch.full((1, 1), 10. if truth else -10.), phase_logits=phase_logits,
        base_execution_logits=torch.zeros(1, 1), base_execution_probability=torch.ones(1, 1),
        base_routing_multiplier=torch.ones(1, 1))
    if previous is not None:
        inputs['policy_state'] = dict(phase_safe_memory_state_ids=torch.tensor([[previous]]),
                                     phase_safe_memory_valid=torch.ones(1, 1, dtype=torch.bool))
    with torch.no_grad():
        result = memory(**inputs)
    return memory, inputs, result


@pytest.mark.parametrize('phase', [0, 1, 2])
@pytest.mark.parametrize('previous', [None, 0, 1, 2])
def test_false_predicate_cannot_start_sticky_completion(phase, previous):
    memory, inputs, result = run_memory(truth=False, phase=phase, previous=previous)
    assert int(result['state_logits'].argmax(-1)) == memory.COMPLETED
    assert not bool(result['completed_sticky'].item())
    assert int(result['next_state_ids'].item()) != memory.COMPLETED
    inputs['policy_state'] = dict(phase_safe_memory_state_ids=result['next_state_ids'],
                                 phase_safe_memory_valid=result['next_state_valid'])
    with torch.no_grad():
        second = memory(**inputs)
    assert not bool(second['completed_sticky'].item())
    assert int(second['next_state_ids'].item()) != memory.COMPLETED


@pytest.mark.parametrize('phase', [0, 1, 2])
@pytest.mark.parametrize('active', [False, True])
def test_existing_completion_stays_sticky(phase, active):
    memory, _, result = run_memory(truth=False, phase=phase, previous=3, active=active)
    assert bool(result['completed_sticky'].item())
    assert int(result['next_state_ids'].item()) == memory.COMPLETED


@pytest.mark.parametrize('phase,expected', [(0, 3), (1, 1), (2, 3)])
def test_true_predicate_retains_release_and_holding_rules(phase, expected):
    _, _, result = run_memory(truth=True, phase=phase)
    assert int(result['next_state_ids'].item()) == expected


def test_released_false_predicate_still_retries():
    memory, _, result = run_memory(truth=False, phase=2, previous=1)
    assert int(result['next_state_ids'].item()) == memory.RETRY
    assert bool(result['released_unsatisfied_retry'].item())
