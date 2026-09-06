"""Exercise new replay reconstruction and real production training gradients."""
import copy
import pytest
import torch

from test_robotwin_eraf_fg_bridge import case
from experiments.robotwin.eraf_fg_training import backward_example
from experiments.robotwin.eraf_fg_bridge import trainable_parameters
from experiments.robotwin.compact_replay import ReplayPayloads, capture_delta


def test_new_compact_payload_preserves_proprio_and_tail_mask(case, tmp_path):
    _, captured, reference, _ = case
    captured['policy_guard_state'] = None
    original = tmp_path / 'parent.pt'
    torch.save({'captured': {'target': captured}}, original)
    changed = copy.deepcopy(captured)
    changed['video_inputs']['x'] += 1
    changed['proprio'] += 2
    body = lambda c: {k: c[k] for k in ('video_inputs', 'action_inputs')}
    path = tmp_path / 'child.pt'
    valid = torch.tensor([[True, True, False, False]])
    torch.save({'format': 'robotwin_eraf_fg_compact_v1', 'parent_payload': str(original),
        'capture_deltas': {'target': capture_delta(body(changed), body(captured))},
        'capture_extras': {'target': {'proprio': changed['proprio'], 'policy_guard_state': None}},
        'references': {'target': reference}, 'valid': {'target': valid}}, path)
    actual = ReplayPayloads([{'id': 'child', 'payload': str(path)}], 'cpu')['child']
    assert torch.equal(actual['captured']['target']['video_inputs']['x'], changed['video_inputs']['x'])
    assert torch.equal(actual['captured']['target']['proprio'], changed['proprio'])
    assert torch.equal(actual['valid']['target'], valid)
    assert actual['captured']['target']['policy_guard_state'] is None


def test_full_goal_actual_training_updates_only_interface(case):
    model, captured, noise, time = case
    selected = trainable_parameters(model, 'interface')
    report = backward_example(model, {'fg_correction': True, 'frame_index': 0},
        {'captured': {'target': captured}, 'references': {'target': torch.zeros_like(noise)},
         'valid': {'target': torch.tensor([[True, True, False, False]])}}, noise, time, teachers={})
    assert report['endpoint_objective'] >= 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in selected.values())
    assert all(p.grad is None for p in model.mot.parameters())


def test_same_state_flag_cannot_hide_different_observations(case):
    model, captured, noise, time = case
    target = copy.deepcopy(captured)
    target['video_inputs']['x'] += 1
    with pytest.raises(ValueError, match='exact same observation'):
        backward_example(model, {'initial_observations_exactly_equal': True},
            {'captured': {'source': captured, 'target': target},
             'references': {'source': noise, 'target': noise}}, noise, time, teachers={})


def test_local_ablation_rejects_a_later_full_goal_window(case):
    model, captured, noise, time = case
    with pytest.raises(ValueError, match='failure-start'):
        backward_example(model, {'fg_correction': True, 'frame_index': 8},
            {'captured': {'target': captured}, 'references': {'target': noise}},
            noise, time, teachers={}, fg='local')
