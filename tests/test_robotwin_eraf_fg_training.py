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


def test_local_control_cannot_read_future_labels_through_noisy_action_input(case, monkeypatch):
    from types import SimpleNamespace
    from experiments.robotwin import eraf_fg_bridge
    production, _, _, time = case
    model = SimpleNamespace(train_action_scheduler=production.train_action_scheduler,
                            device='cpu', torch_dtype=torch.float32)
    parameter = torch.nn.Parameter(torch.tensor(.2))
    calls = []

    def predict(model, captured, noisy, time, **kwargs):
        calls.append(noisy.detach().clone())
        # A prediction that can attend to every action token exposes the
        # leakage that a per-token loss mask alone cannot prevent.
        return parameter * noisy.mean(dim=1, keepdim=True).expand_as(noisy)

    monkeypatch.setattr(eraf_fg_bridge, 'predict', predict)
    noise = torch.linspace(-1, 1, 32 * 14).reshape(1, 32, 14)
    ordinary = torch.ones_like(noise)
    changed_future = ordinary.clone()
    changed_future[:, 12:] = 1000

    def evaluate(reference, mode):
        calls.clear()
        parameter.grad = None
        report = backward_example(model, {'fg_correction': True, 'frame_index': 0},
            {'captured': {'target': {}}, 'references': {'target': reference}},
            noise, time, teachers={}, fg=mode)
        return report, parameter.grad.clone(), [value.clone() for value in calls]

    a, b = evaluate(ordinary, 'local'), evaluate(changed_future, 'local')
    assert a[0] == b[0] and torch.equal(a[1], b[1])
    assert all(torch.equal(x, y) for x, y in zip(a[2], b[2], strict=True))
    full_a, full_b = evaluate(ordinary, 'full'), evaluate(changed_future, 'full')
    assert not torch.equal(full_a[2][0], full_b[2][0])


def test_local_and_full_draw_identical_scenes_and_preservation_examples():
    from experiments.robotwin.eraf_fg_training import mixture_stream
    rows = []
    for kind in ('native_retention', 'cf_retention', 'pair', 'fg_correction'):
        for scene in range(3):
            for frame in (0, 8, 16):
                rows.append({'id': f'{kind}_{scene}_{frame}', 'pair_id': kind,
                             'task_config': 'demo_clean', 'scene_seed': scene,
                             'replay_split': 'train', 'frame_index': frame, kind: True})
    full, local = mixture_stream(rows, 42, 'full'), mixture_stream(rows, 42, 'local')
    for _ in range(10):
        for a, b in zip(next(full), next(local), strict=True):
            assert (a['pair_id'], a['scene_seed']) == (b['pair_id'], b['scene_seed'])
            if a.get('fg_correction'):
                assert b['frame_index'] == 0
            else:
                assert a['id'] == b['id']


def test_resume_preserves_sub_bfloat16_updates_and_optimizer_moments():
    from experiments.robotwin.eraf_fg_bridge import MasterAdamW
    a = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    original = MasterAdamW([a], lr=1e-5)
    for _ in range(100):
        original.zero_grad(); a.grad = torch.tensor([.2, -.3], dtype=a.dtype); original.step()
    state = copy.deepcopy(original.state_dict())
    b = torch.nn.Parameter(a.detach().clone())
    resumed = MasterAdamW([b], lr=1e-5)
    resumed.load_state_dict(state)
    for index in range(100):
        grad = torch.tensor([.2 + index * .01, -.3], dtype=a.dtype)
        for optimizer, live in ((original, a), (resumed, b)):
            optimizer.zero_grad(); live.grad = grad.clone(); optimizer.step()
    assert torch.equal(original.master[0], resumed.master[0])
    assert torch.equal(a, b)
