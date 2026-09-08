from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch

from experiments.robotwin.five_task_action import (
    TASKS, FG_TASKS, OBJECTIVE, mixture_stream, pools, target_payload, backward_example,
)


def rows():
    result = []
    for task in TASKS:
        for kind in ['ordinary', 'cf'] + (['fg'] if task in FG_TASKS else []):
            for scene in [1, 2, 3]:
                for frame in range(6):
                    r = dict(id=f'{task}_{kind}_{scene}_{frame}', source_task=task, pair_id=task,
                             task_config='demo_clean', scene_seed=scene, frame_index=frame,
                             replay_split='replay_holdout' if scene == 3 else 'train',
                             payload='/not_read_by_sampling')
                    if kind == 'cf': r['cf_retention'] = True
                    if kind == 'fg': r['fg_correction'] = True
                    result.append(r)
    return result


def test_all_five_receive_expert_actions_and_controls_match_temporal_strata():
    data = rows(); before = deepcopy(data)
    full, off = mixture_stream(data, 42), mixture_stream(data, 42, 'off')
    counts = {task: 0 for task in TASKS}; phases = {task: set() for task in TASKS}
    for _ in range(200):
        a, b = next(full), next(off)
        assert len(a) == len(b) == 12
        assert sum(bool(r.get('cf_retention')) for r in a) == 4
        assert sum(bool(r.get('fg_correction')) for r in a) == 4
        for x, y in zip(a, b):
            assert x['scene_seed'] != 3 and y['scene_seed'] != 3
            if x.get('fg_correction'):
                assert y.get('ordinary_cf_control') and not y.get('fg_correction')
                for k in ['pair_id', 'task_config', 'replay_trajectory_third']:
                    assert x[k] == y[k]
            else:
                assert x == y
            if x.get('ordinary_target_trajectory'):
                counts[x['source_task']] += 1
                phases[x['source_task']].add(x['replay_trajectory_third'])
    assert counts == dict.fromkeys(TASKS, 160)
    assert all(v == {0, 1, 2} for v in phases.values())
    assert data == before


@pytest.mark.parametrize('corruption', ['missing_expert', 'initial_only', 'duplicate', 'split_overlap'])
def test_data_errors_are_rejected(corruption):
    data = rows()
    if corruption == 'missing_expert':
        data = [r for r in data if r['source_task'] != TASKS[0] or r.get('cf_retention') or r.get('fg_correction')]
    elif corruption == 'initial_only':
        data = [r for r in data if r['frame_index'] == 0]
    elif corruption == 'duplicate':
        data.append(dict(data[0]))
    else:
        data.append(dict(data[0], id='cross_split', replay_split='replay_holdout'))
    with pytest.raises(ValueError): pools(data)


def payload():
    return dict(captured={'source': {'name': 'source'}, 'target': {'name': 'target'}},
                references={'source': torch.full((1,32,14), 99.), 'target': torch.full((1,32,14), 3.)},
                valid={k: torch.ones((1,32), dtype=torch.bool) for k in ['source', 'target']})


@pytest.mark.parametrize('kind,target,gradient', [('expert', 'expert_action', -6.), ('cf', 'teacher_action', -2.)])
def test_expert_gradient_is_not_replaced_with_retention_teacher(monkeypatch, kind, target, gradient):
    import experiments.robotwin.deployed_action_objective as deployed
    model = SimpleNamespace(torch_dtype=torch.float32, x=torch.nn.Parameter(torch.tensor(0.)))
    calls = []
    def sample(m,c,n,**kwargs):
        assert c['name'] == 'target'
        return m.x.expand_as(n)
    def teacher(t,c,n):
        calls.append('teacher')
        return torch.ones_like(n)
    monkeypatch.setattr(deployed, 'sample', sample)
    monkeypatch.setattr(deployed, 'teacher_action', teacher)
    row = {'ordinary_target_trajectory': True} if kind == 'expert' else {'cf_retention': True}
    p = payload()
    result = backward_example(model, row, p, torch.zeros((1,32,14)), None, teachers={'cf': object()})
    assert model.x.grad.item() == pytest.approx(gradient)
    assert result['supervision_target'] == target and result['action_objective'] == OBJECTIVE
    assert calls == ([] if kind == 'expert' else ['teacher'])
    assert set(p['captured']) == {'source', 'target'}


def test_fg_and_ordinary_control_have_identical_target_loss(monkeypatch):
    import experiments.robotwin.deployed_action_objective as deployed
    monkeypatch.setattr(deployed, 'sample', lambda m,c,n,**kwargs: m.x.expand_as(n))
    gradients = []
    for row in [{'fg_correction': True}, {'ordinary_cf_control': True}]:
        model = SimpleNamespace(torch_dtype=torch.float32, x=torch.nn.Parameter(torch.tensor(0.)))
        backward_example(model, row, payload(), torch.zeros((1,32,14)), None, teachers={})
        gradients.append(model.x.grad.item())
    assert gradients == [-6., -6.]


def test_target_payload_never_accepts_unlabelled_or_native_rows():
    for row in [{}, {'native_retention': True}]:
        with pytest.raises(ValueError): target_payload(row, payload())
