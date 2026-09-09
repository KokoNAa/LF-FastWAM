from types import SimpleNamespace

import pytest
import torch

from experiments.robotwin import deployed_action_objective as objective
from experiments.robotwin.native_teacher import NativeTeacher
from experiments.robotwin.parent_eraf_teacher import validate_teacher_binding


def fixture(tmp_path):
    guard = torch.nn.Linear(1, 1, bias=False)
    guard.register_buffer('scale', torch.tensor([3.]))
    guard.weight.data.fill_(2.)
    adapter = torch.nn.Parameter(torch.tensor([5.]))
    model = SimpleNamespace(policy_guard_modules=guard, adapter=adapter)
    payload = dict(stage='joint', optimizer_steps=200, fg_supervision='off',
                   provenance={'eraf': 'on'}, mot_trainable={'a': torch.tensor([7.])},
                   policy_guard={'weight': torch.tensor([[11.]]), 'scale': torch.tensor([13.])})
    path = tmp_path / 'parent.pt'
    torch.save(payload, path)
    return model, payload, path


def test_full_teacher_uses_frozen_adapters_guard_and_buffers_then_restores_student(tmp_path, monkeypatch):
    model, _, path = fixture(tmp_path)
    teacher = NativeTeacher(model, {'a': model.adapter}, path, eraf=True)
    pointers = {n: p.data_ptr() for n, p in teacher.parameters.items()}
    def sample(m, captured, noise, *, eraf, checkpoint):
        assert eraf is True and checkpoint is False and not torch.is_grad_enabled()
        return noise * (m.adapter + m.policy_guard_modules.weight.flatten()) * m.policy_guard_modules.scale
    monkeypatch.setattr(objective, 'sample', sample)
    expected = torch.tensor([(7. + 11.) * 13.])
    assert torch.equal(objective.teacher_action(teacher, {}, torch.ones(1)), expected)
    assert all(p.data_ptr() == pointers[n] for n, p in teacher.parameters.items())
    assert model.adapter.item() == 5. and model.policy_guard_modules.weight.item() == 2.
    assert model.policy_guard_modules.scale.item() == 3.
    assert all(not v.requires_grad and v.grad is None for v in teacher.values.values())
    # Accumulated student gradients survive later teacher calls; targets stay fixed.
    loss = (model.adapter + model.policy_guard_modules.weight.flatten()).square().sum()
    loss.backward()
    old_grad = model.adapter.grad.clone()
    with torch.no_grad():
        model.adapter.add_(1.)
        model.policy_guard_modules.weight.add_(1.)
    assert torch.equal(objective.teacher_action(teacher, {}, torch.ones(1)), expected)
    assert torch.equal(model.adapter.grad, old_grad)
    assert model.policy_guard_modules.weight.grad.item() == 14.


def test_guard_storage_restored_on_failure_and_native_default_kept(tmp_path, monkeypatch):
    model, _, path = fixture(tmp_path)
    teacher = NativeTeacher(model, {'a': model.adapter}, path, eraf=True)
    def fail(*args, **kwargs):
        assert model.policy_guard_modules.weight.item() == 11.
        assert model.policy_guard_modules.scale.item() == 13.
        raise RuntimeError('failed')
    monkeypatch.setattr(objective, 'sample', fail)
    with pytest.raises(RuntimeError, match='failed'):
        objective.teacher_action(teacher, {}, torch.ones(1))
    assert model.policy_guard_modules.weight.item() == 2.
    assert model.policy_guard_modules.scale.item() == 3.
    native = NativeTeacher(model, {'a': model.adapter}, path)
    def off(m, c, n, *, eraf, checkpoint):
        assert eraf is False and m.policy_guard_modules.weight.item() == 2.
        return n * m.adapter
    monkeypatch.setattr(objective, 'sample', off)
    assert objective.teacher_action(native, {}, torch.ones(1)).item() == 7.


def test_full_teacher_rejects_incomplete_or_incompatible_guard(tmp_path):
    model, payload, path = fixture(tmp_path)
    for patch in ({'provenance': {'eraf': 'off'}}, {'optimizer_steps': 0},
                  {'policy_guard': {'weight': torch.tensor([[11.]])}}):
        torch.save(payload | patch, path)
        with pytest.raises(ValueError):
            NativeTeacher(model, {'a': model.adapter}, path, eraf=True)


def test_parent_teacher_requires_exact_joint_continuation_and_hash(tmp_path):
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    _, _, path = fixture(tmp_path)
    plan = dict(cf_teacher_mode='parent_eraf', action_objective='five_task_expert_rollout_v1',
                continue_eraf_with_fg=True, stage='joint', eraf='on', fg='full',
                policy_scope='action', interface_scope='all', fg_gradient_route='joint',
                checkpoint=str(path), cf_teacher=str(path), continuation_parent_sha256=file_sha256(path))
    assert validate_teacher_binding(plan) == file_sha256(path)
    for patch in ({'cf_teacher': str(tmp_path / 'old.pt')}, {'continuation_parent_sha256': '0'*64},
                  {'fg_gradient_route': 'eraf_only'}, {'resume_state': 'optimizer.pt'},
                  {'continue_eraf_with_fg': False}, {'cf_teacher_mode': 'unknown'}):
        with pytest.raises(ValueError):
            validate_teacher_binding(plan | patch)
    assert validate_teacher_binding({}) is None


def test_full_trial_preserves_matched_200_step_recipe_and_uses_completed_parent(tmp_path):
    from scripts.run_robotwin_five_task_repair import training_command
    from scripts.run_robotwin_parent_eraf_trial import candidate_command, PARENT_SHA
    plan = dict(manifest='/manifest', source_bank='/bank', strongest_checkpoint='/no_eraf', steps=200,
                arms={'eraf_fg': dict(parent='/eraf200', parent_sha256=PARENT_SHA, eraf='on', fg='full', gpus=[0,1,2])})
    old = training_command(plan, tmp_path, 'eraf_fg')
    new = candidate_command(plan, tmp_path, '/eraf200')
    expected = list(old)
    expected[expected.index('--cf-teacher') + 1] = '/eraf200'
    assert new == expected + ['--cf-teacher-mode', 'parent_eraf']
    assert plan['steps'] == 200 and plan['strongest_checkpoint'] == '/no_eraf'
