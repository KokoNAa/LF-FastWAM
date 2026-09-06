from types import SimpleNamespace
import pytest
from experiments.robotwin.eraf_fg_data import retention_language, validate_retention_scene
from experiments.robotwin import eraf_fg_collection as collection


def record(**updates):
    return dict(native_retention=True, full_native_episode_success=True, retention_condition='correct',
                source_task='place_a2b_left', task_config='demo_clean', scene_seed=80600000,
                capture_path='/capture.npz', capture_sha256='abc', teacher_checkpoint_sha256='teacher',
                source_instruction='left', counterfactual_instruction='right', frame_index=0, **updates)


def test_native_capture_cannot_be_mislabeled_as_cf_retention():
    assert retention_language(record()) == 'source'
    assert retention_language({'cf_retention': True, 'full_cf_episode_success': True}) == 'target'
    for row in [record() | {'cf_retention': True},
                record() | {'full_native_episode_success': False},
                record() | {'retention_condition': 'counterfactual'}]:
        with pytest.raises(ValueError):
            retention_language(row)


def test_retention_scene_keeps_one_teacher_condition_and_exact_state_indices():
    rows = [record(), record() | {'frame_index': 1}]
    assert validate_retention_scene(rows) == 'source'
    for change in [{'teacher_checkpoint_sha256': 'different'}, {'frame_index': 0},
                   {'source_instruction': 'wrong'}]:
        with pytest.raises(ValueError):
            validate_retention_scene([rows[0], rows[1] | change])


def test_rollout_stop_condition_and_release_match_selected_goal(monkeypatch):
    spec = SimpleNamespace(source_variant='native', counterfactual_variant='cf')
    task = SimpleNamespace(check_success=lambda: 'original',
        robot=SimpleNamespace(is_left_gripper_open=lambda: True, is_right_gripper_open=lambda: True))
    original = task.check_success
    monkeypatch.setattr(collection, 'check_variant', lambda task, spec, variant: variant == 'native')
    with collection.goal_audit(task, spec, selected_goal='source') as audit:
        assert task.check_success() is True
        assert audit == {'source': True, 'target': False}
    assert task.check_success is original
    with collection.goal_audit(task, spec) as audit:
        assert task.check_success() is False  # Existing CF default is preserved.
        assert audit['source'] and not audit['target']
    assert collection.full_goal(task, spec, selected_goal='source')
    assert not collection.full_goal(task, spec)
    task.robot.is_right_gripper_open = lambda: False
    assert not collection.full_goal(task, spec, selected_goal='source')
