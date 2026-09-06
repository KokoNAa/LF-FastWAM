import sys
import json
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import pytest

from experiments.robotwin.target_goal_branches import action_difference, require_same_observation


def test_action_difference_separates_a_common_prefix_from_immediate_goal_choices():
    source, target = np.zeros((60, 14)), np.zeros((40, 14))
    target[25:, 2] = .4
    result = action_difference(source, target)
    assert result['first_different_frame'] == 25
    assert not result['informative_first24'] and result['action_rmse_first24'] == 0
    assert result['action_rmse_first32'] > 0
    target[8, 2] = .4
    assert action_difference(source, target)['informative_first24']
    assert action_difference(source, source)['first_different_frame'] is None
    with pytest.raises(ValueError):
        action_difference(source[:12], target)


def test_paired_experts_cannot_change_camera_or_proprio_at_the_branch():
    first = {'qpos': np.zeros(14), 'images': {'head': np.zeros((2, 2, 3))}}
    require_same_observation(first, first, ['head'])
    for second in (first | {'qpos': np.ones(14)},
                   first | {'images': {'head': np.ones((2, 2, 3))}}):
        with pytest.raises(ValueError, match='observation'):
            require_same_observation(first, second, ['head'])


@pytest.mark.parametrize('language,expected', [('source', 'rgb'), ('target', 'bgr')])
def test_expert_continuation_clears_slots_for_the_selected_ranking_goal(monkeypatch, language, expected):
    from experiments.robotwin import eraf_fg_collection as collection
    monkeypatch.setitem(sys.modules, 'envs.utils', SimpleNamespace(ArmTag=str))
    actors = [SimpleNamespace(get_pose=lambda x=x: SimpleNamespace(p=np.array([x, 0., .75])))
              for x in (-.1, 0., .1)]
    cleared, variants = [], []
    task = SimpleNamespace(block1_target_pose=[-.1, 0, .75], block2_target_pose=[0, 0, .75],
                           block3_target_pose=[.1, 0, .75], table_z_bias=0,
                           pgc_scene_actors=lambda: actors, move=lambda *args: None,
                           open_gripper=lambda **kwargs: None, move_by_displacement=lambda **kwargs: None,
                           pick_and_place_block=lambda actor, pose: cleared.append(actor))
    spec = SimpleNamespace(source_task='blocks_ranking_rgb', source_variant='rgb', counterfactual_variant='bgr')
    monkeypatch.setattr(collection, 'play_variant', lambda task, spec, variant: variants.append(variant))
    collection.continue_to_goal(task, spec, selected_goal=language)
    assert variants == [expected] and task._pgc_active_variant == expected
    assert len(cleared) == (0 if language == 'source' else 2)
    if cleared:
        assert cleared[0] is actors[0] and cleared[1] is actors[2]


@pytest.mark.parametrize('case', ['verified', 'source_failed', 'changed_camera'])
def test_probe_requires_both_replayed_goals_at_one_observation(tmp_path, monkeypatch, case):
    from experiments.robotwin import eraf_fg_collection as collection, pgc_task_variants
    from scripts import collect_pgc_robotwin_pairs as pairs, probe_robotwin_target_goal_branches as probe
    from experiments.robotwin.eraf_fg_contract import CAMERAS
    from test_robotwin_eraf_fg_contract import correction
    record_path, output = tmp_path / 'record.json', tmp_path / 'out'
    record = correction() | {'source_instruction': 'RGB', 'counterfactual_instruction': 'BGR',
                             'frame_path': str(tmp_path / 'correction.npz')}
    record_path.write_text(json.dumps(record))
    np.savez(tmp_path / 'failure_rollout.npz', initial=np.zeros(50), actions=np.zeros((48, 14)),
             capture_steps=np.array([48]), states=np.zeros((1, 50)))
    task = SimpleNamespace(reached=None, plan_success=True, pgc_eraf_snapshot=lambda: {'roles': np.zeros(2)})
    task.setup_demo = lambda **kwargs: setattr(task, 'reached', None)

    def obs():
        value = int(case == 'changed_camera' and task._pgc_active_variant == 'bgr')
        return {'joint_action': {'vector': np.zeros(14)},
                'observation': {c: {'rgb': np.full((2, 3, 3), value, dtype=np.uint8)} for c in CAMERAS}}

    task.get_obs = obs
    monkeypatch.setattr(pairs, '_load_robotwin_args', lambda **kwargs: (task, {}))
    monkeypatch.setattr(pairs, '_capture_data_type', lambda config: {})
    monkeypatch.setattr(pairs, '_close', lambda task: None)
    monkeypatch.setattr(pgc_task_variants, 'install_pgc_task_contract', lambda *args: None)
    monkeypatch.setattr(collection, 'physical_state', lambda task: np.zeros(50))
    monkeypatch.setattr(collection, 'replay_prefix', lambda *args: 0.)
    monkeypatch.setattr(collection, 'full_goal', lambda task, spec, selected_goal: task.reached == selected_goal)
    active = {}

    @contextmanager
    def recording(task):
        active['controls'], active['frames'] = [], []
        yield active['controls'], active['frames']

    def continue_goal(task, spec, *, selected_goal):
        for _ in range(40):
            active['frames'].append({'qpos': np.full(14, int(selected_goal == 'target')),
                'images': {c: np.zeros((2, 3, 3), dtype=np.uint8) for c in CAMERAS},
                'grounding': {'roles': np.zeros(2)}})
        active['controls'].append({'goal': selected_goal})
        task.reached = None if case == 'source_failed' and selected_goal == 'source' else selected_goal

    replays = []

    def replay(task, controls):
        task.reached = controls[0]['goal']
        replays.append(task.reached)
        return 0.

    monkeypatch.setattr(collection, 'record_continuation', recording)
    monkeypatch.setattr(collection, 'continue_to_goal', continue_goal)
    monkeypatch.setattr(collection, 'replay_continuation', replay)
    monkeypatch.setattr(sys, 'argv', ['probe', '--record', str(record_path), '--output', str(output),
                                    '--robotwin-root', str(tmp_path)])
    probe.main()
    report = json.loads((output / 'results.json').read_text())
    assert report['complete'] and report['paired_goals_verified'] == (case == 'verified')
    if case == 'verified':
        assert replays == ['source', 'source', 'target', 'target']
        assert report['initial_observations_equal'] and report['informative_first24']
        for language in ('source', 'target'):
            assert report['conditions'][language]['verified_replays'] == 2
    elif case == 'changed_camera':
        assert 'observation' in report['conditions']['target']['error']
        assert replays == ['source', 'source']
    else:
        assert replays == ['target', 'target']
        assert report['conditions']['source']['full_goal_success'] is False
