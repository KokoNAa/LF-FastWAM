import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from experiments.robotwin.pad_counterfactual import PAD_ACTORS, geometry, play_front
from experiments.robotwin.pgc_data import ROBOTWIN_TEN_TASK_NAMES, pair_spec_from_source_task
from experiments.robotwin.pgc_task_variants import install_pgc_observation_contract, _play_ranking
from experiments.robotwin.language_interventions import GoalObserver, load_intervention_manifest

ROOT = Path(__file__).resolve().parents[1]


class Actor:
    def __init__(self, p, q):
        self.pose = SimpleNamespace(p=np.array(p, dtype=float), q=np.array(q, dtype=float))
    def get_pose(self):
        return self.pose


def scene(name):
    robot = SimpleNamespace(is_left_gripper_open=lambda: True, is_right_gripper_open=lambda: True)
    env = SimpleNamespace(robot=robot, table_z_bias=.013, is_left_gripper_open=lambda: True,
                          is_right_gripper_open=lambda: True, check_success=lambda: False)
    actor_name, pad_name = PAD_ACTORS[name]
    q = [0, 0, .7, .7] if name == 'place_mouse_pad' else [.5, .5, .5, .5]
    setattr(env, actor_name, Actor([.12, -.07, .754], q))
    setattr(env, pad_name, Actor([.12, -.07, .753], [1, 0, 0, 0]))
    return env


@pytest.mark.parametrize('name', list(PAD_ACTORS))
def test_native_geometry_matches_vendored_predicate_and_cf_is_exclusive(name):
    env = scene(name)
    module = ast.parse((ROOT/'third_party/RoboTwin/envs'/f'{name}.py').read_text())
    cls = next(x for x in module.body if isinstance(x, ast.ClassDef))
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'check_success')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<native-predicate>', 'exec'), namespace)
    native = namespace['check_success']
    actor = getattr(env, PAD_ACTORS[name][0])
    original = actor.pose.p.copy()
    for offset in [0, .01, .025, .05, -.13]:
        actor.pose.p = original + [0, offset, 0]
        assert geometry(env, name, 'native_pad')[0] == bool(native(env))
        assert not (geometry(env, name, 'native_pad')[0] and geometry(env, name, 'front_pad')[0])
    assert geometry(env, name, 'front_pad')[0]


@pytest.mark.parametrize('name', list(PAD_ACTORS))
def test_selected_goal_does_not_recurse_or_move_reference(name):
    env = scene(name)
    install_pgc_observation_contract(env, pair_spec_from_source_task(name))
    pairs = load_intervention_manifest(ROOT/'configs/eval/robotwin_cis_ten_tasks.json')
    pair = next(x for x in pairs if x.source_task == name)
    observer = GoalObserver(env, pair)
    observer.install_selected_goal('counterfactual')
    pad = getattr(env, PAD_ACTORS[name][1]); before = pad.pose.p.copy()
    assert not env.check_success()
    getattr(env, PAD_ACTORS[name][0]).pose.p[1] -= .13
    assert env.check_success()
    assert not observer.update().source.success
    np.testing.assert_array_equal(pad.pose.p, before)
    observer.restore_native_goal()


def test_expert_offsets_command_only_and_restores_on_failure():
    env = scene('move_stapler_pad'); calls = []
    target = [.12, -.07, .753, 1, 0, 0, 0]
    def place(actor, *, target_pose):
        calls.append(np.array(target_pose))
        raise RuntimeError('planner failure')
    env.place_actor = place
    env.play_once = lambda: env.place_actor(env.stapler, target_pose=target)
    with pytest.raises(RuntimeError, match='planner failure'):
        play_front(env, 'move_stapler_pad')
    assert env.place_actor is place
    assert target[1] == -.07
    np.testing.assert_allclose(calls[0], [.12, -.20, .753, 1, 0, 0, 0])
    np.testing.assert_allclose(env.pad.pose.p, [.12, -.07, .753])


def test_ten_distinct_tasks_preserve_five_historical_pairs():
    old = load_intervention_manifest(ROOT/'configs/eval/robotwin_cis_v939_four_tasks.json')
    new = load_intervention_manifest(ROOT/'configs/eval/robotwin_cis_ten_tasks.json',
                                     robotwin_root=ROOT/'third_party/RoboTwin')
    assert len(new) == len(set(ROBOTWIN_TEN_TASK_NAMES)) == 10
    assert new[:5] == old


def test_size_expert_swaps_slots_and_keeps_size_instruction_labels():
    env = SimpleNamespace(block1=object(), block2=object(), block3=object(),
        block1_target_pose=[1], block2_target_pose=[2], block3_target_pose=[3], info={})
    calls = []
    env.pick_and_place_block = lambda actor, slot: calls.append((actor, slot)) or 'left'
    _play_ranking(env, 'small_to_large')
    assert calls == [(env.block3, [1]), (env.block2, [2]), (env.block1, [3])]
    assert env.info['info']['{A}'] == 'large block'
    assert env.info['info']['{C}'] == 'small block'
