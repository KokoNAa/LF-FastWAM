from types import SimpleNamespace as NS
import pytest

from experiments.robotwin.manipulation_metrics import EventAccumulator, ManipulationObserver, actor_attributes


def sample(lift=.04, arm='left', placed=False, contacts=2):
    return dict(lift_m=lift, grasp_arm=arm, placement=placed,
                contact_links_count=contacts, contact_links=['finger1', 'finger2'] if arm else [])


def test_sustained_physical_lift_then_release_is_required():
    s = EventAccumulator(['target'], .004)
    for _ in range(24):
        s.update({'target': sample()}, full_goal=False)
    assert not s.report()['any_correct_object_lifted']
    s.update({'target': sample()}, full_goal=False)
    assert s.report()['any_correct_object_lifted']
    assert not s.report()['any_object_placed_after_lift']
    s.update({'target': sample(lift=0, arm=None, placed=True, contacts=0)}, full_goal=True)
    r = s.report()
    assert r['full_goal_after_any_lift'] and r['all_objects_placed_after_lift']
    assert r['objects']['target']['first_lift_tick'] == 25
    assert r['objects']['target']['first_placement_tick'] == 26


@pytest.mark.parametrize('samples', [[sample(lift=.01)]*30, [sample(arm=None)]*30,
                                    [sample()]*24+[sample(arm=None)]+[sample()]*24,
                                    [sample(arm='left')]*20+[sample(arm='right')]*20])
def test_touch_push_brief_lift_or_cross_arm_streak_not_grasp(samples):
    s = EventAccumulator(['target'], .004)
    for x in samples:
        s.update({'target': x}, full_goal=False)
    assert not s.report()['any_correct_object_lifted']


def test_goal_without_lift_is_not_attributed_to_pick_place():
    s = EventAccumulator(['target'], .004)
    s.update({'target': sample(lift=0, arm=None, placed=True)}, full_goal=True)
    assert not s.report()['full_goal_after_any_lift']
    assert not s.report()['any_object_placed_after_lift']


def test_multi_object_partial_and_final_drop():
    s = EventAccumulator(['red', 'blue'], .004)
    for _ in range(25):
        s.update({'red': sample(), 'blue': sample(arm=None)}, full_goal=False)
    s.update({'red': sample(0, None, True), 'blue': sample(arm=None)}, full_goal=False)
    r = s.report()
    assert r['any_correct_object_lifted'] and not r['all_instruction_objects_lifted']
    assert r['any_object_placed_after_lift'] and not r['full_goal_after_any_lift']
    s.update({'red': sample(0, None, False), 'blue': sample(arm=None)}, full_goal=False)
    r = s.report()['objects']['red']
    assert r['placed_after_lift'] and not r['final_placement']


def test_duplicate_box_names_are_distinguished_by_physical_id():
    observer = object.__new__(ManipulationObserver)
    observer.attributes = ['red', 'green']
    observer.actor_ids = {10: 'red', 11: 'green'}
    observer.fingers = {'left': {21: 'f1', 22: 'f2'}, 'right': {31: 'f1', 32: 'f2'}}
    def contact(a, b, impulse=1):
        return NS(bodies=[NS(entity=NS(per_scene_id=x, name='box')) for x in (a, b)],
                  points=[NS(impulse=[impulse, 0, 0])])
    observer.env = NS(scene=NS(get_contacts=lambda: [contact(10,21),contact(22,10),
                          contact(11,31,0),contact(11,32)]))
    assert observer.contacts() == {'red': {'left': {'f1','f2'}, 'right': set()},
                                   'green': {'left': set(), 'right': {'f2'}}}


def test_instruction_objects_and_unsupported_tasks():
    assert actor_attributes(dict(type='stacked_on',actor='red',reference='green')) == ['red','green']
    assert actor_attributes(dict(type='relative_pose',actor='a',reference='b')) == ['a']
    with pytest.raises(ValueError):
        actor_attributes(dict(type='unknown'))
