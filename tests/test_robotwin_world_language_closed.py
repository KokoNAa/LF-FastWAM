from copy import deepcopy
import pytest
from experiments.robotwin.manipulation_metrics import PROTOCOL
from scripts.audit_robotwin_world_language_closed import validate_episode


def valid_episode():
    return dict(selected_goal='source',source_goal_ever_success=True,source_goal_final_success=True,
        counterfactual_goal_ever_success=False,counterfactual_goal_final_success=False,
        initial_source_goal_success=False,initial_counterfactual_goal_success=False,
        selected_goal_success=True,native_eval_success=True,steps=100,step_limit=400,
        goal_evaluation_calls=102,manipulation_metrics=dict(protocol=deepcopy(PROTOCOL),
            physics_samples=100,simulation_seconds=.4,
            objects={'object':dict(max_lift_m=.05,correctly_lifted=True,first_lift_tick=30,
                placed_after_lift=False,first_placement_tick=None,final_placement=False,
                lift_arm='left',lift_contact_links=['finger1','finger2'],max_contact_links=2)},
            gripper_contact_links={'left':{'1':'finger1','2':'finger2'},'right':{'3':'finger3','4':'finger4'}},
            any_correct_object_lifted=True,all_instruction_objects_lifted=True,
            any_object_placed_after_lift=False,all_objects_placed_after_lift=False,
            full_goal_after_any_lift=True,first_full_goal_after_lift_tick=100,
            correct_placement_after_lift=False,first_correct_placement_tick=None))


def test_benchmark_success_need_not_satisfy_strict_placement():
    result=validate_episode(valid_episode())
    assert result['full_goal_after_any_lift'] and not result['correct_placement_after_lift']


@pytest.mark.parametrize('mutation,message',[
    (lambda x:x.update(native_eval_success=False),'termination'),
    (lambda x:x.update(goal_evaluation_calls=99),'sampling clocks'),
    (lambda x:x['manipulation_metrics']['objects']['object'].update(lift_contact_links=['finger1']),'bilateral'),
    (lambda x:x['manipulation_metrics']['objects']['object'].update(lift_contact_links=['finger1','unknown']),'Unknown grasp'),
    (lambda x:x['manipulation_metrics']['objects']['object'].update(placed_after_lift=True,first_placement_tick=29),'precedes lift'),
    (lambda x:x['manipulation_metrics'].update(any_correct_object_lifted=False),'aggregate'),
    (lambda x:x['manipulation_metrics'].update(first_full_goal_after_lift_tick=101),'outside lifted'),
    (lambda x:x['manipulation_metrics']['protocol'].update(lift_height_m=.01),'Changed physical'),
])
def test_rejects_inconsistent_physical_or_goal_evidence(mutation,message):
    row=valid_episode();mutation(row)
    with pytest.raises(ValueError,match=message):validate_episode(row)
