import json
import pytest
from scripts.report_robotwin_formal_five40 import summarize_cell,report,markdown
from scripts.run_robotwin_formal_five40 import write,records,TASKS
from test_robotwin_formal_five40 import matrix


def episode(cf,lift,after=False,strict=False):
    return dict(counterfactual_goal_ever_success=cf,manipulation_metrics=dict(
        any_correct_object_lifted=lift,full_goal_after_any_lift=after,correct_placement_after_lift=strict,
        physics_samples=10,all_instruction_objects_lifted=False,final_strict_slot_assignment=None))


def test_task_and_strict_placement_denominators_stay_separate():
    x=summarize_cell([episode(True,False),episode(True,True,True),episode(False,True)])
    assert x['cf']==2 and x['lift']==2 and x['cf_after_lift']==1 and x['strict_release_placement']==0
    assert x['task_placement_given_lift_rate']==.5 and x['strict_release_given_lift_rate']==0
    assert x['cf_without_verified_lift']==1 and x['lift_without_later_task_success']==1


def test_zero_lift_does_not_invent_a_zero_conditional_success_rate():
    x=summarize_cell([episode(False,False)])
    assert x['task_placement_given_lift_rate'] is None and x['strict_release_given_lift_rate'] is None


def test_inconsistent_success_attribution_is_rejected():
    with pytest.raises(ValueError):summarize_cell([episode(False,True,True)])
    with pytest.raises(ValueError):summarize_cell([episode(True,True,False,True)])


def test_complete_six_hundred_report_is_paired_and_named(matrix):
    root,plan=matrix;plan['tasks']=TASKS
    write(root/'frozen_protocol.json',plan);write(root/'status.json',dict(complete=True,status='complete'))
    write(root/'terminal_verification.json',dict(complete=True,episodes=600))
    for arm in plan['models']:
        for task in TASKS:
            base=root/'evaluation'/arm/task/'demo_clean/counterfactual'
            c=json.loads((base/'complete.json').read_text());c.update(policy_kind='repair',memory_mode='carry',
                  deployment=dict(action_horizon=32,replan_steps=24,inference_steps=10));write(base/'complete.json',c)
            data=records(base/'episodes.jsonl')
            for row in data:
                row.update(source_task=task,condition='counterfactual',instruction_goal='counterfactual',selected_goal='counterfactual')
                for obj in row['manipulation_metrics']['objects'].values():obj['final_placement']=obj['placed_after_lift']
            (base/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
    r=report(root)
    assert r['episodes']==600 and len(r['cells'])==15
    assert all(x['cf']==50 and x['lift']==100 and x['task_placement_given_lift_rate']==.5 for x in r['pooled'].values())
    assert all(x['gains']==x['losses']==0 for x in r['paired'])
    assert '50/200（25.0%）' in markdown(r) and '严格脱手放置确认' in markdown(r)
    write(root/'status.json',dict(complete=False,status='evaluating_fixed_models'))
    with pytest.raises(ValueError,match='complete600'):report(root)
