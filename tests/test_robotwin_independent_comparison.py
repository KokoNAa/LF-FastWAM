from copy import deepcopy
import json

import pytest

from scripts.compare_robotwin_independent_test import TASKS,build_report


def fixture(root):
    def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj))
    exclusions=root/'train.json';write(exclusions,dict(states=[dict(scene_seed=86000100)]))
    methods={}
    for method in ('baseline','candidate'):
        folder=root/method;checkpoint='/server/'+method+'.pt';cells=[]
        methods[method]=str(folder)
        for task_id,task in enumerate(sorted(TASKS)):
            dest=folder/task/'demo_clean/counterfactual'
            rows=[]
            for i in range(10):
                success=method=='candidate' and i==0
                row=dict(scene_seed=91700000+1000*task_id+i,episode_index=i,source_task=task,
                    source_instruction='source',counterfactual_instruction='target',policy_instruction='target',
                    instruction_goal='counterfactual',selected_goal='counterfactual',condition='counterfactual',
                    selected_goal_success=success,steps=100,counterfactual_goal_final_success=success,
                    source_goal_final_success=False,
                    final_source_goal=dict(assignments=[dict(actor=a,planar_distance=.12) for a in ('hamburg','frenchfries')]),
                    final_counterfactual_goal=dict(assignments=[dict(actor=a,planar_distance=.01) for a in ('hamburg','frenchfries')]))
                rows.append(row)
            write(dest/'complete.json',dict(complete=True,checkpoint=checkpoint,checkpoint_sha256=('a' if method=='baseline' else 'b')*64,
                canonical_sha256='c'*64,deployment=dict(seed=42),eraf='off' if method=='baseline' else 'on',policy_kind='repair'))
            write(dest/'initial_states.json',[dict(scene_seed=r['scene_seed'],state=[0,1]) for r in rows])
            (dest/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            cells.append(dict(source_task=task,condition='counterfactual',episodes=10,selected_goal_successes=sum(r['selected_goal_success'] for r in rows)))
        write(folder/'summary.json',dict(complete=True,checkpoint=checkpoint,cells=cells,episodes=100))
    return dict(target='candidate',methods=methods,exclude_records=[str(exclusions)])


def test_complete_matched_test_reports_counts_and_never_claims_selection_audit(tmp_path):
    result=build_report(fixture(tmp_path))
    assert result['complete'] and result['methods']['candidate']['ten_task_macro_cf']==.1
    assert result['paired_target_vs_controls']['baseline']['gains']==10
    assert result['paired_target_vs_controls']['baseline']['losses']==0
    assert result['supplementary_final_slots']['candidate']['strict_final_slot_successes']==1
    assert not result['checkpoint_selection_freeze_verified'] and not result['goal_achievement_claim']


@pytest.mark.parametrize('corruption',['excluded','namespace','duplicate','instruction','initial','checkpoint','partial','goal'])
def test_rejects_leakage_partial_and_unpaired_test_evidence(tmp_path,corruption):
    config=fixture(tmp_path)
    task=sorted(TASKS)[0];folder=tmp_path/'candidate'/task/'demo_clean/counterfactual'
    path=folder/'episodes.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
    if corruption=='excluded':
        p=tmp_path/'train.json';p.write_text(json.dumps(dict(states=[dict(scene_seed=rows[0]['scene_seed'])])))
    if corruption=='namespace':rows[0]['scene_seed']=91300000
    if corruption=='duplicate':rows[0]['scene_seed']=rows[1]['scene_seed']
    if corruption=='instruction':rows[0]['policy_instruction']='source'
    if corruption=='goal':rows[0]['instruction_goal']='source'
    if corruption=='partial':rows.pop()
    if corruption=='initial':
        p=folder/'initial_states.json';v=json.loads(p.read_text());v[0]['state']=[1,2];p.write_text(json.dumps(v))
    if corruption=='checkpoint':
        p=folder/'complete.json';v=json.loads(p.read_text());v['checkpoint_sha256']='d'*64;p.write_text(json.dumps(v))
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):build_report(config)
