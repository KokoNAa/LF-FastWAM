#!/usr/bin/env python3
"""Independently validate paired closed-loop artifacts and physical-event records."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from experiments.robotwin.manipulation_metrics import PROTOCOL

TASKS=['blocks_ranking_rgb','stack_blocks_two','place_a2b_left','place_a2b_right','place_burger_fries']
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def validate_policy_mode(proof, model):
    expected={'released':'legacy','no_eraf':'repair'}[model]
    require(proof['eraf']=='off' and proof['policy_kind']==expected and not proof['skip_file_hashes'],'Wrong policy/binding mode')

def validate_episode(row):
    """Check cross-field constraints; summaries alone cannot prove physical events."""
    selected=row['selected_goal'];require(selected in ['source','counterfactual'],'Unknown selected goal')
    for goal in ['source','counterfactual']:
        for ending in ['ever_success','final_success']:
            require(type(row[f'{goal}_goal_{ending}']) is bool,'Nonboolean goal field')
        require(not row[f'initial_{goal}_goal_success'],'Initially satisfied goal')
        require(not row[f'{goal}_goal_final_success'] or row[f'{goal}_goal_ever_success'],'Final goal absent from ever record')
    require(row['selected_goal_success']==row[selected+'_goal_ever_success'],'Selected goal mismatch')
    require(row['native_eval_success']==row['selected_goal_success']==row[selected+'_goal_final_success'],'Selected immediate-termination outcome mismatch')
    require(0<row['steps']<=row['step_limit'],'Invalid action-step count')
    m=row['manipulation_metrics'];require(m['protocol']==PROTOCOL,'Changed physical event protocol')
    n=m['physics_samples'];require(isinstance(n,int) and n>0,'No physics samples')
    require(row['goal_evaluation_calls']==n+2,'Goal and physical sampling clocks differ')
    dt=m['simulation_seconds']/n;require(0<dt<=.02 and math.isfinite(dt),'Invalid simulator timestep')
    objects=m['objects'];require(bool(objects),'No instruction objects')
    for name,obj in objects.items():
        require(math.isfinite(obj['max_lift_m']) and obj['max_lift_m']>=0,'Invalid object lift')
        lifted=obj['correctly_lifted'];tick=obj['first_lift_tick'];placed=obj['placed_after_lift'];pt=obj['first_placement_tick']
        require(type(lifted) is bool and type(placed) is bool,'Nonboolean object event')
        require(lifted==(tick is not None) and placed==(pt is not None),'Event/tick mismatch')
        if lifted:
            require(isinstance(tick,int) and math.ceil(PROTOCOL['sustained_grasp_seconds']/dt)<=tick<=n,'Lift outside physical sample clock')
            require(obj['max_lift_m']>=PROTOCOL['lift_height_m'],'Lift height below protocol')
            require(obj['lift_arm'] in ['left','right'],'No physical grasp arm')
            require(len(set(obj['lift_contact_links']))>=2 and obj['max_contact_links']>=2,'No bilateral finger-contact evidence')
            known=set(m['gripper_contact_links'][obj['lift_arm']].values())
            require(set(obj['lift_contact_links'])<=known,'Unknown grasp contact link')
        if placed:require(lifted and isinstance(pt,int) and tick<pt<=n,'Placement precedes lift')
        require(not obj['final_placement'] or placed,'Final placement absent from event record')
    for field,attr,fn in [('any_correct_object_lifted','correctly_lifted',any),('all_instruction_objects_lifted','correctly_lifted',all),('any_object_placed_after_lift','placed_after_lift',any),('all_objects_placed_after_lift','placed_after_lift',all)]:
        require(m[field]==fn(o[attr] for o in objects.values()),'Object/aggregate event mismatch: '+field)
    for field,tickfield in [('full_goal_after_any_lift','first_full_goal_after_lift_tick'),('correct_placement_after_lift','first_correct_placement_tick')]:
        require(m[field]==(m[tickfield] is not None),'Goal-placement event/tick mismatch')
        if m[field]:
            require(m['any_correct_object_lifted'] and row['selected_goal_success'],'Placement without lift/selected success')
            require(min(o['first_lift_tick'] for o in objects.values() if o['correctly_lifted'])<=m[tickfield]<=n,'Goal-placement event outside lifted interval')
    require(not m['correct_placement_after_lift'] or m['full_goal_after_any_lift'],'Strict placement without full goal after lift')
    return m

def audit(root,allow_partial=False):
    plan_path=root/'probes/plan.json';plan=read(plan_path);status=read(root/'status.json')
    require(list(plan['noise_seeds'])==[42,43,44] and set(plan['models'])=={'released','no_eraf'},'Unexpected study arms')
    outputs={};completed={};pending=[];initial_obs={};catalog={}
    def bind(p):outputs[str(p)]=sha(p)
    bind(plan_path)
    for task in TASKS:
        p=root/'probes/catalog'/task/'demo_randomized/correct/episodes.jsonl';rows=lines(p)
        require(len(rows)==10 and len({x['scene_seed'] for x in rows})==10,'Catalog count/duplicate')
        require({x['scene_seed'] for x in rows}=={x['scene_seed'] for x in plan['scenes'] if x['task']==task},'Catalog scene set differs from frozen plan')
        catalog[task]=(p,rows);bind(p)
    for model in plan['models']:
      for seed in plan['noise_seeds']:
       for v in ['source','target']:
        for action in ['source','target']:
            name=f'{model}-closed-v_{v}-a_{action}-seed{seed}';folder=root/'closed_loop'/name
            marker=folder/'world_language_complete.json';job=status['jobs'].get(name)
            if job and job.get('status')=='exited':require(job.get('exit_code')==0,'Closed worker failed: '+name)
            if not marker.exists() or not job or job.get('status')!='exited':pending.append(name);continue
            require(read(marker)==dict(complete=True,video_language=v,noise_seed=seed,intervention='Video text only; action text and all other production inputs retained.'),'Changed intervention marker')
            command=job['command']
            for flag,val in [('--world-video-language',v),('--policy-seed',str(seed)),('--output',str(folder)),('--episodes','10'),('--task-config','demo_randomized'),('--eraf','off'),('--policy-kind','legacy' if model=='released' else 'repair')]:
                require(flag in command and command[command.index(flag)+1]==val,'Worker command mismatch: '+flag)
            bind(marker);inp_path=folder/'world_language_inputs.jsonl';inputs=lines(inp_path);bind(inp_path)
            by_key={(x['source_task'],x['scene_seed']):x for x in inputs};require(len(inputs)==len(by_key)==50,'First-input coverage')
            counts=Counter();cell_results=[]
            for task in TASKS:
                condition='correct' if action=='source' else 'counterfactual';selected='source' if action=='source' else 'counterfactual'
                cell=folder/task/'demo_randomized'/condition;p,canonical=catalog[task]
                proof=read(cell/'complete.json');records=lines(cell/'episodes.jsonl');initial=read(cell/'initial_states.json')
                require(proof['complete'] and proof['manipulation_metrics'] and proof['instruction_type']=='canonical','Incomplete or changed evaluation')
                validate_policy_mode(proof,model)
                require(proof['checkpoint_sha256']==plan['checkpoint_sha256'][model] and proof['canonical_sha256']==sha(p),'Checkpoint/catalog binding mismatch')
                require(proof['deployment']==dict(action_horizon=32,replan_steps=24,inference_steps=10),'Changed production sampling')
                require(proof['task_config']=='demo_randomized' and len(records)==len(initial)==10,'Domain/count mismatch')
                cell_count=Counter()
                for row,c,h in zip(records,canonical,initial,strict=True):
                    require(row['source_task']==task and row['condition']==condition and row['selected_goal']==selected and row['instruction_goal']==selected,'Wrong episode condition')
                    for key in ['episode_index','scene_seed','source_instruction','counterfactual_instruction']:
                        require(row[key]==c[key],'Canonical episode identity mismatch: '+key)
                    require(h['scene_seed']==c['scene_seed'] and h['sha256']==c['initial_physical_state_sha256'],'Initial physics mismatch')
                    key=(task,c['scene_seed']);inp=by_key[key];digest=inp['initial_observation_sha256']
                    require(initial_obs.setdefault(key,digest)==digest,'Cross-arm initial observation mismatch')
                    require(inp['video_language']==v and inp['noise_seed']==seed,'Video/noise intervention mismatch')
                    require(inp['video_instruction']==c['source_instruction' if v=='source' else 'counterfactual_instruction'],'Wrong video text')
                    text=c['source_instruction' if action=='source' else 'counterfactual_instruction']
                    require(inp['action_instruction']==row['policy_instruction']==text,'Wrong action text')
                    m=validate_episode(row)
                    cell_count.update(dict(episodes=1,source_success=int(row['source_goal_ever_success']),cf_success=int(row['counterfactual_goal_ever_success']),selected_success=int(row['selected_goal_success']),any_correct_lift=int(m['any_correct_object_lifted']),all_instruction_lift=int(m['all_instruction_objects_lifted']),full_goal_after_lift=int(m['full_goal_after_any_lift']),strict_placement_after_lift=int(m['correct_placement_after_lift'])))
                for f in ['complete.json','episodes.jsonl','initial_states.json','summary.json']:bind(cell/f)
                cell_results.append(dict(task=task,**cell_count));counts.update(cell_count)
            completed[name]=dict(**counts,tasks=cell_results,process_exit=job)
    require(allow_partial or not pending,'Closed-loop study incomplete: '+str(len(pending))+' arms pending')
    result=dict(format='robotwin_world_language_closed_data_audit_v1',complete=not pending,allow_partial=allow_partial,plan_sha256=sha(plan_path),expected_episodes=1200,validated_episodes=sum(x['episodes'] for x in completed.values()),completed_arms=completed,pending_arms=pending,artifact_sha256=outputs,scope='Terminal closed workers only; cross-input identities, checkpoint/catalog bindings and physical-event consistency. Final source/asset hashes and independent termination-source review remain separate. Aggregate fields are checked against logged evidence; no raw contact traces were recorded for independent replay.')
    out=root/('closed_data_audit_partial.json' if allow_partial else 'closed_data_audit.json');out.write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
    r=audit(args.root,args.allow_partial);print(json.dumps({k:r[k] for k in ['complete','validated_episodes','pending_arms']}))
if __name__=='__main__':main()
