#!/usr/bin/env python3
"""Collect cup FG from real failed policy prefixes with direct physical replays."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('output','manifest','checkpoint','robotwin-root'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--start-seed',type=int,default=82000000)
    ap.add_argument('--scenes',type=int,default=2)
    ap.add_argument('--task-config',choices=['demo_clean','demo_randomized'],default='demo_clean')
    ap.add_argument('--split',choices=['train','replay_holdout'],default='train')
    ap.add_argument('--candidates',type=int,default=4)
    ap.add_argument('--candidate-order',choices=['late_first','early_first'],default='late_first')
    args=ap.parse_args()
    lower=82000000 if args.split=='train' else 83000000
    if not (lower<=args.start_seed and args.start_seed+args.scenes<=lower+1000000
            and 1<=args.scenes<=30 and 1<=args.candidates<=20):
        ap.error('Invalid split seed range or bounded scene/candidate count')
    for key in ('output','manifest','checkpoint','robotwin_root'):
        setattr(args,key,getattr(args,key).resolve())
    import numpy as np
    from experiments.robotwin.cup_counterfactual import SOURCE_INSTRUCTION,FRONT_INSTRUCTION
    from experiments.robotwin.cup_full_goal import validate_cup_correction
    from experiments.robotwin.eraf_fg_collection import (
        physical_state,run_failure_rollout,replay_prefix,record_continuation,
        continue_to_goal,replay_continuation,full_goal)
    from experiments.robotwin.eraf_fg_contract import candidate_replans,verify_replayed_state,CAMERAS
    from experiments.robotwin.pgc_data import (pair_spec_from_source_task,
        ROBOTWIN_ERAF_PAIR_SPECS,ROBOTWIN_REPLACEMENT_PAIR_SPECS)
    from experiments.robotwin.pgc_task_variants import install_pgc_task_contract,play_variant
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args,_capture_data_type,_close
    from scripts.train_robotwin_cf_decision_adapter import load_policy
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text())
    tasks={s.pair_id:s.source_task for s in (*ROBOTWIN_ERAF_PAIR_SPECS,*ROBOTWIN_REPLACEMENT_PAIR_SPECS)}
    excluded={(r.get('source_task',tasks[r['pair_id']]),r['task_config'],int(r['scene_seed']))
              for r in manifest['states']}
    spec=pair_spec_from_source_task('place_empty_cup')
    task,config=_load_robotwin_args(robotwin_root=args.robotwin_root,task_name='place_empty_cup',
                                  task_config=args.task_config,output_root=args.output)
    install_pgc_task_contract(task,spec)
    config.update(data_type=_capture_data_type(config),eval_mode=True,need_plan=True,
                  save_data=False,render_freq=0)
    report={'format':'robotwin_cup_full_goal_direct_replay_v1','complete':False,
            'checkpoint':str(args.checkpoint),'split':args.split,'task_config':args.task_config,
            'requested_scenes':args.scenes,'records':[],'attempts':[],'hash_scans':False}
    def save():
        (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    opened=False
    def setup(seed):
        nonlocal opened
        opened=True
        task._pgc_active_variant=spec.counterfactual_variant
        task.setup_demo(now_ep_num=0,seed=seed,is_test=False,**deepcopy(config))
    def close():
        nonlocal opened
        if opened:
            try:_close(task)
            finally:opened=False
    policy=None
    save()
    for seed in range(args.start_seed,args.start_seed+args.scenes):
        if ('place_empty_cup',args.task_config,seed) in excluded:raise ValueError('Scene already used')
        folder=args.output/f'scene_{seed}';folder.mkdir()
        attempt={'scene_seed':seed,'candidates':[]};report['attempts'].append(attempt)
        try:
            setup(seed);initial=physical_state(task)
            play_variant(task,spec,spec.counterfactual_variant)
            reachable=bool(task.plan_success and full_goal(task,spec));close()
            attempt['initial_expert_success']=reachable
            if not reachable:continue
            if policy is None:
                policy=load_policy(SimpleNamespace(checkpoint=str(args.checkpoint),seed=42),manifest)
                policy.task_name,policy.task_config='place_empty_cup',args.task_config
            setup(seed);verify_replayed_state(initial,physical_state(task))
            trace=run_failure_rollout(task,policy,spec,FRONT_INSTRUCTION)
            np.savez_compressed(folder/'failure_rollout.npz',initial=trace['initial'],actions=trace['actions'],
                capture_steps=np.array(list(trace['states'])),states=np.stack(list(trace['states'].values())))
            attempt['failure_audit']=trace['audit'];close()
            if trace['audit']['target']:continue
            candidates=candidate_replans(trace['states'],limit=args.candidates)
            if args.candidate_order=='early_first':candidates.sort()
            for step in candidates:
                candidate={'prefix_steps':step};attempt['candidates'].append(candidate)
                try:
                    setup(seed);error=replay_prefix(task,spec,trace,step)
                    with record_continuation(task) as (controls,frames):continue_to_goal(task,spec)
                    okay=bool(task.plan_success and full_goal(task,spec) and len(frames)>=12)
                    candidate.update(plan_success=bool(task.plan_success),full_goal=full_goal(task,spec),frames=len(frames))
                    close()
                    if not okay:continue
                    for repeat in range(2):
                        setup(seed);error=max(error,replay_prefix(task,spec,trace,step))
                        error=max(error,replay_continuation(task,controls))
                        verified=full_goal(task,spec);close()
                        if not verified:raise ValueError('Control replay did not finish the full goal')
                    arrays={'actions':np.stack([f['qpos'] for f in frames]).astype(np.float32)}
                    arrays.update({c:np.stack([f['images'][c] for f in frames]) for c in CAMERAS})
                    arrays.update({'grounding/'+k:np.stack([f['grounding'][k] for f in frames])
                                   for k in frames[0]['grounding']})
                    np.savez_compressed(folder/'correction.npz',**arrays)
                    with (folder/'controls.pkl').open('xb') as f:pickle.dump(controls,f,protocol=5)
                    row={'source_task':'place_empty_cup','pair_id':spec.pair_id,'scene_seed':seed,
                         'task_config':args.task_config,'replay_split':args.split,'fg_correction':True,
                         'capture_action_index':step,'prefix_action_count':step,
                         'source_goal_ever_success':trace['audit']['source'],'counterfactual_goal_ever_success':False,
                         'full_goal_verified':True,'counterfactual_goal_final_success':True,
                         'both_grippers_open_final':True,'verified_replay_count':2,
                         'recorded_action_count':len(frames),'replay_state_max_abs':error,'state_atol':1e-7,
                         'source_instruction':SOURCE_INSTRUCTION,'counterfactual_instruction':FRONT_INSTRUCTION,
                         'frame_path':str(folder/'correction.npz'),'control_path':str(folder/'controls.pkl'),
                         'rollout_path':str(folder/'failure_rollout.npz'),'rollout_checkpoint':str(args.checkpoint),
                         'verification_binding':'direct_physics_replay','image_color_space':'RGB'}
                    if not 0<=error<=1e-7:raise ValueError('Replay drift')
                    row=validate_cup_correction(row)
                    (folder/'record.json').write_text(json.dumps(row,indent=2)+'\n')
                    report['records'].append(row)
                    print('[verified]',seed,'prefix',step,'frames',len(frames),flush=True)
                    break
                except Exception as error:
                    candidate['error']=repr(error)
                    print('[candidate-failed]',seed,step,repr(error),flush=True)
                finally:
                    close();save()
        except Exception as error:
            attempt['error']=repr(error)
            print('[scene-failed]',seed,repr(error),flush=True)
        finally:
            close();save()
    report['complete']=True
    report['accepted_scenes']=len(report['records'])
    report['collection_target_met']=len(report['records'])==args.scenes;save()
    print(json.dumps({'complete':True,'accepted_scenes':len(report['records']),
                      'attempted_scenes':args.scenes}),flush=True)


if __name__=='__main__':main()
