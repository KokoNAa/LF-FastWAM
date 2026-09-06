#!/usr/bin/env python3
"""Compare expert arm choices from the same saved late failed cup states."""
from pathlib import Path
import argparse
from copy import deepcopy
import json
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('collection','output','robotwin-root'):ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args()
    args.output=args.output.resolve();args.output.mkdir(exist_ok=False,parents=True)
    collection=json.loads(args.collection.resolve().read_text())
    if not collection['complete'] or len(collection['records'])!=1:raise ValueError('Expected one completed smoke scene')
    row=collection['records'][0]
    from experiments.robotwin.pgc_data import pair_spec_from_source_task
    from experiments.robotwin.pgc_task_variants import install_pgc_task_contract
    from experiments.robotwin.eraf_fg_collection import replay_prefix,continue_to_goal,full_goal
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args,_close
    spec=pair_spec_from_source_task('place_empty_cup')
    task,config=_load_robotwin_args(robotwin_root=args.robotwin_root.resolve(),task_name='place_empty_cup',
                                  task_config=row['task_config'],output_root=args.output)
    install_pgc_task_contract(task,spec)
    config.update(eval_mode=True,need_plan=True,save_data=False,render_freq=0)
    with np.load(row['rollout_path']) as a:
        trace={'initial':a['initial'],'actions':a['actions'],
               'states':{int(k):v for k,v in zip(a['capture_steps'],a['states'],strict=True)}}
    step=max(trace['states'])
    reports=[]
    def snapshot():
        rotation=task.cup.get_pose().to_transformation_matrix()[:3,:3]
        return {'plan_success':bool(task.plan_success),'full_goal':full_goal(task,spec),
                'cup_point':task.cup.get_functional_point(0,'pose').p.tolist(),
                'coaster_point':task.coaster.get_functional_point(0,'pose').p.tolist(),
                'upright':float((rotation@task._cup_cf_local_up)[2]),
                'left_open':bool(task.is_left_gripper_open()),'right_open':bool(task.is_right_gripper_open())}
    for strategy in ('current','initial'):
        phases=[];original=task.move
        try:
            task._pgc_active_variant=spec.counterfactual_variant
            task.setup_demo(now_ep_num=0,seed=row['scene_seed'],is_test=False,**deepcopy(config))
            error=replay_prefix(task,spec,trace,step)
            initial=snapshot();task._cup_cf_recovery_arm_strategy=strategy
            def move(*a,**kw):
                result=original(*a,**kw);phases.append(snapshot());return result
            task.move=move
            continue_to_goal(task,spec)
            result={'scene_seed':row['scene_seed'],'strategy':strategy,'prefix_steps':step,
                    'replay_state_max_abs':error,'initial':initial,'phases':phases,'final':snapshot()}
            reports.append(result);print(json.dumps(result),flush=True)
        finally:
            task.move=original;_close(task)
            (args.output/'results.json').write_text(json.dumps({'complete':False,'records':reports},indent=2))
    (args.output/'results.json').write_text(json.dumps({'complete':True,'records':reports},indent=2))


if __name__=='__main__':main()
