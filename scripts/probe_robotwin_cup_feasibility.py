#!/usr/bin/env python3
"""Screen a replacement CF task using experts only, without modifying its scenes."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import MethodType

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--robotwin-root', type=Path, default=REPO/'third_party/RoboTwin')
    parser.add_argument('--task-config', choices=['demo_clean', 'demo_randomized'], default='demo_clean')
    parser.add_argument('--start-seed', type=int, default=92000000)
    parser.add_argument('--scenes', type=int, default=2)
    parser.add_argument('--relation', choices=['behind', 'front'], default='behind')
    args = parser.parse_args()
    if not 92000000 <= args.start_seed < 92000100 or not 1 <= args.scenes <= 4:
        parser.error('Use the dedicated diagnostic-only seed range and at most four scenes.')
    import numpy as np
    from experiments.robotwin.cup_counterfactual import (
        initialize_geometry, counterfactual_success, play_counterfactual,
        SOURCE_INSTRUCTION, TARGET_INSTRUCTION, FRONT_INSTRUCTION)
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args, _close
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    task, config = _load_robotwin_args(robotwin_root=args.robotwin_root.resolve(),
        task_name='place_empty_cup', task_config=args.task_config, output_root=output)
    config.update(eval_mode=True, need_plan=True, save_data=False, render_freq=0)
    config['data_type'] = dict(config.get('data_type') or {}, rgb=True, qpos=True,
                               actor_segmentation_ids=False, pgc_entity_state=False)
    native_success = type(task).check_success
    results = []
    for seed in range(args.start_seed, args.start_seed + args.scenes):
        initial = None
        for variant in ('source', 'target'):
            task.check_success = MethodType(native_success if variant == 'source' else counterfactual_success, task)
            try:
                task.setup_demo(now_ep_num=0, seed=seed, **deepcopy(config))
                initialize_geometry(task, direction=1 if args.relation == 'behind' else -1)
                observation = task.get_obs()
                state = {'qpos': np.array(observation['joint_action']['vector']),
                         'cup': np.r_[task.cup.get_pose().p, task.cup.get_pose().q],
                         'coaster': np.r_[task.coaster.get_pose().p, task.coaster.get_pose().q]}
                state.update({c: np.array(v['rgb']) for c, v in observation['observation'].items() if 'rgb' in v})
                if initial is None:
                    initial = state
                else:
                    assert initial.keys() == state.keys() and all(np.array_equal(initial[k], state[k]) for k in state), 'Initial observations differ'
                assert not native_success(task) and not counterfactual_success(task), 'Goal already true initially'
                if variant == 'source':
                    task.play_once()
                else:
                    play_counterfactual(task)
                source_ok, target_ok = bool(native_success(task)), bool(counterfactual_success(task))
                row = {'seed': seed, 'variant': variant, 'plan_success': bool(task.plan_success),
                       'source_success': source_ok, 'target_success': target_ok,
                       'success': bool(task.plan_success and (source_ok if variant == 'source' else target_ok)),
                       'initial_observations_equal': True,
                       'cup_point': task.cup.get_functional_point(0, 'pose').p.tolist(),
                       'coaster_point': task.coaster.get_functional_point(0, 'pose').p.tolist()}
                print(json.dumps(row), flush=True)
                results.append(row)
            finally:
                _close(task)
            (output/'results.json').write_text(json.dumps({'complete': False, 'records': results}, indent=2))
    report = {'complete': True, 'task': 'place_empty_cup', 'task_config': args.task_config,
              'relation': args.relation, 'source_instruction': SOURCE_INSTRUCTION,
              'counterfactual_instruction': TARGET_INSTRUCTION if args.relation == 'behind' else FRONT_INSTRUCTION,
              'scope': 'Expert feasibility only. Diagnostic seeds are excluded from training and reported development/test sets.',
              'records': results, 'jointly_feasible_scenes': sum(all(v['success'] for v in results if v['seed']==seed)
                  for seed in range(args.start_seed, args.start_seed+args.scenes))}
    (output/'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
