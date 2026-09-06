#!/usr/bin/env python3
"""Test two complete goals from one real policy state while it holds the cup."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--record', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--robotwin-root', type=Path, required=True)
    ap.add_argument('--prefix-steps', type=int, default=120)
    args = ap.parse_args()
    import numpy as np
    from experiments.robotwin.cup_full_goal import validate_cup_correction
    from experiments.robotwin.eraf_fg_collection import (
        physical_state, replay_prefix, record_continuation, replay_continuation, full_goal)
    from experiments.robotwin.eraf_fg_contract import CAMERAS, verify_replayed_state
    from experiments.robotwin.pgc_data import pair_spec_from_source_task
    from experiments.robotwin.pgc_task_variants import install_pgc_task_contract
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args, _capture_data_type, _close
    record = validate_cup_correction(json.loads(args.record.read_text()))
    with np.load(record['rollout_path'], allow_pickle=False) as data:
        trace = {'initial': data['initial'], 'actions': data['actions'],
                 'states': dict(zip(data['capture_steps'].tolist(), data['states']))}
    if args.prefix_steps not in trace['states']: ap.error('Prefix absent from the recorded real policy rollout')
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    spec = pair_spec_from_source_task('place_empty_cup')
    task, config = _load_robotwin_args(robotwin_root=args.robotwin_root.resolve(), task_name=spec.source_task,
        task_config=record['task_config'], output_root=args.output)
    from envs.utils import ArmTag
    install_pgc_task_contract(task, spec)
    config.update(data_type=_capture_data_type(config), eval_mode=True, need_plan=True,
                  save_data=False, render_freq=0)
    report = {'format': 'robotwin_cup_held_goal_branch_probe_v1', 'complete': False,
        'parent_record': str(args.record.resolve()), 'scene_seed': record['scene_seed'],
        'task_config': record['task_config'], 'replay_split': record['replay_split'],
        'prefix_steps': args.prefix_steps, 'conditions': {}, 'hash_scans': False, 'optimizer_updates': 0}
    opened = False

    def save():
        (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')

    def setup(language):
        nonlocal opened
        opened = True
        task._pgc_active_variant = spec.source_variant if language == 'source' else spec.counterfactual_variant
        task.setup_demo(now_ep_num=0, seed=record['scene_seed'], is_test=False, **deepcopy(config))
        return replay_prefix(task, spec, trace, args.prefix_steps)

    def close():
        nonlocal opened
        if opened:
            try: _close(task)
            finally: opened = False

    def frame():
        obs = task.get_obs()
        return {'qpos': np.array(obs['joint_action']['vector'], copy=True),
                'images': {c: np.array(obs['observation'][c]['rgb'], copy=True) for c in CAMERAS},
                'grounding': deepcopy(task.pgc_eraf_snapshot())}

    initial_frame = initial_state = None
    actions = {}
    save()
    for language in ('source', 'target'):
        row = {}; report['conditions'][language] = row
        try:
            error = setup(language)
            state, first = physical_state(task), frame()
            if initial_frame is None: initial_frame, initial_state = first, state
            else:
                verify_replayed_state(initial_state, state)
                if (not np.array_equal(first['qpos'], initial_frame['qpos']) or
                        any(not np.array_equal(first['images'][c], initial_frame['images'][c]) for c in CAMERAS)):
                    raise ValueError('Goal branches changed the real policy observation')
            row.update(prefix_replay_error=error, initial_goal=full_goal(task, spec, selected_goal=language),
                       cup_height_above_initial=float(state[2]-trace['initial'][2]),
                       initial_arm=task._cup_cf_initial_arm)
            if row['cup_height_above_initial'] < .03:
                raise ValueError('The selected state does not visibly hold the cup above the table')
            task.need_plan = True; task.plan_success = True
            arm = ArmTag(task._cup_cf_initial_arm)
            target = np.array(task.coaster.get_functional_point(0), dtype=float, copy=True)
            if language == 'target':
                target[1] -= .13; target[2] = task._cup_cf_floor_z
            with record_continuation(task) as (controls, frames):
                frames.append(first)
                # Continue carrying the already grasped cup; never release/regrasp first.
                task.move(task.place_actor(task.cup, arm, target_pose=target,
                                          functional_point_id=0, pre_dis=.05))
                task.move(task.move_by_displacement(arm, z=.05, move_axis='arm'))
                task.delay(30)
            okay = bool(task.plan_success and full_goal(task, spec, selected_goal=language))
            row.update(plan_success=bool(task.plan_success), full_goal_success=okay, frames=len(frames))
            close()
            if not okay or len(frames) < 32: continue
            for repeat in range(2):
                error = max(error, setup(language), replay_continuation(task, controls))
                if not full_goal(task, spec, selected_goal=language):
                    raise ValueError('Complete goal branch failed control replay')
                close()
            row.update(verified_replays=2, replay_state_max_abs=error, initial_observations_equal=True)
            arrays = {'actions': np.stack([f['qpos'] for f in frames]).astype(np.float32)}
            arrays.update({c: np.stack([f['images'][c] for f in frames]) for c in CAMERAS})
            arrays.update({'grounding/'+k: np.stack([f['grounding'][k] for f in frames])
                           for k in frames[0]['grounding']})
            path = args.output/(language+'.npz'); np.savez_compressed(path, **arrays)
            controls_path = args.output/(language+'_controls.pkl')
            with controls_path.open('xb') as f: pickle.dump(controls, f, protocol=5)
            row.update(frame_path=str(path), control_path=str(controls_path))
            actions[language] = arrays['actions']
        except Exception as error:
            row['error'] = repr(error)
        finally:
            close(); save()
        print('[goal-branch]', language, json.dumps(row), flush=True)
    report['both_goals_verified'] = set(actions) == {'source', 'target'}
    if report['both_goals_verified']:
        report['action_difference_rmse24'] = float(np.sqrt(np.mean((actions['source'][:24]-actions['target'][:24])**2)))
    report['complete'] = True; save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
