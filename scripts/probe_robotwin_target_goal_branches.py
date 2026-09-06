#!/usr/bin/env python3
"""Compare complete source/CF experts from the same recorded training failure.

This is a data-feasibility probe, not a policy success-rate evaluation. It reads
existing original-task failure traces, never independent test rollouts.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('record', 'output', 'robotwin-root'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--prefix-steps', type=int)
    ap.add_argument('--continuation', choices=['release-regrasp', 'held-placement'], default='release-regrasp')
    ap.add_argument('--held-arm', choices=['left', 'right'])
    args = ap.parse_args()
    import numpy as np
    from experiments.robotwin.eraf_fg_contract import CAMERAS, validate_correction, verify_replayed_state
    from experiments.robotwin.eraf_fg_data import file_metadata
    from experiments.robotwin.eraf_fg_collection import (
        physical_state, replay_prefix, record_continuation, replay_continuation, full_goal, continue_to_goal)
    from experiments.robotwin.target_goal_branches import (
        FORMAT, action_difference, require_same_observation, continue_held_placement)
    from experiments.robotwin.pgc_data import pair_spec_from_source_task
    from experiments.robotwin.pgc_task_variants import install_pgc_task_contract
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args, _capture_data_type, _close
    record = validate_correction(json.loads(args.record.read_text()))
    if (record['source_task'] not in {'place_a2b_left', 'blocks_ranking_rgb'} or
            not 80000000 <= record['scene_seed'] < 81000000 or record['replay_split'] != 'train'):
        ap.error('Use original-task training failure records, excluding data-holdout and evaluation scenes.')
    if args.continuation == 'held-placement' and (record['source_task'] != 'place_a2b_left' or not args.held_arm):
        ap.error('Held placement requires the left task and an explicit --held-arm.')
    step = record['prefix_action_count'] if args.prefix_steps is None else args.prefix_steps
    trace_path = Path(record['frame_path']).parent / 'failure_rollout.npz'
    with np.load(trace_path, allow_pickle=False) as data:
        trace = {'initial': data['initial'], 'actions': data['actions'],
                 'states': dict(zip(data['capture_steps'].tolist(), data['states']))}
    if step <= 0 or step not in trace['states'] or step > len(trace['actions']):
        ap.error('Prefix must be a recorded noninitial physical state of this policy rollout.')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {'format': FORMAT, 'complete': False, 'paired_goals_verified': False,
              'source_task': record['source_task'], 'task_config': record['task_config'],
              'scene_seed': record['scene_seed'], 'replay_split': record['replay_split'],
              'parent_record': str(args.record.resolve()), 'trace_metadata': file_metadata(trace_path),
              'source_instruction': record['source_instruction'],
              'counterfactual_instruction': record['counterfactual_instruction'],
              'prefix_steps': step, 'continuation': args.continuation, 'held_arm': args.held_arm,
              'conditions': {}, 'hash_scans': False, 'optimizer_updates': 0}

    def save():
        temp = root / 'results.json.tmp'
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(root / 'results.json')

    save()
    spec = pair_spec_from_source_task(record['source_task'])
    task, config = _load_robotwin_args(robotwin_root=args.robotwin_root.resolve(), task_name=spec.source_task,
        task_config=record['task_config'], output_root=root)
    install_pgc_task_contract(task, spec)
    config.update(data_type=_capture_data_type(config), eval_mode=True, need_plan=True,
                  save_data=False, render_freq=0)
    opened = False

    def setup(language):
        nonlocal opened
        opened = True
        task._pgc_active_variant = spec.source_variant if language == 'source' else spec.counterfactual_variant
        task.setup_demo(now_ep_num=0, seed=record['scene_seed'], **deepcopy(config))
        return replay_prefix(task, spec, trace, step)

    def close():
        nonlocal opened
        if opened:
            try:
                _close(task)
            finally:
                opened = False

    def frame():
        observation = task.get_obs()
        return {'qpos': np.array(observation['joint_action']['vector'], copy=True),
                'images': {c: np.array(observation['observation'][c]['rgb'], copy=True) for c in CAMERAS},
                'grounding': deepcopy(task.pgc_eraf_snapshot())}

    first_state = first_frame = None
    actions = {}
    for language in ('source', 'target'):
        row = {}
        report['conditions'][language] = row
        try:
            error = setup(language)
            state, initial = physical_state(task), frame()
            if first_state is None:
                first_state, first_frame = state, initial
            else:
                error = max(error, verify_replayed_state(first_state, state))
                require_same_observation(first_frame, initial, CAMERAS)
            row['initial_goal'] = full_goal(task, spec, selected_goal=language)
            if row['initial_goal']:
                raise ValueError('The selected branch goal is already complete at the capture.')
            with record_continuation(task) as (controls, frames):
                frames.append(initial)
                if args.continuation == 'held-placement':
                    continue_held_placement(task, spec, selected_goal=language, arm_name=args.held_arm,
                                           initial_object_z=float(trace['initial'][2]))
                else:
                    continue_to_goal(task, spec, selected_goal=language)
            row.update(plan_success=bool(task.plan_success),
                       full_goal_success=full_goal(task, spec, selected_goal=language), frames=len(frames))
            close()
            if not row['plan_success'] or not row['full_goal_success'] or len(frames) < 32:
                raise ValueError('Complete goal continuation was not obtained.')
            for _ in range(2):
                error = max(error, setup(language))
                require_same_observation(initial, frame(), CAMERAS)
                error = max(error, replay_continuation(task, controls))
                if not full_goal(task, spec, selected_goal=language):
                    raise ValueError('Complete goal failed physical control replay.')
                close()
            arrays = {'actions': np.stack([f['qpos'] for f in frames]).astype(np.float32)}
            arrays.update({c: np.stack([f['images'][c] for f in frames]) for c in CAMERAS})
            arrays.update({'grounding/' + k: np.stack([f['grounding'][k] for f in frames])
                           for k in frames[0]['grounding']})
            path, control_path = root / (language + '.npz'), root / (language + '_controls.pkl')
            np.savez_compressed(path, **arrays)
            with control_path.open('xb') as stream:
                pickle.dump(controls, stream, protocol=5)
            row.update(verified_replays=2, replay_state_max_abs=error,
                       replay_observations_equal=True, frame_path=str(path), control_path=str(control_path),
                       frame_metadata=file_metadata(path), control_metadata=file_metadata(control_path))
            actions[language] = arrays['actions']
        except Exception as error:
            row['error'] = repr(error)
        finally:
            close()
            save()
        print('[target-goal-branch]', language, json.dumps(row), flush=True)
    report['paired_goals_verified'] = set(actions) == {'source', 'target'}
    if report['paired_goals_verified']:
        report.update(initial_observations_equal=True, **action_difference(actions['source'], actions['target']))
    report['complete'] = True
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
