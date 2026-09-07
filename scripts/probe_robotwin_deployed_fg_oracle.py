#!/usr/bin/env python3
"""Execute recorded expert labels through the policy joint-action interface."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--record', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--robotwin-root', type=Path, required=True)
    args = ap.parse_args()
    import numpy as np
    from experiments.robotwin.eraf_fg_contract import CAMERAS, validate_correction
    from experiments.robotwin.eraf_fg_collection import physical_state, replay_prefix, full_goal
    from experiments.robotwin.eraf_fg_data import file_metadata
    from experiments.robotwin.pgc_data import pair_spec_from_source_task
    from experiments.robotwin.pgc_task_variants import install_pgc_task_contract
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args, _capture_data_type, _close
    record = validate_correction(json.loads(args.record.read_text()))
    if record['replay_split'] != 'train' or not 80000000 <= record['scene_seed'] < 81000000:
        ap.error('Oracle diagnostics are restricted to training scenes')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame_path = Path(record['frame_path'])
    trace_path = frame_path.parent / 'failure_rollout.npz'
    with np.load(trace_path, allow_pickle=False) as arrays:
        trace = dict(initial=arrays['initial'], actions=arrays['actions'],
                     states=dict(zip(arrays['capture_steps'].tolist(), arrays['states'])))
    with np.load(frame_path, allow_pickle=False) as arrays:
        actions = arrays['actions']
        initial_images = {camera: arrays[camera][0] for camera in CAMERAS}
    if len(actions) != record['recorded_action_count'] or actions.shape[1:] != (14,) or not np.isfinite(actions).all():
        raise ValueError('Invalid expert label sequence')
    spec = pair_spec_from_source_task(record['source_task'])
    task, config = _load_robotwin_args(robotwin_root=args.robotwin_root.resolve(),
        task_name=spec.source_task, task_config=record['task_config'], output_root=output)
    install_pgc_task_contract(task, spec)
    config.update(data_type=_capture_data_type(config), eval_mode=True, need_plan=True,
                  save_data=False, render_freq=0)
    report = dict(format='robotwin_deployed_fg_oracle_v1', complete=False, learned_policy_evaluated=False,
        source_task=spec.source_task, scene_seed=record['scene_seed'], parent_record=str(args.record.resolve()),
        frame_metadata=file_metadata(frame_path), original_dense_expert_full_goal_verified=True,
        prefix_actions=record['prefix_action_count'], available_expert_labels=len(actions),
        expert_labels_executed=0, hash_scans=False)

    def save():
        temporary = output / 'results.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(output / 'results.json')

    try:
        task._pgc_active_variant = spec.counterfactual_variant
        task.setup_demo(now_ep_num=0, seed=record['scene_seed'], **deepcopy(config))
        report['policy_step_limit'] = int(task.step_lim)
        report['remaining_policy_steps'] = max(0, int(task.step_lim) - record['prefix_action_count'])
        report['prefix_replay_max_abs'] = replay_prefix(task, spec, trace, record['prefix_action_count'])
        observation = task.get_obs()
        qpos = np.asarray(observation['joint_action']['vector'], dtype=np.float32)
        if not np.array_equal(qpos, actions[0]) or any(
                not np.array_equal(initial_images[c], observation['observation'][c]['rgb']) for c in CAMERAS):
            raise ValueError('Oracle start differs from the recorded training observation')
        report.update(initial_observations_equal=True, selected_goal_initial=full_goal(task, spec),
                      physical_start=physical_state(task).tolist())
        if report['selected_goal_initial']:
            raise ValueError('Selected oracle goal was already complete at the correction start')
        save()
        for action in actions:
            if task.take_action_cnt >= task.step_lim or task.eval_success:
                break
            before = task.take_action_cnt
            task.take_action(action, action_type='qpos')
            if task.take_action_cnt != before + 1:
                raise ValueError('Oracle policy action was not executed exactly once')
            report['expert_labels_executed'] += 1
        report.update(complete=True, selected_goal_final=full_goal(task, spec),
                      source_goal_final=full_goal(task, spec, selected_goal='source'),
                      final_policy_action_count=int(task.take_action_cnt),
                      policy_budget_exhausted=task.take_action_cnt >= task.step_lim,
                      physical_final=physical_state(task).tolist())
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        _close(task)
        save()
    print(json.dumps({k: v for k, v in report.items() if not k.startswith('physical_')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
