#!/usr/bin/env python3
"""Fixed source-expert-selected cup catalogs and actual paired policy rollouts."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]
from experiments.robotwin.cup_evaluation import (
    DEPLOYMENT, initial_snapshot, require_same_initial, summarize_records)
from experiments.robotwin.cup_counterfactual import (
    SOURCE_INSTRUCTION, FRONT_INSTRUCTION, initialize_geometry,
    counterfactual_success, play_counterfactual)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def setup(task, config, seed, episode=0):
    task.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **deepcopy(config))
    initialize_geometry(task, direction=-1)


def state_record(task):
    return {'step': int(task.take_action_cnt),
            'cup': list(map(float, task.cup.get_functional_point(0, 'pose').p)),
            'coaster': list(map(float, task.coaster.get_functional_point(0, 'pose').p)),
            'left_open': bool(task.is_left_gripper_open()),
            'right_open': bool(task.is_right_gripper_open())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['catalog', 'worker', 'summarize'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--robotwin-root', type=Path,
                    default=Path('/root/gpufree-data/LF-FastWAM/third_party/RoboTwin'))
    ap.add_argument('--domain', choices=['demo_clean', 'demo_randomized'], default='demo_clean')
    ap.add_argument('--split', choices=['dev', 'test'], default='dev')
    ap.add_argument('--start-seed', type=int)
    ap.add_argument('--episodes', type=int, default=6)
    ap.add_argument('--max-attempts', type=int, default=24)
    ap.add_argument('--catalog', type=Path)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--policy-kind', choices=['legacy', 'repair'], default='legacy')
    ap.add_argument('--episode-start', type=int, default=0)
    ap.add_argument('--episode-count', type=int)
    ap.add_argument('--workers', type=Path, nargs='+')
    args = ap.parse_args()
    for key in ('output', 'robotwin_root', 'catalog', 'manifest', 'checkpoint'):
        p = getattr(args, key)
        if p is not None: setattr(args, key, p.expanduser().resolve())
    if args.mode != 'catalog' and not args.catalog: ap.error('--catalog is required')
    if args.mode == 'worker' and not (args.checkpoint and args.manifest):
        ap.error('Worker needs --checkpoint and --manifest')
    catalog = json.loads(args.catalog.read_text()) if args.catalog else None
    if catalog and (not catalog['complete'] or catalog['relation'] != 'front'):
        ap.error('Expected a complete front relation catalog')
    if args.mode == 'summarize':
        if not args.workers: ap.error('Specify completed worker JSON files')
        reports = [json.loads(p.read_text()) for p in args.workers]
        if any(not r['complete'] or r['catalog'] != str(args.catalog) for r in reports):
            raise ValueError('Incomplete or mismatched workers')
        identities = {(r['checkpoint'], r['policy_kind']) for r in reports}
        if len(identities) != 1: raise ValueError('Mixed models')
        result = summarize_records(catalog, [r for w in reports for r in w['records']])
        result.update(checkpoint=reports[0]['checkpoint'], policy_kind=reports[0]['policy_kind'],
                      catalog=str(args.catalog), split=catalog['split'], task='place_empty_cup',
                      relation='front', domain=catalog['domain'])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result); print(json.dumps(result), flush=True)
        return
    import numpy as np
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args, _close
    args.output.mkdir(parents=True, exist_ok=False)
    domain = catalog['domain'] if catalog else args.domain
    task, config = _load_robotwin_args(robotwin_root=args.robotwin_root,
        task_name='place_empty_cup', task_config=domain, output_root=args.output)
    config.update(eval_mode=True, need_plan=True, save_data=False, render_freq=0,
                  eval_video_log=False, eval_video_save_dir=None)
    config['data_type'] = dict(config.get('data_type') or {}, rgb=True, qpos=True,
                               actor_segmentation_ids=False, pgc_entity_state=False)
    native_success = type(task).check_success
    if args.mode == 'catalog':
        lower = 93000000 if args.split == 'dev' else 94000000
        start = args.start_seed if args.start_seed is not None else lower
        if not (lower <= start < lower+1000000 and 1 <= args.episodes <= 20
                and args.episodes <= args.max_attempts <= 100
                and start+args.max_attempts <= lower+1000000):
            ap.error('Invalid reserved split range or catalog size')
        report = {'format': 'robotwin_cup_front_catalog_v1', 'complete': False,
                  'split': args.split, 'domain': domain, 'relation': 'front',
                  'selection': 'First native-expert-solvable scenes; CF expert results never filter scenes.',
                  'source_instruction': SOURCE_INSTRUCTION, 'target_instruction': FRONT_INSTRUCTION,
                  'start_seed': start, 'episodes': [], 'screening': []}
        for seed in range(start, start+args.max_attempts):
            episode = len(report['episodes']); event = {'scene_seed': seed}
            task.check_success = MethodType(native_success, task)
            try:
                setup(task, config, seed, episode)
                snap = initial_snapshot(task, task.get_obs())
                if native_success(task) or counterfactual_success(task):
                    raise ValueError('Goal already true initially')
                task.play_once()
                event['source_expert_success'] = bool(task.plan_success and native_success(task))
            finally:
                _close(task)
            report['screening'].append(event)
            if event['source_expert_success']:
                snapshot_path = args.output/f'initial_{episode:03d}.npz'
                np.savez_compressed(snapshot_path, **snap)
                task.check_success = MethodType(native_success, task)
                try:
                    setup(task, config, seed, episode)
                    task.check_success = MethodType(counterfactual_success, task)
                    require_same_initial(snap, initial_snapshot(task, task.get_obs()))
                    play_counterfactual(task)
                    cf_ok = bool(task.plan_success and counterfactual_success(task))
                finally:
                    _close(task)
                report['episodes'].append({'episode_index': episode, 'scene_seed': seed,
                    'initial_snapshot': str(snapshot_path), 'cf_expert_success': cf_ok,
                    'source_expert_success': True})
                print(f'[catalog] accepted={episode+1}/{args.episodes} seed={seed} cf_expert={cf_ok}', flush=True)
            write_json(args.output/'catalog.json', report)
            if len(report['episodes']) == args.episodes: break
        report['complete'] = len(report['episodes']) == args.episodes
        write_json(args.output/'catalog.json', report)
        if not report['complete']: raise RuntimeError('Source-expert catalog incomplete')
        return
    manifest = json.loads(args.manifest.read_text())
    if args.policy_kind == 'repair':
        from experiments.robotwin.eraf_fg_bridge import load_policy
        policy = load_policy(args.checkpoint, manifest)
        policy.model.policy_guard_enabled = False
    else:
        from scripts.train_robotwin_cf_decision_adapter import load_policy
        policy = load_policy(SimpleNamespace(checkpoint=str(args.checkpoint), seed=42), manifest)
    policy.task_name, policy.task_config = 'place_empty_cup', domain
    if policy.closed_loop_capture_dir is not None or policy.initial_observation_audit is not None:
        raise ValueError('Disable unrelated historical capture/audit environment variables')
    episodes = catalog['episodes'][args.episode_start:]
    if args.episode_count is not None: episodes = episodes[:args.episode_count]
    if not episodes: raise ValueError('Empty worker shard')
    report = {'format': 'robotwin_cup_front_policy_v1', 'complete': False,
              'checkpoint': str(args.checkpoint), 'policy_kind': args.policy_kind,
              'catalog': str(args.catalog), 'records': []}
    for item in episodes:
        with np.load(item['initial_snapshot']) as loaded:
            expected = {k: loaded[k] for k in loaded.files}
        for condition in ('source', 'target'):
            reached = {'source': False, 'target': False}
            def check(env):
                reached['source'] |= bool(native_success(env))
                reached['target'] |= bool(counterfactual_success(env))
                return reached[condition]
            task.check_success = MethodType(native_success, task)
            trace, actions = [], []
            original_take = task.take_action
            try:
                setup(task, config, item['scene_seed'], item['episode_index'])
                task.check_success = MethodType(check, task)
                obs = task.get_obs()
                require_same_initial(expected, initial_snapshot(task, obs))
                if native_success(task) or counterfactual_success(task):
                    raise ValueError('Goal already true initially')
                task.set_instruction(catalog['source_instruction'] if condition=='source' else catalog['target_instruction'])
                policy.reset()
                def take(action, *a, **kw):
                    actions.append(np.array(action, copy=True)); return original_take(action, *a, **kw)
                task.take_action = take
                stem = f"episode_{item['episode_index']:03d}_{condition}"
                frames = args.output/(stem+'_frames'); frames.mkdir()
                from PIL import Image
                while task.take_action_cnt < task.step_lim:
                    obs = task.get_obs() if policy.should_request_observation() else None
                    if obs is not None:
                        trace.append(state_record(task))
                        Image.fromarray(obs['observation']['head_camera']['rgb']).save(
                            frames/f'{int(task.take_action_cnt):04d}.jpg', quality=85)
                    before = int(task.take_action_cnt)
                    policy.step(task, obs)
                    if task.take_action_cnt <= before: raise RuntimeError('Policy did not execute an action')
                    if task.eval_success: break
                check(task)
                trace.append(state_record(task))
                Image.fromarray(task.get_obs()['observation']['head_camera']['rgb']).save(
                    frames/'final.jpg', quality=90)
                np.save(args.output/(stem+'_actions.npy'), np.asarray(actions, dtype=np.float32))
                write_json(args.output/(stem+'_trace.json'), trace)
                row = {'scene_seed': item['scene_seed'], 'episode_index': item['episode_index'],
                       'condition': condition, 'policy_instruction': task.get_instruction(),
                       'success': bool(reached[condition]), 'source_success': reached['source'],
                       'target_success': reached['target'], 'steps': int(task.take_action_cnt),
                       'step_limit': int(task.step_lim), 'initial_observations_equal': True,
                       'deployment': DEPLOYMENT, 'final_state': trace[-1]}
                report['records'].append(row)
                write_json(args.output/'worker.json', report)
                print(json.dumps(row), flush=True)
            finally:
                task.take_action = original_take
                _close(task)
    report['complete'] = True
    write_json(args.output/'worker.json', report)


if __name__ == '__main__':
    main()
