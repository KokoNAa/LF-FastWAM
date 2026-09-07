#!/usr/bin/env python3
"""Screen fixed scenes with both experts before evaluating any learned policy."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--start-seed', type=int, required=True)
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--max-attempts', type=int, default=25)
    ap.add_argument('--split', choices=['dev', 'test'], default='dev')
    ap.add_argument('--exclude-records', action='append', default=[],
                    help='Training JSON manifest or prior catalog JSONL; repeat for every source. Required for test.')
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--deadline', required=True)
    ap.add_argument('--robotwin-root', type=Path, default=Path('/root/gpufree-data/LF-FastWAM/third_party/RoboTwin'))
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None:
        ap.error('Declare an absolute deadline.')
    if args.episodes < 1 or args.max_attempts < args.episodes:
        ap.error('Invalid episode or attempt count')
    from experiments.robotwin.catalog_protocol import validate_namespace, excluded_scene_records
    validate_namespace(args.split, args.start_seed, args.max_attempts)
    excluded, exclusion_receipts = excluded_scene_records(args.exclude_records, split=args.split)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
    import numpy as np
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args, _close
    from scripts.eval_robotwin_eraf_fg import official_module
    from experiments.robotwin.pgc_data import pair_spec_from_source_task, array_sha256
    from experiments.robotwin.pgc_task_variants import install_pgc_task_contract, play_variant, check_variant
    from experiments.robotwin.eraf_fg_collection import physical_state
    from experiments.robotwin.language_interventions import EPISODE_FORMAT
    spec = pair_spec_from_source_task(args.task)
    task, options = _load_robotwin_args(robotwin_root=args.robotwin_root, task_name=args.task,
                                        task_config='demo_clean', output_root=args.output)
    install_pgc_task_contract(task, spec)
    options.update(eval_mode=True, render_freq=0, need_plan=True, save_data=False)
    official = official_module(args.robotwin_root)
    root = args.output / args.task / 'demo_clean/correct'
    root.mkdir(parents=True, exist_ok=False)
    count = 0
    with (root/'episodes.jsonl').open('x', buffering=1) as journal, (root/'expert_screening.jsonl').open('x', buffering=1) as events:
        for seed in range(args.start_seed, args.start_seed + args.max_attempts):
            if time.time() >= cutoff.timestamp():
                raise TimeoutError('Authorized work cutoff reached')
            if seed in excluded:
                events.write(json.dumps({'seed': seed, 'accepted': False, 'error': 'Excluded training/development seed'})+'\n')
                continue
            states, audit, source_info = [], [], None
            scene_open = False
            try:
                for variant in (spec.source_variant, spec.counterfactual_variant):
                    task._pgc_active_variant = None
                    scene_open = True
                    task.setup_demo(now_ep_num=count, seed=seed, is_test=True, **deepcopy(options))
                    if any(check_variant(task, spec, v) for v in (spec.source_variant, spec.counterfactual_variant)):
                        raise ValueError('An initial goal is already satisfied')
                    states.append(physical_state(task).copy())
                    if len(states) == 2 and not np.array_equal(states[0], states[1]):
                        raise ValueError('Source and CF expert initial states differ')
                    task._pgc_active_variant = variant
                    info = task.play_once() if variant == spec.source_variant else play_variant(task, spec, variant)
                    if not task.plan_success or not check_variant(task, spec, variant):
                        raise ValueError(f'{variant} expert failed full goal')
                    opposite = spec.counterfactual_variant if variant == spec.source_variant else spec.source_variant
                    if check_variant(task, spec, opposite):
                        raise ValueError('The two goals are not exclusive at the endpoint')
                    audit.append({'variant': variant, 'full_goal_success': True, 'opposite_goal_success': False})
                    if variant == spec.source_variant:
                        source_info = deepcopy(info['info'])
                    _close(task)
                    scene_open = False
                source = official._deterministic_instruction(task_name=args.task, episode_info=source_info,
                                                             instruction_type='unseen', scene_seed=seed)
                row = dict(format=EPISODE_FORMAT, pair_id=spec.pair_id, source_task=args.task,
                    counterfactual_task=spec.counterfactual_task, task_config='demo_clean', condition='correct',
                    episode_index=count, scene_seed=seed, instruction_type='unseen', source_instruction=source,
                    counterfactual_instruction=spec.counterfactual_instruction, policy_instruction=source,
                    instruction_goal='source', selected_goal='source', initial_source_goal_success=False,
                    initial_counterfactual_goal_success=False, initial_physical_state_sha256=array_sha256(states[0]),
                    catalog_only=True, selection='both expert goals feasible; no learned policy selection',
                    catalog_split=args.split,
                    expert_checks=audit, matched_expert_initial_states=True)
                journal.write(json.dumps(row)+'\n'); count += 1
                events.write(json.dumps({'seed':seed,'accepted':True,'expert_checks':audit})+'\n')
                print(f'[catalog] {args.task} {count}/{args.episodes} seed={seed}',flush=True)
            except Exception as error:
                events.write(json.dumps({'seed':seed,'accepted':False,'error':repr(error),'expert_checks':audit})+'\n')
            finally:
                if scene_open:
                    _close(task)
            if count == args.episodes:
                break
    report = dict(complete=count == args.episodes, episodes=count, required_episodes=args.episodes,
                  selection='matched physical starts; both expert endpoints satisfy their selected goal and reject the opposite; no policy selection',
                  control_replay_verified=False, start_seed=args.start_seed, catalog_split=args.split,
                  max_attempts=args.max_attempts, exclusion_records=exclusion_receipts,
                  excluded_seed_count=len(excluded),
                  code_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    (root/'catalog_complete.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['complete']:
        raise RuntimeError(f'Incomplete expert-validated catalog: {count}/{args.episodes}')


if __name__ == '__main__':
    main()
