#!/usr/bin/env python3
"""Evaluate explicit repair checkpoints through the existing RoboTwin CIS loop.

This bypasses only model construction. Goal checks, matched seeds, prompts,
rollout termination, observation requests and episode results use the existing
official simulation loop. It does not modify the legacy policy symlink.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def official_module(robotwin_root):
    path = REPO / 'third_party/RoboTwin/script/eval_policy.py'
    spec = importlib.util.spec_from_file_location('eraf_fg_official_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Keep the versioned CIS loop, but resolve camera/video assets from the
    # actual simulator installation, just like _load_robotwin_args below.
    module.parent_directory = str(Path(robotwin_root) / 'script')
    return module


def install_memory_reset(policy):
    """Expose the offline training history condition as an explicit diagnostic."""
    original = policy._infer_action_chunk

    def infer_without_history(*args, **kwargs):
        policy.policy_guard_state = None
        return original(*args, **kwargs)

    policy._infer_action_chunk = infer_without_history


def file_metadata(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['catalog', 'worker', 'summarize'])
    ap.add_argument('--output', required=True)
    ap.add_argument('--robotwin-root', default='/root/gpufree-data/LF-FastWAM/third_party/RoboTwin')
    ap.add_argument('--interventions', default=str(REPO / 'configs/eval/robotwin_cis_v939_four_tasks.json'))
    ap.add_argument('--manifest')
    ap.add_argument('--checkpoint')
    ap.add_argument('--catalog-root')
    ap.add_argument('--tasks', nargs='+', default=['place_a2b_left', 'place_a2b_right', 'place_burger_fries', 'stack_blocks_two', 'blocks_ranking_rgb'])
    ap.add_argument('--conditions', nargs='+', choices=['correct', 'counterfactual'], default=['correct', 'counterfactual'])
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--start-seed', type=int)
    ap.add_argument('--max-attempts', type=int, default=300)
    ap.add_argument('--eraf', choices=['on', 'off'], default='on')
    ap.add_argument('--policy-kind', choices=['repair', 'legacy'], default='repair')
    ap.add_argument('--memory-mode', choices=['carry', 'reset'], default='carry')
    ap.add_argument('--videos', action='store_true')
    ap.add_argument('--skip-file-hashes', action='store_true',
                    help='Bind checkpoint/catalog file metadata without repeated full-file scans.')
    args = ap.parse_args()
    if args.mode == 'worker' and args.memory_mode == 'reset' and (args.eraf != 'on' or args.policy_kind != 'repair'):
        ap.error('Memory reset requires an ERAF-on repair checkpoint.')
    for key in ('output', 'robotwin_root', 'interventions', 'manifest', 'checkpoint', 'catalog_root'):
        value = getattr(args, key)
        if value:
            setattr(args, key, str(Path(value).resolve()))
    root = Path(args.output)
    if args.mode in ('worker', 'summarize') and not all((args.checkpoint, args.catalog_root)):
        ap.error('Evaluation requires checkpoint and catalog root.')
    if args.mode == 'summarize':
        from experiments.robotwin.eraf_fg_bridge import file_sha256
        cells = []
        checkpoint_hash = None if args.skip_file_hashes else file_sha256(args.checkpoint)
        checkpoint_metadata = file_metadata(args.checkpoint) if args.skip_file_hashes else None
        signatures = []
        for task in args.tasks:
            initial_by_condition = []
            for condition in args.conditions:
                path = root / task / 'demo_clean' / condition
                report = json.loads((path / 'complete.json').read_text())
                if not report['complete'] or report['checkpoint'] != args.checkpoint:
                    raise ValueError('Mixed or incomplete checkpoint evaluation.')
                if bool(report.get('skip_file_hashes', False)) != args.skip_file_hashes:
                    raise ValueError('Mixed file binding protocols in evaluation.')
                if (report.get('checkpoint_metadata') != checkpoint_metadata if args.skip_file_hashes
                        else report['checkpoint_sha256'] != checkpoint_hash):
                    raise ValueError('Evaluation checkpoint changed.')
                signatures.append((report['eraf'], report['policy_kind'], report.get('memory_mode', 'carry'),
                                   json.dumps(report['deployment'], sort_keys=True)))
                canonical_path = Path(args.catalog_root) / task / 'demo_clean/correct/episodes.jsonl'
                if (report.get('canonical_metadata') != file_metadata(canonical_path) if args.skip_file_hashes
                        else report['canonical_sha256'] != file_sha256(canonical_path)):
                    raise ValueError('Matched scene/instruction catalog changed.')
                records = [json.loads(line) for line in (path / 'episodes.jsonl').read_text().splitlines()]
                canonical = [json.loads(line) for line in canonical_path.read_text().splitlines()]
                fields = ('scene_seed', 'source_instruction', 'counterfactual_instruction', 'episode_index')
                if len(records) != args.episodes or any(
                        any(a[k] != b[k] for k in fields) for a, b in zip(records, canonical, strict=True)):
                    raise ValueError('Evaluation did not use the exact matched seeds and instructions.')
                initial_by_condition.append(json.loads((path / 'initial_states.json').read_text()))
                cell = json.loads((path / 'summary.json').read_text())
                if cell['total_episodes'] != args.episodes:
                    raise ValueError('Cell episode count changed.')
                cells.append(cell | {'episodes': cell['total_episodes']})
            if any(x != initial_by_condition[0] for x in initial_by_condition):
                raise ValueError('Correct and CF initial physical states differ.')
        if len(set(signatures)) != 1:
            raise ValueError('Mixed deployment modes in the evaluation matrix.')
        summary = {'complete': True, 'checkpoint': args.checkpoint, 'cells': cells,
                   'episodes': sum(r['episodes'] for r in cells)}
        (root / 'summary.json').write_text(json.dumps(summary, indent=2))
        if args.episodes == 3 and len(args.tasks) == 5 and len(args.conditions) == 2:
            from experiments.robotwin.eraf_fg_contract import acceptance
            decision = acceptance(summary)
            (root / 'acceptance.json').write_text(json.dumps(decision, indent=2))
            print(json.dumps(decision), flush=True)
        return
    if args.mode == 'worker' and not all((args.manifest, args.checkpoint, args.catalog_root)):
        ap.error('Worker requires manifest, checkpoint and catalog root.')
    if args.mode == 'catalog' and (args.start_seed is None or not 90000000 <= args.start_seed < 92000000):
        ap.error('New dev/test catalogs use reserved seeds [90000000,92000000).')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args, _close
    from experiments.robotwin.language_interventions import (
        GoalObserver, EPISODE_FORMAT, load_intervention_manifest, select_intervention_pair, load_matched_episode_records)
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    from experiments.robotwin.eraf_fg_collection import physical_state
    from experiments.robotwin.pgc_data import pair_spec_from_source_task, array_sha256
    from experiments.robotwin.pgc_task_variants import install_pgc_observation_contract
    import numpy as np
    root.mkdir(parents=True, exist_ok=True)
    pairs = load_intervention_manifest(args.interventions, robotwin_root=Path(args.robotwin_root))
    policy = None
    checkpoint_hash = file_sha256(args.checkpoint) if args.checkpoint and not args.skip_file_hashes else None
    checkpoint_metadata = file_metadata(args.checkpoint) if args.checkpoint and args.skip_file_hashes else None
    for task_name in args.tasks:
        task, options = _load_robotwin_args(robotwin_root=Path(args.robotwin_root), task_name=task_name,
                                          task_config='demo_clean', output_root=root)
        install_pgc_observation_contract(task, pair_spec_from_source_task(task_name))
        official = official_module(args.robotwin_root)
        pair = select_intervention_pair(pairs, source_task=task_name)
        options.update(eval_mode=True, render_freq=0, need_plan=True, save_data=False,
                       policy_name='eraf_fg_explicit_policy', ckpt_setting=args.checkpoint,
                       eval_video_log=args.videos)
        video_size = official.get_eval_video_size(options) if args.videos else None
        if args.mode == 'catalog':
            directory = root / task_name / 'demo_clean' / 'correct'
            directory.mkdir(parents=True, exist_ok=False)
            journal = (directory / 'episodes.jsonl').open('x', buffering=1)
            events = (directory / 'screening.jsonl').open('x', buffering=1)
            count = 0
            for seed in range(args.start_seed, args.start_seed + args.max_attempts):
                try:
                    task.setup_demo(now_ep_num=count, seed=seed, is_test=True, **deepcopy(options))
                    initial = GoalObserver(task, pair).update()
                    if initial.source.success or initial.counterfactual.success:
                        raise ValueError('Initial scene already satisfies a goal.')
                    state_hash = array_sha256(physical_state(task))
                    info = task.play_once()
                    if not task.plan_success or not task.check_success() or not GoalObserver(task, pair).update().source.success:
                        raise ValueError('Source expert could not solve the scene.')
                    source = official._deterministic_instruction(task_name=pair.source_task,
                        episode_info=info['info'], instruction_type='unseen', scene_seed=seed)
                    target = (str(pair.counterfactual_instruction) if pair.counterfactual_instruction is not None else
                              official._deterministic_instruction(task_name=pair.counterfactual_task,
                                  episode_info=info['info'], instruction_type='unseen', scene_seed=seed))
                    row = {'format': EPISODE_FORMAT, 'pair_id': pair.pair_id, 'source_task': task_name,
                           'counterfactual_task': pair.counterfactual_task, 'task_config': 'demo_clean',
                           'condition': 'correct', 'episode_index': count, 'scene_seed': seed,
                           'instruction_type': 'unseen', 'source_instruction': source,
                           'counterfactual_instruction': target, 'policy_instruction': source,
                           'instruction_goal': 'source', 'selected_goal': 'source',
                           'initial_source_goal_success': False, 'initial_counterfactual_goal_success': False,
                           'initial_physical_state_sha256': state_hash,
                           'catalog_only': True, 'selection': 'native expert feasibility; no learned policy selection'}
                    journal.write(json.dumps(row) + '\n')
                    count += 1
                    print(f'[catalog] {task_name} accepted={count}/{args.episodes} seed={seed}', flush=True)
                except Exception as exc:
                    events.write(json.dumps({'scene_seed': seed, 'error': repr(exc)}) + '\n')
                finally:
                    _close(task)
                if count == args.episodes:
                    break
            journal.close(); events.close()
            if count != args.episodes:
                raise ValueError(f'Catalog incomplete: {task_name} {count}/{args.episodes}')
            (directory / 'catalog_complete.json').write_text(json.dumps({'complete': True, 'episodes': count,
                'selection': 'source expert only', 'start_seed': args.start_seed}))
            continue
        canonical_path = Path(args.catalog_root) / task_name / 'demo_clean/correct/episodes.jsonl'
        canonical = load_matched_episode_records(canonical_path, expected_pair_id=pair.pair_id,
            expected_source_task=task_name, expected_counterfactual_task=pair.counterfactual_task,
            expected_task_config='demo_clean', expected_instruction_type='unseen', expected_episodes=args.episodes)
        if policy is None:
            manifest = json.loads(Path(args.manifest).read_text())
            if args.policy_kind == 'repair':
                from experiments.robotwin.eraf_fg_bridge import load_policy
                policy = load_policy(args.checkpoint, manifest)
                policy.model.policy_guard_enabled = args.eraf == 'on'
            else:
                from types import SimpleNamespace
                from scripts.train_robotwin_cf_decision_adapter import load_policy
                policy = load_policy(SimpleNamespace(checkpoint=args.checkpoint, seed=42), manifest)
            if args.memory_mode == 'reset':
                install_memory_reset(policy)
            print(f'[evaluation-runtime] eraf={args.eraf} memory={args.memory_mode}', flush=True)
        policy.task_name, policy.task_config = task_name, 'demo_clean'
        from experiments.robotwin.fastwam_policy.deploy_policy import eval as evaluate, reset_model
        official.eval_function_decorator = lambda name, function: {'eval': evaluate, 'reset_model': reset_model}[function]
        for condition in args.conditions:
            directory = root / task_name / 'demo_clean' / condition
            directory.mkdir(parents=True, exist_ok=False)
            options['eval_video_save_dir'] = directory if args.videos else None
            initial_hashes = []
            original_begin = policy.begin_episode

            def begin(env, metadata):
                digest = array_sha256(physical_state(env))
                expected = canonical[metadata['episode_index']].get('initial_physical_state_sha256')
                if expected is not None and digest != expected:
                    raise ValueError('Catalog initial physical state changed.')
                initial_hashes.append({'scene_seed': metadata['scene_seed'], 'sha256': digest})
                return original_begin(env, metadata)

            policy.begin_episode = begin
            try:
                goal = 'source' if condition == 'correct' else 'counterfactual'
                _, _, records = official.eval_policy(task_name, task, deepcopy(options), policy, 0,
                    test_num=args.episodes, video_size=video_size,
                    instruction_type='unseen', skip_get_obs_within_replan=True, condition=condition,
                    intervention_pair=pair, instruction_goal=goal, selected_goal=goal,
                    episode_results_path=directory / 'episodes.jsonl', matched_episode_records=canonical)
            finally:
                policy.begin_episode = original_begin
            summary = official._summarize_intervention_records(records=records, pair=pair, condition=condition,
                task_config='demo_clean', instruction_type='unseen', checkpoint=args.checkpoint)
            (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
            (directory / 'initial_states.json').write_text(json.dumps(initial_hashes, indent=2))
            (directory / 'complete.json').write_text(json.dumps({'complete': len(records) == args.episodes,
                'checkpoint': args.checkpoint, 'checkpoint_sha256': checkpoint_hash,
                'canonical_sha256': None if args.skip_file_hashes else file_sha256(canonical_path),
                'checkpoint_metadata': checkpoint_metadata,
                'interventions': args.interventions,
                'interventions_metadata': file_metadata(args.interventions),
                'canonical_metadata': file_metadata(canonical_path) if args.skip_file_hashes else None,
                'skip_file_hashes': args.skip_file_hashes, 'eraf': args.eraf, 'policy_kind': args.policy_kind,
                'memory_mode': args.memory_mode,
                'deployment': {'action_horizon': 32, 'replan_steps': 24, 'inference_steps': 10}}, indent=2))
    if args.mode == 'catalog':
        (root / 'catalog_plan.json').write_text(json.dumps(vars(args), indent=2))


if __name__ == '__main__':
    main()
