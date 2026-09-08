#!/usr/bin/env python3
"""Validate physical ID/contact APIs on expert rollouts outside the formal test.

The expert render hook is only an API smoke test. Formal policy metrics sample
the existing per-physics-step goal check, which is tested separately.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args, _close
    from experiments.robotwin.pgc_data import pair_spec_from_source_task
    from experiments.robotwin.pgc_task_variants import install_pgc_task_contract, play_variant
    from experiments.robotwin.language_interventions import load_intervention_manifest, select_intervention_pair, evaluate_goal
    from experiments.robotwin.manipulation_metrics import ManipulationObserver
    root = Path('/root/gpufree-data/LF-FastWAM/third_party/RoboTwin')
    pair = select_intervention_pair(load_intervention_manifest(REPO/'configs/eval/robotwin_cis_ten_tasks.json', robotwin_root=root), source_task=args.task)
    spec = pair_spec_from_source_task(args.task)
    env, options = _load_robotwin_args(robotwin_root=root, task_name=args.task, task_config='demo_clean', output_root=args.output.parent)
    install_pgc_task_contract(env, spec)
    options.update(eval_mode=True, render_freq=0, need_plan=True, save_data=False)
    errors = []
    for seed in range(args.seed, args.seed+20):
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **deepcopy(options))
            observer = ManipulationObserver(env, pair.counterfactual_goal)
            assert not observer.report()['any_correct_object_lifted']
            original = env._update_render
            def record():
                result = original()
                observer.sample(evaluate_goal(env, pair.counterfactual_goal))
                return result
            env._update_render = record
            try:
                play_variant(env, spec, spec.counterfactual_variant)
            finally:
                env._update_render = original
            report = observer.report()
            if not env.plan_success or not evaluate_goal(env, pair.counterfactual_goal).success:
                errors.append(dict(seed=seed, expert_failure=True))
                continue
            if not report['any_correct_object_lifted']:
                raise RuntimeError('Successful expert produced no sustained physical lift: '+json.dumps(report))
            args.output.write_text(json.dumps(dict(complete=True, task=args.task, seed=seed,
                expert_goal=True, metrics=report, screening=errors,
                sampling='expert render-hook API smoke only; not formal policy counts'), indent=2)+'\n')
            print(json.dumps(dict(complete=True, task=args.task, lifted={k:v['correctly_lifted'] for k,v in report['objects'].items()})), flush=True)
            return
        finally:
            _close(env)
    raise RuntimeError('No feasible smoke scene in declared range')


if __name__ == '__main__':
    main()
