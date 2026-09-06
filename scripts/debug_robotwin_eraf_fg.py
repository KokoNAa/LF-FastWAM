#!/usr/bin/env python3
"""Diagnose expert continuation on one recorded failure prefix without a model."""
import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, default=24)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--initial-expert", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args, _capture_data_type, _close
    from experiments.robotwin.pgc_data import pair_spec_from_source_task
    from experiments.robotwin.pgc_task_variants import install_pgc_task_contract, play_variant
    from experiments.robotwin.eraf_fg_collection import replay_prefix, record_continuation, continue_to_goal, full_goal
    root = Path(args.trace).resolve().parent
    with np.load(args.trace) as handle:
        trace = {"initial": handle["initial"], "actions": handle["actions"],
                 "states": dict(zip(handle["capture_steps"].tolist(), handle["states"]))}
    task, options = _load_robotwin_args(robotwin_root=Path(args.robotwin_root), task_name=args.task,
                                      task_config="demo_clean", output_root=root / "debug")
    spec = pair_spec_from_source_task(args.task)
    install_pgc_task_contract(task, spec)
    options.update(data_type=_capture_data_type(options), eval_mode=True, need_plan=True, save_data=False, render_freq=0)
    task._pgc_active_variant = spec.counterfactual_variant
    task.setup_demo(now_ep_num=0, seed=args.seed, **options)
    if not args.initial_expert:
        print("prefix", args.step, "drift", replay_prefix(task, spec, trace, args.step), flush=True)
    print("actors", [a.get_pose().p.tolist() for a in task.pgc_scene_actors()], flush=True)
    original, count = task.move, 0

    def move(*a, **kw):
        nonlocal count
        count += 1
        result = original(*a, **kw)
        print("move", count, "plan_success", task.plan_success, flush=True)
        return result

    task.move = move
    with record_continuation(task) as (controls, frames):
        try:
            if args.initial_expert:
                play_variant(task, spec, spec.counterfactual_variant)
            else:
                continue_to_goal(task, spec)
        except Exception as exc:
            print("exception", repr(exc), flush=True)
    print(json.dumps({"plan_success": bool(task.plan_success), "full_goal": full_goal(task, spec),
                      "frames": len(frames), "controls": len(controls)}), flush=True)
    _close(task)


if __name__ == "__main__":
    main()
