#!/usr/bin/env python3
"""Collect full-goal corrections from failed warm-policy rollouts on new scenes."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import sys
import traceback
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]


def historical_scene_keys(rows):
    from experiments.robotwin.pgc_data import ROBOTWIN_ERAF_PAIR_SPECS
    from experiments.robotwin.eraf_fg_contract import scene_key
    tasks = {spec.pair_id: spec.source_task for spec in ROBOTWIN_ERAF_PAIR_SPECS}
    keys = set()
    for row in rows:
        task = tasks[row["pair_id"]]
        if row.get("source_task", task) != task:
            raise ValueError("Historical scene task contradicts the authoritative pair specification.")
        keys.add(scene_key(dict(row, source_task=task)))
    return keys


def instructions(task, spec):
    from experiments.robotwin.decision_language_replay import (
        bound_spatial_instruction_pairs, seen_instruction_pairs)
    if spec.source_task.startswith("place_a2b_"):
        slots = {"{A}": f"{task.selected_modelname_A}/base{task.selected_model_id_A}",
                 "{B}": f"{task.selected_modelname_B}/base{task.selected_model_id_B}"}
        return bound_spatial_instruction_pairs(REPO, spec.pair_id, slots)[0]
    return seen_instruction_pairs(REPO, spec.pair_id)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--task", choices=["place_a2b_left", "blocks_ranking_rgb"], required=True)
    ap.add_argument("--task-config", choices=["demo_clean", "demo_randomized"], default="demo_clean")
    ap.add_argument("--robotwin-root", default=str(REPO / "third_party/RoboTwin"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--start-seed", type=int, required=True)
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--holdout-scenes", type=int, default=6)
    ap.add_argument("--max-attempts", type=int, default=120)
    args = ap.parse_args()
    if not 80000000 <= args.start_seed < 81000000 or args.start_seed + args.max_attempts >= 81000000:
        ap.error("Use the reserved new-training seed range [80000000,81000000).")
    if not 0 <= args.holdout_scenes < args.scenes or args.scenes > args.max_attempts:
        ap.error("Require 0 <= holdout < scenes <= attempts.")
    args.output = str(Path(args.output).resolve())
    args.checkpoint = str(Path(args.checkpoint).resolve())
    args.manifest = str(Path(args.manifest).resolve())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/root/gpufree-data/fastwam/FastWAM/checkpoints")
    import numpy as np
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    from experiments.robotwin.eraf_fg_contract import (
        FORMAT, FAILURE_ORIGINS, CAMERAS, candidate_replans, failure_kind,
        validate_correction, assert_scene_disjoint, scene_key)
    from experiments.robotwin.eraf_fg_collection import (
        run_failure_rollout, replay_prefix, record_continuation, continue_to_goal,
        replay_continuation, full_goal)
    from experiments.robotwin.pgc_data import array_sha256, pair_spec_from_source_task
    from experiments.robotwin.pgc_task_variants import install_pgc_task_contract
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args, _capture_data_type, _close
    from scripts.train_robotwin_cf_decision_adapter import load_policy
    manifest = json.loads(Path(args.manifest).read_text())
    excluded = historical_scene_keys(manifest["states"])
    policy = load_policy(SimpleNamespace(checkpoint=args.checkpoint, seed=42), manifest)
    spec = pair_spec_from_source_task(args.task)
    task, task_args = _load_robotwin_args(robotwin_root=Path(args.robotwin_root).resolve(),
        task_name=args.task, task_config=args.task_config, output_root=root)
    install_pgc_task_contract(task, spec)
    task_args.update(data_type=_capture_data_type(task_args), eval_mode=True,
                     need_plan=True, save_data=False, render_freq=0)
    checkpoint_hash = file_sha256(args.checkpoint)
    (root / "plan.json").write_text(json.dumps(vars(args) | {"rollout_checkpoint_sha256": checkpoint_hash}, indent=2))
    accepted, failures = [], []

    def setup(seed):
        task._pgc_active_variant = spec.counterfactual_variant
        task.setup_demo(now_ep_num=0, seed=seed, **deepcopy(task_args))

    def close():
        _close(task)

    with (root / "events.jsonl").open("x", buffering=1) as journal:
        for seed in range(args.start_seed, args.start_seed + args.max_attempts):
            if len(accepted) >= args.scenes:
                break
            scene = {"source_task": args.task, "task_config": args.task_config, "scene_seed": seed}
            assert_scene_disjoint([scene], excluded)
            folder = root / f"scene_{seed}"
            folder.mkdir()
            try:
                setup(seed)
                texts = instructions(task, spec)
                trace = run_failure_rollout(task, policy, spec, texts["target"])
                np.savez_compressed(folder / "failure_rollout.npz", initial=trace["initial"], actions=trace["actions"],
                    capture_steps=np.array(list(trace["states"])), states=np.stack(list(trace["states"].values())))
                close()
                if trace["audit"]["target"]:
                    journal.write(json.dumps(scene | {"status": "already_cf_success"}) + "\n")
                    continue
                kind = failure_kind(source_ever_success=trace["audit"]["source"], cf_ever_success=False)
                candidates = candidate_replans(trace["states"])
                obtained = False
                for tried, step in enumerate(candidates, 1):
                    try:
                        setup(seed)
                        error = replay_prefix(task, spec, trace, step)
                        with record_continuation(task) as (controls, frames):
                            continue_to_goal(task, spec)
                        okay = task.plan_success and full_goal(task, spec) and len(frames) >= 12
                        close()
                        if not okay:
                            raise ValueError("Expert continuation did not complete/release the full goal.")
                        for _ in range(2):
                            setup(seed)
                            error = max(error, replay_prefix(task, spec, trace, step))
                            replay_continuation(task, controls)
                            verified = full_goal(task, spec)
                            close()
                            if not verified:
                                raise ValueError("Full-goal control replay failed.")
                        actions = np.stack([f["qpos"] for f in frames]).astype(np.float32)
                        record = validate_correction(scene | {
                            "format": FORMAT, "pair_id": spec.pair_id, "failure_kind": kind,
                            "capture_origin": FAILURE_ORIGINS[kind], "capture_action_index": step,
                            "prefix_action_count": step, "source_goal_ever_success": trace["audit"]["source"],
                            "counterfactual_goal_ever_success": False,
                            "initial_state_sha256": array_sha256(trace["initial"]),
                            "capture_state_sha256": array_sha256(trace["states"][step]),
                            "prefix_action_sha256": array_sha256(trace["actions"][:step]),
                            "correction_action_sha256": array_sha256(actions),
                            "rollout_checkpoint_sha256": checkpoint_hash, "verification_policy": "full_goal",
                            "full_goal_verified": True, "counterfactual_goal_final_success": True,
                            "both_grippers_open_final": True, "verified_replay_count": 2,
                            "captured_state_count": len(trace["states"]), "candidate_count": tried,
                            "recorded_action_count": len(actions), "state_atol": 1e-7,
                            "replay_state_max_abs": error,
                            "replay_split": "replay_holdout" if len(accepted) < args.holdout_scenes else "train",
                            "source_instruction": texts["source"], "counterfactual_instruction": texts["target"],
                            "frame_path": str(folder / "correction.npz"), "image_color_space": "RGB",
                            "save_freq": task_args.get("save_freq")})
                        arrays = {"actions": actions}
                        arrays.update({c: np.stack([f["images"][c] for f in frames]) for c in CAMERAS})
                        arrays.update({"grounding/" + k: np.stack([f["grounding"][k] for f in frames])
                                       for k in frames[0]["grounding"]})
                        np.savez_compressed(folder / "correction.npz", **arrays)
                        with (folder / "controls.pkl").open("xb") as handle:
                            pickle.dump(controls, handle, protocol=5)
                        record["frame_sha256"] = file_sha256(folder / "correction.npz")
                        record["controls_sha256"] = file_sha256(folder / "controls.pkl")
                        (folder / "record.json").write_text(json.dumps(record, indent=2))
                        accepted.append(record)
                        journal.write(json.dumps(record | {"status": "verified"}) + "\n")
                        print(f"[verified] {args.task} seed={seed} prefix={step} frames={len(frames)} scenes={len(accepted)}/{args.scenes}", flush=True)
                        obtained = True
                        break
                    except Exception as exc:
                        try:
                            close()
                        except Exception:
                            pass
                        journal.write(json.dumps(scene | {"status": "candidate_rejected", "step": step,
                                                         "error": repr(exc)}) + "\n")
                if not obtained:
                    failures.append(scene | {"reason": "all_corrections_rejected"})
            except Exception as exc:
                try:
                    close()
                except Exception:
                    pass
                traceback.print_exc()
                failures.append(scene | {"reason": repr(exc)})
                journal.write(json.dumps(failures[-1] | {"status": "scene_failed"}) + "\n")
    result = {"format": FORMAT, "complete": len(accepted) == args.scenes,
              "records": accepted, "failures": failures, "requested_scenes": args.scenes}
    (root / "manifest.json").write_text(json.dumps(result, indent=2))
    if not result["complete"]:
        raise RuntimeError(f"FG collection incomplete: {len(accepted)}/{args.scenes}; inspect events.jsonl")


if __name__ == "__main__":
    main()
