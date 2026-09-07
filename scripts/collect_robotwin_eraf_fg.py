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
import subprocess
from datetime import datetime

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]


def historical_scene_keys(rows):
    from experiments.robotwin.pgc_data import ROBOTWIN_TEN_TASK_SPECS
    from experiments.robotwin.eraf_fg_contract import scene_key
    tasks = {spec.pair_id: spec.source_task for spec in ROBOTWIN_TEN_TASK_SPECS}
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
    from experiments.robotwin.expanded_fg import (
        FORMAT as EXPANDED_FORMAT, TASKS, validate_scope, canonical_instructions,
        load_collection_policy, check_budget, CollectionBudgetExceeded)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--task", choices=["place_a2b_left", "blocks_ranking_rgb", *TASKS], required=True)
    ap.add_argument('--protocol', choices=['legacy_v1', 'expanded_v2'], default='legacy_v1')
    ap.add_argument('--split', choices=['train', 'replay_holdout'], default='train')
    ap.add_argument('--policy-kind', choices=['legacy', 'repair'], default='legacy')
    ap.add_argument('--eraf', choices=['on', 'off'], default='off')
    ap.add_argument('--candidates', type=int, default=20)
    ap.add_argument('--deadline', help='Absolute timezone-aware cutoff; no server shutdown.')
    ap.add_argument('--reserve-gib', type=float, default=3.)
    ap.add_argument("--task-config", choices=["demo_clean", "demo_randomized"], default="demo_clean")
    ap.add_argument("--robotwin-root", default=str(REPO / "third_party/RoboTwin"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--start-seed", type=int, required=True)
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--holdout-scenes", type=int, default=6)
    ap.add_argument("--max-attempts", type=int, default=120)
    ap.add_argument("--candidate-order", choices=["late_first", "early_first"], default="late_first")
    args = ap.parse_args()
    expanded = args.protocol == 'expanded_v2'
    if expanded:
        try:
            validate_scope(args.task, args.split, args.start_seed, args.max_attempts)
        except ValueError as exc:
            ap.error(str(exc))
        if args.holdout_scenes != 0 or args.deadline is None:
            ap.error('Expanded collection requires --holdout-scenes 0, explicit --split and --deadline.')
    elif (args.task not in ['place_a2b_left', 'blocks_ranking_rgb']
          or not 80000000 <= args.start_seed < 81000000
          or args.start_seed + args.max_attempts >= 81000000):
        ap.error('Legacy collection retains the original tasks and [80000000,81000000) seeds.')
    if args.policy_kind == 'legacy' and args.eraf != 'off':
        ap.error('ERAF requires --policy-kind repair.')
    if not 1 <= args.candidates <= 20 or args.reserve_gib < 1:
        ap.error('Require 1..20 candidates and at least 1GiB reserve.')
    if args.deadline is not None:
        cutoff = datetime.fromisoformat(args.deadline)
        if cutoff.tzinfo is None or cutoff <= datetime.now().astimezone():
            ap.error('Deadline must be future and timezone-aware.')
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
        replay_continuation, full_goal, physical_state)
    from experiments.robotwin.pgc_data import array_sha256, pair_spec_from_source_task
    from experiments.robotwin.pgc_task_variants import install_pgc_task_contract, play_variant
    from experiments.robotwin.eraf_fg_contract import verify_replayed_state
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args, _capture_data_type, _close
    manifest = json.loads(Path(args.manifest).read_text())
    excluded = historical_scene_keys(manifest["states"])
    policy = None
    spec = pair_spec_from_source_task(args.task)
    task, task_args = _load_robotwin_args(robotwin_root=Path(args.robotwin_root).resolve(),
        task_name=args.task, task_config=args.task_config, output_root=root)
    install_pgc_task_contract(task, spec)
    task_args.update(data_type=_capture_data_type(task_args), eval_mode=True,
                     need_plan=True, save_data=False, render_freq=0)
    checkpoint_hash = file_sha256(args.checkpoint)
    lineage = dict(policy_kind=args.policy_kind, eraf_mode=args.eraf, memory_mode='carry', policy_seed=42,
                   collector_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                   input_manifest_sha256=file_sha256(args.manifest))
    protocol_format = EXPANDED_FORMAT if expanded else FORMAT
    (root / "plan.json").write_text(json.dumps(vars(args) | lineage | {"rollout_checkpoint_sha256": checkpoint_hash}, indent=2))
    accepted, failures = [], []
    def save_result():
        result = {"format": protocol_format, "complete": len(accepted) == args.scenes,
                  "records": accepted, "failures": failures, "requested_scenes": args.scenes}
        temp = root / 'manifest.json.tmp'
        temp.write_text(json.dumps(result, indent=2))
        temp.replace(root / 'manifest.json')
        return result
    save_result()

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
                check_budget(root, args.deadline, args.reserve_gib)
                setup(seed)
                texts = canonical_instructions(spec) if expanded else instructions(task, spec)
                initial = physical_state(task)
                if expanded and (full_goal(task, spec) or full_goal(task, spec, selected_goal='source')):
                    raise ValueError('An initial goal is already satisfied.')
                play_variant(task, spec, spec.counterfactual_variant)
                reachable = bool(task.plan_success and full_goal(task, spec))
                close()
                screening = scene | {"initial_cf_expert_reachable": reachable,
                    "purpose": "initial scene feasibility screening only; not an FG correction",
                    "initial_state_sha256": array_sha256(initial)}
                (folder / "initial_expert_screen.json").write_text(json.dumps(screening, indent=2))
                if not reachable:
                    journal.write(json.dumps(screening | {"status": "initial_expert_rejected"}) + "\n")
                    print(f"[screen-rejected] {args.task} seed={seed}", flush=True)
                    continue
                if policy is None:
                    policy = load_collection_policy(args.checkpoint, manifest,
                        policy_kind=args.policy_kind, eraf_mode=args.eraf,
                        task=args.task, task_config=args.task_config)
                setup(seed)
                verify_replayed_state(initial, physical_state(task))
                trace = run_failure_rollout(task, policy, spec, texts["target"])
                np.savez_compressed(folder / "failure_rollout.npz", initial=trace["initial"], actions=trace["actions"],
                    capture_steps=np.array(list(trace["states"])), states=np.stack(list(trace["states"].values())))
                (folder / "failure_audit.json").write_text(json.dumps(trace["audit"] | {
                    "source_instruction": texts["source"], "target_instruction": texts["target"],
                    "rollout_checkpoint_sha256": checkpoint_hash, **lineage}, indent=2))
                close()
                if trace["audit"]["target"]:
                    journal.write(json.dumps(scene | {"status": "already_cf_success"}) + "\n")
                    continue
                kind = failure_kind(source_ever_success=trace["audit"]["source"], cf_ever_success=False)
                candidates = candidate_replans(trace["states"], limit=args.candidates)
                if args.candidate_order == "early_first":
                    candidates.sort()
                obtained = False
                for tried, step in enumerate(candidates, 1):
                    try:
                        check_budget(root, args.deadline, args.reserve_gib)
                        setup(seed)
                        error = replay_prefix(task, spec, trace, step)
                        with record_continuation(task) as (controls, frames):
                            continue_to_goal(task, spec)
                        details = {"plan_success": bool(task.plan_success),
                                   "full_goal": full_goal(task, spec), "frames": len(frames)}
                        okay = details["plan_success"] and details["full_goal"] and len(frames) >= 12
                        close()
                        if not okay:
                            raise ValueError(f"Expert continuation failed: {details}")
                        for _ in range(2):
                            setup(seed)
                            error = max(error, replay_prefix(task, spec, trace, step))
                            error = max(error, replay_continuation(task, controls))
                            verified = full_goal(task, spec)
                            close()
                            if not verified:
                                raise ValueError("Full-goal control replay failed.")
                        actions = np.stack([f["qpos"] for f in frames]).astype(np.float32)
                        record = validate_correction(scene | lineage | {
                            "format": protocol_format, "pair_id": spec.pair_id, "failure_kind": kind,
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
                            "replay_split": args.split if expanded else ("replay_holdout" if len(accepted) < args.holdout_scenes else "train"),
                            "source_instruction": texts["source"], "counterfactual_instruction": texts["target"],
                            "frame_path": str(folder / "correction.npz"), "image_color_space": "RGB",
                            "save_freq": task_args.get("save_freq")})
                        arrays = {"actions": actions}
                        arrays.update({c: np.stack([f["images"][c] for f in frames]) for c in CAMERAS})
                        arrays.update({"grounding/" + k: np.stack([f["grounding"][k] for f in frames])
                                       for k in frames[0]["grounding"]})
                        check_budget(root, args.deadline, args.reserve_gib)
                        np.savez_compressed(folder / "correction.npz", **arrays)
                        with (folder / "controls.pkl").open("xb") as handle:
                            pickle.dump(controls, handle, protocol=5)
                        record["frame_sha256"] = file_sha256(folder / "correction.npz")
                        record["controls_sha256"] = file_sha256(folder / "controls.pkl")
                        (folder / "record.json").write_text(json.dumps(record, indent=2))
                        accepted.append(record)
                        save_result()
                        journal.write(json.dumps(record | {"status": "verified"}) + "\n")
                        print(f"[verified] {args.task} seed={seed} prefix={step} frames={len(frames)} scenes={len(accepted)}/{args.scenes}", flush=True)
                        obtained = True
                        break
                    except CollectionBudgetExceeded:
                        raise
                    except Exception as exc:
                        try:
                            close()
                        except Exception:
                            pass
                        journal.write(json.dumps(scene | {"status": "candidate_rejected", "step": step,
                                                         "error": repr(exc)}) + "\n")
                if not obtained:
                    failures.append(scene | {"reason": "all_corrections_rejected"})
            except CollectionBudgetExceeded as exc:
                failures.append(scene | {'reason': str(exc), 'budget_stop': True})
                journal.write(json.dumps(failures[-1]) + '\n')
                save_result()
                break
            except Exception as exc:
                try:
                    close()
                except Exception:
                    pass
                traceback.print_exc()
                failures.append(scene | {"reason": repr(exc)})
                journal.write(json.dumps(failures[-1] | {"status": "scene_failed"}) + "\n")
            finally:
                save_result()
    result = save_result()
    if not result["complete"]:
        raise RuntimeError(f"FG collection incomplete: {len(accepted)}/{args.scenes}; inspect events.jsonl")


if __name__ == "__main__":
    main()
