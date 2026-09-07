#!/usr/bin/env python3
"""Keep successful teacher rollouts for explicit Correct or CF retention."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]


TASKS = ('place_a2b_left', 'blocks_ranking_rgb', 'place_a2b_right',
         'place_burger_fries', 'stack_blocks_two')


def validate_request(args):
    if not 80200000 <= args.start_seed < args.start_seed + args.max_attempts <= 81000000:
        raise ValueError('Use the reserved fresh policy-retention training seeds [80200000, 81000000).')
    if not 0 < args.scenes <= args.max_attempts:
        raise ValueError('Require positive scenes <= attempts.')
    if args.task not in TASKS or args.condition not in {'correct', 'counterfactual'}:
        raise ValueError('Unknown task or retention condition.')
    if args.max_seconds <= 0:
        raise ValueError('Require a positive collection time budget.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "checkpoint", "output", "robotwin-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument('--condition', choices=['correct', 'counterfactual'], default='counterfactual')
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=60)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--max-seconds', type=int, default=3600,
                        help='Stop admitting new scenes after this budget; an active scene finishes.')
    args = parser.parse_args()
    try:
        validate_request(args)
    except ValueError as error:
        parser.error(str(error))
    native = args.condition == 'correct'
    language = 'source' if native else 'target'
    kind = 'native' if native else 'cf'
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    args.manifest, args.checkpoint = str(Path(args.manifest).resolve()), str(Path(args.checkpoint).resolve())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/root/gpufree-data/fastwam/FastWAM/checkpoints")
    import numpy as np
    from experiments.robotwin.eraf_fg_contract import CAMERAS, assert_scene_disjoint
    from experiments.robotwin.eraf_fg_collection import run_failure_rollout, full_goal
    from experiments.robotwin.eraf_fg_data import file_metadata, validate_retention_scene
    from experiments.robotwin.pgc_data import pair_spec_from_source_task
    from experiments.robotwin.pgc_task_variants import install_pgc_task_contract
    from scripts.collect_robotwin_eraf_fg import historical_scene_keys, instructions
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args, _capture_data_type, _close
    from scripts.train_robotwin_cf_decision_adapter import load_policy
    manifest = json.loads(Path(args.manifest).read_text())
    excluded = historical_scene_keys(manifest["states"])
    # Reject the entire request before loading a large teacher or admitting any
    # records, including collisions with earlier data-holdout scenes.
    assert_scene_disjoint([{'source_task': args.task, 'task_config': 'demo_clean', 'scene_seed': seed}
        for seed in range(args.start_seed, args.start_seed + args.max_attempts)], excluded)
    checkpoint_metadata = file_metadata(args.checkpoint)
    (root / 'plan.json').write_text(json.dumps(vars(args) | {
        'teacher_checkpoint_metadata': checkpoint_metadata, 'hash_scans': False,
        'purpose': 'successful training-scene retention collection, not success-rate evaluation'}, indent=2))
    policy = load_policy(SimpleNamespace(checkpoint=args.checkpoint, seed=42), manifest)
    task, options = _load_robotwin_args(robotwin_root=Path(args.robotwin_root), task_name=args.task,
                                      task_config="demo_clean", output_root=root)
    spec = pair_spec_from_source_task(args.task)
    install_pgc_task_contract(task, spec)
    options.update(data_type=_capture_data_type(options), eval_mode=True, need_plan=True, save_data=False, render_freq=0)
    accepted, episodes, attempts = [], 0, 0
    deadline = time.monotonic() + args.max_seconds
    result = {'format': 'robotwin_policy_retention_v2', 'complete': False,
              'condition': args.condition, 'source_task': args.task, 'successful_scenes': 0,
              'requested_scenes': args.scenes, 'attempted_scenes': 0, 'states': accepted,
              'hash_scans': False, 'teacher_checkpoint_metadata': checkpoint_metadata}

    def save():
        result.update(successful_scenes=episodes, attempted_scenes=attempts)
        temporary = root / 'manifest.json.tmp'
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(root / 'manifest.json')

    save()
    journal = (root / "events.jsonl").open("x", buffering=1)
    for seed in range(args.start_seed, args.start_seed + args.max_attempts):
        if episodes == args.scenes:
            break
        if time.monotonic() >= deadline:
            result['stop_reason'] = 'collection_time_budget'
            break
        attempts += 1
        scene = {"source_task": args.task, "task_config": "demo_clean", "scene_seed": seed}
        assert_scene_disjoint([scene], excluded)
        captured = []
        original = policy._infer_action_chunk

        def infer(observation, instruction):
            action = original(observation, instruction)
            captured.append({"state": np.array(observation["joint_action"]["vector"], copy=True),
                             "action": action.copy(),
                             'step': int(task.take_action_cnt),
                             "images": {c: np.array(observation["observation"][c]["rgb"], copy=True) for c in CAMERAS}})
            return action

        try:
            task._pgc_active_variant = spec.source_variant if native else spec.counterfactual_variant
            task.setup_demo(now_ep_num=0, seed=seed, **deepcopy(options))
            initial_goal = full_goal(task, spec, selected_goal=language)
            if initial_goal:
                raise ValueError('The selected retention goal is already complete before any action.')
            texts = ({"source": spec.source_instruction, "target": spec.counterfactual_instruction}
                     if args.task == "place_burger_fries" else instructions(task, spec))
            policy._infer_action_chunk = infer
            trace = run_failure_rollout(task, policy, spec, texts[language], selected_goal=language)
            okay = bool(trace["audit"][language] and full_goal(task, spec, selected_goal=language))
            if okay:
                if not captured:
                    raise ValueError('An initially satisfied goal cannot provide action retention.')
                frames = sorted(set(np.rint(np.linspace(0, len(captured)-1, min(12, len(captured)))).astype(int)))
                items = [captured[i] for i in frames]
                path = root / f"scene_{seed}.npz"
                np.savez_compressed(path, state=np.stack([r["state"] for r in items]),
                    reference_action_raw=np.stack([r["action"] for r in items]),
                    capture_action_steps=np.array([r['step'] for r in items]),
                    initial_physical_state=trace['initial'], executed_actions=trace['actions'],
                    **{c: np.stack([r["images"][c] for r in items]) for c in CAMERAS})
                metadata = file_metadata(path)
                rows = [scene | {"id": f"{kind}_retention_{args.task}_{seed}_{i}", "pair_id": spec.pair_id,
                    kind + "_retention": True, "retention_condition": args.condition,
                    'retention_format': 'robotwin_policy_retention_v2',
                    "replay_split": "train", "source_instruction": texts["source"],
                    "counterfactual_instruction": texts["target"], "capture_path": str(path), "frame_index": i,
                    'capture_action_index': int(items[i]['step']),
                    "capture_metadata": metadata, "teacher_checkpoint": args.checkpoint,
                    "teacher_checkpoint_metadata": checkpoint_metadata, "full_" + kind + "_episode_success": True,
                    "initial_selected_goal_success": False,
                    "image_color_space": "RGB", "reference_kind": f"deployed_successful_{kind}_policy_32_actions"}
                    for i in range(len(items))]
                validate_retention_scene(rows)
                accepted.extend(rows)
                episodes += 1
            journal.write(json.dumps(scene | {kind + "_success": okay, "condition": args.condition,
                'source_ever_success': trace['audit']['source'],
                'target_ever_success': trace['audit']['target'],
                'executed_actions': len(trace['actions']), "accepted_scenes": episodes}) + "\n")
            print(f"[{kind}-retention] {args.task} seed={seed} success={okay} scenes={episodes}/{args.scenes}", flush=True)
        except Exception as exc:
            journal.write(json.dumps(scene | {"error": repr(exc)}) + "\n")
            print(f"[{kind}-retention-error] {args.task} seed={seed} {exc!r}", flush=True)
        finally:
            policy._infer_action_chunk = original
            try:
                _close(task)
            except Exception:
                pass
            save()
    journal.close()
    if file_metadata(args.checkpoint) != checkpoint_metadata:
        result['stop_reason'] = 'teacher_checkpoint_changed'
        save()
        raise ValueError('The teacher checkpoint changed during collection.')
    result['complete'] = episodes == args.scenes
    result.setdefault('stop_reason', 'scene_quota' if result['complete'] else 'attempt_budget')
    save()
    if not result["complete"]:
        raise RuntimeError(f"{kind}-retention scene target was not reached.")


if __name__ == "__main__":
    main()
