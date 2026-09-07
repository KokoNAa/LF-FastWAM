#!/usr/bin/env python3
"""Pretrain fresh RoboTwin ERAF semantic heads while retaining warm policy weights."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]


def balanced_rows(rows, seed, *, task_balanced=False):
    if task_balanced:
        from experiments.robotwin.eraf_fg_training import balanced_group_stream
        from random import Random
        selected = [r for r in rows if r['replay_split'] == 'train' and not any(r.get(k)
                    for k in ('native_retention', 'cf_retention', 'fg_correction'))]
        languages = Random(seed + 7919)
        for row in balanced_group_stream(selected, seed, task_balanced=True):
            yield row, languages.choice(('source', 'target'))
        return
    groups = defaultdict(list)
    for row in rows:
        if row["replay_split"] == "train" and not any(row.get(k) for k in
                ('native_retention', 'cf_retention', 'fg_correction')):
            groups[row["pair_id"], row["task_config"]].append(row)
    if not groups:
        raise ValueError('No original expert training rows.')
    rng = random.Random(seed)
    while True:
        keys = sorted(groups)
        rng.shuffle(keys)
        for key in keys:
            yield rng.choice(groups[key]), rng.choice(("source", "target"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source-bank", "checkpoint", "output", "label-cache"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--global-batch", type=int, default=12)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step-offset", type=int, default=0,
                    help="Prior semantic optimizer steps; extension restarts optimizer explicitly.")
    ap.add_argument('--geometry-only', action='store_true')
    ap.add_argument('--position-weight', type=float)
    ap.add_argument('--anchor-weight', type=float)
    from experiments.robotwin.pgc_data import ROBOTWIN_TEN_TASK_NAMES, pair_spec_from_source_task
    ap.add_argument('--qualification-tasks', nargs='+', choices=ROBOTWIN_TEN_TASK_NAMES,
                    default=['place_a2b_left', 'blocks_ranking_rgb'])
    ap.add_argument('--qualify-each-language', action='store_true')
    ap.add_argument('--task-balanced', action='store_true')
    ap.add_argument('--skip-file-hashes', action='store_true')
    ap.add_argument('--geometry-replay', choices=['off', 'ordinary', 'fg'], default='off',
                    help='Replace declared slots with matched target-only partial geometry supervision.')
    ap.add_argument('--geometry-replay-slots', type=int, default=3)
    args = ap.parse_args()
    if len(set(args.qualification_tasks)) != len(args.qualification_tasks):
        ap.error('Duplicate qualification task')
    required_pairs = [pair_spec_from_source_task(task).pair_id for task in args.qualification_tasks]
    world, rank, local = (int(os.environ.get(k, d)) for k, d in
                           (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if min(args.steps, args.save_every, args.global_batch) < 1 or args.global_batch % world:
        ap.error("Positive counts and global batch divisible by world size required.")
    if args.geometry_replay != 'off' and not 0 < args.geometry_replay_slots < args.global_batch:
        ap.error('Geometry replay needs positive slots and a nonempty ordinary expert core.')
    if any(value is not None and (not __import__('math').isfinite(value) or value <= 0)
           for value in (args.position_weight, args.anchor_weight)):
        ap.error('Geometry loss weights must be finite and positive.')
    import torch
    import torch.distributed as dist
    from experiments.robotwin.eraf_fg_bridge import load_policy, trainable_parameters, MasterAdamW, save_repair_checkpoint
    from experiments.robotwin.eraf_fg_data import RawReplay, grounding_outputs
    from experiments.robotwin.compact_replay import ReplayPayloads
    from experiments.robotwin.same_state_repair import move_cache
    from scripts.train_robotwin_cf_decision_adapter import average_gradients
    from fastwam.models.wan22.entity_relation_affordance import entity_relation_affordance_loss, masks_to_patch_targets
    from experiments.robotwin.eraf_geometry_metrics import geometry_parameter, geometry_errors, summarize_geometry, semantic_qualification
    from experiments.robotwin.fg_geometry_replay import SCHEMA, geometry_mixture, partial_geometry_loss, validate_scene_splits
    torch.cuda.set_device(local)
    torch.manual_seed(args.seed)
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local}"))

    def barrier():
        if world > 1:
            dist.barrier(device_ids=[local])

    root = Path(args.output).resolve()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    barrier()
    manifest = json.loads(Path(args.manifest).read_text())
    rows = manifest["states"]
    policy = load_policy(args.checkpoint, manifest, device=f"cuda:{local}", seed=args.seed)
    model = policy.model
    selected = trainable_parameters(model, "grounding")
    if args.geometry_only:
        for name, parameter in selected.items():
            parameter.requires_grad_(geometry_parameter(name))
        selected = {name: parameter for name, parameter in selected.items() if parameter.requires_grad}
        if not selected:
            raise ValueError('Geometry calibration selected no parameters.')
    overrides = {name: value for name, value in
        [('position', args.position_weight), ('anchor', args.anchor_weight)] if value is not None}
    model.policy_guard_eraf_loss_weights = replace(model.policy_guard_eraf_loss_weights, **overrides)
    optimizer = MasterAdamW(selected.values(), lr=args.learning_rate)
    payloads = ReplayPayloads(rows, model.device)
    raw = RawReplay(args.source_bank, label_cache=args.label_cache, fg_geometry=args.geometry_replay != 'off')
    stream = balanced_rows(rows, args.seed, task_balanced=args.task_balanced)
    geometry_stream = (geometry_mixture(rows, args.seed, batch_size=args.global_batch,
        slots=args.geometry_replay_slots, mode=args.geometry_replay, task_balanced=args.task_balanced)
        if args.geometry_replay != 'off' else None)
    # Audit whole-scene separation before consuming either correction or expert data.
    validate_scene_splits(rows)
    fg_validation = [r for r in rows if r['replay_split'] == 'replay_holdout' and r.get('fg_correction')]
    if args.geometry_replay != 'off' and not fg_validation:
        raise ValueError('Geometry replay requires separate FG holdout observations.')
    validation = [r for r in rows if r["replay_split"] == "replay_holdout" and not any(r.get(k)
        for k in ('native_retention', 'cf_retention', 'fg_correction'))]
    if args.steps <= 2:
        validation = list({r["pair_id"]: r for r in validation}.values())
        fg_validation = list({r['pair_id']: r for r in fg_validation}.values())
    if {r["pair_id"] for r in validation} != {r["pair_id"] for r in rows}:
        raise ValueError("Grounding holdout must cover every replay task.")
    if not set(required_pairs) <= {r['pair_id'] for r in validation}:
        raise ValueError('A declared semantic qualification task has no holdout observations.')
    if rank == 0:
        (root / "plan.json").write_text(json.dumps(vars(args) | {"world_size": world,
            "geometry_replay_schema": SCHEMA if geometry_stream is not None else None,
            "code_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "manifest_sha256": __import__("hashlib").sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "geometry_replay_validation_states": len(fg_validation) if geometry_stream is not None else 0,
            "geometry_replay_selection": "Report FG holdout separately; retain original expert qualification and geometry selection.",
            "trainable_parameters": list(selected), "validation_states": len(validation),
            "policy_frozen": True, "optimizer_precision": "FP32 master weights",
            "optimizer_restart": bool(args.step_offset),
            "loss_weights": asdict(model.policy_guard_eraf_loss_weights),
            "qualification_pairs": required_pairs,
            "selection_rule": "Minimum equal-task/language position and goal error after the declared semantic qualification; baseline included; no action-test selection."}, indent=2))

    def evaluate(step):
        reports = []
        with torch.no_grad():
            for row in validation:
                payload = raw.attach_proprio(payloads[row["id"]], row, policy)
                for language in ("source", "target"):
                    labels = move_cache(raw.grounding(row, language), model.device)
                    outputs, _ = grounding_outputs(model, payload["captured"][language])
                    valid = labels["clause_valid"].bool()
                    hit, total = 0., 0
                    for role in ("subject", "reference"):
                        mask, present = masks_to_patch_targets(labels[role + "_masks"],
                            token_count=outputs[role + "_attention"].shape[-1])
                        keep = valid & present & labels[role + "_mask_valid"].bool()
                        index = outputs[role + "_attention"].argmax(-1)
                        correct = torch.gather(mask > 0, -1, index.unsqueeze(-1)).squeeze(-1)
                        hit += int((correct & keep).sum()); total += int(keep.sum())
                    relation = ((outputs["predicate_logits"].argmax(-1) == labels["predicate_ids"]) & valid)
                    reports.append({"id": row["id"], "pair_id": row['pair_id'], "language": language, "role_hits": hit,
                                    "role_count": total, "relation_hits": int(relation.sum()),
                                    "relation_count": int(valid.sum()),
                                    "geometry_cm": geometry_errors(outputs, labels)})
        role = sum(r["role_hits"] for r in reports) / max(1, sum(r["role_count"] for r in reports))
        relation = sum(r["relation_hits"] for r in reports) / max(1, sum(r["relation_count"] for r in reports))
        qualification = semantic_qualification(reports, required_pairs, each_language=args.qualify_each_language)
        result = {"step": step, "role_accuracy": role, "relation_accuracy": relation,
                  "eligible": qualification['target_tasks_eligible'],
                  "qualification": qualification, "rows": reports,
                  "geometry": summarize_geometry(reports)}
        if geometry_stream is not None:
            partial_reports = []
            with torch.no_grad():
                for row in fg_validation:
                    payload = raw.attach_proprio(payloads[row['id']], row, policy)
                    labels = move_cache(raw.grounding(row, 'target'), model.device)
                    outputs, _ = grounding_outputs(model, payload['captured']['target'])
                    loss, _ = partial_geometry_loss(outputs, labels, model.policy_guard_eraf_loss_weights)
                    partial_reports.append({'id': row['id'], 'pair_id': row['pair_id'], 'language': 'target',
                                            'loss': float(loss), 'geometry_cm': geometry_errors(outputs, labels)})
            result['fg_geometry_holdout'] = {'rows': partial_reports, 'geometry': summarize_geometry(partial_reports)}
        (root / f"grounding_eval_{step:06d}.json").write_text(json.dumps(result, indent=2))
        print(f"[grounding-eval] step={step} roles={role:.4f} relations={relation:.4f} geometry_cm={result['geometry']['selection_score_cm']:.3f}", flush=True)
        return result

    if rank == 0:
        evaluations = [evaluate(args.step_offset)]
    barrier()
    start = time.monotonic()
    journal = (root / f"rank{rank}.jsonl").open("x", buffering=1)
    for step in range(1, args.steps + 1):
        cumulative_step = args.step_offset + step
        optimizer.zero_grad()
        batch = (next(geometry_stream) if geometry_stream is not None else
                 [(r, language, False) for r, language in (next(stream) for _ in range(args.global_batch))])
        losses, ids = [], []
        for row, language, partial in batch[rank::world]:
            payload = raw.attach_proprio(payloads[row["id"]], row, policy)
            labels = move_cache(raw.grounding(row, language), model.device)
            outputs, _ = grounding_outputs(model, payload["captured"][language])
            loss_fn = partial_geometry_loss if partial else entity_relation_affordance_loss
            loss, metrics = loss_fn(outputs, labels, weights=model.policy_guard_eraf_loss_weights)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite ERAF grounding loss.")
            (loss / (args.global_batch // world)).backward()
            losses.append(float(loss.detach())); ids.append([row["id"], language])
        average_gradients(selected.values())
        norm = optimizer.step()
        record = {"step": cumulative_step, "local_step": step, "mean_loss": sum(losses) / len(losses), "grad_norm": norm,
                  "samples": ids, "elapsed": time.monotonic() - start}
        if geometry_stream is not None:
            record['partial_geometry_slots'] = [partial for _, _, partial in batch[rank::world]]
        journal.write(json.dumps(record) + "\n")
        if rank == 0 and (step == 1 or step % 25 == 0):
            print(f"[grounding] step={step} loss={record['mean_loss']:.6f} elapsed={record['elapsed']:.1f}s", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            barrier()
            if rank == 0:
                save_repair_checkpoint(model, root / f"step_{cumulative_step:06d}.pt", stage="grounding", steps=cumulative_step,
                    parent=args.checkpoint, fg_supervision="full" if args.geometry_replay == 'fg' else "off",
                    provenance={"plan": str(root / "plan.json"), "geometry_replay_schema": SCHEMA if geometry_stream is not None else None},
                    record_hashes=not args.skip_file_hashes)
                evaluations.append(evaluate(cumulative_step))
            barrier()
    journal.close()
    if rank == 0:
        eligible = [r for r in evaluations if r['eligible']]
        best = min(eligible, key=lambda r: (r['geometry']['selection_score_cm'], r['step'])) if eligible else None
        (root / 'selection.json').write_text(json.dumps({'complete': True,
            'selected_step': best['step'] if best else None,
            'selected_checkpoint': (args.checkpoint if best['step'] == args.step_offset else
                str(root / f"step_{best['step']:06d}.pt")) if best else None,
            'baseline_score_cm': evaluations[0]['geometry']['selection_score_cm'],
            'selected_score_cm': best['geometry']['selection_score_cm'] if best else None,
            'improved_over_parent': bool(best and best['geometry']['selection_score_cm'] < evaluations[0]['geometry']['selection_score_cm'])}, indent=2))
        (root / "complete.json").write_text(json.dumps({"complete": True,
            "optimizer_steps": args.step_offset + args.steps, "local_optimizer_steps": args.steps}))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
