#!/usr/bin/env python3
"""Pretrain fresh RoboTwin ERAF semantic heads while retaining warm policy weights."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]


def balanced_rows(rows, seed):
    groups = defaultdict(list)
    for row in rows:
        if row["replay_split"] == "train" and not row.get("native_retention"):
            groups[row["pair_id"], row["task_config"]].append(row)
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
    args = ap.parse_args()
    world, rank, local = (int(os.environ.get(k, d)) for k, d in
                           (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if min(args.steps, args.save_every, args.global_batch) < 1 or args.global_batch % world:
        ap.error("Positive counts and global batch divisible by world size required.")
    import torch
    import torch.distributed as dist
    from experiments.robotwin.eraf_fg_bridge import load_policy, trainable_parameters, MasterAdamW, save_repair_checkpoint
    from experiments.robotwin.eraf_fg_data import RawReplay, grounding_outputs
    from experiments.robotwin.compact_replay import ReplayPayloads
    from experiments.robotwin.same_state_repair import move_cache
    from scripts.train_robotwin_cf_decision_adapter import average_gradients
    from fastwam.models.wan22.entity_relation_affordance import entity_relation_affordance_loss, masks_to_patch_targets
    torch.cuda.set_device(local)
    torch.manual_seed(args.seed)
    if world > 1:
        dist.init_process_group("nccl")

    def barrier():
        if world > 1:
            dist.barrier()

    root = Path(args.output).resolve()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    barrier()
    manifest = json.loads(Path(args.manifest).read_text())
    rows = manifest["states"]
    policy = load_policy(args.checkpoint, manifest, device=f"cuda:{local}", seed=args.seed)
    model = policy.model
    selected = trainable_parameters(model, "grounding")
    optimizer = MasterAdamW(selected.values(), lr=args.learning_rate)
    payloads = ReplayPayloads(rows, model.device)
    raw = RawReplay(args.source_bank, label_cache=args.label_cache)
    stream = balanced_rows(rows, args.seed)
    validation = [r for r in rows if r["replay_split"] == "replay_holdout" and not r.get("native_retention")]
    if args.steps <= 2:
        validation = list({r["pair_id"]: r for r in validation}.values())
    if {r["pair_id"] for r in validation} != {r["pair_id"] for r in rows}:
        raise ValueError("Grounding holdout must cover all five tasks.")
    if rank == 0:
        (root / "plan.json").write_text(json.dumps(vars(args) | {"world_size": world,
            "trainable_parameters": list(selected), "validation_states": len(validation),
            "policy_frozen": True, "optimizer_precision": "FP32 master weights"}, indent=2))

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
                    reports.append({"id": row["id"], "language": language, "role_hits": hit,
                                    "role_count": total, "relation_hits": int(relation.sum()),
                                    "relation_count": int(valid.sum())})
        role = sum(r["role_hits"] for r in reports) / max(1, sum(r["role_count"] for r in reports))
        relation = sum(r["relation_hits"] for r in reports) / max(1, sum(r["relation_count"] for r in reports))
        result = {"step": step, "role_accuracy": role, "relation_accuracy": relation,
                  "eligible": role >= .8 and relation >= .9, "rows": reports}
        (root / f"grounding_eval_{step:06d}.json").write_text(json.dumps(result, indent=2))
        print(f"[grounding-eval] step={step} roles={role:.4f} relations={relation:.4f}", flush=True)

    if rank == 0:
        evaluate(0)
    barrier()
    start = time.monotonic()
    journal = (root / f"rank{rank}.jsonl").open("x", buffering=1)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        batch = [next(stream) for _ in range(args.global_batch)]
        losses, ids = [], []
        for row, language in batch[rank::world]:
            payload = raw.attach_proprio(payloads[row["id"]], row, policy)
            labels = move_cache(raw.grounding(row, language), model.device)
            outputs, _ = grounding_outputs(model, payload["captured"][language])
            loss, metrics = entity_relation_affordance_loss(outputs, labels, weights=model.policy_guard_eraf_loss_weights)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite ERAF grounding loss.")
            (loss / (args.global_batch // world)).backward()
            losses.append(float(loss.detach())); ids.append([row["id"], language])
        average_gradients(selected.values())
        norm = optimizer.step()
        record = {"step": step, "mean_loss": sum(losses) / len(losses), "grad_norm": norm,
                  "samples": ids, "elapsed": time.monotonic() - start}
        journal.write(json.dumps(record) + "\n")
        if rank == 0 and (step == 1 or step % 25 == 0):
            print(f"[grounding] step={step} loss={record['mean_loss']:.6f} elapsed={record['elapsed']:.1f}s", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            barrier()
            if rank == 0:
                save_repair_checkpoint(model, root / f"step_{step:06d}.pt", stage="grounding", steps=step,
                    parent=args.checkpoint, fg_supervision="off", provenance={"plan": str(root / "plan.json")})
                evaluate(step)
            barrier()
    journal.close()
    if rank == 0:
        (root / "complete.json").write_text(json.dumps({"complete": True, "optimizer_steps": args.steps}))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
