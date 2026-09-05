#!/usr/bin/env python3
"""Short multi-scene Video/Action LoRA repair using cached frozen inputs.

Each sample contains both expert positives, at a shared initial observation or
at each branch's own later observation. Keep ordinary action flow training and
add positive MSE at the pure-noise endpoint.
The cache stops before trainable Video operations, which are recomputed.
No video reconstruction, sample hashing, or repeated deployment audits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
FOCUS = ("blocks_ranking_rgb_to_bgr", "stack_blocks_two_green_on_red_to_red_on_green")


def pair_stream(rows, seed, focus_repeats=2):
    """Visit every training scene; modestly upsample early goal decisions."""
    groups = defaultdict(list)
    for row in rows:
        if row["replay_split"] == "train":
            groups[row["pair_id"], row["task_config"]].append(row)
    if not groups or focus_repeats < 1:
        raise ValueError("Need training pairs and a positive sampling factor.")
    rng = random.Random(seed)
    while True:
        cycle = []
        for key, values in sorted(groups.items()):
            shuffled = [row for row in values for _ in range(int(row.get("sampling_weight", 1)))]
            rng.shuffle(shuffled)
            copies = focus_repeats if key[0] in FOCUS else 1
            cycle.extend(shuffled * copies)
        rng.shuffle(cycle)
        yield from cycle


def small_validation_set(rows):
    selected = {}
    for row in rows:
        if row["replay_split"] == "replay_holdout":
            selected.setdefault((row["pair_id"], row["task_config"]), row)
    return list(selected.values())


def average_gradients(parameters):
    """Synchronize accumulated paired gradients in one collective per step."""
    import torch
    import torch.distributed as dist
    if not dist.is_initialized():
        return
    grads = []
    for p in parameters:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat.div_(dist.get_world_size())
    offset = 0
    for grad in grads:
        count = grad.numel()
        grad.copy_(flat[offset:offset + count].view_as(grad))
        offset += count


def load_policy(args, manifest, device="cuda:0"):
    import torch
    from omegaconf import OmegaConf
    from fastwam.utils.config_resolvers import register_default_resolvers
    from scripts.probe_robotwin_no_eraf import inference_bootstrap_configs
    from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy

    register_default_resolvers()
    cfg = OmegaConf.load(manifest["original_train_config"])
    model_cfg, processor_cfg = inference_bootstrap_configs(cfg)
    return WorldActionRobotWinPolicy(
        model_cfg=model_cfg, processor_cfg=processor_cfg,
        checkpoint_path=args.checkpoint, dataset_stats_path=Path(manifest["stats_path"]),
        device=device, model_dtype=torch.bfloat16, action_horizon=32, replan_steps=24,
        num_inference_steps=10, sigma_shift=None, seed=args.seed,
        text_cfg_scale=1., negative_prompt="", rand_device="cpu", tiled=False,
        timing_enabled=False, num_video_frames=9,
        task_name="cf_decision_repair", task_config="demo_clean")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--pairs-per-step", type=int, default=2, help="Global pairs across all GPUs.")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--endpoint-weight", type=float, default=.25)
    ap.add_argument("--conditional-gain", type=float, default=1.,
                    help="Weight for language-dependent action difference within endpoint MSE.")
    ap.add_argument("--focus-repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42027)
    ap.add_argument("--resume-state")
    ap.add_argument("--seen-language-augmentation", action="store_true",
                    help="Mix seen-template paraphrases into ranking/stacking training positives.")
    args = ap.parse_args()
    if min(args.steps, args.pairs_per_step, args.save_every, args.eval_every) < 1:
        ap.error("Step counts and intervals must be positive.")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.pairs_per_step % world:
        ap.error("Global pairs per step must be divisible by the GPU count.")
    local_pairs = args.pairs_per_step // world
    if world == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/root/gpufree-data/fastwam/FastWAM/checkpoints")
    for key in list(os.environ):
        if key.startswith("PGC_ROBOTWIN_CLOSED_LOOP_CAPTURE"):
            os.environ.pop(key)
    import numpy as np
    import torch
    import torch.distributed as dist
    from experiments.robotwin.joint_adapter_repair import configure_adapters, paired_backward, build_cache
    from experiments.robotwin.same_state_repair import move_cache, noise_tensor, sample_cached_actions

    torch.manual_seed(args.seed)
    torch.cuda.set_device(local_rank)
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
    validation = small_validation_set(rows) if rank == 0 else []
    plan = vars(args) | {"training_pairs": sum(r["replay_split"] == "train" for r in rows),
        "focus_pairs": FOCUS, "validation_states": len(validation), "world_size": world,
        "scope": "Replay holdout belongs to original training scenes; only closed-loop evaluation establishes CF success."}
    if rank == 0:
        (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    policy = load_policy(args, manifest, f"cuda:{local_rank}")
    model = policy.model
    selected = configure_adapters(model, ["video", "action"])
    optimizer = torch.optim.AdamW(list(selected.values()), lr=args.learning_rate, weight_decay=0.)
    start = 0
    if args.resume_state:
        state = torch.load(args.resume_state, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        start = int(state["step"])
        del state
    payloads = {r["id"]: move_cache(torch.load(r["payload"], map_location="cpu", weights_only=True), model.device)
                for r in rows if r["replay_split"] == "train" or r in validation}
    from experiments.robotwin.decision_language_replay import build_seen_contexts, replace_language
    seen_contexts = (build_seen_contexts(model, REPO, [r['pair_id'] for r in rows])
                     if args.seen_language_augmentation else {})
    stream = itertools.islice(pair_stream(rows, args.seed, args.focus_repeats),
                             start * args.pairs_per_step + rank, None, world)

    def save(step):
        path = root / f"step_{step:06d}.pt"
        payload = {"format": "fastwam_lora_adapter_v1", "mot_trainable": model._lora_adapter_state_dict(),
            "lora_config": model.lora_config, "base_checkpoint": model.lora_base_checkpoint,
            "step": step, "torch_dtype": str(model.torch_dtype),
            "decision_repair": {"parent_checkpoint": args.checkpoint, "optimizer_steps": step,
                                "plan": str(root / "plan.json")}}
        torch.save(payload, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        torch.save({"step": step, "checkpoint": str(path), "optimizer": optimizer.state_dict()}, root / "optimizer.tmp")
        (root / "optimizer.tmp").replace(root / "optimizer_last.pt")
        print(f"[checkpoint] step={step} path={path}", flush=True)

    def evaluate(step):
        reports = []
        with torch.no_grad():
            for row in validation:
                p = payloads[row["id"]]
                refs = {k: v[0].float().cpu().numpy() for k, v in p["references"].items()}
                values = {}
                for language in ("source", "target"):
                    cache = build_cache(model, p["captured"][language])
                    values[language] = sample_cached_actions(model, cache, 42, 10)
                    del cache
                delta, gold = values["target"] - values["source"], refs["target"] - refs["source"]
                rmse = {k: float(np.mean((values[k] - refs[k]) ** 2) ** .5) for k in values}
                reports.append({"id": row["id"], "pair_id": row["pair_id"], "task_config": row["task_config"],
                    "source_rmse": rmse["source"], "target_rmse": rmse["target"],
                    "expert_separation": float(np.mean(gold ** 2) ** .5),
                    "delta_projection": float(np.sum(delta * gold) / max(float(np.sum(gold ** 2)), 1e-12))})
        (root / f"action_eval_{step:06d}.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(f"[action_eval] step={step} " + json.dumps(reports), flush=True)

    if rank == 0:
        evaluate(start)
    barrier()
    started = time.monotonic()
    log_path = root / "training.jsonl" if rank == 0 else Path(os.devnull)
    with log_path.open("x" if rank == 0 else "w", buffering=1) as log:
        for step in range(start + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            terms, ids = [], []
            for i in range(local_pairs):
                row = next(stream)
                ids.append(row["id"])
                p = payloads[row["id"]]
                seed = args.seed + (step - 1) * args.pairs_per_step + i * world + rank
                captured = p['captured']
                variants = seen_contexts.get(row['pair_id'], [])
                rng = random.Random(seed)
                if variants and rng.random() < .5:
                    variant = rng.choice(variants)
                    captured = {language: replace_language(captured[language], *variant[language])
                                for language in ('source', 'target')}
                noise = noise_tensor((1, 32, 14), seed, model)
                generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_000)
                u = torch.rand((1,), generator=generator)
                scheduler = model.train_action_scheduler
                t = (scheduler._phi(u, scheduler.shift) * scheduler.num_train_timesteps).to(device=model.device, dtype=model.torch_dtype)
                refs = {k: v.to(model.torch_dtype) for k, v in p["references"].items()}
                terms.append(paired_backward(model, captured, refs, noise, t,
                    float(scheduler.training_weight(t).item()) / local_pairs,
                    args.endpoint_weight / local_pairs,
                    args.conditional_gain if row.get("initial_observations_exactly_equal", True) else 1.))
            average_gradients(list(selected.values()))
            norm = torch.nn.utils.clip_grad_norm_(list(selected.values()), 1., error_if_nonfinite=True)
            optimizer.step()
            report = {"step": step, "ids": ids, "gradient_norm": float(norm), "terms": terms}
            log.write(json.dumps(report) + "\n")
            if rank == 0 and (step == start + 1 or step % 10 == 0):
                print(f"[train] step={step}/{args.steps} seconds_per_step={(time.monotonic()-started)/(step-start):.2f} "
                      f"endpoint_mse={sum(t['common_mse'] + t['conditional_mse'] for t in terms)/len(terms):.6f}", flush=True)
            if rank == 0 and (step % args.save_every == 0 or step == args.steps):
                save(step)
            if rank == 0 and (step % args.eval_every == 0 or step == args.steps):
                evaluate(step)
            if step % args.save_every == 0 or step % args.eval_every == 0 or step == args.steps:
                barrier()
    if rank == 0:
        (root / "complete.json").write_text(json.dumps({"complete": True, "steps": args.steps,
            "checkpoint": str(root / f"step_{args.steps:06d}.pt"), "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
