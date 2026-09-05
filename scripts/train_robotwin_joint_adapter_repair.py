#!/usr/bin/env python3
"""Matched Action-only versus Video+Action repair of one audited decision state.

This isolates trainable scope, not the original full-dataset sampling objective.
Both arms use the same paired positives, shared noise, flow and endpoint losses.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
ARMS = {"action": ["action"], "joint": ["video", "action"]}


def worker(args):
    from scripts.probe_robotwin_no_eraf import load_probe_policy, read_json, sha256, write_json
    plan = read_json(Path(args.output) / "plan.json")
    source = read_json(Path(plan["source_probe"]) / "plan.json")
    for p, digest in plan["source_artifact_sha256"].items():
        if sha256(p) != digest:
            raise ValueError("Source artifacts changed.")
    root = Path(args.output) / args.arm
    root.mkdir(exist_ok=False)
    policy, audit = load_probe_policy(source, "step1000", args.gpu)
    import numpy as np
    import torch
    from experiments.robotwin.joint_adapter_repair import configure_adapters, capture_inputs, build_cache, paired_backward
    from experiments.robotwin.same_state_repair import (
        move_cache, noise_tensor, sample_cached_actions, action_rows, checkpoint_score,
        frozen_versions, audit_frozen,
    )
    from experiments.robotwin.no_eraf_probe import CAMERAS, observation_hash, difference, typed_hash
    from scripts.inspect_robotwin_same_state_repair import paired_components
    model = policy.model
    from collections import Counter
    loaded_dtypes = dict(Counter(str(p.dtype) for n, p in model.mot.named_parameters()
                                if n.endswith((".lora_A", ".lora_B"))))
    selected = configure_adapters(model, ARMS[args.arm])
    protected = frozen_versions(model)
    audit.update(experts=ARMS[args.arm], trainable_parameters=list(selected), loaded_adapter_dtypes=loaded_dtypes,
                 all_adapter_arithmetic="float32 in both arms; frozen base bfloat16")
    write_json(root / "checkpoint_audit.json", audit)
    norm = policy.processor.normalizer.normalizers["action"][policy.processor.shape_meta["action"][0]["key"]]
    states = plan["states"]
    observations, refs, inputs, baseline = {}, {}, {}, {}

    def predict_raw(obs, instruction, seed):
        policy.seed = seed
        policy.policy_guard_state = None
        torch.manual_seed(seed)
        np.random.seed(seed)
        return policy._infer_action_chunk(obs, instruction)

    def normalize(raw):
        return (torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0) * norm.scale + norm.offset)[0].numpy()

    for state in states:
        if sha256(state["file"]) != state["sha256"]:
            raise ValueError("Fixed state file changed.")
        with np.load(state["file"], allow_pickle=False) as f:
            arrays = dict(f)
        if observation_hash(arrays) != state["observation_sha256"]:
            raise ValueError("Fixed observation changed.")
        sid = state["id"]
        observations[sid] = {"joint_action": {"vector": arrays["state"]}, "observation": {
            k: {"rgb": arrays[k]} for k in CAMERAS}}
        refs[sid] = {k: norm.forward(torch.as_tensor(arrays[k + "_reference"], dtype=torch.float32).unsqueeze(0))[0].numpy()
                     for k in ("source", "target")}
        inputs[sid] = {}
        for k, field in (("source", "source_instruction"), ("target", "counterfactual_instruction")):
            raw, captured, calls = capture_inputs(model, lambda: predict_raw(observations[sid], state[field], 42))
            with torch.no_grad():
                cache = build_cache(model, move_cache(captured, model.device))
                prediction = sample_cached_actions(model, cache, 42, 10)
            if calls != 10 or difference(normalize(raw), prediction)["max_abs"] > 1e-5:
                raise ValueError("Fresh rebuilt cache does not replay production.")
            inputs[sid][k] = captured
            del cache, captured
        print(f"[inputs] {args.arm} {sid}", flush=True)
    optimizer = torch.optim.AdamW(list(selected.values()), lr=plan["learning_rate"], weight_decay=0.)
    scores = []

    def evaluate(step, stage="train"):
        rows, replays = [], []
        for state in states:
            sid = state["id"]
            with torch.no_grad():
                caches = {k: build_cache(model, move_cache(v, model.device)) for k, v in inputs[sid].items()}
                for seed in plan["eval_seeds"]:
                    values = {k: sample_cached_actions(model, cache, seed, 10) for k, cache in caches.items()}
                    if step == 0:
                        baseline[sid, seed] = values
                    old = baseline[sid, seed]
                    np.savez_compressed(root / f"{stage}{step:06d}_{sid}_seed{seed}.npz", **values,
                                        source_reference=refs[sid]["source"], target_reference=refs[sid]["target"])
                    for row in action_rows(values["source"], values["target"], refs[sid]["source"], refs[sid]["target"], old["source"], old["target"]):
                        h = row["horizon"]
                        parts = paired_components(values["source"][:h], values["target"][:h], refs[sid]["source"][:h], refs[sid]["target"][:h], old["source"][:h], old["target"][:h])
                        rows.append({"id": sid, "split": state["split"], "arm": args.arm, "step": step,
                                     "seed": seed, "stage": stage, **row, **parts})
                    if seed == 42:
                        for k, field in (("source", "source_instruction"), ("target", "counterfactual_instruction")):
                            current = normalize(predict_raw(observations[sid], state[field], seed))
                            error = difference(current, values[k])["max_abs"]
                            if error > 1e-5:
                                raise ValueError(f"Production replay mismatch {sid}/{k}: {error}")
                            replays.append({"id": sid, "language": k, "max_abs": error})
                del caches
        score = checkpoint_score(rows)
        score.update(step=step, stage=stage)
        scores.append(score)
        write_json(root / f"evaluation_{stage}_{step:06d}.json", {"rows": rows, "score": score, "production_replays": replays})
        print(f"[eval] {args.arm} {stage} step={step} fit={score['fit_both_correct']}/{score['fit_rows']} guard={score['guard_pass']} rmse={score['fit_mean_correct_rmse']:.6f}", flush=True)
        return score

    def save(step):
        payload = {"format": "fastwam_lora_adapter_v1", "mot_trainable": model._lora_adapter_state_dict(),
                   "lora_config": model.lora_config, "base_checkpoint": model.lora_base_checkpoint,
                   "step": 1000 + step, "torch_dtype": str(model.torch_dtype),
                   "joint_adapter_repair": {"arm": args.arm, "optimizer_steps": step, "plan": str(Path(args.output) / "plan.json")}}
        path = root / f"step_{step:06d}.pt"
        torch.save(payload, path)
        return path

    evaluate(0)
    fit = [s for s in states if s["split"] == "repair"]
    with (root / "training.jsonl").open("x") as log:
        for step in range(1, plan["steps"] + 1):
            optimizer.zero_grad(set_to_none=True)
            draws, terms = [], []
            for i, state in enumerate(fit):
                sid, seed = state["id"], plan["train_seed"] + step * len(fit) + i
                captured = move_cache(inputs[sid], model.device)
                targets = {k: torch.as_tensor(v, device=model.device, dtype=model.torch_dtype).unsqueeze(0) for k, v in refs[sid].items()}
                noise = noise_tensor((1, 32, 14), seed, model)
                u = torch.rand((1,), generator=torch.Generator(device="cpu").manual_seed(1_000_000_000 + seed))
                scheduler = model.train_action_scheduler
                time = (scheduler._phi(u, scheduler.shift) * scheduler.num_train_timesteps).to(device=model.device, dtype=model.torch_dtype)
                weight = float(scheduler.training_weight(time).item())
                terms.append(paired_backward(model, captured, targets, noise, time, weight / len(fit),
                                             plan["anchor_weight"] / len(fit), plan["conditional_gain"]))
                draws.append({"id": sid, "seed": seed, "time": float(time.item()), "weight": weight,
                              "noise_sha256": typed_hash(noise[0].float().cpu().numpy())})
                del captured, targets, noise
            grad_by_expert = {}
            zero_weight_step = plan["anchor_weight"] == 0 and all(d["weight"] == 0 for d in draws)
            for expert in ARMS[args.arm]:
                grads = [p.grad for n, p in selected.items() if n.split('.')[1] == expert and p.grad is not None]
                if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                    raise ValueError("Missing/nonfinite expert gradients.")
                grad_by_expert[expert] = float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt())
                if grad_by_expert[expert] == 0 and not zero_weight_step:
                    raise ValueError("Zero expert gradient.")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(list(selected.values()), 1., error_if_nonfinite=True))
            optimizer.step()
            audit_frozen(model, protected)
            log.write(json.dumps({"step": step, "draws": draws, "terms": terms, "gradient_norm": grad_norm,
                                  "gradient_by_expert": grad_by_expert, "zero_weight_step": zero_weight_step}, allow_nan=False) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0:
                print(f"[train] {args.arm} step={step}/{plan['steps']} grad={grad_by_expert}", flush=True)
            if step % plan["eval_every"] == 0 or step == plan["steps"]:
                evaluate(step)
                save(step)
    audit_frozen(model, protected)
    # Explicit precision ablation, NOT the normal loader: it injects fp32 LoRA
    # parameters even when the frozen backbone and saved source weights are bf16.
    # This intentional final conversion is excluded from the frozen training audit.
    del optimizer
    model.zero_grad(set_to_none=True)
    for n, p in model.mot.named_parameters():
        if n.endswith((".lora_A", ".lora_B")):
            p.data = p.data.to(model.torch_dtype)
    evaluate(plan["steps"], "forced_bf16")
    write_json(root / "complete.json", {"complete": True, "arm": args.arm, "steps": plan["steps"],
               "scores": scores, "frozen_training_parameters_unchanged": True,
               "all_evaluations_replayed_production": True})


def main():
    from scripts.probe_robotwin_no_eraf import read_json, write_json, sha256
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["run", "worker"])
    ap.add_argument("--source-probe", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    args = ap.parse_args()
    if args.mode == "worker":
        return worker(args)
    if args.steps <= 0 or args.eval_every <= 0 or not 0 < args.learning_rate <= 1e-3:
        ap.error("Invalid training limits.")
    source = Path(args.source_probe).resolve()
    if read_json(source / "summary.json").get("complete") is not True:
        raise ValueError("Source probe incomplete.")
    states = [s for s in read_json(source / "states.json") if s["frame_index"] == 0]
    if len(states) != 10 or not all(s["dual_reference_valid"] and s["initial_observations_exactly_equal"] for s in states):
        raise ValueError("Require ten audited initial states.")
    for state in states:
        state["split"] = "repair" if state["profile"] == "historical" and state["pair_id"] == "stack_blocks_two_green_on_red_to_red_on_green" else "guard"
    fit = [s for s in states if s["split"] == "repair"]
    if len(fit) != 1 or not fit[0]["own_reference_in_training_split"]:
        raise ValueError("Require one historical training state.")
    pair = next(p for p in read_json(source / "plan.json")["pairs"] if p["profile"] == fit[0]["profile"] and p["pair_id"] == fit[0]["pair_id"] and p["episode_index"] == fit[0]["episode_index"])
    if not pair["native_in_training_split"] or not pair["counterfactual_in_training_split"]:
        raise ValueError("Both expert references must be in the original training split.")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = {"format": "robotwin_joint_adapter_repair_v1", "source_probe": str(source), "states": states,
            "source_artifact_sha256": {str(source / p): sha256(source / p) for p in ["plan.json", "summary.json", "states.json"]},
            "steps": args.steps, "eval_every": args.eval_every, "train_seed": 17000, "eval_seeds": [42, 43, 44],
            "learning_rate": args.learning_rate, "anchor_weight": .25, "conditional_gain": 4., "arms": ARMS,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "scope": "One training initial state, nine regression states; matched scope intervention, no Video loss or original four-pool sampler. Not closed-loop success or held-out-scene generalization."}
    write_json(root / "plan.json", plan)
    workers = []
    for gpu, arm in enumerate(ARMS):
        log = (root / f"{arm}.log").open("x")
        command = [sys.executable, "-u", __file__, "worker", "--source-probe", str(source), "--output", str(root), "--arm", arm, "--gpu", str(gpu)]
        p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        workers.append((p, log, arm))
        print(f"[start] {arm} gpu={gpu} pid={p.pid}", flush=True)
    for p, log, arm in workers:
        rc = p.wait(); log.close()
        print(f"[exit] {arm} code={rc}", flush=True)
    if any(p.returncode for p, _, _ in workers):
        raise RuntimeError("Failed workers; partial results not summarized.")
    completions = {arm: read_json(root / arm / "complete.json") for arm in ARMS}
    logs = {arm: [json.loads(line) for line in (root / arm / "training.jsonl").read_text().splitlines()] for arm in ARMS}
    if any([r["step"] for r in data] != list(range(1, args.steps + 1)) for data in logs.values()):
        raise ValueError("Missing/duplicate optimizer steps.")
    if [r["draws"] for r in logs["action"]] != [r["draws"] for r in logs["joint"]]:
        raise ValueError("Training draws differ.")
    a, b = [read_json(root / arm / "evaluation_train_000000.json")["rows"] for arm in ARMS]
    keys = ["id", "seed", "horizon", "source_correct_rmse", "target_correct_rmse", "delta_projection"]
    if [[r[k] for k in keys] for r in a] != [[r[k] for k in keys] for r in b]:
        raise ValueError("Initial predictions differ between arms.")
    write_json(root / "summary.json", {"format": plan["format"], "complete": True, "arms": completions,
               "matched_training_draws": True, "identical_initial_evaluations": True, "scope": plan["scope"]})
    print("[complete]", root / "summary.json", flush=True)


if __name__ == "__main__":
    main()
