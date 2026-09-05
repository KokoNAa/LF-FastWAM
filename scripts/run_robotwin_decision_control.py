#!/usr/bin/env python3
"""Run and audit two matched full-training arms on the two existing GPUs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]


def summarize(root):
    from scripts.probe_robotwin_no_eraf import read_json, write_json
    plan = read_json(root / "plan.json")
    runs = {name: read_json(root / name / "complete.json") for name in plan["arms"]}
    if not all(r.get("complete") is True and r["steps"] == plan["steps"] for r in runs.values()):
        raise ValueError("Training arms are incomplete.")
    initials = [read_json(root / name / "decision_replay/initial_rank0.json") for name in plan["arms"]]
    if initials[0] != initials[1]:
        raise ValueError("Initial adapters or runtime scopes differ between arms.")
    logs = {name: [json.loads(line) for line in (root / name / "decision_replay/draws_rank0.jsonl").read_text().splitlines()]
            for name in plan["arms"]}
    expected = list(range(plan["steps"] * 16))
    for rows in logs.values():
        if [r["position"] for r in rows] != expected:
            raise ValueError("Unexpected microstep count/order.")
    keys = ["position", "id", "seed", "sample_sha256", "rng_before", "rng_after_original"]
    first, second = list(logs.values())
    if [[r[k] for k in keys] for r in first] != [[r[k] for k in keys] for r in second]:
        raise ValueError("Original sample or diffusion/dropout random stream changed between arms.")
    if [r["original_loss"] for r in first[:16]] != [r["original_loss"] for r in second[:16]]:
        raise ValueError("Original losses differ before the first optimizer update.")
    result = {"format": "robotwin_full_training_decision_control_v1", "complete": True,
        "runs": runs, "matched_initial_adapters": True, "matched_original_samples_and_rng": True,
        "matched_original_losses_before_first_update": True, "microsteps_per_arm": len(first),
        "scope": "Training completion and control validity only; CF success requires separate closed-loop evaluation."}
    write_json(root / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    from scripts.probe_robotwin_no_eraf import sha256, write_json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["run", "summarize"])
    ap.add_argument("--manifest")
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--base-port", type=int, default=29680)
    args = ap.parse_args()
    root = Path(args.output).resolve()
    if args.mode == "summarize":
        return summarize(root)
    if not args.manifest or args.steps < 1 or args.save_every < 1:
        ap.error("Require manifest and positive training limits.")
    root.mkdir(parents=True, exist_ok=False)
    arms = {"original": 0., "decision_replay": .25}
    plan = {"format": "robotwin_full_training_decision_control_v1", "steps": args.steps,
        "save_every": args.save_every, "arms": arms, "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256(args.manifest), "base_port": args.base_port,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "world_size_per_arm": 1, "gradient_accumulation": 16, "effective_batch": 16,
        "scope": "Both arms retain all original tasks, four-pool data, and Video/Action losses. Treatment adds balanced paired sigma1 positives. No extra conditional gain."}
    write_json(root / "plan.json", plan)
    children = []
    for gpu, (name, weight) in enumerate(arms.items()):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + str(REPO)
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [sys.executable, "-m", "accelerate.commands.launch", "--config_file",
            "scripts/accelerate_configs/accelerate_zero1_ds.yaml", "--num_processes", "1",
            "--main_process_port", str(args.base_port + gpu), "scripts/train_robotwin_decision_replay.py",
            "--manifest", str(Path(args.manifest).resolve()), "--output", str(root / name),
            "--weight", str(weight), "--steps", str(args.steps), "--save-every", str(args.save_every)]
        log = (root / f"{name}.log").open("x")
        child = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append((name, child, log))
        print(f"[start] {name} gpu={gpu} pid={child.pid}", flush=True)
    for name, child, log in children:
        code = child.wait()
        log.close()
        print(f"[exit] {name} code={code}", flush=True)
    if any(child.returncode for _, child, _ in children):
        raise RuntimeError("One or more training arms failed. Partial results are not compared.")
    summarize(root)


if __name__ == "__main__":
    main()
