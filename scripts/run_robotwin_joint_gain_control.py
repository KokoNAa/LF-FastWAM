#!/usr/bin/env python3
"""Change only endpoint difference gain in an existing joint-adapter protocol."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from scripts.probe_robotwin_no_eraf import read_json, write_json, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gain", type=float, default=1.)
    args = ap.parse_args()
    reference, root = args.reference_run.resolve(), args.output.resolve()
    source = read_json(reference / "plan.json")
    if source["format"] != "robotwin_joint_adapter_repair_v1" or args.gain < 1 or args.gain >= source["conditional_gain"]:
        raise ValueError("Require a smaller gain >=1 and the audited joint-adapter protocol.")
    if not (reference / "joint/evaluation_train_000000.json").is_file():
        raise ValueError("Reference has no completed initial evaluation.")
    root.mkdir(parents=True, exist_ok=False)
    plan = {**source, "conditional_gain": args.gain, "arms": {"joint": ["video", "action"]},
            "gain_control_reference": str(reference), "reference_gain": source["conditional_gain"],
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
    plan["source_artifact_sha256"] = {**source["source_artifact_sha256"], str(reference / "plan.json"): sha256(reference / "plan.json")}
    write_json(root / "plan.json", plan)
    with (root / "joint.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(REPO / "scripts/train_robotwin_joint_adapter_repair.py"),
            "worker", "--source-probe", plan["source_probe"], "--output", str(root), "--arm", "joint", "--gpu", str(args.gpu)],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        print(f"[start] gain={args.gain} gpu={args.gpu} pid={process.pid}", flush=True)
        rc = process.wait()
    if rc:
        raise RuntimeError(f"Gain-control worker failed: {rc}")
    completed = read_json(root / "joint/complete.json")
    other = read_json(reference / "joint/complete.json")
    if not completed["complete"] or not other["complete"]:
        raise ValueError("Both joint runs must complete before comparison.")
    logs = [[json.loads(line) for line in (p / "joint/training.jsonl").read_text().splitlines()] for p in [root, reference]]
    if any([r["step"] for r in rows] != list(range(1, plan["steps"] + 1)) for rows in logs):
        raise ValueError("Incomplete optimizer steps.")
    if [r["draws"] for r in logs[0]] != [r["draws"] for r in logs[1]]:
        raise ValueError("Training draws differ.")
    fields = ["id", "seed", "horizon", "source_correct_rmse", "target_correct_rmse", "delta_projection"]
    initial = [[[r[k] for k in fields] for r in read_json(p / "joint/evaluation_train_000000.json")["rows"]] for p in [root, reference]]
    if initial[0] != initial[1]:
        raise ValueError("Initial evaluations differ.")
    write_json(root / "summary.json", {"format": "robotwin_joint_gain_control_v1", "complete": True,
               "gain": args.gain, "reference_gain": source["conditional_gain"], "reference_run": str(reference),
               "matched_training_draws": True, "identical_initial_evaluations": True,
               "control": completed, "reference": other})
    print("[complete]", root / "summary.json", flush=True)


if __name__ == "__main__":
    main()
