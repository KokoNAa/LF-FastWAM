#!/usr/bin/env python3
"""Factorial Video/Action adapter ablation on fixed RoboTwin observations.

No training. Full and zero-adapter controls must replay saved step1000 and
Base actions at the original seed. Hybrids isolate each adapter's functional
contribution; they are not independently trained models or success rates.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
VARIANTS = {"base_zero": (), "full": ("video", "action"),
            "video_only": ("video",), "action_only": ("action",)}


def select_adapters(model, saved, active):
    import torch
    if not set(active) <= {"video", "action"}:
        raise ValueError("Unknown adapter expert.")
    with torch.no_grad():
        current = dict(model.mot.named_parameters())
        for name, original in saved.items():
            expert = name.split(".")[1]
            current[name].copy_(original if expert in active else torch.zeros_like(original))


def worker(args):
    from scripts.probe_robotwin_no_eraf import load_probe_policy, read_json, sha256, write_json
    source = Path(args.source_probe).resolve()
    plan = read_json(source / "plan.json")
    all_states = [s for s in read_json(source / "states.json") if s["frame_index"] == 0]
    if len(all_states) != 10 or not all(s["dual_reference_valid"] for s in all_states):
        raise ValueError("Expected ten audited paired initial states.")
    root = Path(args.output).resolve() / f"shard{args.shard}"
    root.mkdir(parents=True, exist_ok=False)
    policy, audit = load_probe_policy(plan, "step1000", args.gpu)
    import numpy as np
    import torch
    from experiments.robotwin.no_eraf_probe import CAMERAS, observation_hash, difference
    from experiments.robotwin.same_state_repair import action_rows
    from scripts.inspect_robotwin_same_state_repair import paired_components

    model = policy.model
    model.eval()
    saved = {n: p.detach().clone() for n, p in model.mot.named_parameters()
             if n.startswith(("mixtures.video.", "mixtures.action.")) and n.endswith(".lora_B")}
    if {n.split(".")[1] for n in saved} != {"video", "action"}:
        raise ValueError("Both adapter groups are required.")
    protected = {n: p._version for n, p in model.mot.named_parameters() if n not in saved}
    normalizer = policy.processor.normalizer.normalizers["action"][policy.processor.shape_meta["action"][0]["key"]]
    write_json(root / "checkpoint_audit.json", audit)
    records, replays = [], []
    for state in all_states[args.shard::args.shards]:
        if sha256(state["file"]) != state["sha256"]:
            raise ValueError("Observation artifact changed.")
        with np.load(state["file"], allow_pickle=False) as data:
            arrays = dict(data)
        if observation_hash(arrays) != state["observation_sha256"]:
            raise ValueError("Observation fingerprint mismatch.")
        obs = {"joint_action": {"vector": arrays["state"]}, "observation": {
            camera: {"rgb": arrays[camera]} for camera in CAMERAS}}
        refs = {k: normalizer.forward(torch.as_tensor(arrays[k + "_reference"]).unsqueeze(0))[0].numpy()
                for k in ("source", "target")}
        originals = {}
        for variant, old_model in (("base_zero", "base"), ("full", "step1000")):
            path = source / old_model / f"{state['id']}.npz"
            with np.load(path, allow_pickle=False) as data:
                originals[variant] = dict(data)
            for k in refs:
                if difference(refs[k], originals[variant][k + "_reference"])["max_abs"] > 1e-5:
                    raise ValueError("Reference normalization mismatch.")
        for seed in args.seeds:
            predictions = {}
            for variant, active in VARIANTS.items():
                select_adapters(model, saved, active)
                values = {}
                for language, field in (("source", "source_instruction"), ("target", "counterfactual_instruction")):
                    policy.seed = seed
                    policy.policy_guard_state = None
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    raw = policy._infer_action_chunk(obs, state[field])
                    values[language] = (torch.as_tensor(raw).unsqueeze(0) * normalizer.scale + normalizer.offset)[0].numpy()
                    if seed == plan["probe_seed"] and variant in originals:
                        error = difference(values[language], originals[variant][language])["max_abs"]
                        replays.append({"id": state["id"], "variant": variant, "language": language, "max_abs": error})
                        if error > 1e-5:
                            write_json(root / "replay_failure.json", replays[-1])
                            raise ValueError("Full/zero adapter control did not replay original model.")
                predictions[variant] = values
                baseline = predictions["base_zero"]
                for row in action_rows(values["source"], values["target"], refs["source"], refs["target"],
                                       baseline["source"], baseline["target"]):
                    h = row["horizon"]
                    parts = paired_components(values["source"][:h], values["target"][:h], refs["source"][:h],
                                              refs["target"][:h], baseline["source"][:h], baseline["target"][:h])
                    records.append({"id": state["id"], "pair_id": state["pair_id"], "profile": state["profile"],
                                    "variant": variant, "seed": seed, **row, **parts})
                np.savez_compressed(root / f"{state['id']}_{seed}_{variant}.npz", **values,
                                    source_reference=refs["source"], target_reference=refs["target"])
            print(f"[attribution] shard={args.shard} {state['id']} seed={seed}", flush=True)
    select_adapters(model, saved, ("video", "action"))
    if protected != {n: p._version for n, p in model.mot.named_parameters() if n not in saved}:
        raise ValueError("A non-ablated tensor changed.")
    write_json(root / "records.json", records)
    write_json(root / "complete.json", {"complete": True, "rows": len(records), "control_replays": replays,
               "source_plan_sha256": sha256(source / "plan.json"), "states_sha256": sha256(source / "states.json"),
               "non_ablated_tensors_unchanged": True, "adapter_B_tensor_count": len(saved)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("run", "worker"))
    ap.add_argument("--source-probe", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=2)
    args = ap.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        ap.error("Require distinct nonnegative GPU indices.")
    if args.mode == "worker":
        return worker(args)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    from scripts.probe_robotwin_no_eraf import read_json, write_json
    source = Path(args.source_probe).resolve()
    if not read_json(source / "summary.json").get("complete"):
        raise ValueError("Source probe incomplete.")
    if read_json(source / "plan.json")["probe_seed"] not in args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Unique seeds must include the original replay seed.")
    workers = []
    for shard, gpu in enumerate(args.gpus):
        log = (root / f"shard{shard}.log").open("x")
        command = [sys.executable, "-u", __file__, "worker", "--source-probe", str(source), "--output", str(root),
                   "--gpu", str(gpu), "--shard", str(shard), "--shards", str(len(args.gpus)),
                   "--seeds", *map(str, args.seeds)]
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        workers.append((proc, log, shard))
        print(f"[start] shard={shard} gpu={gpu} pid={proc.pid}", flush=True)
    failed = []
    for proc, log, shard in workers:
        rc = proc.wait(); log.close()
        print(f"[exit] shard={shard} code={rc}", flush=True)
        if rc:
            failed.append(shard)
    if failed:
        raise RuntimeError(f"Failed workers {failed}; partial results are not summarized.")
    rows = []
    for _, _, shard in workers:
        complete = read_json(root / f"shard{shard}/complete.json")
        shard_rows = read_json(root / f"shard{shard}/records.json")
        if complete["rows"] != len(shard_rows):
            raise ValueError("Incomplete shard.")
        rows.extend(shard_rows)
    expected = {(s["id"], v, seed, h) for s in read_json(source / "states.json") if s["frame_index"] == 0
                for v in VARIANTS for seed in args.seeds for h in (24, 32)}
    keys = [(r["id"], r["variant"], r["seed"], r["horizon"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Missing or duplicated factorial observations.")
    write_json(root / "summary.json", {"format": "robotwin_adapter_attribution_v1", "complete": True,
               "source_probe": str(source), "variants": VARIANTS, "seeds": args.seeds, "rows": rows,
               "interpretation": "Functional ablations on ten initial observations; no training or closed-loop evaluation."})
    with (root / "attribution.csv").open("x") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print("[complete]", root / "summary.json", flush=True)


if __name__ == "__main__":
    main()
