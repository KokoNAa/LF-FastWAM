#!/usr/bin/env python3
"""Probe noisy expert actions using captured production observation caches.

No training, gradients, future demonstration video, or simulator. By default
run only prints a plan; --execute loads Base/step500/step1000 on listed GPUs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.probe_robotwin_no_eraf import load_probe_policy, read_json, sha256, write_json

DEFAULT_PAIRS = ["stack_blocks_two_green_on_red_to_red_on_green", "blocks_ranking_rgb_to_bgr"]


def resolve_source_probe(value, search_root=None):
    if value != "latest":
        return Path(value).expanduser().resolve()
    search_root = Path(search_root) if search_root is not None else REPO / "runs/robotwin_same_state_probe"
    candidates = []
    for path in search_root.glob("*/summary.json"):
        try:
            data = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if (data.get("complete") is True and data.get("format") == "robotwin_no_eraf_same_state_probe_v1"
                and all((path.parent / name).is_file() for name in ("plan.json", "states.json"))):
            candidates.append(path)
    if not candidates:
        raise ValueError(f"No completed same-state probe under {search_root}; specify its absolute path.")
    return max(candidates, key=lambda path: path.stat().st_mtime).parent.resolve()


def build_plan(args):
    root = resolve_source_probe(args.source_probe)
    source = read_json(root / "plan.json")
    if source.get("format") != "robotwin_no_eraf_same_state_probe_v1":
        raise ValueError("Expected an audited same-state probe plan.")
    if not read_json(root / "summary.json").get("complete"):
        raise ValueError("Source probe has not completed.")
    all_states = read_json(root / "states.json")
    states = [state for state in all_states if state["profile"] == "historical"
              and state["frame_index"] == 0 and state["dual_reference_valid"]
              and state["pair_id"] in args.pairs]
    if len(states) != len(args.pairs) or {s["pair_id"] for s in states} != set(args.pairs):
        raise ValueError("Need exactly one matched historical initial state per requested pair.")
    models = list(source["checkpoints"])
    if models != ["base", "step500", "step1000"]:
        raise ValueError("This diagnostic compares the completed Base/step500/step1000 run.")
    saved = {}
    for model in models:
        if read_json(root / model / "complete.json")["states"] != len(all_states):
            raise ValueError(f"Incomplete source model: {model}")
        records = [json.loads(line) for line in (root / model / "records.jsonl").read_text().splitlines()]
        if len(records) != len(all_states) or {r["id"] for r in records} != {s["id"] for s in all_states}:
            raise ValueError(f"Incomplete/duplicate source records: {model}")
        by_id = {r["id"]: r for r in records}
        saved[model] = {}
        for state in states:
            row = by_id[state["id"]]
            if row["observation_sha256"] != state["observation_sha256"] or not row["dual_reference_valid"]:
                raise ValueError(f"Source observation mismatch: {model}/{state['id']}")
            saved[model][state["id"]] = {"path": row["actions_file"], "sha256": sha256(row["actions_file"])}
    return {"format": "robotwin_deployment_cache_denoising_v1", "source_probe": str(root), "source_plan": source,
            "output": str(Path(args.output).expanduser().resolve()), "states": states,
            "models": models, "saved_predictions": saved, "gpus": args.gpus,
            "sigmas": args.sigmas, "noise_seeds": args.noise_seeds,
            "interpretation": "Oracle-noised action reconstruction under production observation caches. This is not full training loss, rollout success, or a sigma<1 goal-selection test."}


def worker(args):
    plan = read_json(args.plan)
    root = Path(plan["output"]) / args.model
    root.mkdir(exist_ok=False)
    policy, audit = load_probe_policy(plan["source_plan"], args.model, args.gpu)
    import numpy as np
    import torch
    from experiments.robotwin.denoising_probe import capture_action_cache, denoising_metrics
    from experiments.robotwin.no_eraf_probe import CAMERAS, difference, observation_hash, typed_hash

    write_json(root / "checkpoint_audit.json", audit)
    model = policy.model
    scheduler = model.train_action_scheduler
    key = policy.processor.shape_meta["action"][0]["key"]
    normalizer = policy.processor.normalizer.normalizers["action"][key]

    def to_numpy(value):
        return value.detach().float().cpu().numpy()[0]

    count = 0
    with (root / "records.jsonl").open("x") as log, torch.no_grad():
        for state in plan["states"]:
            if sha256(state["file"]) != state["sha256"]:
                raise ValueError("Fixed state changed.")
            with np.load(state["file"], allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            if observation_hash(arrays) != state["observation_sha256"]:
                raise ValueError("Fixed observation fingerprint changed.")
            saved_path = plan["saved_predictions"][args.model][state["id"]]
            if sha256(saved_path["path"]) != saved_path["sha256"]:
                raise ValueError("Prior predictions changed after planning.")
            with np.load(saved_path["path"], allow_pickle=False) as data:
                saved = {key: data[key] for key in data.files}
            for kind in ("source", "target"):
                reference = normalizer.forward(torch.as_tensor(arrays[kind + "_reference"]).unsqueeze(0))
                if difference(to_numpy(reference), saved[kind + "_reference"])["max_abs"] > 1e-5:
                    raise ValueError("Expert normalization differs from the original probe.")
            obs = {"joint_action": {"vector": arrays["state"]}, "observation": {
                camera: {"rgb": arrays[camera]} for camera in CAMERAS}}
            caches, replay = {}, {}
            for language, instruction in (("source", state["source_instruction"]),
                                          ("target", state["counterfactual_instruction"])):
                torch.manual_seed(policy.seed)
                np.random.seed(policy.seed)
                policy.policy_guard_state = None
                raw, cache, calls = capture_action_cache(model, lambda: policy._infer_action_chunk(obs, instruction))
                normalized = torch.as_tensor(raw).unsqueeze(0) * normalizer.scale + normalizer.offset
                error = difference(to_numpy(normalized), saved[language])
                if calls != policy.num_inference_steps or error["max_abs"] > 1e-5:
                    write_json(root / "replay_failure.json", {"id": state["id"], "language": language,
                               "predictor_calls": calls, "error": error})
                    raise ValueError("Captured production inference differs from the original saved prediction.")
                caches[language] = cache
                replay[language] = {"error": error, "predictor_calls": calls,
                                    "video_seq_len": cache["video_seq_len"]}
            write_json(root / f"{state['id']}_replay.json", replay)
            for kind in ("source", "target"):
                clean = torch.as_tensor(saved[kind + "_reference"], device=model.device,
                                        dtype=model.torch_dtype).unsqueeze(0)
                wrong_language = "target" if kind == "source" else "source"
                for sigma in plan["sigmas"]:
                    timestep = torch.tensor([sigma * scheduler.num_train_timesteps], device=model.device,
                                            dtype=model.torch_dtype)
                    effective_sigma = float(timestep.float().item()) / scheduler.num_train_timesteps
                    for seed in plan["noise_seeds"]:
                        noise = torch.randn(clean.shape, generator=torch.Generator(device="cpu").manual_seed(seed),
                                            dtype=torch.float32).to(device=model.device, dtype=model.torch_dtype)
                        noisy = scheduler.add_noise(clean, noise, timestep)
                        velocity_target = scheduler.training_target(clean, noise, timestep)
                        correct = model._predict_action_noise_with_cache(latents_action=noisy, timestep_action=timestep,
                                                                         **caches[kind])
                        wrong = model._predict_action_noise_with_cache(latents_action=noisy, timestep_action=timestep,
                                                                       **caches[wrong_language])
                        metrics = denoising_metrics(to_numpy(clean), to_numpy(noisy), to_numpy(velocity_target),
                                                    to_numpy(correct), to_numpy(wrong), effective_sigma)
                        record = {"id": state["id"], "pair_id": state["pair_id"], "model": args.model,
                                  "reference": kind, "sigma": sigma, "effective_sigma": effective_sigma,
                                  "noise_seed": seed, "noisy_action_sha256": typed_hash(to_numpy(noisy)),
                                  "training_weight": float(scheduler.training_weight(timestep).float().item()),
                                  "metrics": metrics}
                        log.write(json.dumps(record, allow_nan=False) + "\n")
                        log.flush()
                        count += 1
                    print(f"[denoise] {args.model} {state['pair_id']} {kind} sigma={sigma}", flush=True)
            del caches
    write_json(root / "complete.json", {"records": count})


def summarize(output):
    from statistics import fmean
    root = Path(output)
    plan = read_json(root / "plan.json")
    expected = {(state["id"], kind, sigma, seed) for state in plan["states"] for kind in ("source", "target")
                for sigma in plan["sigmas"] for seed in plan["noise_seeds"]}
    groups = {}
    input_hashes = {}
    for model in plan["models"]:
        if read_json(root / model / "complete.json")["records"] != len(expected):
            raise ValueError(f"Incomplete denoising worker: {model}")
        rows = [json.loads(line) for line in (root / model / "records.jsonl").read_text().splitlines()]
        keys = [(r["id"], r["reference"], r["sigma"], r["noise_seed"]) for r in rows]
        if len(rows) != len(expected) or set(keys) != expected:
            raise ValueError(f"Incomplete/duplicated denoising rows: {model}")
        for row, key in zip(rows, keys):
            # At sigma=1 both references must receive exactly the same noise.
            hash_key = (key[0], "pure_noise" if row["sigma"] == 1 else key[1], key[2], key[3])
            previous = input_hashes.setdefault(hash_key, row["noisy_action_sha256"])
            if previous != row["noisy_action_sha256"]:
                raise ValueError("Noisy actions differ across models or pure-noise references.")
            if [m["horizon"] for m in row["metrics"]] != [24, 32]:
                raise ValueError("Missing 24/32-step metrics.")
            for metrics in row["metrics"]:
                group = (model, row["pair_id"], row["reference"], row["sigma"], metrics["horizon"])
                groups.setdefault(group, []).append(metrics)
    summary = []
    for key, metrics in sorted(groups.items()):
        row = dict(zip(("model", "pair_id", "reference", "sigma", "horizon"), key))
        row["noise_draws"] = len(metrics)
        for field in metrics[0]:
            if field != "horizon":
                row[field] = fmean(m[field] for m in metrics)
        row["correct_flow_rmse_min"] = min(m["correct_flow_rmse"] for m in metrics)
        row["correct_flow_rmse_max"] = max(m["correct_flow_rmse"] for m in metrics)
        summary.append(row)
    with (root / "denoising_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_json(root / "summary.json", {"format": plan["format"], "complete": True,
               "rows": summary, "interpretation": plan["interpretation"]})
    print(f"[complete] {root / 'denoising_summary.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--source-probe", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    run.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    run.add_argument("--sigmas", type=float, nargs="+", default=[.1, .5, .9, 1.])
    run.add_argument("--noise-seeds", type=int, nargs="+", default=[42, 43, 44])
    run.add_argument("--execute", action="store_true")
    work = sub.add_parser("worker")
    work.add_argument("--plan", required=True)
    work.add_argument("--model", required=True)
    work.add_argument("--gpu", type=int, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("output")
    args = parser.parse_args()
    if args.command == "worker":
        return worker(args)
    if args.command == "summarize":
        return summarize(args.output)
    if (any(len(set(values)) != len(values) for values in (args.gpus, args.pairs, args.sigmas, args.noise_seeds))
            or min(args.gpus) < 0 or min(args.noise_seeds) < 0 or any(not 0 < sigma <= 1 for sigma in args.sigmas)):
        parser.error("Invalid/duplicate GPUs, pairs, sigmas or seeds.")
    plan = build_plan(args)
    root = Path(plan["output"])
    if root.exists():
        raise FileExistsError(f"Use a fresh output directory: {root}")
    print(json.dumps({"source_probe": plan["source_probe"], "output": str(root), "states": [s["id"] for s in plan["states"]],
                      "models": plan["models"], "sigmas": plan["sigmas"], "noise_seeds": plan["noise_seeds"]}, indent=2), flush=True)
    if not args.execute:
        print("PLAN ONLY. Add --execute to run GPU inference.")
        return
    root.mkdir(parents=True)
    plan["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    write_json(root / "plan.json", plan)

    def gpu_queue(index):
        codes = []
        for model in plan["models"][index::len(args.gpus)]:
            print(f"[start] {model} gpu={args.gpus[index]}", flush=True)
            log_path = root / f"{model}.log"
            with log_path.open("x") as log:
                code = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--plan",
                    str(root / "plan.json"), "--model", model, "--gpu", str(args.gpus[index])],
                    cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"[exit] {model} code={code}", flush=True)
            codes.append(code)
            if code:
                print("\n".join(log_path.read_text(errors="replace").splitlines()[-45:]), flush=True)
                break
        return codes

    with ThreadPoolExecutor(max_workers=min(len(plan["models"]), len(args.gpus))) as pool:
        codes = list(pool.map(gpu_queue, range(min(len(plan["models"]), len(args.gpus)))))
    if any(code for values in codes for code in values):
        raise SystemExit("Denoising probe failed; worker error tails are printed above.")
    summarize(root)


if __name__ == "__main__":
    main()
