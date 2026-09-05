#!/usr/bin/env python3
"""Freeze audited production inputs for a balanced multi-task decision replay."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]


def main():
    from scripts.probe_robotwin_no_eraf import (
        build_plan, prepare_states, read_json, write_json, sha256, load_probe_policy,
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-probe", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--train-per-stratum", type=int, default=8)
    ap.add_argument("--holdout-per-stratum", type=int, default=2)
    args = ap.parse_args()
    if args.train_per_stratum < 1 or args.holdout_per_stratum < 1:
        ap.error("Require positive train/holdout counts.")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = read_json(Path(args.source_probe) / "plan.json")
    expert = source["pairs"][0]["native"]["hdf5"].split("/historical/")[0]
    if expert == source["pairs"][0]["native"]["hdf5"]:
        raise ValueError("Cannot locate the historical expert root.")
    selected, plans = [], {}
    for domain in ("demo_clean", "demo_randomized"):
        folder = root / domain
        folder.mkdir()
        options = SimpleNamespace(
            train_run=str(Path(source["train_config"]).parent), base_checkpoint=source["base_checkpoint"],
            stats_path=source["stats_path"], expert_root=expert, profiles=["historical"],
            task_config=domain, pairs=None, episodes_per_pair=args.train_per_stratum + args.holdout_per_stratum,
            steps=[1000], output=str(folder), inference_steps=10, seed=42, fractions=[0.], gpus=[args.gpu])
        plan = build_plan(options)
        if len({p["pair_id"] for p in plan["pairs"]}) != 5:
            raise ValueError("Expected five task pairs in each domain.")
        write_json(folder / "plan.json", plan)
        states = prepare_states(plan)
        write_json(folder / "states.json", states)
        counts = {}
        for state in states:
            if state["frame_index"] != 0:
                continue
            if not state["dual_reference_valid"] or not state["initial_observations_exactly_equal"]:
                raise ValueError("Initial complete observations differ.")
            pair = next(p for p in plan["pairs"] if p["pair_id"] == state["pair_id"] and p["episode_index"] == state["episode_index"])
            if not pair["native_in_training_split"] or not pair["counterfactual_in_training_split"]:
                raise ValueError("Replay reference is outside the original training split.")
            count = counts.get(state["pair_id"], 0)
            counts[state["pair_id"]] = count + 1
            state.update(id=domain + "_" + state["id"], task_config=domain,
                         replay_split="train" if count < args.train_per_stratum else "replay_holdout")
            selected.append(state)
        plans[domain] = plan

    policy, audit = load_probe_policy(source, "step1000", args.gpu)
    import numpy as np
    import torch
    from experiments.robotwin.joint_adapter_repair import capture_inputs, build_cache
    from experiments.robotwin.same_state_repair import move_cache, sample_cached_actions
    from experiments.robotwin.no_eraf_probe import CAMERAS, observation_hash, difference
    model = policy.model
    norm = policy.processor.normalizer.normalizers["action"][policy.processor.shape_meta["action"][0]["key"]]
    (root / "payloads").mkdir()
    for state in selected:
        if sha256(state["file"]) != state["sha256"]:
            raise ValueError("State changed during preparation.")
        with np.load(state["file"], allow_pickle=False) as handle:
            arrays = dict(handle)
        if observation_hash(arrays) != state["observation_sha256"]:
            raise ValueError("Observation hash mismatch.")
        obs = {"joint_action": {"vector": arrays["state"]}, "observation": {c: {"rgb": arrays[c]} for c in CAMERAS}}
        refs = {k: norm.forward(torch.as_tensor(arrays[k + "_reference"], dtype=torch.float32).unsqueeze(0)).cpu()
                for k in ("source", "target")}
        captured, errors = {}, {}
        for language, field in (("source", "source_instruction"), ("target", "counterfactual_instruction")):
            policy.seed = 42
            policy.policy_guard_state = None
            raw, inputs, calls = capture_inputs(model, lambda: policy._infer_action_chunk(obs, state[field]))
            with torch.no_grad():
                cache = build_cache(model, move_cache(inputs, model.device))
                replay = sample_cached_actions(model, cache, 42, 10)
            normalized = norm.forward(torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0))[0].numpy()
            error = difference(normalized, replay)["max_abs"]
            if calls != 10 or error > 1e-5:
                raise ValueError("Frozen inputs cannot reproduce current production inference.")
            errors[language] = error
            captured[language] = inputs
            del cache
        payload = root / "payloads" / (state["id"] + ".pt")
        torch.save({"captured": captured, "references": refs}, payload)
        state.update(payload=str(payload), payload_sha256=sha256(payload), production_replay_errors=errors,
                     expert_separation_rmse=float((refs["source"] - refs["target"]).square().mean().sqrt()))
        print(f"[prepared] {state['id']} split={state['replay_split']} separation={state['expert_separation_rmse']:.6f}", flush=True)
    write_json(root / "manifest.json", {
        "format": "robotwin_decision_replay_bank_v1", "complete": True,
        "source_probe": str(Path(args.source_probe).resolve()), "capture_checkpoint_audit": audit,
        "original_train_config": source["train_config"], "original_train_config_sha256": source["train_config_sha256"],
        "stats_path": source["stats_path"], "stats_sha256": source["stats_sha256"],
        "base_checkpoint": source["base_checkpoint"], "base_checkpoint_sha256": source["checkpoint_sha256"]["base"],
        "train_per_stratum": args.train_per_stratum, "holdout_per_stratum": args.holdout_per_stratum,
        "states": selected,
        "scope": "Five tasks x two domains. Both positive experts belong to the original training split. Replay holdout is NOT a scene holdout from original training. Capture excludes every trainable operation. No full-goal labels."})
    print("[complete]", root / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
