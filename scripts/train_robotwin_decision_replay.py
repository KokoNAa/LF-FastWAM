#!/usr/bin/env python3
"""Original full multi-task trainer with one explicit decision-replay addition.

Launch each arm with accelerate/ZeRO1, one GPU and accumulation16. Both arms
start at the same released Base, seed, four-pool samples, and full objectives.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]


def main():
    from scripts.probe_robotwin_no_eraf import read_json, sha256, write_json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weight", type=float, choices=[0., .25], required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=250)
    args = ap.parse_args()
    if args.steps < 1 or args.save_every < 1:
        ap.error("Positive step limits required.")
    import os
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Matched control launcher requires one GPU per arm and accumulation16.")
    manifest = read_json(args.manifest)
    config_path = manifest["original_train_config"]
    if sha256(config_path) != manifest["original_train_config_sha256"]:
        raise ValueError("Original training config changed.")
    for path, expected in ((manifest["base_checkpoint"], manifest["base_checkpoint_sha256"]),
                           (manifest["stats_path"], manifest["stats_sha256"])):
        if sha256(path) != expected:
            raise ValueError("Original Base or statistics changed.")
    from omegaconf import OmegaConf, open_dict
    from fastwam.utils.config_resolvers import register_default_resolvers
    from fastwam.utils.pytorch_utils import set_global_seed
    from fastwam.runtime import run_training
    register_default_resolvers()
    cfg = OmegaConf.load(config_path)
    original = OmegaConf.to_container(cfg, resolve=True)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    if (cfg.data.train._target_ != "fastwam.datasets.lerobot.robotwin_no_eraf_dataset.RoboTwinNoERAFFourPoolDataset"
            or cfg.model._target_ != "fastwam.runtime.create_fastwam"
            or cfg.resume != manifest["base_checkpoint"] or int(cfg.batch_size) != 1
            or cfg.model.policy_guard.enabled or cfg.model.transition_contract.enabled
            or cfg.model.lora.extra_trainable_patterns):
        raise ValueError("Require the original Base/no-ERAF/shared-LoRA four-pool experiment.")
    with open_dict(cfg):
        cfg.output_dir = str(root)
        cfg.max_steps = args.steps
        cfg.save_every = args.save_every
        cfg.log_every = 1 if args.steps <= 2 else 10
        cfg.gradient_accumulation_steps = 16
        cfg.model._target_ = "experiments.robotwin.decision_replay.create_fastwam"
        cfg.model.decision_replay = {"manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": sha256(args.manifest), "weight": args.weight,
            "seed": 27000, "log_dir": str(root / "decision_replay")}
    changed = OmegaConf.to_container(cfg, resolve=True)
    for key in original:
        if key not in {"output_dir", "max_steps", "save_every", "log_every", "gradient_accumulation_steps", "model"} and original[key] != changed[key]:
            raise ValueError(f"Unintended change: {key}")
    original_model = dict(original["model"])
    changed_model = dict(changed["model"])
    changed_model.pop("decision_replay")
    changed_model["_target_"] = original_model["_target_"]
    if changed_model != original_model:
        raise ValueError("Original model or loss configuration changed.")
    write_json(root / "intervention.json", {"original_config": config_path,
        "original_config_sha256": sha256(config_path), "weight": args.weight,
        "original_data_and_losses_unchanged": True, "global_batch": 16,
        "new_optimizer_steps": args.steps, "initialization_seed_set_before_model_creation": int(cfg.seed),
        "scope": "Matched fresh training from Base, not an optimizer-state continuation of the old run."})
    # The original trainer seeds AFTER model creation. Set it before creation
    # too, so the two fresh runs receive exactly identical initial LoRA A.
    set_global_seed(int(cfg.seed))
    run_training(cfg)
    checkpoint = root / "checkpoints/weights" / f"step_{args.steps:06d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    write_json(root / "complete.json", {"complete": True, "steps": args.steps,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)})


if __name__ == "__main__":
    main()
