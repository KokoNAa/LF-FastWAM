#!/usr/bin/env python3
"""Run two matched V9.30 CF loss ablations, each starting from V9.28 step10000.

Clone the actual resolved CONTROL config, not a newly reconstructed data recipe.
No dataset is filtered, no checkpoint is overwritten, and no eval is auto-started.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from omegaconf import OmegaConf

from fastwam.utils.cf_ablation import MODES, CONTRACT, checkpoint_mode, validate_mode
from fastwam.utils.config_resolvers import register_default_resolvers

# Running this file directly puts scripts/ on sys.path. Tests may import it as
# scripts.train_libero_eraf_cf_ablation instead.
try:
    from scripts.train_libero_eraf_safe_gain_v930_from_config import launch_spec
except ModuleNotFoundError:
    from train_libero_eraf_safe_gain_v930_from_config import launch_spec


DEFAULT_MODES = ("mask_lift_corrective", "mask_corrective_ranking")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_control(config_path):
    register_default_resolvers()
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))
    eraf = cfg.model.policy_guard.entity_relation_grounding
    if (
        eraf.grounding_objective_version != 30
        or not eraf.safe_gain_training
        or eraf.get("cf_ablation", "none") != "none"
    ):
        raise ValueError("Source must be the unablated V9.30 run-root/config.yaml.")
    if (
        not cfg.get("resume")
        or cfg.get("weight_only_start_step") not in (None, 0)
        or not 1 <= int(cfg.max_steps) <= 500
        or eraf.safe_gain_injector_training_steps != cfg.max_steps
        or eraf.safe_gain_gate_calibration_steps != 0
        or not cfg.data.train.pgc_v9_safe_gain_counterfactual_replay
    ):
        raise ValueError("Expected a fresh short V9.30 injector-only control with its V9.28 resume.")
    return cfg


def training_config(control, mode, output_dir):
    """Only three experiment fields differ; all training hyperparameters remain."""
    validate_mode(mode)
    cfg = OmegaConf.create(OmegaConf.to_container(control, resolve=True))
    cfg.model.policy_guard.entity_relation_grounding.cf_ablation = mode
    cfg.output_dir = str(Path(output_dir).resolve())
    cfg.wandb.name = f"v930_cf_{mode}"
    # Resolved run configs omit Hydra's own operational settings. Match train.yaml.
    cfg.hydra = {"job": {"chdir": False}, "run": {"dir": "."}, "output_subdir": None}
    return cfg


def training_command(run_dir, gpus, cfg):
    return [
        "bash", "scripts/train_zero1.sh", str(gpus),
        # train_zero1 prepends overrides. Keep ALL overrides contiguous before
        # option flags: Hydra/argparse rejects a second overrides group after
        # --config-path when the first group has already started.
        f"output_dir={cfg.output_dir}", f"wandb.name={cfg.wandb.name}",
        "--config-path", str(Path(run_dir).resolve()),
        "--config-name", "ablation_train",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="actual completed V9.30 CONTROL run-root/config.yaml")
    parser.add_argument("gpus", type=int, help="same GPU count as the control training, not eval")
    parser.add_argument("run_tag", help="new unique directory name, no slashes")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(DEFAULT_MODES))
    parser.add_argument("--dry-run", action="store_true", help="inspect config/commands only; no files, CUDA or jobs")
    args = parser.parse_args()
    if args.gpus < 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_tag):
        parser.error("Use a positive GPU count and a plain unique run_tag.")
    if len(set(args.modes)) != len(args.modes):
        parser.error("Each mode may run only once.")
    if os.environ.get("NNODES", "1") != "1":
        parser.error("This matched experiment runner supports one machine only.")
    source = args.config.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    control = load_control(source)
    root = repo / "runs/libero_eraf_cf_ablation" / args.run_tag
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {root}")
    commands = {}
    configs = {}
    for mode in args.modes:
        run_dir = root / mode
        configs[mode] = training_config(control, mode, run_dir)
        commands[mode] = training_command(run_dir, args.gpus, configs[mode])
    plan = {
        "contract": CONTRACT,
        "source_config": str(source),
        "source_config_sha256": file_sha256(source),
        "warm_checkpoint": str(control.resume),
        "output_root": str(root), "modes": args.modes,
        "gpus": args.gpus, "seed": int(control.seed),
        "steps_per_run": int(control.max_steps),
        "learning_rate": float(control.learning_rate),
        "effective_batch_size": args.gpus * int(control.batch_size) * int(control.gradient_accumulation_steps),
        "commands": commands,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("DRY_RUN: config inspection only; checkpoint/data/CUDA preflight NOT run.", flush=True)
        return

    # Reuse the existing sidecar/workspace/action-convention checks, but do not
    # let the old shell reconstruct the actual experiment's hyperparameters.
    preflight, env = launch_spec(source, args.gpus, os.environ, source_objective=30)
    env["ERAF_SAFE_GAIN_PREFLIGHT_ONLY"] = "1"
    env["PYTHON_BIN"] = sys.executable
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["ERAF_SAFE_GAIN_MAX_STEPS"] = str(control.max_steps)
    subprocess.run(preflight, env=env, check=True)
    env.pop("ERAF_SAFE_GAIN_PREFLIGHT_ONLY")
    from fastwam.datasets.pgc_libero import load_pgc_closed_loop_corrective_index

    dataset = Path(control.data.train.pgc_closed_loop_corrective_dataset_dirs[0])
    records = load_pgc_closed_loop_corrective_index(dataset)
    counts = Counter(r["verification_kind"] for r in records.values())
    if "mask_lift_corrective" in args.modes and not counts["target_lift"]:
        raise ValueError("Ablation A requires actual audited target_lift corrective episodes.")
    index_path = dataset / "meta/pgc_v8_closed_loop/index.json"
    plan["corrective_verification_counts"] = dict(counts)
    plan["corrective_index_sha256"] = file_sha256(index_path)
    plan["warm_checkpoint_sha256"] = file_sha256(control.resume)
    root.mkdir(parents=True, exist_ok=False)
    (root / "experiment.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"ROOT={root}\nVERIFICATION_COUNTS={dict(counts)}", flush=True)

    for mode in args.modes:
        if (
            file_sha256(control.resume) != plan["warm_checkpoint_sha256"]
            or file_sha256(index_path) != plan["corrective_index_sha256"]
        ):
            raise RuntimeError("Warm checkpoint/corrective index changed between experiment arms.")
        run_dir = root / mode
        run_dir.mkdir(exist_ok=False)
        OmegaConf.save(configs[mode], run_dir / "ablation_train.yaml")
        print(f"[TRAIN_START] mode={mode} warm={control.resume} output={run_dir}", flush=True)
        env["RUN_ID"] = f"{args.run_tag}-{mode}"
        # Sequential fresh processes: identical seed/optimizer start, no chained
        # continuation from A to B, and no GPU contention between experiment arms.
        subprocess.run(commands[mode], env=env, check=True)
        checkpoint = run_dir / "checkpoints/weights" / f"step_{int(control.max_steps):06d}.pt"
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("step") != int(control.max_steps) or checkpoint_mode(
            payload.get("architecture_metadata") or {}
        ) != mode:
            raise RuntimeError(f"Final checkpoint step/ablation mismatch: {checkpoint}")
        provenance = payload["architecture_metadata"].get("eraf_preservation_source") or {}
        if provenance.get("checkpoint_sha256") != plan["warm_checkpoint_sha256"]:
            raise RuntimeError(f"Final checkpoint has a different V9.28 teacher: {checkpoint}")
        del payload
        print(f"[TRAIN_DONE] mode={mode} checkpoint={checkpoint}", flush=True)
    print(f"[ALL_TRAIN_DONE] runs={len(args.modes)} ROOT={root}; evaluation not started", flush=True)


if __name__ == "__main__":
    main()
