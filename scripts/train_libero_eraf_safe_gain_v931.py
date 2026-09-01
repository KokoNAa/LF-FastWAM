#!/usr/bin/env python3
"""Run the matched V9.31 selective full-goal preservation probe.

The input is the actual resolved V9.30-B run config.  The runner preserves its
data recipe, optimizer, seed, batch size, 250-step schedule and V9.28 resume;
only the objective and declared V9.31 preservation terms change.
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

from fastwam.datasets.pgc_libero import load_pgc_closed_loop_corrective_index
from fastwam.models.wan22 import eraf_preservation as preservation
from fastwam.utils import cf_ablation
from fastwam.utils.config_resolvers import register_default_resolvers

try:
    from scripts.train_libero_eraf_safe_gain_v930_from_config import launch_spec
except ModuleNotFoundError:
    from train_libero_eraf_safe_gain_v930_from_config import launch_spec


FULL_GOAL_WEIGHTS = {
    "full_goal_action_preservation_weight": 1.0,
    "full_goal_token_preservation_weight": 0.1,
    "full_goal_context_preservation_weight": 1.0,
    "full_goal_preservation_margin": 0.0,
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v930_b(config_path):
    register_default_resolvers()
    cfg = OmegaConf.create(
        OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    )
    eraf = cfg.model.policy_guard.entity_relation_grounding
    if (
        int(eraf.grounding_objective_version) != 30
        or not eraf.safe_gain_training
        or eraf.get("cf_ablation", "none") != "mask_corrective_ranking"
    ):
        raise ValueError(
            "Source must be the completed V9.30-B mask_corrective_ranking config.yaml."
        )
    if (
        not cfg.get("resume")
        or cfg.get("weight_only_start_step") not in (None, 0)
        or int(cfg.max_steps) != 250
        or int(eraf.safe_gain_injector_training_steps) != 250
        or int(eraf.safe_gain_gate_calibration_steps) != 0
        or int(eraf.safe_gain_noise_levels) < 2
        or float(cfg.learning_rate) != 2.0e-6
        or not cfg.data.train.pgc_v9_safe_gain_counterfactual_replay
    ):
        raise ValueError(
            "Expected the fresh 250-step V9.30-B run with its V9.28 resume and data recipe."
        )
    return cfg


def training_config(source, output_dir):
    cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
    eraf = cfg.model.policy_guard.entity_relation_grounding
    eraf.grounding_objective_version = 31
    eraf.cf_ablation = "mask_corrective_ranking"
    for name, value in FULL_GOAL_WEIGHTS.items():
        eraf[name] = value
    cfg.output_dir = str(Path(output_dir).resolve())
    cfg.wandb.name = "v931_selective_full_goal_preservation"
    cfg.hydra = {
        "job": {"chdir": False},
        "run": {"dir": "."},
        "output_subdir": None,
    }
    return cfg


def training_command(run_dir, gpus, cfg):
    return [
        "bash",
        "scripts/train_zero1.sh",
        str(gpus),
        f"output_dir={cfg.output_dir}",
        f"wandb.name={cfg.wandb.name}",
        "--config-path",
        str(Path(run_dir).resolve()),
        "--config-name",
        "v931_train",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="actual completed V9.30-B run-root/config.yaml"
    )
    parser.add_argument("gpus", type=int, help="must be 3 to preserve effective batch 12")
    parser.add_argument("run_tag", help="new unique directory name, no slashes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpus != 3:
        parser.error("The matched V9.31 probe requires exactly 3 GPUs (effective batch 12).")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_tag):
        parser.error("run_tag must be a plain unique directory name.")
    if os.environ.get("NNODES", "1") != "1":
        parser.error("This matched runner supports one machine only.")

    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    source_path = args.config.expanduser().resolve()
    source = load_v930_b(source_path)
    root = repo / "runs/libero_eraf_safe_gain_v931_2cam224" / args.run_tag
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {root}")
    cfg = training_config(source, root)
    command = training_command(root, args.gpus, cfg)
    plan = {
        "contract": preservation.SELECTIVE_FULL_GOAL_CONTRACT,
        "source_v930_b_config": str(source_path),
        "source_v930_b_config_sha256": file_sha256(source_path),
        "warm_v928_checkpoint": str(Path(source.resume).resolve()),
        "output": str(root),
        "gpus": args.gpus,
        "effective_batch_size": (
            args.gpus * int(source.batch_size) * int(source.gradient_accumulation_steps)
        ),
        "seed": int(source.seed),
        "steps": int(source.max_steps),
        "learning_rate": float(source.learning_rate),
        "cf_ablation": "mask_corrective_ranking",
        "full_goal_weights": FULL_GOAL_WEIGHTS,
        "command": command,
    }
    if plan["effective_batch_size"] != 12:
        raise ValueError(
            "Resolved V9.30-B config no longer yields the audited effective batch 12."
        )
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("DRY_RUN: no files, checkpoint reads, CUDA or training jobs.", flush=True)
        return

    preflight, env = launch_spec(
        source_path, args.gpus, os.environ, source_objective=30
    )
    env.update({
        "ERAF_SAFE_GAIN_PREFLIGHT_ONLY": "1",
        "PYTHON_BIN": sys.executable,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
        "ERAF_SAFE_GAIN_MAX_STEPS": str(source.max_steps),
    })
    subprocess.run(preflight, env=env, check=True)
    env.pop("ERAF_SAFE_GAIN_PREFLIGHT_ONLY")

    corrective_dataset = Path(
        source.data.train.pgc_closed_loop_corrective_dataset_dirs[0]
    )
    records = load_pgc_closed_loop_corrective_index(corrective_dataset)
    counts = Counter(record["verification_kind"] for record in records.values())
    if not counts["counterfactual_goal"] or not counts["target_lift"]:
        raise ValueError(
            "V9.31 requires audited counterfactual_goal and target_lift rows so the "
            "selective mask and its hard negative exclusion are both exercised."
        )
    index_path = corrective_dataset / "meta/pgc_v8_closed_loop/index.json"
    plan.update({
        "corrective_verification_counts": dict(counts),
        "corrective_index_sha256": file_sha256(index_path),
        "warm_v928_checkpoint_sha256": file_sha256(source.resume),
    })
    root.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, root / "v931_train.yaml")
    (root / "experiment.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[TRAIN_START] output={root} counts={dict(counts)}", flush=True)
    env["RUN_ID"] = args.run_tag
    subprocess.run(command, env=env, check=True)

    checkpoint = root / "checkpoints/weights/step_000250.pt"
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("architecture_metadata") or {}
    if (
        payload.get("step") != 250
        or metadata.get("eraf_grounding_objective_version") != 31
        or cf_ablation.checkpoint_mode(metadata) != "mask_corrective_ranking"
        or metadata.get("eraf_selective_full_goal_preservation_contract")
        != preservation.SELECTIVE_FULL_GOAL_CONTRACT
        or metadata.get("eraf_preservation_source", {}).get("checkpoint_sha256")
        != plan["warm_v928_checkpoint_sha256"]
    ):
        raise RuntimeError(f"Final V9.31 checkpoint contract mismatch: {checkpoint}")
    preservation.validate_teacher_payload(payload)
    for step in (50, 100, 250):
        expected = root / "checkpoints/weights" / f"step_{step:06d}.pt"
        if not expected.is_file():
            raise FileNotFoundError(f"Missing required V9.31 checkpoint: {expected}")
    print(f"[TRAIN_DONE] checkpoint={checkpoint}; evaluation not started", flush=True)


if __name__ == "__main__":
    main()
