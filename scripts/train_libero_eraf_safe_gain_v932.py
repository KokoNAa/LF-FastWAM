#!/usr/bin/env python3
"""Run the matched verified-ranking experiments from warm V9.28.

The input is the resolved V9.30 preservation template.  Newer artifacts record
the V9.30-B ``mask_corrective_ranking`` mode explicitly; the original server
artifact predates that field and is accepted only when the field is absent.
Derived objectives write their own loss mode explicitly and keep the template's
data, optimizer, seed, effective batch, frozen teacher, full-goal preservation
and trainable scope. Objectives 32--35 use the 25/50 diagnostic schedule;
V9.36--V9.37 use a shorter 25-step schedule with early checkpoints.
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
TARGET_EFFECTIVE_BATCH_SIZE = 12
MAX_STEPS = 50
SAVE_STEPS = (25, 50)
SUPPORTED_OBJECTIVES = frozenset({32, 33, 34, 35, 36, 37})
PAIRED_SEMANTIC_CONTRAST_WEIGHT = 0.1
PAIRED_SEMANTIC_CONTRAST_MARGIN = 0.1
PAIRED_SEMANTIC_NON_VIOLATION_WEIGHT = 0.5
EXPLICIT_V930_B_SOURCE_CONTRACT = "explicit_mask_corrective_ranking"
LEGACY_V930_TEMPLATE_SOURCE_CONTRACT = "legacy_unrecorded_cf_ablation"


def ranking_gradient_contract(objective):
    if objective in {34, 35, 36, 37}:
        return cf_ablation.UNIVERSAL_POSITIVE_ONLY_RANKING_CONTRACT
    if objective == 33:
        return cf_ablation.POSITIVE_ONLY_RANKING_CONTRACT
    return "legacy_detached_correct_error_wrong_error_gradient"


def max_steps_for_objective(objective):
    return 25 if objective in {36, 37} else MAX_STEPS


def save_steps_for_objective(objective):
    return (10, 15, 20, 25) if objective in {36, 37} else SAVE_STEPS


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_cf_ablation_contract(eraf):
    if "cf_ablation" not in eraf:
        return LEGACY_V930_TEMPLATE_SOURCE_CONTRACT
    if str(eraf.cf_ablation) == "mask_corrective_ranking":
        return EXPLICIT_V930_B_SOURCE_CONTRACT
    raise ValueError(
        "V9.30 source cf_ablation must be explicitly mask_corrective_ranking "
        "or absent in the legacy server template."
    )


def load_v930_b(config_path):
    register_default_resolvers()
    cfg = OmegaConf.create(
        OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    )
    eraf = cfg.model.policy_guard.entity_relation_grounding
    if int(eraf.grounding_objective_version) != 30 or not eraf.safe_gain_training:
        raise ValueError(
            "Source must be the completed V9.30 preservation config.yaml."
        )
    source_cf_ablation_contract(eraf)
    if (
        not cfg.get("resume")
        or cfg.get("weight_only_start_step") not in (None, 0)
        or int(cfg.max_steps) != 250
        or int(eraf.safe_gain_injector_training_steps) != 250
        or int(eraf.safe_gain_gate_calibration_steps) != 0
        or int(eraf.safe_gain_noise_levels) < 2
        or float(cfg.learning_rate) != 2.0e-6
        or 3 * int(cfg.batch_size) * int(cfg.gradient_accumulation_steps)
        != TARGET_EFFECTIVE_BATCH_SIZE
        or not cfg.data.train.pgc_v9_safe_gain_counterfactual_replay
    ):
        raise ValueError(
            "Expected the fresh 250-step V9.30 preservation run with its "
            "V9.28 resume and data recipe."
        )
    return cfg


def training_config(source, output_dir, gpus=3, *, objective=32):
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError("Verified ranking runner supports V9.32 through V9.37.")
    cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
    per_accumulation_batch = int(gpus) * int(cfg.batch_size)
    if (
        gpus < 1
        or per_accumulation_batch <= 0
        or TARGET_EFFECTIVE_BATCH_SIZE % per_accumulation_batch
    ):
        raise ValueError(
            f"Cannot preserve effective batch {TARGET_EFFECTIVE_BATCH_SIZE} with "
            f"gpus={gpus} and per-GPU batch={cfg.batch_size}."
        )
    cfg.gradient_accumulation_steps = (
        TARGET_EFFECTIVE_BATCH_SIZE // per_accumulation_batch
    )
    max_steps = max_steps_for_objective(objective)
    cfg.max_steps = max_steps
    cfg.save_every = 5 if objective in {36, 37} else SAVE_STEPS[0]
    eraf = cfg.model.policy_guard.entity_relation_grounding
    eraf.grounding_objective_version = objective
    eraf.cf_ablation = "mask_lift_ranking"
    eraf.safe_gain_injector_training_steps = max_steps
    for name, value in FULL_GOAL_WEIGHTS.items():
        eraf[name] = value
    if objective in {35, 36, 37}:
        eraf.paired_semantic_contrast_weight = PAIRED_SEMANTIC_CONTRAST_WEIGHT
        eraf.paired_semantic_contrast_margin = PAIRED_SEMANTIC_CONTRAST_MARGIN
    if objective == 37:
        eraf.paired_semantic_non_violation_weight = (
            PAIRED_SEMANTIC_NON_VIOLATION_WEIGHT
        )
    cfg.output_dir = str(Path(output_dir).resolve())
    cfg.wandb.name = f"v9{objective}_verified_full_goal_ranking"
    cfg.hydra = {
        "job": {"chdir": False},
        "run": {"dir": "."},
        "output_subdir": None,
    }
    return cfg


def training_command(run_dir, gpus, cfg, *, objective=32):
    return [
        "bash",
        "scripts/train_zero1.sh",
        str(gpus),
        f"output_dir={cfg.output_dir}",
        f"wandb.name={cfg.wandb.name}",
        "--config-path",
        str(Path(run_dir).resolve()),
        "--config-name",
        f"v9{objective}_train",
    ]


def main(*, objective=32):
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError("Verified ranking runner supports V9.32 through V9.37.")
    version = f"V9.{objective}"
    version_slug = f"v9{objective}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="actual completed V9.30 run-root/config.yaml"
    )
    parser.add_argument(
        "gpus", type=int, help="GPU count; accumulation is adjusted to keep batch 12"
    )
    parser.add_argument("run_tag", help="new unique directory name, no slashes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error("GPU count must be positive.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_tag):
        parser.error("run_tag must be a plain unique directory name.")
    if os.environ.get("NNODES", "1") != "1":
        parser.error("This matched runner supports one machine only.")

    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    source_path = args.config.expanduser().resolve()
    source = load_v930_b(source_path)
    root = repo / f"runs/libero_eraf_safe_gain_{version_slug}_2cam224" / args.run_tag
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {root}")
    cfg = training_config(source, root, args.gpus, objective=objective)
    max_steps = max_steps_for_objective(objective)
    save_steps = save_steps_for_objective(objective)
    command = training_command(root, args.gpus, cfg, objective=objective)
    plan = {
        "contract": preservation.SELECTIVE_FULL_GOAL_CONTRACT,
        "ranking_route": {
            "target_lift": {"action": True, "ranking": False},
            "counterfactual_goal": {"action": True, "ranking": True},
        },
        "source_v930_template_config": str(source_path),
        "source_v930_template_config_sha256": file_sha256(source_path),
        "source_v930_cf_ablation_contract": source_cf_ablation_contract(
            source.model.policy_guard.entity_relation_grounding
        ),
        "warm_v928_checkpoint": str(Path(source.resume).resolve()),
        "output": str(root),
        "gpus": args.gpus,
        "effective_batch_size": (
            args.gpus * int(cfg.batch_size) * int(cfg.gradient_accumulation_steps)
        ),
        "source_gradient_accumulation_steps": int(
            source.gradient_accumulation_steps
        ),
        "derived_gradient_accumulation_steps": int(
            cfg.gradient_accumulation_steps
        ),
        "seed": int(source.seed),
        "steps": int(cfg.max_steps),
        "save_steps": list(save_steps),
        "learning_rate": float(source.learning_rate),
        "cf_ablation": "mask_lift_ranking",
        "ranking_gradient_contract": ranking_gradient_contract(objective),
        "paired_semantic_contrast": (
            {
                "contract": (
                    cf_ablation.SOFT_ACTION_VIOLATION_SEMANTIC_CONTRAST_CONTRACT
                    if objective == 37
                    else
                    cf_ablation.ACTION_VIOLATION_GATED_SEMANTIC_CONTRAST_CONTRACT
                    if objective == 36
                    else cf_ablation.PAIRED_SEMANTIC_CONTRAST_CONTRACT
                ),
                "weight": PAIRED_SEMANTIC_CONTRAST_WEIGHT,
                "margin": PAIRED_SEMANTIC_CONTRAST_MARGIN,
                "validity": (
                    "soft_detached_used_action_ranking_violation_weight"
                    if objective == 37
                    else
                    "detached_used_positive_action_ranking_violation_only"
                    if objective == 36
                    else "all_bidirectional_semantic_pairs"
                ),
                "non_violation_weight": (
                    PAIRED_SEMANTIC_NON_VIOLATION_WEIGHT
                    if objective == 37
                    else None
                ),
                "trainable": "compressor_plus_context_injector_only",
            }
            if objective in {35, 36, 37}
            else None
        ),
        "full_goal_weights": FULL_GOAL_WEIGHTS,
        "command": command,
    }
    if plan["effective_batch_size"] != TARGET_EFFECTIVE_BATCH_SIZE:
        raise ValueError(
            "Resolved V9.30 config no longer yields the audited effective batch 12."
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
        "ERAF_SAFE_GAIN_MAX_STEPS": str(max_steps),
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
            f"{version} requires audited counterfactual_goal and target_lift rows so "
            "both branches of the selective ranking route are exercised."
        )
    index_path = corrective_dataset / "meta/pgc_v8_closed_loop/index.json"
    plan.update({
        "corrective_verification_counts": dict(counts),
        "corrective_index_sha256": file_sha256(index_path),
        "warm_v928_checkpoint_sha256": file_sha256(source.resume),
    })
    root.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, root / f"{version_slug}_train.yaml")
    (root / "experiment.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[TRAIN_START] output={root} counts={dict(counts)}", flush=True)
    env["RUN_ID"] = args.run_tag
    subprocess.run(command, env=env, check=True)

    checkpoint = root / "checkpoints/weights" / f"step_{max_steps:06d}.pt"
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("architecture_metadata") or {}
    if (
        payload.get("step") != max_steps
        or metadata.get("eraf_grounding_objective_version") != objective
        or cf_ablation.checkpoint_mode(metadata) != "mask_lift_ranking"
        or metadata.get("eraf_selective_full_goal_preservation_contract")
        != preservation.SELECTIVE_FULL_GOAL_CONTRACT
        or metadata.get("eraf_preservation_source", {}).get("checkpoint_sha256")
        != plan["warm_v928_checkpoint_sha256"]
        or (
            objective == 33
            and metadata.get("eraf_corrective_ranking_gradient_contract")
            != cf_ablation.POSITIVE_ONLY_RANKING_CONTRACT
        )
        or (
            objective in {34, 35, 36, 37}
            and metadata.get("eraf_paired_ranking_gradient_contract")
            != cf_ablation.UNIVERSAL_POSITIVE_ONLY_RANKING_CONTRACT
        )
        or (
            objective in {35, 36, 37}
            and (
                metadata.get("eraf_paired_semantic_contrast_contract")
                != (
                    cf_ablation.SOFT_ACTION_VIOLATION_SEMANTIC_CONTRAST_CONTRACT
                    if objective == 37
                    else
                    cf_ablation.ACTION_VIOLATION_GATED_SEMANTIC_CONTRAST_CONTRACT
                    if objective == 36
                    else cf_ablation.PAIRED_SEMANTIC_CONTRAST_CONTRACT
                )
                or metadata.get("eraf_paired_semantic_contrast_weight")
                != PAIRED_SEMANTIC_CONTRAST_WEIGHT
                or metadata.get("eraf_paired_semantic_contrast_margin")
                != PAIRED_SEMANTIC_CONTRAST_MARGIN
                or (
                    objective == 37
                    and metadata.get("eraf_paired_semantic_non_violation_weight")
                    != PAIRED_SEMANTIC_NON_VIOLATION_WEIGHT
                )
            )
        )
    ):
        raise RuntimeError(
            f"Final {version} checkpoint contract mismatch: {checkpoint}"
        )
    preservation.validate_teacher_payload(payload)
    for step in save_steps:
        expected = root / "checkpoints/weights" / f"step_{step:06d}.pt"
        if not expected.is_file():
            raise FileNotFoundError(
                f"Missing required {version} checkpoint: {expected}"
            )
    print(f"[TRAIN_DONE] checkpoint={checkpoint}; evaluation not started", flush=True)


if __name__ == "__main__":
    main()
