#!/usr/bin/env python3
"""Audit and launch formal four-GPU RoboTwin ERAF grounding training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.build_pgc_robotwin_eraf_grounding_manifest import (
    load_grounding_training_matrix,
)


METHOD_VERSION = "V9.39"
GROUNDING_OBJECTIVE_VERSION = 2
DEFAULT_STEPS = 1500
DEFAULT_SAVE_EVERY = 250
EFFECTIVE_BATCH_SIZE = 16
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_prompts(matrix: Mapping[str, object]) -> set[str]:
    prompts: set[str] = set()
    dataset_dirs = list(matrix["dataset_dirs"]) + list(
        matrix["counterfactual_dirs"]
    )
    for dataset in dataset_dirs:
        provenance_path = Path(str(dataset)) / "meta" / "pgc_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        for pair in provenance.get("pairs") or []:
            for key in ("source_instruction", "counterfactual_instruction"):
                instruction = str(pair.get(key, "")).strip()
                if instruction:
                    prompts.add(DEFAULT_PROMPT.format(task=instruction))
    if not prompts:
        raise ValueError("RoboTwin ERAF datasets expose no training prompts.")
    return prompts


def _cache_is_complete(matrix: Mapping[str, object], cache_dir: Path) -> bool:
    return all(
        (
            cache_dir
            / f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}"
            ".t5_len128.wan22ti2v5b.pt"
        ).is_file()
        for prompt in _required_prompts(matrix)
    )


def build_overrides(
    *,
    matrix: Mapping[str, object],
    base_checkpoint: Path,
    stats_path: Path,
    cache_dir: Path,
    seed: int,
    steps: int,
    save_every: int,
    gradient_accumulation_steps: int,
) -> list[str]:
    return [
        "task=robotwin_pgc_3cam_384",
        f"resume={base_checkpoint}",
        "weight_only_start_step=null",
        "data.val=null",
        "data.train._target_=fastwam.datasets.lerobot."
        "robotwin_eraf_grounding_dataset.RoboTwinERAFGroundingDataset",
        "data.train.dataset_dirs="
        + json.dumps(matrix["dataset_dirs"], separators=(",", ":")),
        "data.train.pgc_counterfactual_dataset_dirs="
        + json.dumps(matrix["counterfactual_dirs"], separators=(",", ":")),
        "data.train.pgc_counterfactual_oversample_factor=1",
        "data.train.pgc_balance_native_counterfactual=false",
        "data.train.pgc_pair_balanced_sampling=false",
        "data.train.pgc_entity_relation_supervision_required=true",
        "data.train.pgc_entity_relation_sidecar_dirs="
        + json.dumps(matrix["sidecar_dirs"], separators=(",", ":")),
        "data.train.pgc_v9_balanced_sampling=false",
        "data.train.pgc_v9_structured_role_sampling=false",
        f"data.train.pretrained_norm_stats={stats_path}",
        f"data.train.text_embedding_cache_dir={cache_dir}",
        "batch_size=1",
        "num_workers=2",
        f"seed={seed}",
        f"max_steps={steps}",
        "num_epochs=1",
        "learning_rate=1.0e-4",
        "lr_scheduler_type=constant",
        f"gradient_accumulation_steps={gradient_accumulation_steps}",
        "log_every=10",
        f"save_every={save_every}",
        "eval_every=0",
        "save_training_state=false",
        "wandb.enabled=false",
        "model.skip_dit_load_from_pretrain=true",
        "model.action_dit_config.use_latent_action_queries=false",
        "model.langforce_mvp.enabled=false",
        "model.langforce_mvp.enable_prior=false",
        "model.langforce_mvp.enable_posterior_advantage=false",
        "model.transition_contract.enabled=false",
        "model.policy_guard.enabled=true",
        "model.policy_guard.version=9",
        "model.policy_guard.gate_mode=guarded",
        "model.policy_guard.entity_relation_grounding.training_stage=grounding",
        "model.policy_guard.entity_relation_grounding."
        "initialization_contract=released_base_fresh_eraf",
        "model.policy_guard.entity_relation_grounding."
        f"grounding_objective_version={GROUNDING_OBJECTIVE_VERSION}",
        "model.policy_guard.entity_relation_grounding.completion_only_memory=false",
        "model.policy_guard.entity_relation_grounding.action_joint_training=false",
        "model.lora.enabled=false",
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--grounding-manifest", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/text_embeds_cache/robotwin")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--precompute-text", choices=("auto", "always", "never"), default="auto"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.gpus <= 0 or EFFECTIVE_BATCH_SIZE % args.gpus:
        parser.error(
            f"--gpus must divide effective batch {EFFECTIVE_BATCH_SIZE}; "
            f"got {args.gpus}"
        )
    if args.steps <= 0 or args.save_every <= 0 or args.seed < 0:
        parser.error("steps/save-every must be positive and seed non-negative")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_tag):
        parser.error("--run-tag must be a plain unique directory name")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if os.environ.get("NNODES", "1") != "1":
        raise ValueError("RoboTwin ERAF grounding currently supports one machine.")
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    stats_path = args.stats_path.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    if not base_checkpoint.is_file():
        raise FileNotFoundError(f"RoboTwin Base checkpoint not found: {base_checkpoint}.")
    if not stats_path.is_file():
        raise FileNotFoundError(f"RoboTwin normalization stats not found: {stats_path}.")
    matrix = load_grounding_training_matrix(args.grounding_manifest)
    accumulation = EFFECTIVE_BATCH_SIZE // args.gpus
    overrides = build_overrides(
        matrix=matrix,
        base_checkpoint=base_checkpoint,
        stats_path=stats_path,
        cache_dir=cache_dir,
        seed=args.seed,
        steps=args.steps,
        save_every=args.save_every,
        gradient_accumulation_steps=accumulation,
    )
    precompute = args.precompute_text == "always" or (
        args.precompute_text == "auto" and not _cache_is_complete(matrix, cache_dir)
    )
    precompute_command = [
        sys.executable,
        "scripts/precompute_text_embeds.py",
        *overrides,
        "+overwrite=false",
    ]
    train_command = ["bash", "scripts/train_zero1.sh", str(args.gpus), *overrides]
    run_id = f"pgc-{args.run_tag}"
    output_dir = PROJECT_ROOT / "runs" / "robotwin_pgc_3cam_384" / run_id
    plan = {
        "format": "pgc_robotwin_eraf_grounding_launch_v1",
        "method_version": METHOD_VERSION,
        "stage": "grounding",
        "grounding_objective_version": GROUNDING_OBJECTIVE_VERSION,
        "initialization_contract": "released_base_fresh_eraf",
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": _file_sha256(base_checkpoint),
        "normalization_stats": str(stats_path),
        "normalization_stats_sha256": _file_sha256(stats_path),
        "grounding_manifest": matrix["manifest"],
        "grounding_manifest_sha256": matrix["manifest_sha256"],
        "full_goal_usage": "forbidden_not_present",
        "gpus": args.gpus,
        "micro_batch_per_gpu": 1,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "optimizer_steps": args.steps,
        "save_every": args.save_every,
        "seed": args.seed,
        "pair_count": len(matrix["pair_ids"]),
        "dataset_count": len(matrix["sidecar_dirs"]),
        "total_successful_trajectories": matrix["total_successful_trajectories"],
        "pair_episode_counts": matrix["pair_episode_counts"],
        "sampling": "five_pairs_x_native_counterfactual_equal_weight_domains_3_to_2",
        "precompute_text": precompute,
        "output": str(output_dir),
        "train_command": train_command,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output_dir}.")
    if args.precompute_text == "never" and not _cache_is_complete(matrix, cache_dir):
        raise FileNotFoundError(
            "RoboTwin text cache is incomplete and --precompute-text=never was "
            "requested."
        )
    if precompute:
        subprocess.run(precompute_command, cwd=PROJECT_ROOT, check=True)
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    os.execvpe(train_command[0], train_command, env)


if __name__ == "__main__":
    main()
