#!/usr/bin/env python3
"""Continue admitted RoboTwin objective-2 grounding with objective-3 roles."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.build_pgc_robotwin_eraf_grounding_manifest import (
    load_grounding_training_matrix,
)
from scripts.train_pgc_robotwin_eraf_grounding import (
    EFFECTIVE_BATCH_SIZE,
    _cache_is_complete,
    _file_sha256,
    build_overrides,
)


INPUT_OBJECTIVE = 2
OUTPUT_OBJECTIVE = 3
START_STEP = 1500
DEFAULT_STAGE_STEPS = 1000
DEFAULT_SAVE_EVERY = 250


def validate_grounding_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("RoboTwin grounding checkpoint root must be a mapping.")
    metadata = payload.get("architecture_metadata") or {}
    expected = {
        "format": "fastwam_policy_guard_v9",
        "step": START_STEP,
        "policy_guard_version": 9,
        "eraf_grounding_objective_version": INPUT_OBJECTIVE,
        "eraf_training_stage": "grounding",
        "warm_start_contract": "released_base_fresh_eraf",
        "action_output_dim": 14,
        "proprio_dim": 14,
        "eraf_camera_count": 3,
        "eraf_camera_layout": "robotwin_mosaic",
    }
    actual = {
        "format": payload.get("format"),
        "step": int(payload.get("step", -1)),
        **{name: metadata.get(name) for name in expected if name not in {"format", "step"}},
    }
    mismatches = {
        name: {"expected": value, "actual": actual.get(name)}
        for name, value in expected.items()
        if actual.get(name) != value
    }
    if mismatches:
        raise ValueError(
            "RoboTwin grounding-role requires the admitted objective-2 "
            f"step-1500 checkpoint; mismatches={mismatches}."
        )
    aspect = float(metadata.get("eraf_visual_aspect_ratio", -1.0))
    if not math.isclose(aspect, 5.0 / 6.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            "RoboTwin grounding checkpoint has the wrong visual aspect ratio: "
            f"{aspect}."
        )
    return dict(metadata)


def validate_grounding_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return validate_grounding_payload(payload)


def build_role_overrides(
    *,
    matrix: Mapping[str, object],
    grounding_checkpoint: Path,
    stats_path: Path,
    cache_dir: Path,
    seed: int,
    stage_steps: int,
    save_every: int,
    gradient_accumulation_steps: int,
) -> list[str]:
    return build_overrides(
        matrix=matrix,
        base_checkpoint=grounding_checkpoint,
        stats_path=stats_path,
        cache_dir=cache_dir,
        seed=seed,
        steps=START_STEP + stage_steps,
        save_every=save_every,
        gradient_accumulation_steps=gradient_accumulation_steps,
        grounding_objective_version=OUTPUT_OBJECTIVE,
        weight_only_start_step=START_STEP,
        learning_rate=2.0e-5,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--grounding-checkpoint", type=Path, required=True)
    parser.add_argument("--grounding-manifest", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/text_embeds_cache/robotwin")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage-steps", type=int, default=DEFAULT_STAGE_STEPS)
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
    if args.stage_steps <= 0 or args.save_every <= 0 or args.seed < 0:
        parser.error("stage-steps/save-every must be positive and seed non-negative")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_tag):
        parser.error("--run-tag must be a plain unique directory name")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if os.environ.get("NNODES", "1") != "1":
        raise ValueError("RoboTwin ERAF grounding-role supports one machine only.")
    checkpoint = args.grounding_checkpoint.expanduser().resolve()
    stats_path = args.stats_path.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Grounding checkpoint not found: {checkpoint}.")
    if not stats_path.is_file():
        raise FileNotFoundError(f"RoboTwin normalization stats not found: {stats_path}.")
    metadata = validate_grounding_checkpoint(checkpoint)
    matrix = load_grounding_training_matrix(args.grounding_manifest)
    accumulation = EFFECTIVE_BATCH_SIZE // args.gpus
    overrides = build_role_overrides(
        matrix=matrix,
        grounding_checkpoint=checkpoint,
        stats_path=stats_path,
        cache_dir=cache_dir,
        seed=args.seed,
        stage_steps=args.stage_steps,
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
        "format": "pgc_robotwin_eraf_grounding_role_launch_v1",
        "stage": "grounding-role",
        "input_objective": INPUT_OBJECTIVE,
        "output_objective": OUTPUT_OBJECTIVE,
        "input_step": START_STEP,
        "stage_optimizer_steps": args.stage_steps,
        "final_optimizer_step": START_STEP + args.stage_steps,
        "initialization_contract": metadata["warm_start_contract"],
        "grounding_checkpoint": str(checkpoint),
        "grounding_checkpoint_sha256": _file_sha256(checkpoint),
        "grounding_manifest": matrix["manifest"],
        "grounding_manifest_sha256": matrix["manifest_sha256"],
        "full_goal_usage": "forbidden_not_present",
        "gpus": args.gpus,
        "micro_batch_per_gpu": 1,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "learning_rate": 2.0e-5,
        "role_assignment_weight": 4.0,
        "role_assignment_hard_weight": 2.0,
        "save_every": args.save_every,
        "seed": args.seed,
        "pair_count": len(matrix["pair_ids"]),
        "dataset_count": len(matrix["sidecar_dirs"]),
        "total_successful_trajectories": matrix["total_successful_trajectories"],
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
