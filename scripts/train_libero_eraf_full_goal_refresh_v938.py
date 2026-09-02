#!/usr/bin/env python3
"""Train the V9.38 full-goal corrective refresh for 50 clean steps.

V9.38 intentionally keeps the deployed V9.37 model/loss contract and the
fixed V9.28 teacher.  It replaces the old mixed 30-episode corrective pool
with a replay-verified 35-episode pool that contains five complete
counterfactual-goal trajectories for every observed source-directed failure
pair, and runs a clean 50-step optimization from the same V9.28 start.
"""

from __future__ import annotations

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

from fastwam.models.wan22 import eraf_preservation as preservation
from fastwam.utils import cf_ablation

try:
    from scripts.train_libero_eraf_safe_gain_v930_from_config import launch_spec
    from scripts.train_libero_eraf_safe_gain_v932 import (
        PAIRED_SEMANTIC_CONTRAST_MARGIN,
        PAIRED_SEMANTIC_CONTRAST_WEIGHT,
        PAIRED_SEMANTIC_NON_VIOLATION_WEIGHT,
        TARGET_EFFECTIVE_BATCH_SIZE,
        file_sha256,
        load_v930_b,
        source_cf_ablation_contract,
        training_command,
        training_config,
    )
except ModuleNotFoundError:
    from train_libero_eraf_safe_gain_v930_from_config import launch_spec
    from train_libero_eraf_safe_gain_v932 import (
        PAIRED_SEMANTIC_CONTRAST_MARGIN,
        PAIRED_SEMANTIC_CONTRAST_WEIGHT,
        PAIRED_SEMANTIC_NON_VIOLATION_WEIGHT,
        TARGET_EFFECTIVE_BATCH_SIZE,
        file_sha256,
        load_v930_b,
        source_cf_ablation_contract,
        training_command,
        training_config,
    )


MODEL_OBJECTIVE = 37
METHOD_VERSION = "V9.38"
MAX_STEPS = 50
SAVE_EVERY = 5
REQUIRED_SAVE_STEPS = (10, 15, 20, 25, 30, 35, 40, 45, 50)
EXPECTED_EPISODES = 35
EXPECTED_PAIRS = 7
EXPECTED_EPISODES_PER_PAIR = 5
CORRECTIVE_FORMAT = "pgc_libero_closed_loop_corrective_v2"
SIDECAR_FORMAT = "pgc_libero_entity_relation_v1"
ACTION_CONVENTION = "libero_env_gripper_open_minus1_close_plus1"
REPLAY_TRANSFORM = "identity"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_full_goal_binding(
    source,
    dataset: Path,
    sidecar: Path,
    coverage_path: Path | None = None,
) -> dict:
    dataset = dataset.expanduser().resolve()
    sidecar = sidecar.expanduser().resolve()
    coverage_path = (
        dataset / "meta/v938_full_goal_coverage.json"
        if coverage_path is None
        else coverage_path.expanduser().resolve()
    )
    index_path = dataset / "meta/pgc_v8_closed_loop/index.json"
    sidecar_index_path = sidecar / "index.json"
    index = _json(index_path)
    coverage = _json(coverage_path)
    sidecar_index = _json(sidecar_index_path)

    episodes = list(index.get("episodes") or [])
    pair_counts = Counter(str(row.get("pair_id", "")) for row in episodes)
    if (
        index.get("format") != CORRECTIVE_FORMAT
        or index.get("acquisition_only") is not False
        or int(index.get("episode_count", -1)) != EXPECTED_EPISODES
        or len(episodes) != EXPECTED_EPISODES
        or [int(row.get("episode_index", -1)) for row in episodes]
        != list(range(EXPECTED_EPISODES))
        or any(
            row.get("corrective_verified") is not True
            or row.get("verification_kind") != "counterfactual_goal"
            or row.get("counterfactual_goal_verified") is not True
            for row in episodes
        )
        or len(pair_counts) != EXPECTED_PAIRS
        or set(pair_counts.values()) != {EXPECTED_EPISODES_PER_PAIR}
    ):
        raise ValueError(
            "V9.38 requires exactly 35 dense replay-verified full-goal "
            f"episodes over seven 5-episode pairs: counts={dict(pair_counts)}"
        )

    if (
        coverage.get("format") != "pgc_corrective_full_goal_coverage_v1"
        or coverage.get("complete") is not True
        or int(coverage.get("minimum_full_goal_per_pair", -1))
        != EXPECTED_EPISODES_PER_PAIR
        or int(coverage.get("required_pair_count", -1)) != EXPECTED_PAIRS
        or int(coverage.get("covered_required_pair_count", -1)) != EXPECTED_PAIRS
        or list(coverage.get("missing_required_pairs") or [])
        or Path(str(coverage.get("dataset", ""))).resolve() != dataset
    ):
        raise ValueError("V9.38 full-goal coverage audit is incomplete or stale.")

    source_sidecars = list(source.data.train.pgc_entity_relation_sidecar_dirs)
    if len(source_sidecars) != 5:
        raise ValueError("V9.38 expects the verified five-sidecar V9.30 template.")
    reference_sidecar = _json(Path(source_sidecars[0]).expanduser().resolve() / "index.json")
    if (
        sidecar_index.get("format") != SIDECAR_FORMAT
        or Path(str(sidecar_index.get("dataset", ""))).resolve() != dataset
        or sidecar_index.get("dataset_kind") != "counterfactual"
        or sidecar_index.get("dataset_action_convention") != ACTION_CONVENTION
        or sidecar_index.get("simulator_replay_action_transform")
        != REPLAY_TRANSFORM
        or int(sidecar_index.get("episode_count", -1)) != EXPECTED_EPISODES
        or len(sidecar_index.get("episodes") or []) != EXPECTED_EPISODES
        or sidecar_index.get("workspace_min") != reference_sidecar.get("workspace_min")
        or sidecar_index.get("workspace_max") != reference_sidecar.get("workspace_max")
    ):
        raise ValueError("V9.38 sidecar binding/action/workspace contract mismatch.")

    return {
        "dataset": str(dataset),
        "sidecar": str(sidecar),
        "coverage": str(coverage_path),
        "pair_counts": dict(sorted(pair_counts.items())),
        "corrective_index_sha256": _sha256(index_path),
        "sidecar_index_sha256": _sha256(sidecar_index_path),
        "coverage_sha256": _sha256(coverage_path),
    }


def refreshed_config(source, output_dir: Path, dataset: Path, sidecar: Path, gpus: int):
    cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
    cfg.data.train.pgc_closed_loop_corrective_dataset_dirs = [str(dataset.resolve())]
    sidecars = list(cfg.data.train.pgc_entity_relation_sidecar_dirs)
    sidecars[-1] = str(sidecar.resolve())
    cfg.data.train.pgc_entity_relation_sidecar_dirs = sidecars
    cfg = training_config(cfg, output_dir, gpus, objective=MODEL_OBJECTIVE)
    cfg.max_steps = MAX_STEPS
    cfg.save_every = SAVE_EVERY
    cfg.model.policy_guard.entity_relation_grounding.safe_gain_injector_training_steps = (
        MAX_STEPS
    )
    cfg.wandb.name = "v938_full_goal_coverage_refresh_50steps"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="completed V9.30-B run config.yaml")
    parser.add_argument("dataset", type=Path, help="35-episode full-goal corrective dataset")
    parser.add_argument("sidecar", type=Path, help="ERAF sidecar bound to that dataset")
    parser.add_argument("gpus", type=int)
    parser.add_argument("run_tag")
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error("GPU count must be positive.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_tag):
        parser.error("run_tag must be a plain unique directory name.")
    if os.environ.get("NNODES", "1") != "1":
        parser.error("V9.38 supports one machine only.")

    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    source_path = args.config.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    sidecar = args.sidecar.expanduser().resolve()
    source = load_v930_b(source_path)
    binding = validate_full_goal_binding(source, dataset, sidecar, args.coverage)
    root = repo / "runs/libero_eraf_safe_gain_v938_2cam224" / args.run_tag
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {root}")
    cfg = refreshed_config(source, root, dataset, sidecar, args.gpus)
    command = training_command(root, args.gpus, cfg, objective=MODEL_OBJECTIVE)
    save_steps = REQUIRED_SAVE_STEPS
    plan = {
        "method_version": METHOD_VERSION,
        "model_objective_version": MODEL_OBJECTIVE,
        "change_scope": "full_goal_corrective_refresh_50_step_horizon",
        "warm_v928_checkpoint": str(Path(source.resume).resolve()),
        "source_v930_template_config": str(source_path),
        "source_v930_template_config_sha256": file_sha256(source_path),
        "source_v930_cf_ablation_contract": source_cf_ablation_contract(
            source.model.policy_guard.entity_relation_grounding
        ),
        "full_goal_binding": binding,
        "output": str(root),
        "gpus": args.gpus,
        "effective_batch_size": (
            args.gpus * int(cfg.batch_size) * int(cfg.gradient_accumulation_steps)
        ),
        "steps": int(cfg.max_steps),
        "save_steps": list(save_steps),
        "learning_rate": float(cfg.learning_rate),
        "semantic_contract": cf_ablation.SOFT_ACTION_VIOLATION_SEMANTIC_CONTRAST_CONTRACT,
        "semantic_weight": PAIRED_SEMANTIC_CONTRAST_WEIGHT,
        "semantic_margin": PAIRED_SEMANTIC_CONTRAST_MARGIN,
        "semantic_non_violation_weight": PAIRED_SEMANTIC_NON_VIOLATION_WEIGHT,
        "command": command,
    }
    if plan["effective_batch_size"] != TARGET_EFFECTIVE_BATCH_SIZE:
        raise ValueError("V9.38 must preserve effective batch size 12.")
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("DRY_RUN: validation complete; no CUDA or training job.", flush=True)
        return

    preflight, env = launch_spec(source_path, args.gpus, os.environ, source_objective=30)
    env.update(
        {
            "ERAF_SAFE_GAIN_PREFLIGHT_ONLY": "1",
            "PYTHON_BIN": sys.executable,
            "PATH": str(Path(sys.executable).parent)
            + os.pathsep
            + env.get("PATH", ""),
            "ERAF_SAFE_GAIN_MAX_STEPS": str(cfg.max_steps),
        }
    )
    subprocess.run(preflight, env=env, check=True)
    env.pop("ERAF_SAFE_GAIN_PREFLIGHT_ONLY")

    plan["warm_v928_checkpoint_sha256"] = file_sha256(source.resume)
    root.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, root / "v937_train.yaml")
    (root / "experiment.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[TRAIN_START] output={root} full_goal=35 pairs=7 model_objective=37",
        flush=True,
    )
    env["RUN_ID"] = args.run_tag
    subprocess.run(command, env=env, check=True)

    checkpoint = root / "checkpoints/weights/step_000050.pt"
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("architecture_metadata") or {}
    if (
        payload.get("step") != MAX_STEPS
        or metadata.get("eraf_grounding_objective_version") != MODEL_OBJECTIVE
        or metadata.get("eraf_paired_semantic_contrast_contract")
        != cf_ablation.SOFT_ACTION_VIOLATION_SEMANTIC_CONTRAST_CONTRACT
        or metadata.get("eraf_paired_semantic_non_violation_weight")
        != PAIRED_SEMANTIC_NON_VIOLATION_WEIGHT
        or metadata.get("eraf_preservation_source", {}).get("checkpoint_sha256")
        != plan["warm_v928_checkpoint_sha256"]
    ):
        raise RuntimeError(f"Final V9.38 checkpoint contract mismatch: {checkpoint}")
    preservation.validate_teacher_payload(payload)
    for step in save_steps:
        expected = root / "checkpoints/weights" / f"step_{step:06d}.pt"
        if not expected.is_file():
            raise FileNotFoundError(f"Missing V9.38 checkpoint: {expected}")
    print(f"[TRAIN_DONE] checkpoint={checkpoint}; evaluation not started", flush=True)


if __name__ == "__main__":
    main()
