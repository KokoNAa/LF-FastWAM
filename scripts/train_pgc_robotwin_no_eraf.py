#!/usr/bin/env python3
"""Audit and launch the four-pool RoboTwin LoRA-only/no-ERAF control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.build_pgc_robotwin_no_eraf_manifest import load_no_eraf_manifest


EFFECTIVE_BATCH_SIZE = 16
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)
MODES = {
    "smoke": {"steps": 2, "save_every": 1, "num_workers": 0, "log_every": 1},
    "formal": {
        "steps": 10000,
        "save_every": 250,
        "num_workers": 2,
        "log_every": 10,
    },
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_prompts(matrix: Mapping[str, list[str]]) -> set[str]:
    prompts: set[str] = set()
    for dataset in matrix["dataset_dirs"] + matrix["counterfactual_dirs"]:
        provenance = json.loads(
            (Path(dataset) / "meta/pgc_provenance.json").read_text(encoding="utf-8")
        )
        for pair in provenance.get("pairs") or []:
            for key in ("source_instruction", "counterfactual_instruction"):
                prompts.add(DEFAULT_PROMPT.format(task=str(pair[key])))
    return prompts


def _cache_is_complete(matrix: Mapping[str, list[str]], cache_dir: Path) -> bool:
    return all(
        (
            cache_dir
            / (
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                + ".t5_len128.wan22ti2v5b.pt"
            )
        ).is_file()
        for prompt in _required_prompts(matrix)
    )


def _validate_base_checkpoint(path: Path) -> None:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Cannot read RoboTwin Base checkpoint: {path}.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("RoboTwin Base checkpoint must be a mapping.")
    if str(payload.get("format", "")).startswith("fastwam_policy_guard_"):
        raise ValueError(
            "no-ERAF must start from Base, not an ERAF/policy-guard checkpoint."
        )
    if not isinstance(payload.get("mot"), Mapping) and not isinstance(
        payload.get("dit"), Mapping
    ):
        raise ValueError(
            "RoboTwin Base checkpoint has neither `mot` nor `dit` weights."
        )
    if payload.get("policy_guard"):
        raise ValueError(
            "no-ERAF Base checkpoint unexpectedly contains policy_guard weights."
        )


def build_overrides(
    *,
    matrix: Mapping[str, Any],
    base_checkpoint: Path,
    stats_path: Path,
    cache_dir: Path,
    seed: int,
    mode: str,
) -> list[str]:
    spec = MODES[mode]
    native_dirs = matrix["offline_native_dirs"] + matrix["closed_loop_native_dirs"]
    counterfactual_dirs = matrix["historical_cf_dirs"] + matrix["strict_cf_dirs"]
    counts = matrix["dataset_counts"]
    return [
        "task=robotwin_lora_only_no_eraf_3cam384",
        f"resume={base_checkpoint}",
        "weight_only_start_step=null",
        "data.val=null",
        "data.train.dataset_dirs=" + json.dumps(native_dirs, separators=(",", ":")),
        "data.train.pgc_counterfactual_dataset_dirs="
        + json.dumps(counterfactual_dirs, separators=(",", ":")),
        "data.train.pgc_closed_loop_corrective_dataset_dirs=[]",
        "data.train.pgc_entity_relation_sidecar_dirs="
        + json.dumps(matrix["sidecar_dirs"], separators=(",", ":")),
        "data.train.pgc_robotwin_offline_native_dataset_count="
        + str(counts["offline_native"]),
        "data.train.pgc_robotwin_closed_loop_native_dataset_count="
        + str(counts["closed_loop_native"]),
        "data.train.pgc_robotwin_historical_cf_dataset_count="
        + str(counts["historical_cf"]),
        "data.train.pgc_robotwin_strict_cf_dataset_count=" + str(counts["strict_cf"]),
        "data.train.pgc_balance_native_counterfactual=false",
        "data.train.pgc_v9_balanced_sampling=false",
        "data.train.pgc_pair_balanced_sampling=false",
        "data.train.pgc_v9_closed_loop_rebinding=false",
        "data.train.pgc_v9_phase_safe_memory=false",
        "data.train.pgc_v9_closed_loop_native_dataset_count=0",
        f"data.train.pretrained_norm_stats={stats_path}",
        f"data.train.text_embedding_cache_dir={cache_dir}",
        f"seed={seed}",
        f"max_steps={spec['steps']}",
        f"save_every={spec['save_every']}",
        f"num_workers={spec['num_workers']}",
        f"log_every={spec['log_every']}",
        "num_epochs=1",
        "gradient_accumulation_steps=4",
        "learning_rate=5.0e-6",
        "lr_scheduler_type=cosine",
        "weight_decay=1.0e-2",
        "eval_every=0",
        "save_training_state=false",
        "wandb.enabled=false",
        "model.action_dit_config.use_latent_action_queries=false",
        "model.langforce_mvp.enabled=false",
        "model.langforce_mvp.enable_prior=false",
        "model.langforce_mvp.enable_posterior_advantage=false",
        "model.transition_contract.enabled=false",
        "model.policy_guard.enabled=false",
        "model.lora.enabled=true",
        "model.lora.rank=16",
        "model.lora.alpha=16",
        "model.lora.dropout=0.05",
        "model.lora.experts=[video,action]",
        "model.lora.extra_trainable_patterns=[]",
        "model.lora.paired_language_control.enabled=true",
        "model.lora.paired_language_control.bidirectional_supervision=true",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODES), default="smoke")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/text_embeds_cache/robotwin")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--precompute-text", choices=("auto", "always", "never"), default="auto"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpus <= 0 or EFFECTIVE_BATCH_SIZE % args.gpus:
        parser.error(f"--gpus must divide effective batch {EFFECTIVE_BATCH_SIZE}")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_tag):
        parser.error("--run-tag must be a plain unique directory name")
    if os.environ.get("NNODES", "1") != "1":
        parser.error("RoboTwin no-ERAF launcher supports one machine only")

    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    stats_path = args.stats_path.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    if not base_checkpoint.is_file():
        raise FileNotFoundError(
            f"RoboTwin Base checkpoint not found: {base_checkpoint}."
        )
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"RoboTwin normalization stats not found: {stats_path}."
        )
    _validate_base_checkpoint(base_checkpoint)
    matrix = load_no_eraf_manifest(args.prepared_manifest)

    output = PROJECT_ROOT / "runs/robotwin_lora_only_no_eraf_3cam384" / args.run_tag
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}.")
    overrides = build_overrides(
        matrix=matrix,
        base_checkpoint=base_checkpoint,
        stats_path=stats_path,
        cache_dir=cache_dir,
        seed=args.seed,
        mode=args.mode,
    )
    text_matrix = {
        "dataset_dirs": matrix["offline_native_dirs"]
        + matrix["closed_loop_native_dirs"],
        "counterfactual_dirs": matrix["historical_cf_dirs"] + matrix["strict_cf_dirs"],
    }
    cache_complete = _cache_is_complete(text_matrix, cache_dir)
    precompute = args.precompute_text == "always" or (
        args.precompute_text == "auto" and not cache_complete
    )
    precompute_command = [
        sys.executable,
        "scripts/precompute_text_embeds.py",
        *overrides,
        "+overwrite=false",
    ]
    train_command = [
        "bash",
        "scripts/train_zero1.sh",
        str(args.gpus),
        *overrides,
    ]
    spec = MODES[args.mode]
    plan = {
        "format": "pgc_robotwin_no_eraf_training_launch_v1",
        "mode": args.mode,
        "gpus": args.gpus,
        "per_gpu_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "optimizer_steps": spec["steps"],
        "learning_rate": 5.0e-6,
        "lr_scheduler": "cosine",
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": _file_sha256(base_checkpoint),
        "normalization_stats": str(stats_path),
        "normalization_stats_sha256": _file_sha256(stats_path),
        "manifest": matrix["manifest"],
        "pool_order": matrix["pool_order"],
        "dataset_counts": matrix["dataset_counts"],
        "episode_counts": matrix["episode_counts"],
        "sampling": "deterministic_1_1_1_1",
        "full_goal_usage": "forbidden_not_present",
        "policy_guard": False,
        "lora": "video+action rank16 alpha16 dropout0.05",
        "precompute_text": precompute,
        "output": str(output),
        "train_command": train_command,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if args.precompute_text == "never" and not cache_complete:
        raise FileNotFoundError(
            "RoboTwin source/counterfactual text cache is incomplete and "
            "--precompute-text=never was requested."
        )
    if precompute:
        subprocess.run(precompute_command, cwd=PROJECT_ROOT, check=True)
    env = os.environ.copy()
    env["RUN_ID"] = args.run_tag
    subprocess.run(train_command, cwd=PROJECT_ROOT, env=env, check=True)
    final = output / "checkpoints/weights" / f"step_{int(spec['steps']):06d}.pt"
    if not final.is_file():
        raise FileNotFoundError(f"RoboTwin no-ERAF final checkpoint missing: {final}.")
    print(
        json.dumps(
            {
                "format": "pgc_robotwin_no_eraf_training_complete_v1",
                "mode": args.mode,
                "optimizer_steps": spec["steps"],
                "checkpoint": str(final),
                "full_goal_used": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
