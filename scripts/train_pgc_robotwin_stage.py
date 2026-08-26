#!/usr/bin/env python3
"""Validate and launch staged PGC training on matched RoboTwin datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from fastwam.datasets.pgc_libero import load_pgc_entity_relation_index
from scripts.prepare_pgc_robotwin_datasets import PAIR_IDS


PREPARED_FORMAT = "pgc_robotwin_prepared_matrix_v1"
KINDS = ("native", "counterfactual")
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def _sidecar_signatures(index: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (
            str(index["episodes_by_index"][episode]["initial_state_sha256"]),
            str(index["episodes_by_index"][episode]["pair_id"]),
        )
        for episode in range(int(index["episode_count"]))
    ]


def load_training_matrix(path: Path) -> dict[str, list[str]]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != PREPARED_FORMAT or payload.get("complete") is not True:
        raise ValueError(f"Incomplete/unsupported RoboTwin prepared matrix: {path}.")
    entries = payload.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("Prepared matrix datasets must be a list.")
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Prepared matrix dataset entries must be objects.")
        key = (str(entry.get("pair_id")), str(entry.get("dataset_kind")))
        if key in by_key:
            raise ValueError(f"Duplicate prepared dataset entry: {key}.")
        by_key[key] = entry
    expected = {(pair_id, kind) for pair_id in PAIR_IDS for kind in KINDS}
    if set(by_key) != expected:
        raise ValueError(
            f"Prepared RoboTwin matrix must contain exactly {sorted(expected)}; "
            f"got {sorted(by_key)}."
        )

    datasets: dict[tuple[str, str], str] = {}
    sidecars: dict[tuple[str, str], str] = {}
    indices: dict[tuple[str, str], Mapping[str, Any]] = {}
    for key in sorted(expected):
        entry = by_key[key]
        if entry.get("valid") is not True:
            raise ValueError(f"Prepared dataset is not valid: {key}.")
        dataset = Path(str(entry["dataset"])).expanduser().resolve()
        sidecar = Path(str(entry["sidecar"])).expanduser().resolve()
        if not dataset.is_dir():
            raise FileNotFoundError(f"Prepared LeRobot dataset not found: {dataset}.")
        index = load_pgc_entity_relation_index(sidecar)
        if Path(str(index["dataset"])).resolve() != dataset:
            raise ValueError(f"Sidecar does not bind exact dataset for {key}.")
        pair_id, kind = key
        if str(index["dataset_kind"]) != kind:
            raise ValueError(f"Sidecar dataset_kind mismatch for {key}.")
        if int(index["camera_count"]) != 3 or int(index["action_dim"]) != 14:
            raise ValueError(f"RoboTwin 3-camera/14-D contract mismatch for {key}.")
        record_pair_ids = {
            str(record.get("pair_id", ""))
            for record in index["episodes_by_index"].values()
        }
        if record_pair_ids != {pair_id}:
            raise ValueError(f"Sidecar pair IDs mismatch for {key}: {record_pair_ids}.")
        datasets[key] = str(dataset)
        sidecars[key] = str(sidecar)
        indices[key] = index
    for pair_id in PAIR_IDS:
        if _sidecar_signatures(indices[(pair_id, "native")]) != _sidecar_signatures(
            indices[(pair_id, "counterfactual")]
        ):
            raise ValueError(f"Native/counterfactual scenes are unmatched: {pair_id}.")

    ordered_keys = [
        *((pair_id, "native") for pair_id in PAIR_IDS),
        *((pair_id, "counterfactual") for pair_id in PAIR_IDS),
    ]
    return {
        "dataset_dirs": [datasets[key] for key in ordered_keys[: len(PAIR_IDS)]],
        "counterfactual_dirs": [
            datasets[key] for key in ordered_keys[len(PAIR_IDS) :]
        ],
        "sidecar_dirs": [sidecars[key] for key in ordered_keys],
    }


def build_overrides(
    *, matrix: Mapping[str, list[str]], base_checkpoint: Path, seed: int, steps: int,
    stats_path: Path | None, cache_dir: Path,
) -> list[str]:
    overrides = [
        "task=robotwin_pgc_3cam_384",
        f"resume={base_checkpoint}",
        "weight_only_start_step=null",
        "data.val=null",
        f"data.train.dataset_dirs={json.dumps(matrix['dataset_dirs'], separators=(',', ':'))}",
        "data.train.pgc_counterfactual_dataset_dirs="
        + json.dumps(matrix["counterfactual_dirs"], separators=(",", ":")),
        "data.train.pgc_counterfactual_oversample_factor=1",
        "data.train.pgc_balance_native_counterfactual=true",
        "data.train.pgc_pair_balanced_sampling=true",
        "data.train.pgc_entity_relation_supervision_required=true",
        "data.train.pgc_entity_relation_sidecar_dirs="
        + json.dumps(matrix["sidecar_dirs"], separators=(",", ":")),
        "data.train.pgc_v9_balanced_sampling=false",
        "data.train.pgc_v9_structured_role_sampling=false",
        f"data.train.text_embedding_cache_dir={cache_dir}",
        f"seed={seed}",
        f"max_steps={steps}",
        "num_epochs=1",
        "num_workers=0",
        "gradient_accumulation_steps=1",
        "learning_rate=1.0e-4",
        "lr_scheduler_type=constant",
        "log_every=1",
        f"save_every={steps}",
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
        "model.policy_guard.entity_relation_grounding.initialization_contract=released_base_fresh_eraf",
        "model.policy_guard.entity_relation_grounding.grounding_objective_version=2",
        "model.policy_guard.entity_relation_grounding.completion_only_memory=false",
        "model.policy_guard.entity_relation_grounding.action_joint_training=false",
        "model.lora.enabled=false",
    ]
    overrides.append(
        "data.train.pretrained_norm_stats="
        + ("null" if stats_path is None else str(stats_path))
    )
    return overrides


def _required_prompts(matrix: Mapping[str, list[str]]) -> set[str]:
    prompts: set[str] = set()
    for dataset in matrix["dataset_dirs"] + matrix["counterfactual_dirs"]:
        provenance_path = Path(dataset) / "meta" / "pgc_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        for pair in provenance.get("pairs") or []:
            for key in ("source_instruction", "counterfactual_instruction"):
                prompts.add(DEFAULT_PROMPT.format(task=str(pair[key])))
    return prompts


def _cache_is_complete(matrix: Mapping[str, list[str]], cache_dir: Path) -> bool:
    return all(
        (
            cache_dir
            / f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}.t5_len128.wan22ti2v5b.pt"
        ).is_file()
        for prompt in _required_prompts(matrix)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("grounding",))
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/text_embeds_cache/robotwin"))
    parser.add_argument("--precompute-text", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--run-tag")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpus <= 0 or args.steps <= 0 or args.seed < 0:
        parser.error("--gpus/--steps must be positive and --seed non-negative")
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    if not base_checkpoint.is_file():
        raise FileNotFoundError(f"RoboTwin Base checkpoint not found: {base_checkpoint}.")
    stats_path = args.stats_path.expanduser().resolve() if args.stats_path else None
    if stats_path is not None and not stats_path.is_file():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}.")
    cache_dir = args.cache_dir.expanduser().resolve()
    matrix = load_training_matrix(args.prepared_manifest)
    overrides = build_overrides(
        matrix=matrix,
        base_checkpoint=base_checkpoint,
        seed=args.seed,
        steps=args.steps,
        stats_path=stats_path,
        cache_dir=cache_dir,
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
    train_command = [
        "bash",
        "scripts/train_zero1.sh",
        str(args.gpus),
        *overrides,
    ]
    run_tag = args.run_tag or (
        f"pgc-robotwin-v91-grounding-smoke{args.steps}-seed{args.seed}"
    )
    plan = {
        "format": "pgc_robotwin_training_launch_v1",
        "stage": args.stage,
        "gpus": args.gpus,
        "steps": args.steps,
        "run_tag": run_tag,
        "precompute_text": precompute,
        "dataset_dirs": matrix["dataset_dirs"],
        "counterfactual_dirs": matrix["counterfactual_dirs"],
        "sidecar_dirs": matrix["sidecar_dirs"],
        "train_command": train_command,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if args.precompute_text == "never" and not _cache_is_complete(matrix, cache_dir):
        raise FileNotFoundError(
            "RoboTwin source/counterfactual text cache is incomplete and "
            "--precompute-text=never was requested."
        )
    if precompute:
        subprocess.run(precompute_command, cwd=PROJECT_ROOT, check=True)
    env = os.environ.copy()
    env["RUN_ID"] = run_tag
    os.execvpe(train_command[0], train_command, env)


if __name__ == "__main__":
    main()
