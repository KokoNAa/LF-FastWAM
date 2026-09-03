#!/usr/bin/env python3
"""Collect, validate, convert, and bind five-pair RoboTwin ERAF data.

This entry point produces only matched expert grounding supervision.  Its
manifest explicitly forbids use as no-ERAF, joint-policy, or final full-goal
LoRA data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.robotwin.pgc_data import (
    ROBOTWIN_ERAF_PAIR_IDS,
    ROBOTWIN_ERAF_PAIR_SPECS,
)
from fastwam.datasets.pgc_libero import load_pgc_entity_relation_index
from scripts.build_pgc_robotwin_entity_relations import build_sidecar
from scripts.convert_pgc_robotwin_to_lerobot import convert_dataset
from scripts.validate_pgc_robotwin_raw import validate_raw_dataset


DATASET_KINDS = ("native", "counterfactual")
STAGE_EPISODES = {"smoke": 1, "formal": 5}
MANIFEST_NAME = "pgc_robotwin_eraf_prepared.json"


def raw_dataset_roots(raw_root: Path) -> list[Path]:
    return [
        raw_root / pair_id / dataset_kind
        for pair_id in ROBOTWIN_ERAF_PAIR_IDS
        for dataset_kind in DATASET_KINDS
    ]


def build_plan(
    *,
    robotwin_root: Path,
    work_root: Path,
    stage: str,
    episodes: int,
    task_config: str,
    start_seed: int,
    fps: int,
    video_codec: str,
    skip_collection: bool,
    python: str = sys.executable,
) -> dict[str, Any]:
    if stage not in STAGE_EPISODES:
        raise ValueError(f"Unsupported data stage: {stage!r}.")
    if episodes <= 0 or fps <= 0:
        raise ValueError("episodes and fps must be positive")
    if start_seed < 0:
        raise ValueError("start_seed must be non-negative")
    stage_root = work_root / stage / task_config
    raw_root = stage_root / "raw"
    dataset_root = stage_root / "lerobot"
    sidecar_root = stage_root / "eraf"
    roots = raw_dataset_roots(raw_root)
    collect = [
        python,
        "scripts/collect_pgc_robotwin_pairs.py",
        "--robotwin-root",
        str(robotwin_root),
        "--output-root",
        str(raw_root),
        "--task-config",
        task_config,
        "--episodes",
        str(episodes),
        "--start-seed",
        str(start_seed),
        "--source-tasks",
        *(spec.source_task for spec in ROBOTWIN_ERAF_PAIR_SPECS),
    ]
    return {
        "format": "pgc_robotwin_eraf_data_plan_v1",
        "stage": stage,
        "task_config": task_config,
        "episodes_per_dataset": episodes,
        "pair_count": len(ROBOTWIN_ERAF_PAIR_IDS),
        "dataset_count": len(roots),
        "total_successful_trajectories": episodes * len(roots),
        "start_seed": start_seed,
        "fps": fps,
        "video_codec": video_codec,
        "robotwin_root": str(robotwin_root),
        "stage_root": str(stage_root),
        "raw_root": str(raw_root),
        "dataset_root": str(dataset_root),
        "sidecar_root": str(sidecar_root),
        "prepared_manifest": str(dataset_root / MANIFEST_NAME),
        "skip_collection": skip_collection,
        "commands": {
            "collect": None if skip_collection else collect,
            "validate": [
                python,
                "scripts/validate_pgc_robotwin_raw.py",
                *(str(root) for root in roots),
            ],
        },
    }


def _require_modules(names: Sequence[str]) -> None:
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise ModuleNotFoundError(
            "RoboTwin ERAF data preparation is missing Python modules: "
            + ", ".join(missing)
        )


def _preflight(plan: dict[str, Any]) -> None:
    robotwin_root = Path(plan["robotwin_root"])
    task_config = str(plan["task_config"])
    required_paths = [
        robotwin_root / "script" / "collect_data.py",
        robotwin_root / "task_config" / f"{task_config}.yml",
        *(
            robotwin_root / "envs" / f"{spec.source_task}.py"
            for spec in ROBOTWIN_ERAF_PAIR_SPECS
        ),
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "RoboTwin checkout is incomplete for ERAF collection: " + ", ".join(missing)
        )
    raw_root = Path(plan["raw_root"])
    roots = raw_dataset_roots(raw_root)
    if plan["skip_collection"]:
        missing_raw = [str(path) for path in roots if not path.is_dir()]
        if missing_raw:
            raise FileNotFoundError(
                "--skip-collection requires all ten raw datasets: "
                + ", ".join(missing_raw)
            )
    elif raw_root.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing raw capture. Choose a new "
            f"--work-root or audit and remove this exact root: {raw_root}"
        )
    _require_modules(("h5py", "numpy", "yaml", "torch", "lerobot"))
    if not plan["skip_collection"]:
        _require_modules(("sapien",))


def _prepare_matrix(plan: dict[str, Any]) -> dict[str, Any]:
    raw_root = Path(plan["raw_root"])
    dataset_root = Path(plan["dataset_root"])
    sidecar_root = Path(plan["sidecar_root"])
    reports: list[dict[str, Any]] = []
    signatures: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for pair_id in ROBOTWIN_ERAF_PAIR_IDS:
        for dataset_kind in DATASET_KINDS:
            raw = raw_root / pair_id / dataset_kind
            dataset = dataset_root / pair_id / dataset_kind
            sidecar = sidecar_root / pair_id / dataset_kind
            raw_report = validate_raw_dataset(raw)
            if raw_report["pair_id"] != pair_id:
                raise ValueError(
                    f"Raw pair mismatch at {raw}: {raw_report['pair_id']}."
                )
            if raw_report["dataset_kind"] != dataset_kind:
                raise ValueError(
                    f"Raw dataset_kind mismatch at {raw}: "
                    f"{raw_report['dataset_kind']}."
                )
            signatures[(pair_id, dataset_kind)] = list(
                raw_report["_matched_signatures"]
            )
            dataset_exists = dataset.exists()
            sidecar_exists = (sidecar / "index.json").is_file()
            if dataset_exists != sidecar_exists:
                raise FileExistsError(
                    "Refusing ambiguous partial prepared output; dataset and "
                    f"sidecar must both exist or both be absent: {dataset}"
                )
            if not dataset_exists:
                convert_dataset(
                    raw_root=raw,
                    output=dataset,
                    fps=int(plan["fps"]),
                    video_codec=str(plan["video_codec"]),
                )
                build_sidecar(
                    raw_root=raw,
                    dataset_root=dataset,
                    output_root=sidecar,
                )
            index = load_pgc_entity_relation_index(sidecar)
            if Path(str(index["dataset"])).resolve() != dataset.resolve():
                raise ValueError(f"Sidecar is not bound to {dataset}.")
            if int(index["episode_count"]) != int(raw_report["episodes"]):
                raise ValueError(f"Prepared episode count mismatch at {sidecar}.")
            reports.append(
                {
                    "pair_id": pair_id,
                    "dataset_kind": dataset_kind,
                    "episodes": int(index["episode_count"]),
                    "dataset": str(dataset.resolve()),
                    "sidecar": str(sidecar.resolve()),
                    "camera_count": int(index["camera_count"]),
                    "action_dim": int(index["action_dim"]),
                    "artifact_role": "eraf_grounding_supervision",
                    "full_goal_verified": False,
                    "valid": True,
                }
            )
        if signatures[(pair_id, "native")] != signatures[(pair_id, "counterfactual")]:
            raise ValueError(
                f"Native/counterfactual scenes are unmatched for {pair_id}."
            )
    manifest = {
        "format": "pgc_robotwin_eraf_prepared_matrix_v1",
        "complete": True,
        "artifact_role": "eraf_grounding_supervision",
        "allowed_training_stages": ["grounding"],
        "forbidden_training_stages": [
            "no_eraf",
            "joint",
            "final_short_lora",
        ],
        "full_goal_usage": "not_present",
        "full_goal_verified": False,
        "same_scene_pair_ready": True,
        "pairs": list(ROBOTWIN_ERAF_PAIR_IDS),
        "datasets": reports,
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_root / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def run_plan(plan: dict[str, Any]) -> dict[str, Any]:
    _preflight(plan)
    collect = plan["commands"]["collect"]
    if collect is not None:
        print("[robotwin-eraf-data] collect", flush=True)
        subprocess.run(collect, cwd=PROJECT_ROOT, check=True)
    print("[robotwin-eraf-data] validate and prepare", flush=True)
    manifest = _prepare_matrix(plan)
    expected = int(plan["episodes_per_dataset"])
    bad_counts = [
        f"{row['pair_id']}/{row['dataset_kind']}={row['episodes']}"
        for row in manifest["datasets"]
        if int(row["episodes"]) != expected
    ]
    if bad_counts:
        raise RuntimeError("Prepared episode counts mismatch: " + ", ".join(bad_counts))
    return {
        "format": "pgc_robotwin_eraf_data_ready_v1",
        "stage": plan["stage"],
        "task_config": plan["task_config"],
        "pair_count": int(plan["pair_count"]),
        "dataset_count": int(plan["dataset_count"]),
        "episodes_per_dataset": expected,
        "total_successful_trajectories": int(plan["total_successful_trajectories"]),
        "prepared_manifest": plan["prepared_manifest"],
        "eraf_grounding_ready": True,
        "full_goal_verified": False,
        "allowed_training_stages": ["grounding"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "RoboTwin",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "pgc_robotwin_eraf_v1",
    )
    parser.add_argument("--stage", choices=tuple(STAGE_EPISODES), default="smoke")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--start-seed", type=int, default=4_400_000)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--video-codec",
        choices=("h264", "hevc", "libsvtav1", "h264_nvenc"),
        default="h264",
    )
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    episodes = args.episodes or STAGE_EPISODES[args.stage]
    try:
        plan = build_plan(
            robotwin_root=args.robotwin_root.expanduser().resolve(),
            work_root=args.work_root.expanduser().resolve(),
            stage=args.stage,
            episodes=episodes,
            task_config=args.task_config,
            start_seed=args.start_seed,
            fps=args.fps,
            video_codec=args.video_codec,
            skip_collection=args.skip_collection,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.dry_run:
        print(json.dumps(run_plan(plan), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
