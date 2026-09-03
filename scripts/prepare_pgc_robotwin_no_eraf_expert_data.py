#!/usr/bin/env python3
"""Prepare non-full-goal RoboTwin expert pools for no-ERAF training.

The historical profile produces paired offline-native and historical-CF
datasets.  The strict profile retains the native replay only as a same-scene
audit and exposes only the counterfactual dataset to the strict-CF pool.
Neither profile is allowed to contain a full-goal corrective record.
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
from scripts.collect_pgc_robotwin_pairs import collection_contract
from scripts.convert_pgc_robotwin_to_lerobot import convert_dataset
from scripts.validate_pgc_robotwin_raw import validate_raw_dataset


DATASET_KINDS = ("native", "counterfactual")
FULL_GOAL_INDEX = Path("meta/pgc_robotwin_full_goal/index.json")
PROFILES = {
    "historical": "no_eraf_historical",
    "strict": "no_eraf_strict",
}


def raw_dataset_roots(raw_root: Path) -> list[Path]:
    return [
        raw_root / pair_id / kind
        for pair_id in ROBOTWIN_ERAF_PAIR_IDS
        for kind in DATASET_KINDS
    ]


def build_plan(
    *,
    robotwin_root: Path,
    work_root: Path,
    profile: str,
    task_config: str,
    episodes: int,
    start_seed: int,
    fps: int,
    video_codec: str,
    skip_collection: bool,
    python: str = sys.executable,
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"Unsupported no-ERAF expert profile: {profile!r}.")
    if episodes <= 0 or fps <= 0 or start_seed < 0:
        raise ValueError("episodes/fps must be positive and seed non-negative")
    profile_root = work_root / "expert" / profile / task_config
    raw_root = profile_root / "raw"
    dataset_root = profile_root / "lerobot"
    sidecar_root = profile_root / "sidecars"
    collector_profile = PROFILES[profile]
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
        "--collection-profile",
        collector_profile,
        "--source-tasks",
        *(spec.source_task for spec in ROBOTWIN_ERAF_PAIR_SPECS),
    ]
    return {
        "format": "pgc_robotwin_no_eraf_expert_plan_v1",
        "profile": profile,
        "collector_profile": collector_profile,
        "task_config": task_config,
        "episodes_per_pair_kind": episodes,
        "pair_count": len(ROBOTWIN_ERAF_PAIR_IDS),
        "raw_trajectory_count": episodes * len(ROBOTWIN_ERAF_PAIR_IDS) * 2,
        "start_seed": start_seed,
        "fps": fps,
        "video_codec": video_codec,
        "robotwin_root": str(robotwin_root),
        "profile_root": str(profile_root),
        "raw_root": str(raw_root),
        "dataset_root": str(dataset_root),
        "sidecar_root": str(sidecar_root),
        "prepared_manifest": str(dataset_root / "prepared.json"),
        "skip_collection": skip_collection,
        "commands": {
            "collect": None if skip_collection else collect,
            "validate": [
                python,
                "scripts/validate_pgc_robotwin_raw.py",
                *(str(path) for path in raw_dataset_roots(raw_root)),
            ],
        },
    }


def _require_modules(names: Sequence[str]) -> None:
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise ModuleNotFoundError(
            "RoboTwin no-ERAF expert preparation is missing modules: "
            + ", ".join(missing)
        )


def _preflight(plan: dict[str, Any]) -> None:
    robotwin_root = Path(plan["robotwin_root"])
    task_config = str(plan["task_config"])
    required = [
        robotwin_root / "script" / "collect_data.py",
        robotwin_root / "task_config" / f"{task_config}.yml",
        *(
            robotwin_root / "envs" / f"{spec.source_task}.py"
            for spec in ROBOTWIN_ERAF_PAIR_SPECS
        ),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete RoboTwin checkout: " + ", ".join(missing))
    raw_root = Path(plan["raw_root"])
    if plan["skip_collection"]:
        missing_raw = [
            str(path) for path in raw_dataset_roots(raw_root) if not path.is_dir()
        ]
        if missing_raw:
            raise FileNotFoundError(
                "--skip-collection requires all paired raw datasets: "
                + ", ".join(missing_raw)
            )
    elif raw_root.exists():
        raise FileExistsError(f"Refusing to overwrite raw capture: {raw_root}")
    _require_modules(("h5py", "numpy", "yaml", "torch"))
    if not plan["skip_collection"]:
        _require_modules(("sapien",))


def _prepare(plan: dict[str, Any]) -> dict[str, Any]:
    profile = str(plan["profile"])
    collector_profile = str(plan["collector_profile"])
    raw_root = Path(plan["raw_root"])
    dataset_root = Path(plan["dataset_root"])
    sidecar_root = Path(plan["sidecar_root"])
    expected_episodes = int(plan["episodes_per_pair_kind"])
    signatures: dict[tuple[str, str], list[tuple[int, str]]] = {}
    pools: dict[str, list[dict[str, Any]]] = {
        "offline_native": [],
        "historical_cf": [],
        "strict_cf": [],
    }
    for pair_id in ROBOTWIN_ERAF_PAIR_IDS:
        raw_reports: dict[str, dict[str, Any]] = {}
        for kind in DATASET_KINDS:
            raw = raw_root / pair_id / kind
            report = validate_raw_dataset(raw)
            raw_reports[kind] = report
            if report["pair_id"] != pair_id or report["dataset_kind"] != kind:
                raise ValueError(f"Raw pool identity mismatch: {raw}")
            if int(report["episodes"]) != expected_episodes:
                raise ValueError(f"Raw episode count mismatch: {raw}")
            signatures[(pair_id, kind)] = list(report["_matched_signatures"])
            provenance = json.loads(
                (raw / "meta/pgc_provenance.json").read_text(encoding="utf-8")
            )
            contract = collection_contract(collector_profile, kind)
            if (
                provenance.get("collection_profile") != collector_profile
                or provenance.get("artifact_role") != contract["artifact_role"]
                or provenance.get("allowed_training_stages")
                != contract["allowed_training_stages"]
                or provenance.get("full_goal_verified") is True
            ):
                raise ValueError(f"Raw no-ERAF provenance mismatch: {raw}")

        if signatures[(pair_id, "native")] != signatures[
            (pair_id, "counterfactual")
        ]:
            raise ValueError(f"Same-scene audit failed for {pair_id}.")

        selected_kinds = DATASET_KINDS if profile == "historical" else ("counterfactual",)
        for kind in selected_kinds:
            raw = raw_root / pair_id / kind
            dataset = dataset_root / pair_id / kind
            sidecar = sidecar_root / pair_id / kind
            dataset_exists = dataset.exists()
            sidecar_exists = (sidecar / "index.json").is_file()
            if dataset_exists != sidecar_exists:
                raise FileExistsError(f"Partial prepared output: {dataset}")
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
            if (dataset / FULL_GOAL_INDEX).exists():
                raise ValueError(f"full-goal leaked into no-ERAF pool: {dataset}")
            index = load_pgc_entity_relation_index(sidecar)
            contract = collection_contract(collector_profile, kind)
            if (
                Path(str(index["dataset"])).resolve() != dataset.resolve()
                or index.get("artifact_role") != contract["artifact_role"]
                or index.get("allowed_training_stages")
                != contract["allowed_training_stages"]
                or int(index["episode_count"]) != expected_episodes
            ):
                raise ValueError(f"Prepared no-ERAF sidecar mismatch: {sidecar}")
            pool = (
                "offline_native"
                if kind == "native"
                else ("historical_cf" if profile == "historical" else "strict_cf")
            )
            pools[pool].append(
                {
                    "pair_id": pair_id,
                    "task_config": plan["task_config"],
                    "dataset": str(dataset.resolve()),
                    "sidecar": str(sidecar.resolve()),
                    "episodes": expected_episodes,
                    "dataset_kind": kind,
                    "artifact_role": contract["artifact_role"],
                    "full_goal_verified": False,
                    "valid": True,
                }
            )

    active_pools = (
        ("offline_native", "historical_cf")
        if profile == "historical"
        else ("strict_cf",)
    )
    manifest = {
        "format": "pgc_robotwin_no_eraf_expert_prepared_v1",
        "complete": True,
        "profile": profile,
        "task_config": plan["task_config"],
        "full_goal_usage": "forbidden_not_present",
        "allowed_training_stages": ["no_eraf", "joint", "final_short_lora"],
        "pair_ids": list(ROBOTWIN_ERAF_PAIR_IDS),
        "pools": {name: pools[name] for name in active_pools},
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    output = dataset_root / "prepared.json"
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_plan(plan: dict[str, Any]) -> dict[str, Any]:
    _preflight(plan)
    collect = plan["commands"]["collect"]
    if collect is not None:
        print("[robotwin-no-eraf] collect expert pairs", flush=True)
        subprocess.run(collect, cwd=PROJECT_ROOT, check=True)
    print("[robotwin-no-eraf] validate and prepare expert pools", flush=True)
    manifest = _prepare(plan)
    return {
        "format": "pgc_robotwin_no_eraf_expert_ready_v1",
        "profile": plan["profile"],
        "task_config": plan["task_config"],
        "prepared_manifest": plan["prepared_manifest"],
        "pool_counts": {
            name: sum(int(row["episodes"]) for row in rows)
            for name, rows in manifest["pools"].items()
        },
        "full_goal_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=PROJECT_ROOT / "third_party/RoboTwin",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=PROJECT_ROOT / "data/pgc_robotwin_no_eraf_v1",
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--video-codec",
        choices=("h264", "hevc", "libsvtav1", "h264_nvenc"),
        default="h264",
    )
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        plan = build_plan(
            robotwin_root=args.robotwin_root.expanduser().resolve(),
            work_root=args.work_root.expanduser().resolve(),
            profile=args.profile,
            task_config=args.task_config,
            episodes=args.episodes,
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
