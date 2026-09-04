#!/usr/bin/env python3
"""Assemble and audit the four-pool RoboTwin no-ERAF training manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.robotwin.closed_loop_capture import (
    CAPTURE_ACTION_VIDEO_FREQ_RATIO,
    CAPTURE_FORMAT,
    CAPTURE_FRAME_COUNT,
    CAPTURE_PRODUCTIVE_START_COUNT,
    CAPTURE_STATE_DISTRIBUTION,
    CAPTURE_TEMPORAL_CONTRACT,
)
from experiments.robotwin.pgc_data import ROBOTWIN_ERAF_PAIR_IDS
from fastwam.datasets.pgc_libero import load_pgc_entity_relation_index


MANIFEST_FORMAT = "pgc_robotwin_no_eraf_four_pool_v2"
POOL_ORDER = (
    "offline_native",
    "closed_loop_native",
    "historical_cf",
    "strict_cf",
)
EXPECTED_ROLES = dict(zip(POOL_ORDER, POOL_ORDER, strict=True))
EXPECTED_KINDS = {
    "offline_native": "native",
    "closed_loop_native": "native",
    "historical_cf": "counterfactual",
    "strict_cf": "counterfactual",
}
FULL_GOAL_INDEX = Path("meta/pgc_robotwin_full_goal/index.json")
CLOSED_LOOP_INDEX = Path("meta/pgc_robotwin_closed_loop_native.json")
ALLOWED_STAGES = ["no_eraf", "joint", "final_short_lora"]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}.")
    return payload


def _normalize_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(entry),
        "dataset": str(Path(str(entry["dataset"])).expanduser().resolve()),
        "sidecar": str(Path(str(entry["sidecar"])).expanduser().resolve()),
        "episodes": int(entry["episodes"]),
    }


def _expert_entries(
    manifests: Sequence[Path], *, pool: str, profile: str
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_configs: set[str] = set()
    for raw_path in manifests:
        path = raw_path.expanduser().resolve()
        payload = _load_json(path)
        if (
            payload.get("format") != "pgc_robotwin_no_eraf_expert_prepared_v1"
            or payload.get("complete") is not True
            or payload.get("profile") != profile
            or payload.get("full_goal_usage") != "forbidden_not_present"
        ):
            raise ValueError(f"Invalid no-ERAF expert manifest: {path}.")
        config = str(payload.get("task_config", ""))
        if not config or config in seen_configs:
            raise ValueError(f"Duplicate/missing task_config in {path}.")
        seen_configs.add(config)
        raw_entries = payload.get("pools", {}).get(pool)
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError(f"Manifest {path} has no {pool} pool.")
        for raw_entry in raw_entries:
            entry = _normalize_entry(raw_entry)
            if entry.get("task_config") != config:
                raise ValueError(f"Expert entry/config mismatch in {path}.")
            entries.append(entry)
    return entries


def _closed_loop_entry(dataset: Path, sidecar: Path) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    sidecar = sidecar.expanduser().resolve()
    index = _load_json(dataset / CLOSED_LOOP_INDEX)
    expected = {
        "format": "pgc_robotwin_closed_loop_native_v2",
        "complete": True,
        "state_distribution": CAPTURE_STATE_DISTRIBUTION,
        "full_goal_usage": "forbidden_not_present",
        "capture_format": CAPTURE_FORMAT,
        "capture_frame_count": CAPTURE_FRAME_COUNT,
        "action_video_freq_ratio": CAPTURE_ACTION_VIDEO_FREQ_RATIO,
        "productive_start_count_per_episode": CAPTURE_PRODUCTIVE_START_COUNT,
        "temporal_contract": CAPTURE_TEMPORAL_CONTRACT,
    }
    mismatches = {
        key: (index.get(key), value)
        for key, value in expected.items()
        if index.get(key) != value
    }
    episode_count = int(index.get("episode_count", 0))
    frame_count = int(index.get("frame_count", 0))
    productive_frame_count = int(index.get("productive_frame_count", 0))
    if (
        mismatches
        or episode_count <= 0
        or frame_count != episode_count * CAPTURE_FRAME_COUNT
        or productive_frame_count
        != episode_count * CAPTURE_PRODUCTIVE_START_COUNT
    ):
        raise ValueError(
            "Invalid/productively unusable closed-loop native index: "
            f"{dataset / CLOSED_LOOP_INDEX}; mismatches={mismatches}."
        )
    return {
        "dataset": str(dataset),
        "sidecar": str(sidecar),
        "episodes": episode_count,
        "frames": frame_count,
        "productive_frames": productive_frame_count,
        "capture_format": CAPTURE_FORMAT,
        "capture_frame_count": CAPTURE_FRAME_COUNT,
        "action_video_freq_ratio": CAPTURE_ACTION_VIDEO_FREQ_RATIO,
        "productive_start_count_per_episode": CAPTURE_PRODUCTIVE_START_COUNT,
        "temporal_contract": CAPTURE_TEMPORAL_CONTRACT,
        "stage_counts": dict(index.get("stage_counts") or {}),
        "dataset_kind": "native",
        "artifact_role": "closed_loop_native",
        "full_goal_verified": False,
        "valid": True,
    }


def build_manifest(
    *,
    historical_manifests: Sequence[Path],
    strict_manifests: Sequence[Path],
    closed_loop_dataset: Path,
    closed_loop_sidecar: Path,
) -> dict[str, Any]:
    offline = _expert_entries(
        historical_manifests, pool="offline_native", profile="historical"
    )
    historical = _expert_entries(
        historical_manifests, pool="historical_cf", profile="historical"
    )
    strict = _expert_entries(strict_manifests, pool="strict_cf", profile="strict")
    closed = [_closed_loop_entry(closed_loop_dataset, closed_loop_sidecar)]
    return {
        "format": MANIFEST_FORMAT,
        "complete": True,
        "training_stage": "no_eraf",
        "sampling_contract": "deterministic_1_1_1_1",
        "pool_order": list(POOL_ORDER),
        "full_goal_usage": "forbidden_not_present",
        "allowed_training_stage": "no_eraf",
        "pools": {
            "offline_native": offline,
            "closed_loop_native": closed,
            "historical_cf": historical,
            "strict_cf": strict,
        },
    }


def _entry_pair_ids(index: Mapping[str, Any]) -> set[str]:
    pair_ids = {
        str(record.get("pair_id", "")).split("::", 1)[0].strip()
        for record in index["episodes_by_index"].values()
    }
    if not pair_ids or "" in pair_ids:
        raise ValueError("Every no-ERAF sidecar row must have a pair_id.")
    return pair_ids


def load_no_eraf_manifest(path: Path, *, verify_files: bool = True) -> dict[str, Any]:
    """Load one manifest and enforce full-goal exclusion plus pool ordering."""

    path = path.expanduser().resolve()
    payload = _load_json(path)
    if payload.get("format") != MANIFEST_FORMAT or payload.get("complete") is not True:
        raise ValueError(f"Incomplete/unsupported no-ERAF manifest: {path}.")
    if (
        payload.get("training_stage") != "no_eraf"
        or payload.get("sampling_contract") != "deterministic_1_1_1_1"
        or payload.get("pool_order") != list(POOL_ORDER)
        or payload.get("full_goal_usage") != "forbidden_not_present"
    ):
        raise ValueError("RoboTwin no-ERAF stage/sampling/full-goal contract changed.")
    pools = payload.get("pools")
    if not isinstance(pools, Mapping) or tuple(pools) != POOL_ORDER:
        raise ValueError(
            f"RoboTwin no-ERAF pools must be ordered exactly {POOL_ORDER}."
        )

    datasets: dict[str, list[str]] = {pool: [] for pool in POOL_ORDER}
    sidecars: dict[str, list[str]] = {pool: [] for pool in POOL_ORDER}
    episodes: Counter[str] = Counter()
    pair_ids: dict[str, set[str]] = {pool: set() for pool in POOL_ORDER}
    seen_datasets: set[Path] = set()
    seen_sidecars: set[Path] = set()
    for pool in POOL_ORDER:
        entries = pools[pool]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"RoboTwin no-ERAF pool is empty: {pool}.")
        for entry in entries:
            if not isinstance(entry, Mapping) or entry.get("valid") is not True:
                raise ValueError(f"Invalid RoboTwin no-ERAF {pool} entry.")
            dataset = Path(str(entry.get("dataset", ""))).expanduser().resolve()
            sidecar = Path(str(entry.get("sidecar", ""))).expanduser().resolve()
            if dataset in seen_datasets or sidecar in seen_sidecars:
                raise ValueError("RoboTwin no-ERAF datasets/sidecars must be unique.")
            seen_datasets.add(dataset)
            seen_sidecars.add(sidecar)
            if entry.get("artifact_role") != EXPECTED_ROLES[pool]:
                raise ValueError(f"Wrong artifact_role for {pool}: {dataset}.")
            if entry.get("dataset_kind") != EXPECTED_KINDS[pool]:
                raise ValueError(f"Wrong dataset_kind for {pool}: {dataset}.")
            if entry.get("full_goal_verified") is True:
                raise ValueError(
                    f"full-goal leaked into no-ERAF pool {pool}: {dataset}."
                )
            if int(entry.get("episodes", 0)) <= 0:
                raise ValueError(f"No episodes in no-ERAF pool {pool}: {dataset}.")
            if pool == "closed_loop_native":
                expected_temporal = {
                    "capture_format": CAPTURE_FORMAT,
                    "capture_frame_count": CAPTURE_FRAME_COUNT,
                    "action_video_freq_ratio": CAPTURE_ACTION_VIDEO_FREQ_RATIO,
                    "productive_start_count_per_episode": (
                        CAPTURE_PRODUCTIVE_START_COUNT
                    ),
                    "temporal_contract": CAPTURE_TEMPORAL_CONTRACT,
                }
                temporal_mismatches = {
                    key: (entry.get(key), value)
                    for key, value in expected_temporal.items()
                    if entry.get(key) != value
                }
                expected_frames = int(entry["episodes"]) * CAPTURE_FRAME_COUNT
                expected_productive = (
                    int(entry["episodes"]) * CAPTURE_PRODUCTIVE_START_COUNT
                )
                if (
                    temporal_mismatches
                    or int(entry.get("frames", 0)) != expected_frames
                    or int(entry.get("productive_frames", 0))
                    != expected_productive
                ):
                    raise ValueError(
                        "Closed-loop native entry violates the productive "
                        f"temporal contract: {dataset}; "
                        f"mismatches={temporal_mismatches}."
                    )
            if verify_files:
                if not dataset.is_dir():
                    raise FileNotFoundError(f"no-ERAF dataset not found: {dataset}.")
                if (dataset / FULL_GOAL_INDEX).exists():
                    raise ValueError(f"full-goal index leaked into {pool}: {dataset}.")
                provenance = _load_json(dataset / "meta/pgc_provenance.json")
                if (
                    provenance.get("artifact_role") != EXPECTED_ROLES[pool]
                    or provenance.get("dataset_kind") != EXPECTED_KINDS[pool]
                    or provenance.get("allowed_training_stages") != ALLOWED_STAGES
                    or provenance.get("full_goal_verified") is True
                ):
                    raise ValueError(
                        f"Dataset provenance mismatch in {pool}: {dataset}."
                    )
                if pool == "closed_loop_native":
                    expected_temporal = {
                        "capture_format": CAPTURE_FORMAT,
                        "capture_frame_count": CAPTURE_FRAME_COUNT,
                        "action_video_freq_ratio": CAPTURE_ACTION_VIDEO_FREQ_RATIO,
                        "productive_start_count_per_episode": (
                            CAPTURE_PRODUCTIVE_START_COUNT
                        ),
                        "productive_frame_count": int(entry["productive_frames"]),
                        "temporal_contract": CAPTURE_TEMPORAL_CONTRACT,
                    }
                    if any(
                        provenance.get(key) != value
                        for key, value in expected_temporal.items()
                    ):
                        raise ValueError(
                            "Closed-loop native provenance violates the "
                            f"productive temporal contract: {dataset}."
                        )
                index = load_pgc_entity_relation_index(sidecar)
                if (
                    Path(str(index["dataset"])).resolve() != dataset
                    or index.get("artifact_role") != EXPECTED_ROLES[pool]
                    or index.get("dataset_kind") != EXPECTED_KINDS[pool]
                    or index.get("allowed_training_stages") != ALLOWED_STAGES
                    or index.get("full_goal_verified") is True
                    or int(index.get("camera_count", -1)) != 3
                    or int(index.get("action_dim", -1)) != 14
                    or int(index.get("episode_count", -1)) != int(entry["episodes"])
                ):
                    raise ValueError(f"Sidecar contract mismatch in {pool}: {sidecar}.")
                if (
                    pool == "closed_loop_native"
                    and (
                        index.get("state_distribution")
                        != CAPTURE_STATE_DISTRIBUTION
                        or index.get("capture_format") != CAPTURE_FORMAT
                        or int(index.get("capture_frame_count", -1))
                        != CAPTURE_FRAME_COUNT
                        or int(index.get("action_video_freq_ratio", -1))
                        != CAPTURE_ACTION_VIDEO_FREQ_RATIO
                        or int(index.get("productive_start_count_per_episode", -1))
                        != CAPTURE_PRODUCTIVE_START_COUNT
                        or int(index.get("productive_frame_count", -1))
                        != int(entry["productive_frames"])
                        or index.get("temporal_contract")
                        != CAPTURE_TEMPORAL_CONTRACT
                    )
                ):
                    raise ValueError(
                        "Closed-loop native sidecar has an invalid productive "
                        "temporal contract."
                    )
                pair_ids[pool].update(_entry_pair_ids(index))
            else:
                pair_ids[pool].update(
                    str(value).split("::", 1)[0] for value in entry.get("pair_ids", [])
                )
            datasets[pool].append(str(dataset))
            sidecars[pool].append(str(sidecar))
            episodes[pool] += int(entry["episodes"])

    if verify_files:
        expected_pairs = set(ROBOTWIN_ERAF_PAIR_IDS)
        for pool, actual_pairs in pair_ids.items():
            if actual_pairs != expected_pairs:
                raise ValueError(
                    f"RoboTwin no-ERAF pair coverage mismatch for {pool}: "
                    f"expected={sorted(expected_pairs)} actual={sorted(actual_pairs)}."
                )
    return {
        "manifest": str(path),
        "pool_order": list(POOL_ORDER),
        "offline_native_dirs": datasets["offline_native"],
        "closed_loop_native_dirs": datasets["closed_loop_native"],
        "historical_cf_dirs": datasets["historical_cf"],
        "strict_cf_dirs": datasets["strict_cf"],
        "sidecar_dirs": [value for pool in POOL_ORDER for value in sidecars[pool]],
        "dataset_counts": {pool: len(datasets[pool]) for pool in POOL_ORDER},
        "episode_counts": dict(episodes),
        "full_goal_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("/root/gpufree-data/pgc_robotwin_no_eraf_v1/formal"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.work_root.expanduser().resolve()
    historical = [
        root / "expert/historical" / config / "lerobot/prepared.json"
        for config in ("demo_clean", "demo_randomized")
    ]
    strict = [
        root / "expert/strict" / config / "lerobot/prepared.json"
        for config in ("demo_clean", "demo_randomized")
    ]
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "no_eraf_four_pool_manifest.json"
    )
    payload = build_manifest(
        historical_manifests=historical,
        strict_manifests=strict,
        closed_loop_dataset=root / "closed_loop_native/lerobot",
        closed_loop_sidecar=root / "closed_loop_native/sidecar",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validated = load_no_eraf_manifest(output)
    print(
        json.dumps(
            {
                "format": "pgc_robotwin_no_eraf_four_pool_ready_v2",
                "complete": True,
                **validated,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
