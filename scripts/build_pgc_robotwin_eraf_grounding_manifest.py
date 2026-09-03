#!/usr/bin/env python3
"""Audit and combine formal RoboTwin ERAF grounding data shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.robotwin.pgc_data import ROBOTWIN_ERAF_PAIR_IDS
from fastwam.datasets.pgc_libero import load_pgc_entity_relation_index


SOURCE_FORMAT = "pgc_robotwin_eraf_prepared_matrix_v1"
OUTPUT_FORMAT = "pgc_robotwin_eraf_grounding_matrix_v1"
DATASET_KINDS = ("native", "counterfactual")
DOMAIN_EPISODES = (("demo_clean", 3), ("demo_randomized", 2))
FULL_GOAL_INDEX = Path("meta/pgc_robotwin_full_goal/index.json")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}.")
    return payload


def _sidecar_signatures(index: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (
            str(index["episodes_by_index"][episode]["initial_state_sha256"]),
            str(index["episodes_by_index"][episode]["pair_id"]),
        )
        for episode in range(int(index["episode_count"]))
    ]


def _validate_source_manifest(
    *, path: Path, task_config: str, expected_episodes: int
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[tuple[str, str]]]]:
    payload = _load_json(path)
    if payload.get("format") != SOURCE_FORMAT or payload.get("complete") is not True:
        raise ValueError(f"Incomplete/unsupported ERAF source manifest: {path}.")
    if payload.get("artifact_role") != "eraf_grounding_supervision":
        raise ValueError(f"Source manifest has the wrong artifact role: {path}.")
    if payload.get("allowed_training_stages") != ["grounding"]:
        raise ValueError(f"Source manifest is not grounding-only: {path}.")
    if payload.get("full_goal_verified") is not False:
        raise ValueError(f"Source manifest must explicitly reject full-goal: {path}.")
    if tuple(payload.get("pairs") or ()) != ROBOTWIN_ERAF_PAIR_IDS:
        raise ValueError(f"Source manifest pair catalog mismatch: {path}.")
    entries = payload.get("datasets")
    if not isinstance(entries, list):
        raise ValueError(f"Source manifest datasets must be a list: {path}.")
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"Source dataset entry must be an object: {path}.")
        key = (str(entry.get("pair_id")), str(entry.get("dataset_kind")))
        if key in by_key:
            raise ValueError(f"Duplicate source dataset entry {key}: {path}.")
        by_key[key] = entry
    expected = {
        (pair_id, dataset_kind)
        for pair_id in ROBOTWIN_ERAF_PAIR_IDS
        for dataset_kind in DATASET_KINDS
    }
    if set(by_key) != expected:
        raise ValueError(f"Source manifest matrix mismatch: {path}.")

    normalized: list[dict[str, Any]] = []
    signatures: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for pair_id in ROBOTWIN_ERAF_PAIR_IDS:
        for dataset_kind in DATASET_KINDS:
            entry = by_key[(pair_id, dataset_kind)]
            if entry.get("valid") is not True:
                raise ValueError(
                    f"Source dataset is not valid: {task_config}/{pair_id}/{dataset_kind}."
                )
            if int(entry.get("episodes", -1)) != expected_episodes:
                raise ValueError(
                    f"Unexpected episode count for {task_config}/{pair_id}/"
                    f"{dataset_kind}: {entry.get('episodes')}."
                )
            if entry.get("full_goal_verified") is not False:
                raise ValueError(
                    f"Full-goal leakage in {task_config}/{pair_id}/{dataset_kind}."
                )
            dataset = Path(str(entry.get("dataset", ""))).expanduser().resolve()
            sidecar = Path(str(entry.get("sidecar", ""))).expanduser().resolve()
            if not dataset.is_dir():
                raise FileNotFoundError(f"Prepared dataset not found: {dataset}.")
            if (dataset / FULL_GOAL_INDEX).exists():
                raise ValueError(f"Full-goal index leaked into grounding: {dataset}.")
            provenance_path = dataset / "meta/pgc_provenance.json"
            provenance = _load_json(provenance_path)
            if (
                provenance.get("artifact_role") != "eraf_grounding_supervision"
                or provenance.get("allowed_training_stages") != ["grounding"]
                or provenance.get("full_goal_verified") is not False
                or provenance.get("task_config") != task_config
            ):
                raise ValueError(
                    f"Dataset provenance is not grounding-only: {dataset}."
                )
            index = load_pgc_entity_relation_index(sidecar)
            if Path(str(index["dataset"])).resolve() != dataset:
                raise ValueError(f"Sidecar does not bind exact dataset: {sidecar}.")
            if (
                str(index.get("dataset_kind")) != dataset_kind
                or int(index.get("camera_count", -1)) != 3
                or int(index.get("action_dim", -1)) != 14
                or int(index.get("episode_count", -1)) != expected_episodes
                or index.get("artifact_role") != "eraf_grounding_supervision"
                or index.get("allowed_training_stages") != ["grounding"]
                or index.get("full_goal_verified") is not False
            ):
                raise ValueError(f"Sidecar contract mismatch: {sidecar}.")
            record_pair_ids = {
                str(record.get("pair_id", ""))
                for record in index["episodes_by_index"].values()
            }
            if record_pair_ids != {pair_id}:
                raise ValueError(f"Sidecar pair mismatch at {sidecar}.")
            signatures[(pair_id, dataset_kind)] = _sidecar_signatures(index)
            normalized.append(
                {
                    "task_config": task_config,
                    "pair_id": pair_id,
                    "dataset_kind": dataset_kind,
                    "episodes": expected_episodes,
                    "dataset": str(dataset),
                    "sidecar": str(sidecar),
                    "sidecar_index_sha256": _file_sha256(sidecar / "index.json"),
                    "artifact_role": "eraf_grounding_supervision",
                    "full_goal_verified": False,
                    "valid": True,
                }
            )
    for pair_id in ROBOTWIN_ERAF_PAIR_IDS:
        if signatures[(pair_id, "native")] != signatures[(pair_id, "counterfactual")]:
            raise ValueError(
                f"Native/counterfactual scenes are unmatched for "
                f"{task_config}/{pair_id}."
            )
    return normalized, signatures


def build_grounding_manifest(*, work_root: Path, output: Path | None = None) -> dict:
    work_root = work_root.expanduser().resolve()
    all_entries: list[dict[str, Any]] = []
    source_manifests = []
    domain_signatures = {}
    for task_config, expected_episodes in DOMAIN_EPISODES:
        source_path = (
            work_root
            / "formal"
            / task_config
            / "lerobot"
            / "pgc_robotwin_eraf_prepared.json"
        )
        entries, signatures = _validate_source_manifest(
            path=source_path,
            task_config=task_config,
            expected_episodes=expected_episodes,
        )
        all_entries.extend(entries)
        domain_signatures[task_config] = signatures
        source_manifests.append(
            {
                "task_config": task_config,
                "episodes_per_dataset": expected_episodes,
                "path": str(source_path),
                "sha256": _file_sha256(source_path),
            }
        )

    for pair_id in ROBOTWIN_ERAF_PAIR_IDS:
        clean_states = {
            state for state, _ in domain_signatures["demo_clean"][(pair_id, "native")]
        }
        randomized_states = {
            state
            for state, _ in domain_signatures["demo_randomized"][
                (
                    pair_id,
                    "native",
                )
            ]
        }
        if clean_states & randomized_states:
            raise ValueError(f"Clean/randomized state leakage detected for {pair_id}.")

    ordered_entries = [
        entry
        for dataset_kind in DATASET_KINDS
        for pair_id in ROBOTWIN_ERAF_PAIR_IDS
        for task_config, _ in DOMAIN_EPISODES
        for entry in all_entries
        if entry["dataset_kind"] == dataset_kind
        and entry["pair_id"] == pair_id
        and entry["task_config"] == task_config
    ]
    pair_episode_counts = {
        pair_id: {
            dataset_kind: sum(
                int(entry["episodes"])
                for entry in ordered_entries
                if entry["pair_id"] == pair_id and entry["dataset_kind"] == dataset_kind
            )
            for dataset_kind in DATASET_KINDS
        }
        for pair_id in ROBOTWIN_ERAF_PAIR_IDS
    }
    expected_counts = {
        pair_id: {"native": 5, "counterfactual": 5}
        for pair_id in ROBOTWIN_ERAF_PAIR_IDS
    }
    if pair_episode_counts != expected_counts:
        raise ValueError(f"Formal ERAF pair counts must be 5+5: {pair_episode_counts}.")
    payload = {
        "format": OUTPUT_FORMAT,
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
        "pair_count": len(ROBOTWIN_ERAF_PAIR_IDS),
        "dataset_count": len(ordered_entries),
        "total_successful_trajectories": sum(
            int(entry["episodes"]) for entry in ordered_entries
        ),
        "pairs": list(ROBOTWIN_ERAF_PAIR_IDS),
        "domains": {name: episodes for name, episodes in DOMAIN_EPISODES},
        "pair_episode_counts": pair_episode_counts,
        "source_manifests": source_manifests,
        "datasets": ordered_entries,
    }
    output = (
        output.expanduser().resolve()
        if output is not None
        else work_root / "formal" / "pgc_robotwin_eraf_grounding_manifest.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {**payload, "manifest": str(output), "sha256": _file_sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_grounding_manifest(
        work_root=args.work_root,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
