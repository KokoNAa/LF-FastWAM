#!/usr/bin/env python3
"""Validate hashes and same-scene provenance of raw RoboTwin PGC captures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.robotwin.pgc_data import (
    PGC_ROBOTWIN_RAW_FORMAT,
    array_sha256,
    validate_action_array,
    validate_pair_record,
)
from fastwam.datasets.pgc_libero import PGC_DATA_FORMAT, read_jsonl


def _actions(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        for key in ("joint_action/vector", "observation/joint_action/vector", "action", "actions"):
            if key in handle:
                return validate_action_array(np.asarray(handle[key]))
        matches = []

        def visit(name, value):
            if isinstance(value, h5py.Dataset) and value.ndim == 2 and value.shape[-1] == 14:
                matches.append(name)

        handle.visititems(visit)
        if len(matches) != 1:
            raise ValueError(f"Expected one [T,14] dataset in {path}; got {matches}.")
        return validate_action_array(np.asarray(handle[matches[0]]))


def validate_raw_dataset(root: Path) -> dict:
    provenance_path = root / "meta" / "pgc_provenance.json"
    audit_path = root / "meta" / "pgc_episodes.jsonl"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("format") != PGC_DATA_FORMAT:
        raise ValueError("Raw PGC provenance has an incompatible outer format.")
    if provenance.get("raw_format") != PGC_ROBOTWIN_RAW_FORMAT:
        raise ValueError("Raw PGC provenance is not a RoboTwin same-scene capture.")
    if provenance.get("state_aligned") is not True or int(provenance.get("action_dim", -1)) != 14:
        raise ValueError("Raw RoboTwin PGC provenance lacks state/action alignment.")
    dataset_kind = str(provenance.get("dataset_kind", ""))
    if dataset_kind not in {"native", "counterfactual"}:
        raise ValueError(f"Invalid raw RoboTwin dataset_kind: {dataset_kind!r}.")
    records = read_jsonl(audit_path)
    if len(records) != int(provenance.get("successful_episode_count", -1)):
        raise ValueError("Raw RoboTwin PGC episode count is inconsistent.")
    seeds = set()
    matched_signatures = []
    pair_ids = set()
    for expected_index, raw_record in enumerate(records):
        record = validate_pair_record(raw_record)
        pair_ids.add(str(record["pair_id"]))
        if int(record.get("episode_index", -1)) != expected_index:
            raise ValueError("Raw RoboTwin PGC episode indices must be dense.")
        seed = int(record["scene_seed"])
        if seed in seeds:
            raise ValueError(f"Duplicate RoboTwin PGC scene seed {seed}.")
        seeds.add(seed)
        state_path = root / str(record["source_initial_state_catalog"])
        state = np.load(state_path, allow_pickle=False)
        if array_sha256(state) != record["initial_state_sha256"]:
            raise ValueError(f"Initial-state hash mismatch for episode {expected_index}.")
        action_path = root / str(record["raw_hdf5"])
        actions = _actions(action_path)
        if actions.shape[0] != int(record["action_count"]):
            raise ValueError(f"Action count mismatch for episode {expected_index}.")
        if array_sha256(actions) != record["action_sha256"]:
            raise ValueError(f"Action hash mismatch for episode {expected_index}.")
        matched_signatures.append((seed, str(record["initial_state_sha256"])))
    if len(pair_ids) != 1:
        raise ValueError(f"Raw RoboTwin dataset must contain one pair_id: {pair_ids}.")
    return {
        "format": "pgc_robotwin_raw_validation_v1",
        "root": str(root),
        "episodes": len(records),
        "dataset_kind": dataset_kind,
        "pair_id": next(iter(pair_ids)),
        "_matched_signatures": matched_signatures,
        "unique_scene_seeds": len(seeds),
        "valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_roots", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [
        validate_raw_dataset(path.expanduser().resolve())
        for path in args.dataset_roots
    ]
    by_pair = {}
    for report in reports:
        by_pair.setdefault(report["pair_id"], {})[report["dataset_kind"]] = report
    for pair_id, kinds in by_pair.items():
        if set(kinds) == {"native", "counterfactual"} and (
            kinds["native"]["_matched_signatures"]
            != kinds["counterfactual"]["_matched_signatures"]
        ):
            raise ValueError(
                f"Native/counterfactual scenes are not matched for {pair_id}."
            )
    for report in reports:
        report.pop("_matched_signatures", None)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
