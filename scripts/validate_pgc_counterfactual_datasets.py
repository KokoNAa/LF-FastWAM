#!/usr/bin/env python3
"""Validate direct, state-aligned PGC counterfactual action datasets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_FORMAT = "pgc_counterfactual_actions_v1"
EXPECTED_SUPERVISION = "executed_counterfactual_success_trajectory"
LIBERO_SUITES = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}.")
            records.append(record)
    return records


def validate_dataset(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    meta_dir = dataset_dir / "meta"
    required = (
        meta_dir / "info.json",
        meta_dir / "tasks.jsonl",
        meta_dir / "episodes.jsonl",
        meta_dir / "pgc_episodes.jsonl",
        meta_dir / "pgc_provenance.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            f"Incomplete PGC dataset {dataset_dir}; missing {missing}."
        )

    provenance = json.loads(
        (meta_dir / "pgc_provenance.json").read_text(encoding="utf-8")
    )
    if provenance.get("format") != EXPECTED_FORMAT:
        raise ValueError(
            f"{dataset_dir}: provenance.format must be {EXPECTED_FORMAT!r}."
        )
    if provenance.get("benchmark") != "libero":
        raise ValueError(f"{dataset_dir}: provenance.benchmark must be 'libero'.")
    if provenance.get("action_supervision") != EXPECTED_SUPERVISION:
        raise ValueError(
            f"{dataset_dir}: action_supervision must be {EXPECTED_SUPERVISION!r}."
        )
    if provenance.get("state_aligned") is not True:
        raise ValueError(f"{dataset_dir}: `state_aligned` must be true.")
    if provenance.get("successful_only") is not True:
        raise ValueError(f"{dataset_dir}: `successful_only` must be true.")

    source_suites = set(provenance.get("source_suites") or [])
    if not source_suites or not source_suites.issubset(LIBERO_SUITES):
        raise ValueError(
            f"{dataset_dir}: invalid source_suites={sorted(source_suites)}."
        )
    pairs = provenance.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{dataset_dir}: provenance requires non-empty `pairs`.")
    pair_ids = set()
    pair_suites = set()
    target_instructions = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"{dataset_dir}: pairs[{index}] must be an object.")
        required_pair_fields = {
            "pair_id",
            "source_suite",
            "source_task_id",
            "source_instruction",
            "counterfactual_instruction",
            "counterfactual_goal_state",
        }
        missing_fields = required_pair_fields - set(pair)
        if missing_fields:
            raise ValueError(
                f"{dataset_dir}: pairs[{index}] missing {sorted(missing_fields)}."
            )
        pair_id = str(pair["pair_id"])
        if pair_id in pair_ids:
            raise ValueError(f"{dataset_dir}: duplicate pair_id={pair_id!r}.")
        pair_ids.add(pair_id)
        if str(pair["source_suite"]) not in source_suites:
            raise ValueError(
                f"{dataset_dir}: pair {pair_id} references an undeclared suite."
            )
        pair_suites.add(str(pair["source_suite"]))
        if not isinstance(pair["counterfactual_goal_state"], list) or not pair[
            "counterfactual_goal_state"
        ]:
            raise ValueError(
                f"{dataset_dir}: pair {pair_id} needs a non-empty alternate goal."
            )
        target_instructions.add(str(pair["counterfactual_instruction"]).strip())
    if pair_suites != source_suites:
        raise ValueError(
            f"{dataset_dir}: declared source_suites={sorted(source_suites)} "
            f"but pairs cover {sorted(pair_suites)}."
        )

    task_records = _read_jsonl(meta_dir / "tasks.jsonl")
    episode_records = _read_jsonl(meta_dir / "episodes.jsonl")
    episode_audits = _read_jsonl(meta_dir / "pgc_episodes.jsonl")
    dataset_instructions = {str(record.get("task", "")).strip() for record in task_records}
    if not dataset_instructions or "" in dataset_instructions:
        raise ValueError(f"{dataset_dir}: tasks.jsonl contains empty instructions.")
    uncovered = dataset_instructions - target_instructions
    if uncovered:
        raise ValueError(
            f"{dataset_dir}: task instructions lack provenance pairs: {sorted(uncovered)}."
        )
    if not episode_records:
        raise ValueError(f"{dataset_dir}: no counterfactual episodes were recorded.")

    episode_indices = {int(record["episode_index"]) for record in episode_records}
    if len(episode_indices) != len(episode_records):
        raise ValueError(f"{dataset_dir}: episodes.jsonl has duplicate episode indices.")
    audited_indices = set()
    state_hash_pattern = re.compile(r"^[0-9a-f]{64}$")
    for index, audit in enumerate(episode_audits):
        required_audit_fields = {
            "episode_index",
            "pair_id",
            "source_initial_state_index",
            "initial_state_sha256",
            "initial_state_match",
            "counterfactual_goal_satisfied",
        }
        missing_fields = required_audit_fields - set(audit)
        if missing_fields:
            raise ValueError(
                f"{dataset_dir}: pgc_episodes[{index}] missing "
                f"{sorted(missing_fields)}."
            )
        episode_index = int(audit["episode_index"])
        if episode_index in audited_indices:
            raise ValueError(
                f"{dataset_dir}: duplicate PGC audit for episode {episode_index}."
            )
        audited_indices.add(episode_index)
        if str(audit["pair_id"]) not in pair_ids:
            raise ValueError(
                f"{dataset_dir}: episode {episode_index} references unknown "
                f"pair_id={audit['pair_id']!r}."
            )
        if int(audit["source_initial_state_index"]) < 0:
            raise ValueError(
                f"{dataset_dir}: episode {episode_index} has a negative "
                "source_initial_state_index."
            )
        if not state_hash_pattern.fullmatch(
            str(audit["initial_state_sha256"]).strip().lower()
        ):
            raise ValueError(
                f"{dataset_dir}: episode {episode_index} needs a 64-character "
                "initial_state_sha256."
            )
        if audit["initial_state_match"] is not True:
            raise ValueError(
                f"{dataset_dir}: episode {episode_index} is not state-aligned."
            )
        if audit["counterfactual_goal_satisfied"] is not True:
            raise ValueError(
                f"{dataset_dir}: episode {episode_index} did not satisfy the "
                "counterfactual goal."
            )
    if audited_indices != episode_indices:
        raise ValueError(
            f"{dataset_dir}: PGC episode audit coverage mismatch; "
            f"missing={sorted(episode_indices - audited_indices)} "
            f"extra={sorted(audited_indices - episode_indices)}."
        )

    declared_episodes = provenance.get("successful_episode_count")
    if declared_episodes is not None and int(declared_episodes) != len(episode_records):
        raise ValueError(
            f"{dataset_dir}: successful_episode_count={declared_episodes} but "
            f"episodes.jsonl has {len(episode_records)} records."
        )
    return {
        "dataset": str(dataset_dir),
        "suites": sorted(source_suites),
        "pairs": len(pairs),
        "tasks": len(task_records),
        "episodes": len(episode_records),
    }


def _paths_from_list(path: Path) -> list[Path]:
    paths = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        paths.append(candidate)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", type=Path)
    parser.add_argument("--list", type=Path, dest="list_path")
    parser.add_argument(
        "--require-suite",
        action="append",
        choices=sorted(LIBERO_SUITES),
        default=[],
        help="Require direct counterfactual coverage for this LIBERO suite.",
    )
    args = parser.parse_args()
    datasets = list(args.datasets)
    if args.list_path is not None:
        datasets.extend(_paths_from_list(args.list_path))
    if not datasets:
        parser.error("provide dataset paths or --list")

    summaries = [validate_dataset(path) for path in datasets]
    covered_suites = {
        suite for summary in summaries for suite in summary["suites"]
    }
    missing_suites = set(args.require_suite) - covered_suites
    if missing_suites:
        raise ValueError(
            "Direct counterfactual datasets do not cover required suites: "
            f"{sorted(missing_suites)}."
        )
    for summary in summaries:
        print(
            "Validated PGC dataset: "
            f"{summary['dataset']} suites={summary['suites']} "
            f"pairs={summary['pairs']} tasks={summary['tasks']} "
            f"episodes={summary['episodes']}"
        )


if __name__ == "__main__":
    main()
