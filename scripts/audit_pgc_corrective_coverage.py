#!/usr/bin/env python3
"""Audit replay-verified full-goal corrective coverage without task IDs.

By default every manifest pair is required.  When a counterfactual behavior
summary is supplied, only pairs with at least the configured number of
source-directed failures are required.  This makes the same workflow portable
to another suite while keeping selection based on observed behavior rather
than a hand-written task list.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from fastwam.datasets.pgc_libero import (  # noqa: E402
    load_pgc_closed_loop_corrective_index,
    read_jsonl,
    validate_manifest_record,
)


FORMAT = "pgc_corrective_full_goal_coverage_v1"


def _source_directed_tasks(
    summary: Mapping[str, Any],
    *,
    minimum_failures: int,
) -> set[int]:
    if int(summary.get("total_episodes", 0)) <= 0:
        raise ValueError("Behavior summary has no evaluated episodes.")
    per_task = summary.get("per_task")
    if not isinstance(per_task, Mapping):
        raise ValueError("Behavior summary has no per_task mapping.")
    selected = set()
    for raw_task_id, counts in per_task.items():
        if not isinstance(counts, Mapping):
            raise ValueError(f"Behavior task {raw_task_id!r} is not a mapping.")
        source_directed = int(counts.get("source_goal_success", 0)) + int(
            counts.get("source_object_manipulated_no_completion", 0)
        )
        if source_directed >= minimum_failures:
            selected.add(int(raw_task_id))
    return selected


def build_coverage_report(
    records: list[Mapping[str, Any]],
    indexed: Mapping[int, Mapping[str, Any]],
    *,
    required_task_ids: set[int] | None,
    minimum_full_goal_per_pair: int,
) -> dict[str, Any]:
    if minimum_full_goal_per_pair <= 0:
        raise ValueError("minimum_full_goal_per_pair must be positive.")
    records_by_pair = {str(record["pair_id"]): record for record in records}
    if len(records_by_pair) != len(records):
        raise ValueError("Manifest contains duplicate pair IDs.")
    if required_task_ids is None:
        required_pairs = set(records_by_pair)
    else:
        known_tasks = {int(record["task_id"]) for record in records}
        unknown = sorted(required_task_ids - known_tasks)
        if unknown:
            raise ValueError(f"Behavior summary references unknown tasks: {unknown}.")
        required_pairs = {
            pair_id
            for pair_id, record in records_by_pair.items()
            if int(record["task_id"]) in required_task_ids
        }

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for episode in indexed.values():
        pair_id = str(episode.get("pair_id", ""))
        if pair_id not in records_by_pair:
            raise ValueError(f"Corrective episode references unknown pair {pair_id!r}.")
        counts[pair_id][str(episode.get("verification_kind", ""))] += 1

    pairs = []
    missing = []
    for pair_id, record in sorted(
        records_by_pair.items(), key=lambda item: int(item[1]["task_id"])
    ):
        full_goal = int(counts[pair_id]["counterfactual_goal"])
        row = {
            "pair_id": pair_id,
            "task_id": int(record["task_id"]),
            "required": pair_id in required_pairs,
            "counterfactual_goal": full_goal,
            "target_lift": int(counts[pair_id]["target_lift"]),
            "satisfied": (
                pair_id not in required_pairs
                or full_goal >= minimum_full_goal_per_pair
            ),
        }
        pairs.append(row)
        if not row["satisfied"]:
            missing.append(pair_id)
    return {
        "format": FORMAT,
        "minimum_full_goal_per_pair": minimum_full_goal_per_pair,
        "required_pair_count": len(required_pairs),
        "covered_required_pair_count": len(required_pairs) - len(missing),
        "missing_required_pairs": missing,
        "complete": not missing,
        "pairs": pairs,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--behavior-summary", type=Path)
    parser.add_argument("--minimum-source-directed-failures", type=int, default=1)
    parser.add_argument("--minimum-full-goal-per-pair", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.minimum_source_directed_failures <= 0:
        parser.error("--minimum-source-directed-failures must be positive")
    if args.minimum_full_goal_per_pair <= 0:
        parser.error("--minimum-full-goal-per-pair must be positive")
    return args


def main() -> None:
    args = _parse_args()
    records = [
        record
        for record in read_jsonl(args.manifest.expanduser().resolve())
        if str(record.get("task_suite_name", "")) == args.suite
    ]
    if not records:
        raise ValueError(f"Manifest has no records for suite {args.suite!r}.")
    for record in records:
        validate_manifest_record(record)
    required_task_ids = None
    if args.behavior_summary is not None:
        summary = json.loads(
            args.behavior_summary.expanduser().resolve().read_text(encoding="utf-8")
        )
        required_task_ids = _source_directed_tasks(
            summary,
            minimum_failures=args.minimum_source_directed_failures,
        )
    indexed = load_pgc_closed_loop_corrective_index(
        args.dataset.expanduser().resolve()
    )
    report = build_coverage_report(
        records,
        indexed,
        required_task_ids=required_task_ids,
        minimum_full_goal_per_pair=args.minimum_full_goal_per_pair,
    )
    report.update(
        {
            "suite": args.suite,
            "manifest": str(args.manifest.expanduser().resolve()),
            "dataset": str(args.dataset.expanduser().resolve()),
            "behavior_summary": (
                str(args.behavior_summary.expanduser().resolve())
                if args.behavior_summary is not None
                else None
            ),
        }
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["complete"] and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
