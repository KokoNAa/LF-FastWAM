#!/usr/bin/env python3
"""Aggregate Correct/Null/Shuffled LF-FastWAM rollout results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("gpu*_results.json"))


def summarize_run(label: str, path: Path) -> dict:
    by_condition: dict[str, dict[str, float | list[float]]] = {}
    result_files = _result_files(path)
    if not result_files:
        raise FileNotFoundError(f"No evaluation result JSON files found under {path}")
    for result_path in result_files:
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        condition = str(result.get("instruction_condition", "correct"))
        stats = by_condition.setdefault(
            condition,
            {"episodes": 0.0, "successes": 0.0, "latencies": []},
        )
        stats["episodes"] += float(result["total_episodes"])
        if condition == "shuffled":
            stats["successes"] += float(
                result.get("default_task_successes", result["successes"])
            )
        else:
            stats["successes"] += float(result["successes"])
        stats["latencies"].extend(result.get("inference_latencies_ms", []))

    def rate(condition: str) -> float | None:
        stats = by_condition.get(condition)
        if not stats or not stats["episodes"]:
            return None
        return float(stats["successes"]) / float(stats["episodes"])

    latencies = [
        value
        for stats in by_condition.values()
        for value in stats["latencies"]
    ]
    sr_correct = rate("correct")
    sr_null = rate("null")
    return {
        "model": label,
        "sr_correct": sr_correct,
        "sr_null": sr_null,
        "language_reliance_gap": (
            None if sr_correct is None or sr_null is None else sr_correct - sr_null
        ),
        "dtl_shuffle": rate("shuffled"),
        "counterfactual_instruction_success": rate("counterfactual"),
        "latency_p50_ms": (
            None if not latencies else float(np.percentile(latencies, 50))
        ),
        "latency_p95_ms": (
            None if not latencies else float(np.percentile(latencies, 95))
        ),
        "result_files": len(result_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for each model result directory (for example B0 and M1).",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for raw_run in args.run:
        if "=" not in raw_run:
            raise ValueError(f"Invalid --run {raw_run!r}; expected LABEL=PATH")
        label, raw_path = raw_run.split("=", 1)
        rows.append(summarize_run(label.strip(), Path(raw_path)))

    output_json = args.output_prefix.with_suffix(".json")
    output_csv = args.output_prefix.with_suffix(".csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_json} and {output_csv}")


if __name__ == "__main__":
    main()
