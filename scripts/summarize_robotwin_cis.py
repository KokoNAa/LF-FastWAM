#!/usr/bin/env python3
"""Validate and aggregate matched RoboTwin Correct/DTL/CIS rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.language_interventions import (
    EPISODE_FORMAT,
    VALID_CONDITIONS,
)


JOB_SUMMARY_FORMAT = "robotwin_language_intervention_summary_v1"
MATRIX_SUMMARY_FORMAT = "robotwin_cis_matrix_summary_v1"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_job_output(
    job_dir: str | Path,
    *,
    expected_episodes: int | None = None,
    expected_source_task: str | None = None,
    expected_task_config: str | None = None,
    expected_condition: str | None = None,
    expected_checkpoint: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = Path(job_dir)
    summary_path = directory / "summary.json"
    episodes_path = directory / "episodes.jsonl"
    summary = _load_json(summary_path)
    if summary.get("format") != JOB_SUMMARY_FORMAT:
        raise ValueError(
            f"Unexpected job summary format in {summary_path}: "
            f"{summary.get('format')!r}"
        )

    records: list[dict[str, Any]] = []
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid episode JSON at {episodes_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict) or record.get("format") != EPISODE_FORMAT:
                raise ValueError(
                    f"Unexpected episode record at {episodes_path}:{line_number}"
                )
            records.append(record)

    expected_count = int(summary.get("total_episodes", -1))
    if expected_count != len(records):
        raise ValueError(
            f"Job summary/episode count mismatch in {directory}: "
            f"summary={expected_count}, records={len(records)}"
        )
    if expected_episodes is not None and len(records) != int(expected_episodes):
        raise ValueError(
            f"Incomplete job in {directory}: expected {expected_episodes}, "
            f"found {len(records)}"
        )

    expected_metadata = {
        "source_task": expected_source_task,
        "task_config": expected_task_config,
        "condition": expected_condition,
        "checkpoint": (
            None
            if expected_checkpoint is None
            else str(Path(expected_checkpoint).expanduser().resolve())
        ),
    }
    for field, expected in expected_metadata.items():
        if expected is None:
            continue
        if str(summary.get(field)) != str(expected):
            raise ValueError(
                f"Job metadata mismatch in {directory}: {field}="
                f"{summary.get(field)!r}, expected {expected!r}"
            )
    for index, record in enumerate(records):
        for field in ("source_task", "task_config", "condition"):
            if str(record.get(field)) != str(summary.get(field)):
                raise ValueError(
                    f"Episode {index} metadata mismatch in {episodes_path}: "
                    f"{field}={record.get(field)!r}, summary={summary.get(field)!r}"
                )
        if str(record.get("checkpoint")) != str(summary.get("checkpoint")):
            raise ValueError(f"Episode {index} checkpoint mismatch in {episodes_path}")
        if record.get("initial_source_goal_success") is not False:
            raise ValueError(
                f"Episode {index} starts in the source goal in {episodes_path}"
            )
        if record.get("initial_counterfactual_goal_success") is not False:
            raise ValueError(
                f"Episode {index} starts in the counterfactual goal in {episodes_path}"
            )

    checks = {
        "selected_goal_successes": "selected_goal_success",
        "source_goal_successes": "source_goal_ever_success",
        "counterfactual_goal_successes": "counterfactual_goal_ever_success",
    }
    for summary_field, record_field in checks.items():
        observed = sum(bool(record[record_field]) for record in records)
        if int(summary.get(summary_field, -1)) != observed:
            raise ValueError(
                f"Summary count drift in {summary_path}: {summary_field}="
                f"{summary.get(summary_field)!r}, records={observed}"
            )
    return summary, records


def _mean_rate(successes: int, total: int) -> float | None:
    return None if total == 0 else float(successes) / float(total)


def _expected_keys(
    *, tasks: Iterable[str], task_configs: Iterable[str], conditions: Iterable[str]
) -> set[tuple[str, str, str]]:
    return {
        (str(task), str(task_config), str(condition))
        for task in tasks
        for task_config in task_configs
        for condition in conditions
    }


def summarize_run(
    run_root: str | Path,
    *,
    output_prefix: str | Path | None = None,
    expected_episodes: int | None = None,
    expected_tasks: Iterable[str] | None = None,
    expected_task_configs: Iterable[str] | None = None,
    expected_conditions: Iterable[str] | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    summary_paths = sorted(root.glob("*/*/*/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No RoboTwin CIS job summaries found under {root}")

    jobs: dict[tuple[str, str, str], dict[str, Any]] = {}
    records_by_job: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for summary_path in summary_paths:
        summary, records = load_job_output(
            summary_path.parent,
            expected_episodes=expected_episodes,
        )
        key = (
            str(summary["source_task"]),
            str(summary["task_config"]),
            str(summary["condition"]),
        )
        if key in jobs:
            raise ValueError(f"Duplicate RoboTwin CIS job key under {root}: {key}")
        jobs[key] = summary
        records_by_job[key] = records

    checkpoints = {str(summary.get("checkpoint", "")) for summary in jobs.values()}
    if "" in checkpoints or len(checkpoints) != 1:
        raise ValueError(
            f"RoboTwin CIS run mixes checkpoint identities: {sorted(checkpoints)}"
        )

    expected_task_list = (
        sorted({key[0] for key in jobs})
        if expected_tasks is None
        else [str(value) for value in expected_tasks]
    )
    expected_task_config_list = (
        sorted({key[1] for key in jobs})
        if expected_task_configs is None
        else [str(value) for value in expected_task_configs]
    )
    expected_condition_list = (
        list(VALID_CONDITIONS)
        if expected_conditions is None
        else [str(value) for value in expected_conditions]
    )
    expected = _expected_keys(
        tasks=expected_task_list,
        task_configs=expected_task_config_list,
        conditions=expected_condition_list,
    )
    missing = sorted(expected - set(jobs))
    unexpected = sorted(set(jobs) - expected)
    if require_complete and (missing or unexpected):
        raise ValueError(
            f"RoboTwin CIS matrix is incomplete: missing={missing}, "
            f"unexpected={unexpected}"
        )

    matched_audits: list[dict[str, Any]] = []
    for task in expected_task_list:
        for task_config in expected_task_config_list:
            group_keys = [
                (task, task_config, condition) for condition in expected_condition_list
            ]
            if not all(key in records_by_job for key in group_keys):
                continue
            by_condition = {key[2]: records_by_job[key] for key in group_keys}
            baseline = by_condition[expected_condition_list[0]]
            baseline_seeds = [int(record["scene_seed"]) for record in baseline]
            if len(baseline_seeds) != len(set(baseline_seeds)):
                raise ValueError(
                    f"Duplicate scene seeds for task={task}, task_config={task_config}"
                )
            baseline_by_seed = {
                int(record["scene_seed"]): record for record in baseline
            }
            for condition, records in by_condition.items():
                seeds = [int(record["scene_seed"]) for record in records]
                if seeds != baseline_seeds:
                    raise ValueError(
                        "Unmatched scene seeds across conditions for "
                        f"task={task}, task_config={task_config}, "
                        f"condition={condition}: {seeds} != {baseline_seeds}"
                    )
                for record in records:
                    reference = baseline_by_seed[int(record["scene_seed"])]
                    for field in ("source_instruction", "counterfactual_instruction"):
                        if record[field] != reference[field]:
                            raise ValueError(
                                f"Matched instruction drift for {field}, task={task}, "
                                f"task_config={task_config}, seed={record['scene_seed']}"
                            )
            if "shuffled" in by_condition and "counterfactual" in by_condition:
                for shuffled, counterfactual in zip(
                    by_condition["shuffled"],
                    by_condition["counterfactual"],
                    strict=True,
                ):
                    if (
                        shuffled["policy_instruction"]
                        != counterfactual["policy_instruction"]
                    ):
                        raise ValueError(
                            "Shuffle/Counterfactual policy instruction mismatch for "
                            f"task={task}, task_config={task_config}, "
                            f"seed={shuffled['scene_seed']}"
                        )
            matched_audits.append(
                {
                    "source_task": task,
                    "task_config": task_config,
                    "episodes_per_condition": len(baseline),
                    "scene_seeds": baseline_seeds,
                    "matched": True,
                }
            )

    cells: list[dict[str, Any]] = []
    for key in sorted(jobs):
        summary = jobs[key]
        cells.append(
            {
                "source_task": key[0],
                "task_config": key[1],
                "condition": key[2],
                "counterfactual_task": summary["counterfactual_task"],
                "episodes": int(summary["total_episodes"]),
                "selected_goal_successes": int(summary["selected_goal_successes"]),
                "selected_goal_success_rate": summary["selected_goal_success_rate"],
                "source_goal_success_rate": summary["source_goal_success_rate"],
                "counterfactual_goal_success_rate": summary[
                    "counterfactual_goal_success_rate"
                ],
            }
        )

    aggregate_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "episodes": 0,
            "selected": 0,
            "source": 0,
            "counterfactual": 0,
        }
    )
    for summary in jobs.values():
        key = (str(summary["task_config"]), str(summary["condition"]))
        counts = aggregate_counts[key]
        counts["episodes"] += int(summary["total_episodes"])
        counts["selected"] += int(summary["selected_goal_successes"])
        counts["source"] += int(summary["source_goal_successes"])
        counts["counterfactual"] += int(summary["counterfactual_goal_successes"])

    metrics: list[dict[str, Any]] = []
    for task_config in expected_task_config_list:
        row: dict[str, Any] = {"task_config": task_config}
        for condition in expected_condition_list:
            counts = aggregate_counts.get((task_config, condition))
            if counts is None:
                row[f"{condition}_episodes"] = 0
                row[f"{condition}_selected_goal_rate"] = None
                continue
            row[f"{condition}_episodes"] = counts["episodes"]
            row[f"{condition}_selected_goal_rate"] = _mean_rate(
                counts["selected"], counts["episodes"]
            )
            row[f"{condition}_source_goal_rate"] = _mean_rate(
                counts["source"], counts["episodes"]
            )
            row[f"{condition}_counterfactual_goal_rate"] = _mean_rate(
                counts["counterfactual"], counts["episodes"]
            )
        row["correct_sr"] = row.get("correct_selected_goal_rate")
        row["dtl_shuffle"] = row.get("shuffled_selected_goal_rate")
        row["cis"] = row.get("counterfactual_selected_goal_rate")
        metrics.append(row)

    payload = {
        "format": MATRIX_SUMMARY_FORMAT,
        "run_root": str(root),
        "checkpoint": next(iter(checkpoints)),
        "complete": not missing and not unexpected,
        "expected_jobs": len(expected),
        "completed_jobs": len(jobs),
        "missing_jobs": [list(key) for key in missing],
        "unexpected_jobs": [list(key) for key in unexpected],
        "matched_seed_instruction_audits": matched_audits,
        "metrics": metrics,
        "cells": cells,
    }

    if output_prefix is not None:
        prefix = Path(output_prefix)
        json_path = prefix.with_suffix(".json")
        csv_path = prefix.with_suffix(".csv")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fieldnames = sorted({field for row in metrics for field in row})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--task-config", action="append", dest="task_configs")
    parser.add_argument("--condition", action="append", dest="conditions")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    output_prefix = args.output_prefix or args.run_root / "cis_summary"
    payload = summarize_run(
        args.run_root,
        output_prefix=output_prefix,
        expected_episodes=args.expected_episodes,
        expected_tasks=args.tasks,
        expected_task_configs=args.task_configs,
        expected_conditions=args.conditions,
        require_complete=args.require_complete,
    )
    print(
        f"Validated {payload['completed_jobs']}/{payload['expected_jobs']} jobs; "
        f"wrote {output_prefix.with_suffix('.json')} and "
        f"{output_prefix.with_suffix('.csv')}"
    )


if __name__ == "__main__":
    main()
