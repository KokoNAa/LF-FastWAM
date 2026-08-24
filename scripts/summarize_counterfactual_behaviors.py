#!/usr/bin/env python3
"""Aggregate behavior-level diagnostics from LIBERO CIS rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.counterfactual_diagnostics import (
    COUNTERFACTUAL_BEHAVIOR_CATEGORIES,
    empty_behavior_counts,
)
from experiments.libero.oracle_phase_servo import summarize_oracle_phase_servo


def _result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("gpu*_results.json"))


def _episode_row(
    result: dict[str, Any],
    episode: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    source_targets = set(episode.get("source_target_objects", []))
    counterfactual_targets = set(
        episode.get("counterfactual_target_objects", [])
    )
    grasped = set(episode.get("grasped_objects", []))
    lifted = set(episode.get("lifted_objects", []))
    manipulated = set(episode.get("manipulated_objects", []))
    known_targets = source_targets | counterfactual_targets
    return {
        "task_id": int(result["task_id"]),
        "pair_id": str(result.get("pair_id", "")),
        "episode": int(episode["episode"]),
        "category": str(episode["category"]),
        "source_instruction": str(result.get("task_description", "")),
        "counterfactual_instruction": str(result.get("policy_instruction", "")),
        "counterfactual_goal_achieved": bool(
            episode.get("counterfactual_goal_achieved", False)
        ),
        "source_goal_achieved": bool(
            episode.get("source_goal_achieved", False)
        ),
        "target_object_grasped": bool(grasped & counterfactual_targets),
        "source_object_grasped": bool(grasped & source_targets),
        "other_object_grasped": bool(grasped - known_targets),
        "target_object_lifted": bool(lifted & counterfactual_targets),
        "source_object_lifted": bool(lifted & source_targets),
        "other_object_lifted": bool(lifted - known_targets),
        "grasped_objects": sorted(grasped),
        "lifted_objects": sorted(lifted),
        "manipulated_objects": sorted(manipulated),
        "first_grasp_step": episode.get("first_grasp_step", {}),
        "max_lift_delta_m": episode.get("max_lift_delta_m", {}),
        "policy_steps": int(episode.get("policy_steps", 0)),
        "horizon_timeout": bool(episode.get("horizon_timeout", False)),
        "result_file": str(result_path),
    }


def summarize(path: Path, expected_episodes: int | None = None) -> dict[str, Any]:
    result_files = _result_files(path)
    if not result_files:
        raise FileNotFoundError(f"No evaluation result JSON files found under {path}")

    rows: list[dict[str, Any]] = []
    per_task: dict[int, Counter[str]] = defaultdict(Counter)
    oracle_servo_episodes: list[list[dict[str, Any]]] = []
    for result_path in result_files:
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        episodes = result.get("counterfactual_episode_diagnostics")
        if not isinstance(episodes, list):
            raise ValueError(
                f"Result has no counterfactual diagnostics: {result_path}"
            )
        if len(episodes) != int(result.get("total_episodes", len(episodes))):
            raise ValueError(
                "Diagnostic episode count does not match total_episodes in "
                f"{result_path}."
            )
        for policy_episode in result.get("policy_guard_episode_diagnostics", []):
            oracle_servo_episodes.append(
                [
                    decision["entity_relation_oracle_phase_servo"]
                    for decision in policy_episode.get("decisions", [])
                    if "entity_relation_oracle_phase_servo" in decision
                ]
            )
        for episode in episodes:
            row = _episode_row(result, episode, result_path)
            if row["category"] not in COUNTERFACTUAL_BEHAVIOR_CATEGORIES:
                raise ValueError(
                    f"Unknown counterfactual category {row['category']!r} in "
                    f"{result_path}."
                )
            rows.append(row)
            per_task[row["task_id"]][row["category"]] += 1

    if expected_episodes is not None and len(rows) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} diagnostic episodes, found {len(rows)}."
        )

    behavior_counts = empty_behavior_counts()
    behavior_counts.update(Counter(row["category"] for row in rows))
    total = len(rows)
    event_fields = (
        "counterfactual_goal_achieved",
        "source_goal_achieved",
        "target_object_grasped",
        "source_object_grasped",
        "other_object_grasped",
        "target_object_lifted",
        "source_object_lifted",
        "other_object_lifted",
        "horizon_timeout",
    )
    event_counts = {
        field: sum(bool(row[field]) for row in rows) for field in event_fields
    }
    summary = {
        "result_files": len(result_files),
        "total_episodes": total,
        "behavior_counts": behavior_counts,
        "behavior_rates": {
            category: (count / total if total else None)
            for category, count in behavior_counts.items()
        },
        "event_counts": event_counts,
        "event_rates": {
            field: (count / total if total else None)
            for field, count in event_counts.items()
        },
        "per_task": {
            str(task_id): {
                **empty_behavior_counts(),
                **dict(counts),
            }
            for task_id, counts in sorted(per_task.items())
        },
        "episodes": rows,
    }
    if any(oracle_servo_episodes):
        summary["oracle_phase_servo"] = summarize_oracle_phase_servo(
            oracle_servo_episodes
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int)
    args = parser.parse_args()

    summary = summarize(args.results, args.expected_episodes)
    output_json = args.output_prefix.with_suffix(".json")
    output_csv = args.output_prefix.with_suffix(".csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    rows = summary["episodes"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            for key in (
                "grasped_objects",
                "lifted_objects",
                "manipulated_objects",
                "first_grasp_step",
                "max_lift_delta_m",
            ):
                serialized[key] = json.dumps(
                    serialized[key], ensure_ascii=False, sort_keys=True
                )
            writer.writerow(serialized)

    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))
    print(f"Wrote {output_json} and {output_csv}")


if __name__ == "__main__":
    main()
