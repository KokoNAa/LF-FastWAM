#!/usr/bin/env python3
"""Build a balanced full-goal capture plan from matched RoboTwin CIS failures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FORMAT = "pgc_robotwin_failure_capture_plan_v1"
EPISODE_FORMAT = "robotwin_language_intervention_episode_v1"


def _load_records(run_root: Path) -> dict[str, list[dict[str, Any]]]:
    failures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episodes_path in sorted(run_root.glob("*/*/counterfactual/episodes.jsonl")):
        for line_number, raw_line in enumerate(
            episodes_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if (
                not isinstance(record, dict)
                or record.get("format") != EPISODE_FORMAT
                or record.get("condition") != "counterfactual"
            ):
                raise ValueError(
                    f"Invalid counterfactual episode at {episodes_path}:{line_number}"
                )
            if (
                record.get("source_goal_ever_success") is True
                and record.get("counterfactual_goal_ever_success") is not True
            ):
                failures[str(record["pair_id"])].append(
                    {
                        **record,
                        "episodes_path": str(episodes_path.resolve()),
                    }
                )
    if not failures:
        raise ValueError(f"No source-directed failures found under {run_root}")
    return failures


def _round_robin_domains(
    records: list[dict[str, Any]], *, count: int
) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_domain[str(record["task_config"])].append(record)
    for values in by_domain.values():
        values.sort(key=lambda row: (int(row["scene_seed"]), int(row["episode_index"])))

    domains = sorted(by_domain)
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for domain in domains:
            if by_domain[domain]:
                selected.append(by_domain[domain].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def build_capture_plan(run_root: Path, *, episodes_per_pair: int = 5) -> dict:
    run_root = run_root.expanduser().resolve()
    if episodes_per_pair <= 0:
        raise ValueError("episodes_per_pair must be positive")
    summary_path = run_root / "cis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("complete") is not True:
        raise ValueError(f"RoboTwin CIS matrix is incomplete: {summary_path}")

    failures = _load_records(run_root)
    pairs: dict[str, dict[str, Any]] = {}
    for pair_id, records in sorted(failures.items()):
        selected = _round_robin_domains(records, count=episodes_per_pair)
        if len(selected) < episodes_per_pair:
            raise ValueError(
                f"Pair {pair_id} has only {len(selected)} selectable failures; "
                f"requires {episodes_per_pair}."
            )
        identity = {
            (str(row["source_task"]), str(row["counterfactual_task"]))
            for row in records
        }
        if len(identity) != 1:
            raise ValueError(f"Pair identity drift for {pair_id}: {sorted(identity)}")
        source_task, counterfactual_task = next(iter(identity))
        candidates = []
        for rank, record in enumerate(selected):
            candidates.append(
                {
                    "capture_rank": rank,
                    "pair_id": pair_id,
                    "source_task": source_task,
                    "counterfactual_task": counterfactual_task,
                    "task_config": str(record["task_config"]),
                    "scene_seed": int(record["scene_seed"]),
                    "episode_index": int(record["episode_index"]),
                    "source_instruction": str(record["source_instruction"]),
                    "counterfactual_instruction": str(
                        record["counterfactual_instruction"]
                    ),
                    "baseline_episodes_path": str(record["episodes_path"]),
                    "capture_origin": "source_directed_failure_replan",
                    "replay_and_replan_required": True,
                }
            )
        pairs[pair_id] = {
            "source_task": source_task,
            "counterfactual_task": counterfactual_task,
            "source_directed_failures": len(records),
            "required_successful_full_goal_episodes": episodes_per_pair,
            "selected_capture_candidates": candidates,
        }

    return {
        "format": FORMAT,
        "complete": True,
        "baseline_run_root": str(run_root),
        "baseline_checkpoint": str(summary["checkpoint"]),
        "selection_contract": "round_robin_task_config_then_scene_seed",
        "full_goal_usage": "final_short_lora_only",
        "allowed_training_stages": [],
        "required_pair_count": len(pairs),
        "required_successful_full_goal_episodes": len(pairs) * episodes_per_pair,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--episodes-per-pair", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite capture plan: {output}")
    payload = build_capture_plan(
        args.run_root, episodes_per_pair=args.episodes_per_pair
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
