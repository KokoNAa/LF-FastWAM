#!/usr/bin/env python3
"""Create audited same-scene DTL/CIS pairs from LIBERO-Object tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.language_interventions import (  # noqa: E402
    canonical_goal_state,
    validate_counterfactual_problem,
)
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import bddl_utils as BDDLUtils  # noqa: E402


def _task_bddl_path(task: Any) -> Path:
    return Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file


def _parse_task(task: Any) -> dict[str, Any]:
    return BDDLUtils.robosuite_parse_problem(str(_task_bddl_path(task)))


def _build_candidate_map(
    tasks: list[Any],
    problems: list[dict[str, Any]],
) -> dict[int, list[int]]:
    candidates: dict[int, list[int]] = {}
    task_count = len(tasks)
    for source_id, source_problem in enumerate(problems):
        source_candidates: list[int] = []
        for offset in range(1, task_count):
            target_id = (source_id + offset) % task_count
            try:
                counterfactual_goal = validate_counterfactual_problem(
                    source_problem,
                    problems[target_id],
                )
            except ValueError:
                continue
            if canonical_goal_state(counterfactual_goal) in {
                canonical_goal_state([state])
                for state in source_problem.get("initial_state", [])
            }:
                continue
            source_candidates.append(target_id)
        if not source_candidates:
            raise ValueError(
                f"No executable counterfactual candidates for source task {source_id}: "
                f"{tasks[source_id].language!r}."
            )
        candidates[source_id] = source_candidates
    return candidates


def _perfect_matching(candidates: dict[int, list[int]]) -> dict[int, int]:
    """Find a deterministic one-to-one source-to-counterfactual assignment."""
    target_to_source: dict[int, int] = {}

    def assign(source_id: int, seen: set[int]) -> bool:
        for target_id in candidates[source_id]:
            if target_id in seen:
                continue
            seen.add(target_id)
            previous_source = target_to_source.get(target_id)
            if previous_source is None or assign(previous_source, seen):
                target_to_source[target_id] = source_id
                return True
        return False

    for source_id in sorted(candidates):
        if not assign(source_id, set()):
            raise ValueError(
                "Could not construct a one-to-one executable intervention "
                f"matching; failed at source task {source_id}."
            )
    return {source_id: target_id for target_id, source_id in target_to_source.items()}


def build_manifest(suite_name: str) -> list[dict[str, Any]]:
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"Unknown LIBERO suite: {suite_name!r}.")
    task_suite = benchmark_dict[suite_name]()
    tasks = [task_suite.get_task(task_id) for task_id in range(task_suite.n_tasks)]
    problems = [_parse_task(task) for task in tasks]
    candidates = _build_candidate_map(tasks, problems)
    matching = _perfect_matching(candidates)

    records: list[dict[str, Any]] = []
    for source_id in range(len(tasks)):
        target_id = matching[source_id]
        source_task = tasks[source_id]
        target_task = tasks[target_id]
        counterfactual_goal = validate_counterfactual_problem(
            problems[source_id],
            problems[target_id],
        )
        records.append(
            {
                "pair_id": f"{suite_name}_{source_id:02d}_to_{target_id:02d}",
                "task_suite_name": suite_name,
                "task_id": source_id,
                "task_name": source_task.language,
                "scene_group": f"{suite_name}_source_{source_id:02d}",
                "correct_instruction": source_task.language,
                "shuffled_instruction": target_task.language,
                "counterfactual_instruction": target_task.language,
                "counterfactual_task_suite_name": suite_name,
                "counterfactual_task_id": target_id,
                "counterfactual_task_name": target_task.language,
                "counterfactual_is_executable": True,
                "source_bddl_file": str(_task_bddl_path(source_task)),
                "counterfactual_bddl_file": str(_task_bddl_path(target_task)),
                "counterfactual_goal_state": counterfactual_goal,
                "notes": (
                    "Source simulator state is reused. Only the policy instruction "
                    "and, for CIS, the success goal predicate are replaced."
                ),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/eval/libero_object_dtl_cis.jsonl"),
    )
    args = parser.parse_args()

    records = build_manifest(args.suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} audited DTL/CIS pairs: {args.output}")
    for record in records:
        print(
            f"{record['pair_id']}: {record['correct_instruction']} -> "
            f"{record['counterfactual_instruction']}"
        )


if __name__ == "__main__":
    main()
