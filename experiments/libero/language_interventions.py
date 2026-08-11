"""Manifest and predicate helpers for paired LIBERO language interventions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def load_language_intervention_manifest(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON objects from a JSONL intervention manifest."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Manifest line {line_number} must be a JSON object.")
            record = dict(record)
            record["_line_number"] = line_number
            records.append(record)
    return records


def select_language_intervention_record(
    records: Iterable[Mapping[str, Any]],
    *,
    suite_name: str,
    task_id: int,
    task_description: str,
) -> dict[str, Any]:
    """Select one source-task record, preferring explicit suite/task IDs."""
    matches: list[dict[str, Any]] = []
    for raw_record in records:
        record = dict(raw_record)
        has_explicit_selector = "task_suite_name" in record and "task_id" in record
        if has_explicit_selector:
            try:
                matches_task = str(record["task_suite_name"]) == suite_name and int(
                    record["task_id"]
                ) == int(task_id)
            except (TypeError, ValueError):
                matches_task = False
        else:
            matches_task = (
                str(record.get("task_name", "")).strip().casefold()
                == task_description.strip().casefold()
            )
        if matches_task:
            matches.append(record)

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one language-intervention manifest record for "
            f"{suite_name}/{task_id} ({task_description!r}), found {len(matches)}."
        )

    record = matches[0]
    expected_instruction = str(record.get("correct_instruction", "")).strip()
    if (
        expected_instruction
        and expected_instruction.casefold() != task_description.strip().casefold()
    ):
        line_number = record.get("_line_number", "?")
        raise ValueError(
            f"Manifest line {line_number} correct_instruction does not match "
            f"the source task: {expected_instruction!r} != {task_description!r}."
        )
    return record


def canonical_goal_state(
    goal_state: Sequence[Sequence[Any]],
) -> tuple[tuple[str, ...], ...]:
    """Return a stable, case-insensitive representation of a BDDL goal."""
    return tuple(
        tuple(str(value).strip().casefold() for value in predicate)
        for predicate in goal_state
    )


def problem_entity_names(problem: Mapping[str, Any]) -> set[str]:
    """Return entity names addressable by LIBERO's predicate evaluator."""
    names: set[str] = set()
    for mapping_name in ("objects", "fixtures"):
        mapping = problem.get(mapping_name, {})
        if isinstance(mapping, Mapping):
            for values in mapping.values():
                if isinstance(values, (list, tuple, set)):
                    names.update(str(value) for value in values)
    regions = problem.get("regions", {})
    if isinstance(regions, Mapping):
        names.update(str(name) for name in regions)
    return names


def goal_entity_names(goal_state: Sequence[Sequence[Any]]) -> set[str]:
    """Return object/region operands referenced by simple LIBERO predicates."""
    names: set[str] = set()
    for predicate in goal_state:
        if len(predicate) not in (2, 3):
            raise ValueError(
                "Only unary and binary LIBERO goal predicates are supported, "
                f"got {predicate!r}."
            )
        names.update(str(value) for value in predicate[1:])
    return names


def validate_counterfactual_problem(
    source_problem: Mapping[str, Any],
    counterfactual_problem: Mapping[str, Any],
) -> list[list[Any]]:
    """Validate that an alternate BDDL goal is executable in the source scene.

    The source simulator and its initial state stay unchanged. Therefore only
    the alternate goal predicate is imported, and every predicate operand must
    already be present in the source problem.
    """
    source_problem_name = str(source_problem.get("problem_name", ""))
    counterfactual_problem_name = str(counterfactual_problem.get("problem_name", ""))
    if source_problem_name != counterfactual_problem_name:
        raise ValueError(
            "Counterfactual BDDL uses a different LIBERO environment class: "
            f"{source_problem_name!r} != {counterfactual_problem_name!r}."
        )

    source_goal = source_problem.get("goal_state", [])
    counterfactual_goal = counterfactual_problem.get("goal_state", [])
    if not isinstance(counterfactual_goal, list) or not counterfactual_goal:
        raise ValueError("Counterfactual BDDL has no non-empty goal_state.")
    if canonical_goal_state(source_goal) == canonical_goal_state(counterfactual_goal):
        raise ValueError("Counterfactual goal is identical to the source goal.")

    available_entities = problem_entity_names(source_problem)
    required_entities = goal_entity_names(counterfactual_goal)
    missing_entities = sorted(required_entities - available_entities)
    if missing_entities:
        raise ValueError(
            "Counterfactual goal is not executable in the source scene; "
            f"missing predicate entities: {missing_entities}."
        )

    return [list(predicate) for predicate in counterfactual_goal]
