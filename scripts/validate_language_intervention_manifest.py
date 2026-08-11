#!/usr/bin/env python3
"""Validate LF-FastWAM language-intervention JSONL manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = (
    "pair_id",
    "task_name",
    "correct_instruction",
    "shuffled_instruction",
)


def validate_manifest(path: Path) -> int:
    errors: list[str] = []
    pair_ids: set[str] = set()
    task_keys: set[tuple[str, int] | tuple[str, str]] = set()
    record_count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(record, dict):
                errors.append(f"line {line_number}: record must be a JSON object")
                continue

            for field in REQUIRED_FIELDS:
                if not str(record.get(field, "")).strip():
                    errors.append(f"line {line_number}: missing non-empty `{field}`")

            pair_id = str(record.get("pair_id", "")).strip()
            if pair_id in pair_ids:
                errors.append(f"line {line_number}: duplicate pair_id {pair_id!r}")
            pair_ids.add(pair_id)

            if "task_suite_name" in record and "task_id" in record:
                try:
                    task_key: tuple[str, int] | tuple[str, str] = (
                        str(record["task_suite_name"]),
                        int(record["task_id"]),
                    )
                except (TypeError, ValueError):
                    errors.append(f"line {line_number}: `task_id` must be an integer")
                    task_key = ("invalid", str(line_number))
            else:
                task_key = ("task_name", str(record.get("task_name", "")).casefold())
            if task_key in task_keys:
                errors.append(
                    f"line {line_number}: duplicate task selector {task_key!r}"
                )
            task_keys.add(task_key)

            correct = str(record.get("correct_instruction", "")).strip().casefold()
            shuffled = str(record.get("shuffled_instruction", "")).strip().casefold()
            if correct and correct == shuffled:
                errors.append(
                    f"line {line_number}: shuffled instruction equals correct instruction"
                )

            counterfactual = str(record.get("counterfactual_instruction", "")).strip()
            executable = record.get("counterfactual_is_executable")
            if counterfactual and executable is not True:
                errors.append(
                    f"line {line_number}: counterfactual instruction must be marked executable"
                )
            if counterfactual:
                for field in (
                    "counterfactual_task_suite_name",
                    "counterfactual_task_id",
                    "counterfactual_task_name",
                ):
                    if field not in record or str(record.get(field, "")).strip() == "":
                        errors.append(
                            f"line {line_number}: counterfactual record requires `{field}`"
                        )
                try:
                    counterfactual_task_id = int(record["counterfactual_task_id"])
                    if counterfactual_task_id < 0:
                        raise ValueError
                except (KeyError, TypeError, ValueError):
                    errors.append(
                        f"line {line_number}: `counterfactual_task_id` must be a non-negative integer"
                    )
                if counterfactual.casefold() == correct:
                    errors.append(
                        f"line {line_number}: counterfactual instruction equals correct instruction"
                    )
                counterfactual_task_name = str(
                    record.get("counterfactual_task_name", "")
                ).strip()
                if (
                    counterfactual_task_name
                    and counterfactual_task_name.casefold() != counterfactual.casefold()
                ):
                    errors.append(
                        f"line {line_number}: counterfactual task name does not match instruction"
                    )

    if record_count == 0:
        errors.append("manifest contains no records")
    if errors:
        raise ValueError("\n".join(errors))
    return record_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    count = validate_manifest(args.manifest)
    print(f"Validated {count} language-intervention records: {args.manifest}")


if __name__ == "__main__":
    main()
