#!/usr/bin/env python3
"""Summarize paired native-FastWAM LIBERO-Object language-OOD runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONDITION_SPECS = {
    "correct": ("correct", None),
    "paraphrase_near": ("paraphrase", "near"),
    "paraphrase_sequence": ("paraphrase", "sequence"),
    "paraphrase_goal": ("paraphrase", "goal"),
    "null": ("null", None),
    "shuffled": ("shuffled", None),
}
PRIMARY_OOD_LABELS = (
    "paraphrase_near",
    "paraphrase_sequence",
    "paraphrase_goal",
)


def _load_condition(run_root: Path, label: str) -> dict[int, dict[str, Any]]:
    expected_condition, expected_variant = CONDITION_SPECS[label]
    suite_dir = run_root / label / "libero_object"
    result_paths = sorted(suite_dir.glob("gpu*_task*_results.json"))
    if not result_paths:
        return {}
    records: dict[int, dict[str, Any]] = {}
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        task_id = int(path.stem.split("_task", 1)[1].split("_", 1)[0])
        if task_id in records:
            raise ValueError(f"Duplicate {label} result for task {task_id}.")
        if payload.get("instruction_condition") != expected_condition:
            raise ValueError(
                f"{path}: expected condition {expected_condition!r}, got "
                f"{payload.get('instruction_condition')!r}."
            )
        if payload.get("success_predicate") != "source":
            raise ValueError(f"{path}: language-OOD must retain the source goal.")
        if expected_variant is not None:
            if payload.get("language_ood_variant") != expected_variant:
                raise ValueError(
                    f"{path}: expected OOD variant {expected_variant!r}, got "
                    f"{payload.get('language_ood_variant')!r}."
                )
            if payload.get("language_ood_source_goal_unchanged") is not True:
                raise ValueError(f"{path}: missing unchanged-source-goal audit.")
            if payload.get("language_ood_policy_training_exact_match") is not False:
                raise ValueError(
                    f"{path}: paraphrase is not marked policy-training OOD."
                )
            if payload.get("policy_instruction") == payload.get("task_description"):
                raise ValueError(
                    f"{path}: paraphrase equals the canonical instruction."
                )
        records[task_id] = payload
    if set(records) != set(range(10)):
        raise ValueError(
            f"{label} must contain all LIBERO-Object task IDs 0..9, got "
            f"{sorted(records)}."
        )
    return records


def _success_indices(record: dict[str, Any]) -> set[int]:
    return {int(value) for value in record.get("success_episodes", [])}


def summarize(run_root: Path) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    by_condition = {
        label: records
        for label in CONDITION_SPECS
        if (records := _load_condition(run_root, label))
    }
    if not by_condition:
        raise FileNotFoundError(f"No language-OOD result conditions under {run_root}.")

    manifest_hashes = {
        str(record["language_intervention_manifest_sha256"])
        for records in by_condition.values()
        for record in records.values()
        if record.get("language_intervention_manifest_sha256")
    }
    if len(manifest_hashes) > 1:
        raise ValueError(
            f"Multiple intervention manifests were mixed: {manifest_hashes}"
        )

    id_records = by_condition.get("correct")
    condition_summaries: dict[str, dict[str, Any]] = {}
    for label, records in by_condition.items():
        total_trials = sum(int(record["total_episodes"]) for record in records.values())
        total_successes = sum(int(record["successes"]) for record in records.values())
        success_rate = total_successes / total_trials if total_trials else 0.0
        per_task = {
            str(task_id): {
                "successes": int(record["successes"]),
                "trials": int(record["total_episodes"]),
                "success_rate": (
                    int(record["successes"]) / int(record["total_episodes"])
                ),
                "canonical_instruction": str(record.get("task_description", "")),
                "policy_instruction": str(record.get("policy_instruction", "")),
            }
            for task_id, record in sorted(records.items())
        }
        summary = {
            "successes": total_successes,
            "trials": total_trials,
            "success_rate": success_rate,
            "per_task": per_task,
        }
        if id_records is not None and label != "correct":
            id_trials = sum(
                int(record["total_episodes"]) for record in id_records.values()
            )
            id_successes = sum(
                int(record["successes"]) for record in id_records.values()
            )
            id_rate = id_successes / id_trials if id_trials else 0.0
            paired_id_successes = 0
            paired_both_successes = 0
            for task_id in range(10):
                id_record = id_records[task_id]
                candidate_record = records[task_id]
                if int(id_record["total_episodes"]) != int(
                    candidate_record["total_episodes"]
                ):
                    raise ValueError(
                        f"Paired trial-count mismatch for {label}/task {task_id}."
                    )
                id_success = _success_indices(id_record)
                candidate_success = _success_indices(candidate_record)
                paired_id_successes += len(id_success)
                paired_both_successes += len(id_success & candidate_success)
            summary.update(
                {
                    "absolute_drop_from_id": success_rate - id_rate,
                    "retention_vs_id": success_rate / id_rate if id_rate else None,
                    "p_success_given_id_success": (
                        paired_both_successes / paired_id_successes
                        if paired_id_successes
                        else None
                    ),
                }
            )
        condition_summaries[label] = summary

    primary_rates = [
        condition_summaries[label]["success_rate"]
        for label in PRIMARY_OOD_LABELS
        if label in condition_summaries
    ]
    aggregate: dict[str, Any] = {}
    if primary_rates:
        aggregate["mean_primary_ood_success_rate"] = sum(primary_rates) / len(
            primary_rates
        )
        aggregate["worst_primary_ood_success_rate"] = min(primary_rates)
        if "correct" in condition_summaries:
            id_rate = condition_summaries["correct"]["success_rate"]
            aggregate["mean_primary_ood_retention_vs_id"] = (
                aggregate["mean_primary_ood_success_rate"] / id_rate
                if id_rate
                else None
            )

    return {
        "format": "fastwam_libero_object_language_ood_summary_v1",
        "run_root": str(run_root),
        "manifest_sha256": next(iter(manifest_hashes), None),
        "conditions": condition_summaries,
        "aggregate": aggregate,
    }


def write_summary(summary: dict[str, Any], output_prefix: Path) -> None:
    output_prefix = output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "condition",
                "successes",
                "trials",
                "success_rate_percent",
                "absolute_drop_from_id_pp",
                "retention_vs_id_percent",
                "p_success_given_id_success_percent",
            ),
        )
        writer.writeheader()
        for label, condition in summary["conditions"].items():
            writer.writerow(
                {
                    "condition": label,
                    "successes": condition["successes"],
                    "trials": condition["trials"],
                    "success_rate_percent": 100.0 * condition["success_rate"],
                    "absolute_drop_from_id_pp": (
                        100.0 * condition["absolute_drop_from_id"]
                        if "absolute_drop_from_id" in condition
                        else ""
                    ),
                    "retention_vs_id_percent": (
                        100.0 * condition["retention_vs_id"]
                        if condition.get("retention_vs_id") is not None
                        else ""
                    ),
                    "p_success_given_id_success_percent": (
                        100.0 * condition["p_success_given_id_success"]
                        if condition.get("p_success_given_id_success") is not None
                        else ""
                    ),
                }
            )
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    print(f"Wrote language-OOD summary: {json_path}")
    print(f"Wrote language-OOD table: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    write_summary(summarize(args.run_root), args.output_prefix)


if __name__ == "__main__":
    main()
