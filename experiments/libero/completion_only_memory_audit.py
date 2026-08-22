"""Pure-Python admission report for completion-only PSCM memory."""

from __future__ import annotations

from typing import Any, Mapping


def build_completion_only_memory_report(
    records: list[Mapping[str, Any]],
    *,
    enabled: bool,
    phase_safe_memory: Mapping[str, Any],
    action_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that only monotonic COMPLETED states cross replans."""
    clauses = [
        clause
        for record in records
        for clause in record.get("extended_diagnostics", {}).get("clauses", ())
        if clause.get("phase_safe_memory_available") is True
    ]
    audited = [
        clause
        for clause in clauses
        if clause.get("phase_safe_memory_previous_state_valid") is not None
    ]
    valid_previous = [
        clause
        for clause in audited
        if bool(clause["phase_safe_memory_previous_state_valid"])
    ]
    noncompleted_leaks = [
        clause
        for clause in valid_previous
        if clause.get("phase_safe_memory_previous_state") != 3
    ]
    completed_correct = [
        clause for clause in valid_previous if clause.get("status") == "completed"
    ]
    expected_completed = [
        clause for clause in audited if clause.get("status") == "completed"
    ]
    coverage = float(len(audited) / len(clauses)) if clauses else 0.0
    leakage_rate = (
        float(len(noncompleted_leaks) / len(valid_previous))
        if valid_previous
        else None
    )
    completed_precision = (
        float(len(completed_correct) / len(valid_previous))
        if valid_previous
        else None
    )
    completed_recall = (
        float(
            sum(
                bool(clause["phase_safe_memory_previous_state_valid"])
                and clause.get("phase_safe_memory_previous_state") == 3
                for clause in expected_completed
            )
            / len(expected_completed)
        )
        if expected_completed
        else None
    )
    postgrasp_scheduler = phase_safe_memory.get(
        "postgrasp_clause_scheduler_accuracy"
    )
    geometry_max_abs = phase_safe_memory.get("geometry_max_abs")
    sticky_violation = phase_safe_memory.get(
        "completed_sticky_violation_rate"
    )
    passed = bool(
        enabled
        and coverage == 1.0
        and len(valid_previous) > 0
        and leakage_rate == 0.0
        and completed_precision is not None
        and completed_precision >= 0.90
        and sticky_violation == 0.0
        and postgrasp_scheduler is not None
        and float(postgrasp_scheduler) >= 0.90
        and geometry_max_abs == 0.0
        and int(action_integrity.get("chunks", 0)) == len(records)
        and float(action_integrity.get("exact_rate", 0.0)) == 1.0
        and float(action_integrity.get("max_abs_error", float("inf"))) == 0.0
    )
    return {
        "enabled": bool(enabled),
        "policy_state_mode": (
            "completion_only" if enabled else "recurrent"
        ),
        "completed_memory_exercised": bool(valid_previous),
        "audited_clause_count": len(clauses),
        "previous_state_valid_coverage": coverage,
        "completed_previous_state_count": len(valid_previous),
        "noncompleted_previous_state_leak_count": len(noncompleted_leaks),
        "noncompleted_previous_state_leakage_rate": leakage_rate,
        "completed_previous_state_precision": completed_precision,
        "completed_previous_state_recall": completed_recall,
        "completed_sticky_violation_rate": sticky_violation,
        "postgrasp_clause_scheduler_accuracy": postgrasp_scheduler,
        "geometry_max_abs": geometry_max_abs,
        "base_action_exact_rate": action_integrity.get("exact_rate"),
        "admission_thresholds": {
            "previous_state_valid_coverage": 1.0,
            "completed_previous_state_count_min": 1,
            "noncompleted_previous_state_leakage_rate": 0.0,
            "completed_previous_state_precision": 0.90,
            "completed_sticky_violation_rate": 0.0,
            "postgrasp_clause_scheduler_accuracy": 0.90,
            "geometry_max_abs": 0.0,
            "base_action_exact_rate": 1.0,
        },
        "passed": passed,
        "interpretation": (
            "Only a monotonic completed-clause bitset is recurrent; holding, "
            "retry, and pending predictions are discarded after every replan."
        ),
    }
