"""Pure-Python admission report for the stateless PSCM replan ablation."""

from __future__ import annotations

from typing import Any, Mapping


def build_stateless_replan_report(
    records: list[Mapping[str, Any]],
    *,
    enabled: bool,
    phase_safe_memory: Mapping[str, Any],
    action_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that the recurrent PSCM input channel was cut at every replan."""
    clauses = [
        clause
        for record in records
        for clause in record.get("extended_diagnostics", {}).get("clauses", ())
        if clause.get("phase_safe_memory_available") is True
    ]
    previous_state_valid = [
        clause.get("phase_safe_memory_previous_state_valid")
        for clause in clauses
        if clause.get("phase_safe_memory_previous_state_valid") is not None
    ]
    coverage = (
        float(len(previous_state_valid) / len(clauses)) if clauses else 0.0
    )
    invalid_rate = (
        float(sum(not bool(value) for value in previous_state_valid))
        / len(previous_state_valid)
        if previous_state_valid
        else None
    )
    postgrasp_scheduler = phase_safe_memory.get(
        "postgrasp_clause_scheduler_accuracy"
    )
    geometry_max_abs = phase_safe_memory.get("geometry_max_abs")
    passed = bool(
        enabled
        and coverage == 1.0
        and invalid_rate == 1.0
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
            "reset_each_replan" if enabled else "recurrent"
        ),
        "state_input_channel_cut": bool(
            enabled and coverage == 1.0 and invalid_rate == 1.0
        ),
        "audited_clause_count": len(clauses),
        "previous_state_valid_coverage": coverage,
        "previous_state_invalid_rate": invalid_rate,
        "postgrasp_clause_scheduler_accuracy": postgrasp_scheduler,
        "geometry_max_abs": geometry_max_abs,
        "base_action_exact_rate": action_integrity.get("exact_rate"),
        "admission_thresholds": {
            "previous_state_valid_coverage": 1.0,
            "previous_state_invalid_rate": 1.0,
            "postgrasp_clause_scheduler_accuracy": 0.90,
            "geometry_max_abs": 0.0,
            "base_action_exact_rate": 1.0,
        },
        "passed": passed,
        "interpretation": (
            "PSCM state predictions remain diagnostic outputs only; their "
            "state accuracy is not an admission criterion in this ablation."
        ),
    }
