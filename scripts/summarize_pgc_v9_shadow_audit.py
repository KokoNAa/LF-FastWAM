#!/usr/bin/env python3
"""Aggregate passive closed-loop ERAF records from LIBERO task results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.eraf_shadow_audit import summarize_eraf_shadow_records
from experiments.libero.completion_only_memory_audit import (
    build_completion_only_memory_report,
)
from experiments.libero.stateless_replan_audit import (
    build_stateless_replan_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--offline-report",
        type=Path,
        help="Optional offline grounding-gate JSON evaluated on the same checkpoint.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit 2 unless the canonical grounding gate and Base integrity pass.",
    )
    parser.add_argument(
        "--require-extended",
        action="store_true",
        help="Exit 2 unless every record contains the expanded V2 diagnostics.",
    )
    parser.add_argument(
        "--require-phase-safe-memory",
        action="store_true",
        help=(
            "Exit 2 unless the V9.13 phase-safe memory admission passes. "
            "This gate is independent of the legacy offline grounding gate."
        ),
    )
    parser.add_argument(
        "--require-stateless-replan",
        action="store_true",
        help=(
            "Exit 2 unless the reset-each-replan ablation is declared, every "
            "audited clause receives no previous PSCM state, Base actions stay "
            "exact, geometry stays frozen, and post-grasp scheduling is >=90%."
        ),
    )
    parser.add_argument(
        "--require-completion-only-memory",
        action="store_true",
        help=(
            "Exit 2 unless only completed states cross replans, completed "
            "memory is exercised and sticky, Base actions stay exact, geometry "
            "stays frozen, and post-grasp scheduling is >=90%."
        ),
    )
    return parser.parse_args()


def _result_files(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    files = sorted(path.rglob("gpu*_task*_results.json"))
    if not files:
        raise FileNotFoundError(f"No LIBERO task result JSON under {path}.")
    return files


def _metric_delta(
    online: Mapping[str, Any], offline: Mapping[str, Any]
) -> dict[str, Any]:
    higher_is_better = (
        "subject_top1_in_gt_mask",
        "reference_top1_in_gt_mask",
        "relation_macro_f1",
        "exclusive_role_accuracy",
        "clause_exact_match",
        "multi_clause_exact_match",
    )
    result: dict[str, Any] = {}
    for name in higher_is_better:
        online_value = online.get(name)
        offline_value = offline.get(name)
        if online_value is None or offline_value is None:
            result[name] = None
        else:
            result[name] = float(online_value) - float(offline_value)
    online_anchor = online.get("visible_goal_anchor_median_error_cm")
    offline_anchor = offline.get("visible_goal_anchor_median_error_cm")
    result["visible_goal_anchor_median_error_cm"] = (
        None
        if online_anchor is None or offline_anchor is None
        else float(online_anchor) - float(offline_anchor)
    )
    return result


def build_summary(
    result_files: list[Path],
    *,
    offline_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    action_integrity: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    policy_state_modes: set[str] = set()
    deployed_completion_only_modes: set[bool] = set()
    for path in result_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit = payload.get("eraf_shadow_audit")
        if not isinstance(audit, Mapping) or not audit.get("enabled"):
            raise ValueError(f"Result lacks an ERAF shadow audit: {path}")
        policy_state_mode = str(audit.get("policy_state_mode", "recurrent"))
        if policy_state_mode not in {
            "recurrent",
            "reset_each_replan",
            "completion_only",
        }:
            raise ValueError(
                f"Unknown ERAF policy-state mode {policy_state_mode!r}: {path}"
            )
        declared_stateless = bool(
            audit.get(
                "stateless_replan_ablation",
                policy_state_mode == "reset_each_replan",
            )
        )
        if declared_stateless != (policy_state_mode == "reset_each_replan"):
            raise ValueError(f"Inconsistent stateless ablation metadata: {path}")
        declared_completion_only_ablation = bool(
            audit.get(
                "completion_only_memory_ablation",
                False,
            )
        )
        declared_completion_only_deployed = bool(
            audit.get("completion_only_memory_deployed", False)
        )
        effective_completion_only = (
            not declared_stateless
            and (
                declared_completion_only_ablation
                or declared_completion_only_deployed
            )
        )
        if effective_completion_only != (
            policy_state_mode == "completion_only"
        ):
            raise ValueError(
                f"Inconsistent completion-only policy-state metadata: {path}"
            )
        deployed_completion_only_modes.add(declared_completion_only_deployed)
        policy_state_modes.add(policy_state_mode)
        task_records = audit.get("records")
        if not isinstance(task_records, list) or not task_records:
            raise ValueError(f"Result has no ERAF shadow records: {path}")
        records.extend(dict(item) for item in task_records)
        decisions = [
            decision
            for episode in payload.get("policy_guard_episode_diagnostics", ())
            for decision in episode.get("decisions", ())
        ]
        task_integrity = [
            dict(item["shadow_action_integrity"])
            for item in decisions
            if "shadow_action_integrity" in item
        ]
        if len(task_integrity) != len(task_records):
            raise ValueError(
                f"Shadow action/grounding record mismatch in {path}: "
                f"{len(task_integrity)} != {len(task_records)}."
            )
        action_integrity.extend(task_integrity)
        tasks.append(
            {
                "task_id": int(payload["task_id"]),
                "task_description": str(payload["task_description"]),
                "episodes": int(payload["total_episodes"]),
                "successes": int(payload["successes"]),
                "decisions": len(task_records),
                "shadow_passed": bool(audit.get("summary", {}).get("passed", False)),
                "result_file": str(path),
            }
        )
    summary = summarize_eraf_shadow_records(
        records,
        action_integrity=action_integrity,
    )
    if len(policy_state_modes) != 1:
        raise ValueError(
            "Cannot mix policy-state modes in one shadow summary: "
            f"{sorted(policy_state_modes)}."
        )
    policy_state_mode = next(iter(policy_state_modes))
    if len(deployed_completion_only_modes) != 1:
        raise ValueError(
            "Cannot mix deployed completion-only contracts in one shadow "
            "summary."
        )
    completion_only_memory_deployed = next(
        iter(deployed_completion_only_modes)
    )
    stateless_report = build_stateless_replan_report(
        records,
        enabled=policy_state_mode == "reset_each_replan",
        phase_safe_memory=summary["phase_safe_clause_memory"],
        action_integrity=summary["action_integrity"],
    )
    completion_only_report = build_completion_only_memory_report(
        records,
        enabled=policy_state_mode == "completion_only",
        phase_safe_memory=summary["phase_safe_clause_memory"],
        action_integrity=summary["action_integrity"],
    )
    summary.update(
        {
            "result_files": len(result_files),
            "episodes": sum(item["episodes"] for item in tasks),
            "tasks": sorted(tasks, key=lambda item: item["task_id"]),
            "policy_state_mode": policy_state_mode,
            "completion_only_memory_deployed": (
                completion_only_memory_deployed
            ),
            "stateless_replan_ablation": stateless_report,
            "completion_only_memory_ablation": completion_only_report,
        }
    )
    if offline_report is not None:
        offline_metrics = offline_report.get("metrics")
        if not isinstance(offline_metrics, Mapping):
            raise ValueError("Offline report has no metrics object.")
        online_metrics = summary["grounding_gate"]["metrics"]
        summary["offline_reference"] = dict(offline_metrics)
        summary["online_minus_offline"] = _metric_delta(
            online_metrics,
            offline_metrics,
        )
    return summary


def main() -> None:
    args = _parse_args()
    offline_report = None
    if args.offline_report is not None:
        offline_report = json.loads(
            args.offline_report.expanduser().resolve().read_text(encoding="utf-8")
        )
    summary = build_summary(
        _result_files(args.results),
        offline_report=offline_report,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"ERAF_SHADOW_SUMMARY={output}")
    if args.require_pass and not summary["passed"]:
        raise SystemExit(2)
    extended = summary.get("extended_diagnostics", {})
    if args.require_extended and (
        not extended.get("available")
        or float(extended.get("record_coverage", 0.0)) != 1.0
    ):
        raise SystemExit(2)
    if args.require_phase_safe_memory and not summary.get(
        "phase_safe_memory_admission_passed", False
    ):
        raise SystemExit(2)
    if args.require_stateless_replan and not summary.get(
        "stateless_replan_ablation", {}
    ).get("passed", False):
        raise SystemExit(2)
    if args.require_completion_only_memory and not summary.get(
        "completion_only_memory_ablation", {}
    ).get("passed", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
