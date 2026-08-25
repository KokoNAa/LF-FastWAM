#!/usr/bin/env python3
"""Evaluate the PGC V9 ERAF gate through the deployed RGB-language path."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _macro_f1(targets: Iterable[int], predictions: Iterable[int]) -> float:
    target = np.asarray(list(targets), dtype=np.int64)
    prediction = np.asarray(list(predictions), dtype=np.int64)
    if target.size == 0 or target.shape != prediction.shape:
        return 0.0
    scores = []
    for label in sorted(set(target.tolist())):
        true_positive = int(((target == label) & (prediction == label)).sum())
        false_positive = int(((target != label) & (prediction == label)).sum())
        false_negative = int(((target == label) & (prediction != label)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores)) if scores else 0.0


def _safe_rate(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def _optional_rate(values: list[bool]) -> float | None:
    return float(np.mean(values)) if values else None


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _role_group_summary(
    clauses: list[Mapping[str, Any]],
) -> dict[str, Any]:
    full = [bool(item["full_mask_correct"]) for item in clauses]
    exclusive_items = [item for item in clauses if bool(item["exclusive_mask_valid"])]
    exclusive = [bool(item["exclusive_mask_correct"]) for item in exclusive_items]
    full_failures = [item for item in clauses if not item["full_mask_correct"]]
    recovered = [
        item
        for item in full_failures
        if item["exclusive_mask_valid"] and item["exclusive_mask_correct"]
    ]
    remains_wrong = [
        item
        for item in full_failures
        if item["exclusive_mask_valid"] and not item["exclusive_mask_correct"]
    ]
    ambiguous = [item for item in full_failures if not item["exclusive_mask_valid"]]
    all_entity_available = any(
        "all_entity_exclusive_valid" in item for item in clauses
    )
    all_entity_items = [
        item
        for item in clauses
        if bool(item.get("all_entity_exclusive_valid", False))
    ]
    all_entity = [
        bool(item["all_entity_exclusive_correct"])
        for item in all_entity_items
    ]
    return {
        "clauses": len(clauses),
        "full_mask": {
            "correct": int(sum(full)),
            "accuracy": _optional_rate(full),
        },
        "exclusive_mask": {
            "eligible": len(exclusive_items),
            "coverage": (
                float(len(exclusive_items) / len(clauses)) if clauses else None
            ),
            "correct": int(sum(exclusive)),
            "accuracy": _optional_rate(exclusive),
        },
        "all_entity_exclusive_mask": {
            "available": all_entity_available,
            "eligible": len(all_entity_items),
            "coverage": (
                float(len(all_entity_items) / len(clauses))
                if clauses and all_entity_available
                else None
            ),
            "correct": int(sum(all_entity)),
            "accuracy": _optional_rate(all_entity),
        },
        "full_mask_failure_partition": {
            "total": len(full_failures),
            "exclusive_recovers": len(recovered),
            "exclusive_still_wrong": len(remains_wrong),
            "exclusive_ambiguous": len(ambiguous),
        },
        "mask_overlap_iou": _distribution(
            float(item["mask_overlap_iou"]) for item in clauses
        ),
        "subject_overlap_fraction": _distribution(
            float(item["subject_overlap_fraction"]) for item in clauses
        ),
        "reference_overlap_fraction": _distribution(
            float(item["reference_overlap_fraction"]) for item in clauses
        ),
        "full_subject_margin": _distribution(
            float(item["full_subject_margin"]) for item in clauses
        ),
        "full_reference_margin": _distribution(
            float(item["full_reference_margin"]) for item in clauses
        ),
        "exclusive_subject_margin": _distribution(
            float(item["exclusive_subject_margin"]) for item in exclusive_items
        ),
        "exclusive_reference_margin": _distribution(
            float(item["exclusive_reference_margin"]) for item in exclusive_items
        ),
    }


def _role_residual_audit(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    clauses = [
        dict(clause)
        for record in records
        for clause in record.get("role_audit_clauses", ())
    ]
    if not clauses:
        return {
            "format": "pgc_v9_eraf_role_residual_audit_v1",
            "available": False,
            "diagnosis": "missing_per_clause_role_audit",
        }
    overall = _role_group_summary(clauses)
    grouped: dict[str, dict[str, Any]] = {}
    for field in ("dataset_kind", "predicate", "task"):
        values: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for clause in clauses:
            values[str(clause.get(field, "unknown"))].append(clause)
        grouped[f"by_{field}"] = {
            key: _role_group_summary(items) for key, items in sorted(values.items())
        }

    full_accuracy = overall["full_mask"]["accuracy"]
    all_entity_summary = overall["all_entity_exclusive_mask"]
    use_all_entity = bool(all_entity_summary["available"])
    semantic_summary = (
        all_entity_summary if use_all_entity else overall["exclusive_mask"]
    )
    exclusive_accuracy = semantic_summary["accuracy"]
    exclusive_coverage = semantic_summary["coverage"]
    if exclusive_coverage is None or exclusive_coverage < 0.50:
        diagnosis = "insufficient_exclusive_role_support"
        recommendation = (
            "Inspect mask construction or add higher-resolution role labels before "
            "changing the role objective."
        )
    elif exclusive_accuracy is not None and exclusive_accuracy >= 0.90:
        if full_accuracy is not None and full_accuracy >= 0.90:
            diagnosis = "role_gate_pass"
            recommendation = "Proceed only if every other grounding gate passes."
        else:
            diagnosis = "exclusive_role_gate_pass_full_mask_overlap_diagnostic"
            recommendation = (
                "Use exclusive evidence for semantic role acceptance and retain "
                "full-mask top-1 checks for localization."
            )
    else:
        diagnosis = (
            "all_entity_role_binding_generalization_failure"
            if use_all_entity
            else "role_binding_generalization_failure"
        )
        recommendation = (
            "Train the same-state exclusive all-entity assignment objective with "
            "balanced cross-clause hard negatives; do not enter action training yet."
            if use_all_entity
            else "Train an explicit subject/reference assignment objective with "
            "balanced hard role-swap examples; do not enter action training yet."
        )
    return {
        "format": "pgc_v9_eraf_role_residual_audit_v1",
        "available": True,
        "semantic_role_scope": (
            "exclusive_all_entity" if use_all_entity else "exclusive_pairwise"
        ),
        "overall": overall,
        **grouped,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
    }


def compute_grounding_gate_report(
    records: list[Mapping[str, Any]],
    *,
    require_view_scheduler: bool = False,
) -> dict[str, Any]:
    """Aggregate serializable per-sample ERAF observations into gate metrics."""
    if not records:
        raise ValueError("PGC v9 grounding gate received no records.")
    subject_hits: list[bool] = []
    reference_hits: list[bool] = []
    role_swap: list[bool] = []
    relation_targets: list[int] = []
    relation_predictions: list[int] = []
    goal_anchor_errors_m: list[float] = []
    clause_exact: list[bool] = []
    multi_clause_exact: list[bool] = []
    single_visible_view_selection: list[bool] = []
    clause_scheduler_correct: list[bool] = []
    clause_scheduler_confidence: list[float] = []
    multi_clause_failure_partition: dict[str, int] = defaultdict(int)
    for record in records:
        subject_hits.extend(bool(value) for value in record["subject_top1_hits"])
        reference_hits.extend(bool(value) for value in record["reference_top1_hits"])
        role_swap.extend(bool(value) for value in record["role_swap_correct"])
        relation_targets.extend(int(value) for value in record["relation_targets"])
        relation_predictions.extend(
            int(value) for value in record["relation_predictions"]
        )
        goal_anchor_errors_m.extend(
            float(value) for value in record["goal_anchor_errors_m"]
        )
        exact = bool(record["clause_exact"])
        clause_exact.append(exact)
        if int(record["clause_count"]) > 1:
            multi_clause_exact.append(exact)
            components = record.get("clause_exact_components", {})
            for name in (
                "active",
                "predicate",
                "subject_localization",
                "reference_localization",
                "semantic_role",
            ):
                if not bool(components.get(name, True)):
                    multi_clause_failure_partition[name] += 1
        single_visible_view_selection.extend(
            bool(value)
            for value in record.get("single_visible_view_selection_correct", ())
        )
        clause_scheduler_correct.extend(
            bool(value)
            for value in record.get("clause_scheduler_correct", ())
        )
        clause_scheduler_confidence.extend(
            float(value)
            for value in record.get("clause_scheduler_confidence", ())
        )
    anchor_median_cm = (
        float(np.median(goal_anchor_errors_m) * 100.0)
        if goal_anchor_errors_m
        else float("inf")
    )
    role_residual_audit = _role_residual_audit(records)
    role_overall = role_residual_audit.get("overall", {})
    all_entity_summary = role_overall.get("all_entity_exclusive_mask", {})
    pairwise_exclusive_summary = role_overall.get("exclusive_mask", {})
    use_all_entity = bool(all_entity_summary.get("available", False))
    exclusive_summary = (
        all_entity_summary if use_all_entity else pairwise_exclusive_summary
    )
    exclusive_role_accuracy = exclusive_summary.get("accuracy")
    exclusive_role_coverage = exclusive_summary.get("coverage")
    metrics = {
        "samples": len(records),
        "subject_top1_in_gt_mask": _safe_rate(subject_hits),
        "reference_top1_in_gt_mask": _safe_rate(reference_hits),
        "relation_macro_f1": _macro_f1(relation_targets, relation_predictions),
        # Retained for historical comparison only. V9.7 gates semantic roles
        # with exclusive evidence because full subject/reference masks may
        # overlap in valid in/on states.
        "role_swap_accuracy": _safe_rate(role_swap),
        "full_mask_role_swap_accuracy": _safe_rate(role_swap),
        "exclusive_role_accuracy": exclusive_role_accuracy,
        "exclusive_role_coverage": exclusive_role_coverage,
        "pairwise_exclusive_role_accuracy": pairwise_exclusive_summary.get(
            "accuracy"
        ),
        "all_entity_exclusive_role_accuracy": all_entity_summary.get(
            "accuracy"
        ),
        "all_entity_exclusive_role_coverage": all_entity_summary.get(
            "coverage"
        ),
        "semantic_role_gate_scope": (
            "exclusive_all_entity" if use_all_entity else "exclusive_pairwise"
        ),
        "visible_goal_anchor_median_error_cm": anchor_median_cm,
        "clause_exact_match": _safe_rate(clause_exact),
        "multi_clause_exact_match": _safe_rate(multi_clause_exact),
        "multi_clause_samples": len(multi_clause_exact),
        "single_visible_view_selection_accuracy": _optional_rate(
            single_visible_view_selection
        ),
        "single_visible_view_selection_samples": len(
            single_visible_view_selection
        ),
        "clause_scheduler_accuracy": _optional_rate(clause_scheduler_correct),
        "clause_scheduler_samples": len(clause_scheduler_correct),
        "clause_scheduler_confidence_mean": (
            float(np.mean(clause_scheduler_confidence))
            if clause_scheduler_confidence
            else None
        ),
    }
    checks = {
        "subject_top1_at_least_80pct": (metrics["subject_top1_in_gt_mask"] >= 0.80),
        "reference_top1_at_least_80pct": (metrics["reference_top1_in_gt_mask"] >= 0.80),
        "relation_macro_f1_at_least_90pct": (metrics["relation_macro_f1"] >= 0.90),
        "exclusive_role_coverage_at_least_50pct": (
            exclusive_role_coverage is not None and exclusive_role_coverage >= 0.50
        ),
        "exclusive_role_accuracy_at_least_90pct": (
            exclusive_role_accuracy is not None and exclusive_role_accuracy >= 0.90
        ),
        "visible_goal_anchor_median_at_most_5cm": anchor_median_cm <= 5.0,
        "multi_clause_exact_at_least_80pct": (
            bool(multi_clause_exact) and metrics["multi_clause_exact_match"] >= 0.80
        ),
    }
    if require_view_scheduler:
        checks.update(
            {
                "single_visible_view_selection_at_least_80pct": (
                    bool(single_visible_view_selection)
                    and metrics["single_visible_view_selection_accuracy"] >= 0.80
                ),
                "unfinished_clause_scheduler_at_least_90pct": (
                    bool(clause_scheduler_correct)
                    and metrics["clause_scheduler_accuracy"] >= 0.90
                ),
            }
        )
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks == ["multi_clause_exact_at_least_80pct"]:
        diagnosis = "multi_clause_activation_failure"
        recommendation = (
            "Calibrate frozen clause active logits/cardinality without updating "
            "the validated entity, relation, role, or anchor paths."
        )
    elif not failed_checks:
        diagnosis = "grounding_gate_pass"
        recommendation = "Proceed to grounding-action joint training."
    else:
        diagnosis = "grounding_gate_failure"
        recommendation = (
            "Do not enter action training; inspect the failed grounding checks."
        )
    return {
        "format": "pgc_v9_eraf_grounding_gate_v2",
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "diagnostics": {
            "full_mask_role_swap_at_least_90pct": (
                metrics["full_mask_role_swap_accuracy"] >= 0.90
            )
        },
        "role_residual_audit": role_residual_audit,
        "multi_clause_failure_partition": dict(
            sorted(multi_clause_failure_partition.items())
        ),
    }


def mine_hard_native_rows(
    records: list[Mapping[str, Any]],
    *,
    objective_version: int,
) -> dict[str, Any]:
    """Mine audited native failures for a later hard/easy curriculum."""
    native_records = [
        record for record in records if record.get("_dataset_kind") == "native"
    ]
    audited = sorted({int(record["_raw_index"]) for record in native_records})
    if objective_version == 4:
        hard = sorted(
            {
                int(record["_raw_index"])
                for record in native_records
                if record.get("role_swap_correct")
                and not all(
                    bool(value) for value in record["role_swap_correct"]
                )
            }
        )
        index_format = "pgc_v9_hard_role_index_v1"
        criterion = "any_valid_full_mask_role_swap_failure"
    elif objective_version == 11:
        hard_rows: set[int] = set()
        for record in native_records:
            if int(record.get("clause_count", 0)) <= 1:
                continue
            role_failure = any(
                bool(clause.get("all_entity_exclusive_valid", False))
                and not bool(
                    clause.get("all_entity_exclusive_correct", False)
                )
                for clause in record.get("role_audit_clauses", ())
            )
            localization_failure = not all(
                bool(value)
                for value in (
                    *record.get("subject_top1_hits", ()),
                    *record.get("reference_top1_hits", ()),
                )
            )
            if role_failure or localization_failure:
                hard_rows.add(int(record["_raw_index"]))
        hard = sorted(hard_rows)
        index_format = "pgc_v9_hard_role_index_v2"
        criterion = (
            "multi_clause_any_all_entity_role_or_subject_reference_"
            "localization_failure"
        )
    else:
        raise ValueError(
            "Hard-row mining supports objective-v4 (V9.3) or "
            "objective-v11 (V9.10)."
        )
    return {
        "audited_native_raw_indices": audited,
        "hard_native_raw_indices": hard,
        "format": index_format,
        "criterion": criterion,
    }


def _find_training_config(checkpoint: Path) -> Path:
    for parent in checkpoint.parents:
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find the run config above checkpoint {checkpoint}."
    )


def _model_dtype(name: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _patch_targets(mask: np.ndarray, token_count: int) -> np.ndarray:
    import torch
    from fastwam.models.wan22.entity_relation_affordance import (
        masks_to_patch_targets,
    )

    value, _ = masks_to_patch_targets(
        torch.as_tensor(mask).unsqueeze(0), token_count=token_count
    )
    return value[0].numpy()


def _sample_record(
    diagnostics: Mapping[str, Any],
    sample: Mapping[str, Any],
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    *,
    dataset_kind: str = "unknown",
    dataset_label: str = "unknown",
    predicate_vocabulary: Iterable[str] = (),
    all_entity_role_gate: bool = False,
) -> dict[str, Any]:
    import torch

    def array(name: str) -> np.ndarray:
        value = diagnostics[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        return value[0] if value.ndim > 0 and value.shape[0] == 1 else value

    clause_valid = np.asarray(sample["pgc_eraf_clause_valid"], dtype=bool)
    predicate_ids = np.asarray(sample["pgc_eraf_predicate_ids"], dtype=np.int64)
    active = array("active_logits") > 0
    predicate_prediction = array("predicate_logits").argmax(axis=-1)
    subject_attention = array("subject_attention")
    reference_attention = array("reference_attention")
    token_count = int(subject_attention.shape[-1])
    subject_target = _patch_targets(
        np.asarray(sample["pgc_eraf_subject_masks"]), token_count
    )
    reference_target = _patch_targets(
        np.asarray(sample["pgc_eraf_reference_masks"]), token_count
    )
    subject_valid = clause_valid & np.asarray(
        sample["pgc_eraf_subject_mask_valid"], dtype=bool
    )
    reference_valid = clause_valid & np.asarray(
        sample["pgc_eraf_reference_mask_valid"], dtype=bool
    )
    subject_hits = [
        bool(subject_target[index, subject_attention[index].argmax()] > 0)
        for index in np.flatnonzero(subject_valid)
    ]
    reference_hits = [
        bool(reference_target[index, reference_attention[index].argmax()] > 0)
        for index in np.flatnonzero(reference_valid)
    ]
    single_visible_view_selection_correct: list[bool] = []
    view_diagnostic_names = {
        "camera_ids",
        "subject_view_attention_mass",
        "reference_view_attention_mass",
    }
    v99_enabled = (
        "view_scheduler_enabled" in diagnostics
        and bool(array("view_scheduler_enabled").reshape(-1)[0])
    )
    if v99_enabled and not view_diagnostic_names.difference(diagnostics):
        camera_ids = array("camera_ids").astype(np.int64)
        for role, valid in (
            ("subject", subject_valid),
            ("reference", reference_valid),
        ):
            view_visible_key = f"pgc_eraf_{role}_view_visible"
            if view_visible_key not in sample:
                continue
            view_visible = np.asarray(sample[view_visible_key], dtype=bool)
            predicted_view_mass = array(f"{role}_view_attention_mass")
            for index in np.flatnonzero(valid):
                visible_indices = np.flatnonzero(view_visible[index])
                if len(visible_indices) != 1:
                    continue
                target_camera = int(visible_indices[0])
                if not bool((camera_ids == target_camera).any()):
                    continue
                single_visible_view_selection_correct.append(
                    int(predicted_view_mass[index].argmax()) == target_camera
                )

    clause_scheduler_correct: list[bool] = []
    clause_scheduler_confidence: list[float] = []
    scheduler_names = {
        "clause_execution_probability",
        "clause_routing_residual",
    }
    if v99_enabled and not scheduler_names.difference(diagnostics):
        truth_key = "pgc_eraf_predicate_truth"
        truth_valid_key = "pgc_eraf_predicate_truth_valid"
        if truth_key in sample and truth_valid_key in sample:
            predicate_truth = np.asarray(sample[truth_key], dtype=np.float32)
            predicate_truth_valid = np.asarray(sample[truth_valid_key], dtype=bool)
            unfinished = (
                clause_valid
                & predicate_truth_valid
                & (predicate_truth < 0.5)
            )
            if bool(unfinished.any()):
                target_index = int(np.flatnonzero(unfinished)[0])
                execution_probability = array("clause_execution_probability")
                selected_index = int(
                    np.where(clause_valid, execution_probability, -np.inf).argmax()
                )
                clause_scheduler_correct.append(selected_index == target_index)
                clause_scheduler_confidence.append(
                    float(execution_probability[selected_index])
                )
    subject_entity_ids = np.asarray(
        sample["pgc_eraf_subject_entity_ids"], dtype=np.int64
    )
    reference_entity_ids = np.asarray(
        sample["pgc_eraf_reference_entity_ids"], dtype=np.int64
    )
    # Unary articulated predicates deliberately bind subject/reference to the
    # same fixture. They have no meaningful role-swap negative and must not
    # make the grounding gate mathematically impossible.
    role_valid = (
        subject_valid & reference_valid & (subject_entity_ids != reference_entity_ids)
    )
    role_swap_correct = []
    all_entity_role_correct: list[bool] = []
    role_audit_clauses = []
    all_entity_clause_audit: dict[int, tuple[bool, bool]] = {}
    if all_entity_role_gate:
        candidate_targets = np.concatenate(
            (subject_target, reference_target), axis=0
        )
        candidate_ids = np.concatenate(
            (subject_entity_ids, reference_entity_ids), axis=0
        )
        candidate_valid = np.concatenate(
            (subject_valid, reference_valid), axis=0
        ) & (candidate_ids >= 0)
        exclusive_candidates = np.zeros_like(candidate_targets)
        for candidate_index in np.flatnonzero(candidate_valid):
            competitors = np.flatnonzero(
                candidate_valid
                & (candidate_ids != candidate_ids[candidate_index])
            )
            competing_support = (
                candidate_targets[competitors].max(axis=0)
                if len(competitors)
                else np.zeros_like(candidate_targets[candidate_index])
            )
            exclusive_candidates[candidate_index] = np.clip(
                candidate_targets[candidate_index] - competing_support,
                0.0,
                None,
            )
        exclusive_candidate_valid = candidate_valid & (
            exclusive_candidates.sum(axis=-1) > 1e-8
        )
        query_attentions = np.concatenate(
            (subject_attention, reference_attention), axis=0
        )
        query_ids = candidate_ids
        query_valid = np.concatenate((role_valid, role_valid), axis=0)
        all_entity_scores = query_attentions @ exclusive_candidates.T
        query_correct = np.zeros_like(query_valid, dtype=bool)
        query_auditable = np.zeros_like(query_valid, dtype=bool)
        for query_index in np.flatnonzero(query_valid):
            positives = exclusive_candidate_valid & (
                candidate_ids == query_ids[query_index]
            )
            negatives = exclusive_candidate_valid & (
                candidate_ids != query_ids[query_index]
            )
            if not bool(positives.any() and negatives.any()):
                continue
            query_auditable[query_index] = True
            query_correct[query_index] = bool(
                all_entity_scores[query_index, positives].max()
                > all_entity_scores[query_index, negatives].max()
            )
        clause_slots = len(clause_valid)
        for clause_index in np.flatnonzero(role_valid):
            subject_index = int(clause_index)
            reference_index = clause_slots + int(clause_index)
            auditable = bool(
                query_auditable[subject_index]
                and query_auditable[reference_index]
            )
            correct = bool(
                auditable
                and query_correct[subject_index]
                and query_correct[reference_index]
            )
            all_entity_clause_audit[int(clause_index)] = (auditable, correct)
            if auditable:
                all_entity_role_correct.append(correct)
    predicate_names = tuple(str(value) for value in predicate_vocabulary)
    prompt = str(sample.get("prompt", "unknown"))
    prompt_prefix = (
        "A video recorded from a robot's point of view executing the "
        "following instruction: "
    )
    task = prompt[len(prompt_prefix) :] if prompt.startswith(prompt_prefix) else prompt
    for index in np.flatnonzero(role_valid):
        subject_own = float((subject_attention[index] * subject_target[index]).sum())
        subject_wrong = float(
            (subject_attention[index] * reference_target[index]).sum()
        )
        reference_own = float(
            (reference_attention[index] * reference_target[index]).sum()
        )
        reference_wrong = float(
            (reference_attention[index] * subject_target[index]).sum()
        )
        full_correct = subject_own > subject_wrong and reference_own > reference_wrong
        role_swap_correct.append(full_correct)

        subject_mask = subject_target[index]
        reference_mask = reference_target[index]
        shared = np.minimum(subject_mask, reference_mask)
        union = np.maximum(subject_mask, reference_mask)
        subject_exclusive = np.clip(subject_mask - shared, 0.0, None)
        reference_exclusive = np.clip(reference_mask - shared, 0.0, None)
        subject_support = float(subject_mask.sum())
        reference_support = float(reference_mask.sum())
        shared_support = float(shared.sum())
        exclusive_valid = bool(
            float(subject_exclusive.sum()) > 1e-8
            and float(reference_exclusive.sum()) > 1e-8
        )
        subject_exclusive_own = float(
            (subject_attention[index] * subject_exclusive).sum()
        )
        subject_exclusive_wrong = float(
            (subject_attention[index] * reference_exclusive).sum()
        )
        reference_exclusive_own = float(
            (reference_attention[index] * reference_exclusive).sum()
        )
        reference_exclusive_wrong = float(
            (reference_attention[index] * subject_exclusive).sum()
        )
        exclusive_correct = bool(
            exclusive_valid
            and subject_exclusive_own > subject_exclusive_wrong
            and reference_exclusive_own > reference_exclusive_wrong
        )
        predicate_id = int(predicate_ids[index])
        predicate_name = (
            predicate_names[predicate_id]
            if 0 <= predicate_id < len(predicate_names)
            else str(predicate_id)
        )
        role_audit_clauses.append(
            {
                "clause_index": int(index),
                "predicate_id": predicate_id,
                "predicate": predicate_name,
                "dataset_kind": str(dataset_kind),
                "dataset": str(dataset_label),
                "task": task,
                "full_mask_correct": bool(full_correct),
                "exclusive_mask_valid": exclusive_valid,
                "exclusive_mask_correct": exclusive_correct,
                "mask_overlap_iou": (
                    float(shared_support / float(union.sum()))
                    if float(union.sum()) > 0
                    else 0.0
                ),
                "subject_overlap_fraction": (
                    float(shared_support / subject_support)
                    if subject_support > 0
                    else 0.0
                ),
                "reference_overlap_fraction": (
                    float(shared_support / reference_support)
                    if reference_support > 0
                    else 0.0
                ),
                "full_subject_margin": subject_own - subject_wrong,
                "full_reference_margin": reference_own - reference_wrong,
                "exclusive_subject_margin": (
                    subject_exclusive_own - subject_exclusive_wrong
                ),
                "exclusive_reference_margin": (
                    reference_exclusive_own - reference_exclusive_wrong
                ),
                **(
                    {
                        "all_entity_exclusive_valid": all_entity_clause_audit[
                            int(index)
                        ][0],
                        "all_entity_exclusive_correct": all_entity_clause_audit[
                            int(index)
                        ][1],
                    }
                    if all_entity_role_gate
                    else {}
                ),
            }
        )

    goal_valid = (
        clause_valid
        & reference_valid
        & np.asarray(sample["pgc_eraf_goal_anchor_valid"], dtype=bool)
    )
    prediction = array("goal_anchor")
    target = np.asarray(sample["pgc_eraf_goal_anchors"], dtype=np.float32)
    scale = (workspace_max - workspace_min) / 2.0
    anchor_errors = np.linalg.norm((prediction - target) * scale, axis=-1)
    active_exact = bool(np.array_equal(active, clause_valid))
    predicate_exact = bool(
        np.array_equal(
            predicate_prediction[clause_valid], predicate_ids[clause_valid]
        )
    )
    semantic_exact = active_exact and predicate_exact
    semantic_role_correct = (
        all_entity_role_correct
        if all_entity_role_gate
        else role_swap_correct
    )
    # Clause exact match is intentionally stricter than predicate decoding:
    # every visible semantic role must also land in its own mask and beat its
    # same-state role-swap negative.
    exact = bool(
        semantic_exact
        and all(subject_hits)
        and all(reference_hits)
        and all(semantic_role_correct)
    )
    return {
        "subject_top1_hits": subject_hits,
        "reference_top1_hits": reference_hits,
        "role_swap_correct": role_swap_correct,
        "all_entity_role_correct": all_entity_role_correct,
        "all_entity_role_gate": bool(all_entity_role_gate),
        "role_audit_clauses": role_audit_clauses,
        "relation_targets": predicate_ids[clause_valid].tolist(),
        "relation_predictions": predicate_prediction[clause_valid].tolist(),
        "goal_anchor_errors_m": anchor_errors[goal_valid].tolist(),
        "clause_exact": exact,
        "clause_exact_components": {
            "active": active_exact,
            "predicate": predicate_exact,
            "subject_localization": all(subject_hits),
            "reference_localization": all(reference_hits),
            "semantic_role": all(semantic_role_correct),
        },
        "clause_count": int(clause_valid.sum()),
        "single_visible_view_selection_correct": (
            single_visible_view_selection_correct
        ),
        "clause_scheduler_correct": clause_scheduler_correct,
        "clause_scheduler_confidence": clause_scheduler_confidence,
        "v99_view_scheduler_available": bool(
            v99_enabled
            and not view_diagnostic_names.difference(diagnostics)
            and not scheduler_names.difference(diagnostics)
            and "pgc_eraf_subject_view_visible" in sample
            and "pgc_eraf_reference_view_visible" in sample
            and "pgc_eraf_predicate_truth" in sample
            and "pgc_eraf_predicate_truth_valid" in sample
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict

    from fastwam.utils import misc
    from fastwam.utils.pytorch_utils import set_global_seed
    from fastwam.datasets.pgc_libero import (
        pgc_entity_relation_workspace_bounds,
    )

    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = (
        args.training_config.expanduser().resolve()
        if args.training_config
        else _find_training_config(checkpoint)
    )
    cfg = OmegaConf.load(config_path)
    configured_objective = int(
        OmegaConf.select(
            cfg,
            "model.policy_guard.entity_relation_grounding."
            "grounding_objective_version",
            default=1,
        )
    )
    with open_dict(cfg):
        cfg.model.load_text_encoder = True
        cfg.model.skip_dit_load_from_pretrain = True
        cfg.model.action_dit_pretrained_path = None
        # Objective 26 changes the visual representation consumed by the
        # frozen ERAF through Video-Expert LoRA.  Keep those adapters present
        # when auditing the action-stage checkpoint; disabling them would
        # silently measure the pre-upgrade visual backbone instead.
        if configured_objective < 26:
            cfg.model.lora.enabled = False
        cfg.model.policy_guard.enabled = True
        cfg.model.policy_guard.version = 9
        cfg.model.policy_guard.gate_mode = "base"
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output.parent))
    set_global_seed(args.seed, get_worker_init_fn=False)
    dataset = instantiate(cfg.data.train)
    model = instantiate(
        cfg.model,
        model_dtype=_model_dtype(args.dtype),
        device=args.device,
    )
    payload = model.load_checkpoint(str(checkpoint))
    if payload.get("format") != "fastwam_policy_guard_v9":
        raise ValueError("Grounding gate requires a PGC v9 checkpoint.")
    metadata = payload.get("architecture_metadata") or {}
    objective_version = int(metadata.get("eraf_grounding_objective_version", 1))
    grounding_expected_step = {
        2: 1500,
        3: 2500,
        4: 2500,
        5: 3500,
        6: 3500,
        7: 3000,
        8: 3250,
        9: 3750,
        10: 4750,
        11: 5750,
        12: 6250,
        13: 7250,
        14: 7250,
    }
    action_expected_step = {
        14: 11250,
        15: 13250,
        16: 13750,
        17: 14250,
        18: 15250,
        19: 16250,
        20: 17250,
        21: 18250,
        22: 18750,
        23: 18750,
        24: 18750,
        25: 19750,
        26: 20750,
    }
    audited_training_stage = str(metadata.get("eraf_training_stage", ""))
    expected_step = {
        "grounding": grounding_expected_step,
        "action": action_expected_step,
    }.get(audited_training_stage, {}).get(objective_version)
    checkpoint_step = int(payload.get("step", -1))
    intermediate_checkpoint = bool(args.allow_intermediate) and (
        (objective_version == 4 and checkpoint_step in {1750, 2000, 2250})
        or (objective_version in {5, 6} and checkpoint_step in {2750, 3000, 3250})
        or (objective_version == 7 and checkpoint_step == 2750)
        or (objective_version == 9 and checkpoint_step == 3500)
        or (objective_version == 10 and checkpoint_step in {4000, 4250, 4500})
        or (objective_version == 11 and checkpoint_step in {5000, 5250, 5500})
        or (objective_version == 12 and checkpoint_step == 6000)
        or (
            objective_version == 13
            and checkpoint_step in {6500, 6750, 7000}
        )
    )
    if (
        expected_step is None
        or (checkpoint_step != expected_step and not intermediate_checkpoint)
    ):
        raise ValueError(
            "The grounding gate requires a completed V9 grounding checkpoint "
            "or a completed action-stage checkpoint whose upstream Video/ERAF "
            "representation is being audited; got "
            f"stage={audited_training_stage!r} objective={objective_version} "
            f"step={checkpoint_step}."
        )
    model = model.to(args.device).eval()
    lower, upper = pgc_entity_relation_workspace_bounds(
        dataset.pgc_entity_relation_indices
    )
    workspace_min = np.asarray(lower, dtype=np.float32)
    workspace_max = np.asarray(upper, dtype=np.float32)

    unique_positions = []
    seen = set()
    for position, raw_index in enumerate(dataset._sample_indices):
        if int(raw_index) in seen:
            continue
        seen.add(int(raw_index))
        unique_positions.append(position)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_positions)
    if len(unique_positions) < args.num_samples:
        raise ValueError(
            f"Requested {args.num_samples} unique grounding rows, but the "
            f"audited datasets expose only {len(unique_positions)}."
        )
    selected = sorted(unique_positions[: args.num_samples])
    records = []
    for ordinal, position in enumerate(selected):
        sample = dataset[position]
        with torch.no_grad():
            prediction = model.infer_action(
                prompt=None,
                input_image=sample["video"][:, 0],
                action_horizon=int(sample["action"].shape[0]),
                proprio=sample["proprio"][0],
                context=sample["context"],
                context_mask=sample["context_mask"],
                num_inference_steps=args.num_inference_steps,
                seed=args.seed + ordinal,
                rand_device="cpu",
                tiled=False,
            )
        dataset_index = int(torch.as_tensor(sample["pgc_dataset_index"]).item())
        sidecar = dataset.pgc_entity_relation_indices[dataset_index]
        records.append(
            _sample_record(
                prediction["policy_guard_eraf_diagnostics"],
                sample,
                workspace_min,
                workspace_max,
                dataset_kind=str(sidecar["dataset_kind"]),
                dataset_label=Path(str(sidecar["dataset"])).name,
                predicate_vocabulary=sidecar["predicate_vocabulary"],
                all_entity_role_gate=objective_version >= 11,
            )
        )
        records[-1]["_raw_index"] = int(dataset._sample_indices[position])
        records[-1]["_dataset_index"] = dataset_index
        records[-1]["_dataset_kind"] = str(sidecar["dataset_kind"])
        print(f"ERAF_GATE {ordinal + 1}/{len(selected)}", flush=True)
    report = compute_grounding_gate_report(
        records,
        require_view_scheduler=objective_version >= 10,
    )
    report.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_step": payload.get("step"),
            "audited_training_stage": audited_training_stage,
            "intermediate_checkpoint": intermediate_checkpoint,
            "training_config": str(config_path),
            "seed": args.seed,
        }
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.hard_index_output is not None:
        valid_teacher = (
            (objective_version == 4 and checkpoint_step == 2500)
            or (objective_version == 11 and checkpoint_step == 5750)
        )
        if not valid_teacher:
            raise ValueError(
                "PGC hard-role mining requires the completed clean V9.3 "
                "objective-v4 step-2500 checkpoint or the completed V9.10 "
                "objective-v11 step-5750 checkpoint."
            )
        mined = mine_hard_native_rows(
            records,
            objective_version=objective_version,
        )
        audited_native = mined["audited_native_raw_indices"]
        hard_native = mined["hard_native_raw_indices"]
        if not audited_native:
            raise ValueError("Hard-role mining found no audited native rows.")
        if not hard_native:
            raise ValueError(
                "Hard-role mining found no native failures; "
                "increase --num-samples or inspect the audit inputs."
            )
        hard_payload = {
            "format": mined["format"],
            "teacher_checkpoint": str(checkpoint),
            "teacher_objective_version": objective_version,
            "teacher_step": checkpoint_step,
            "seed": args.seed,
            "audited_native_raw_indices": audited_native,
            "hard_native_raw_indices": hard_native,
            "audited_native_count": len(audited_native),
            "hard_native_count": len(hard_native),
            "hard_native_fraction": float(len(hard_native) / len(audited_native)),
            "criterion": mined["criterion"],
            "native_datasets": sorted(
                str(Path(str(index["dataset"])).expanduser().resolve())
                for index in sidecars
                if str(index["dataset_kind"]) == "native"
            ),
            "native_frame_count": int(dataset.pgc_native_frame_count),
        }
        hard_output = args.hard_index_output.expanduser().resolve()
        hard_output.parent.mkdir(parents=True, exist_ok=True)
        hard_output.write_text(
            json.dumps(hard_payload, indent=2) + "\n", encoding="utf-8"
        )
        report["hard_role_index"] = str(hard_output)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"] and args.hard_index_output is None:
        raise SystemExit(2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hard-index-output",
        type=Path,
        help=(
            "Write audited native V9.3 role-swap failures for V9.6 or V9.10 "
            "clause-tuple failures for the V9.11 four-way curriculum."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-intermediate",
        action="store_true",
        help=(
            "Audit V9.3 steps 1750/2000/2250, V9.4/V9.5 steps "
            "2750/3000/3250, V9.6 step 2750, V9.8 step 3500, V9.9 "
            "steps 4000/4250/4500, V9.10 steps 5000/5250/5500, or V9.11 "
            "step 6000, or V9.12 steps 6500/6750/7000 without "
            "treating them as final action inputs."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    args = parser.parse_args()
    if args.num_samples <= 0 or args.num_inference_steps <= 0:
        parser.error("sample count and inference steps must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
