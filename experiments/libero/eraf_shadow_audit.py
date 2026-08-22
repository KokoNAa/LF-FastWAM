"""Passive closed-loop ERAF audit for LIBERO.

The observer in this module is deliberately privileged and evaluation-only.
It reads MuJoCo element segmentation and BDDL predicates to score ERAF while
the deployed action remains the immutable Base action.  None of the labels
constructed here are passed back into the policy.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.pgc_libero import (
    PGC_ENTITY_RELATION_FORMAT,
    PGC_ENTITY_RELATION_PREDICATES,
    libero_problem_entity_catalog,
    parse_libero_goal_clauses,
)


CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")


def _diagnostic_array(diagnostics: Mapping[str, Any], name: str) -> np.ndarray:
    """Return one-example ERAF diagnostics without assuming torch at import time."""
    value = diagnostics[name]
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    array = np.asarray(value)
    return array[0] if array.ndim > 0 and array.shape[0] == 1 else array


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -60.0, 60.0)))


def _mask_center_pixels(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros(2, dtype=np.float32), False
    return np.asarray([xs.mean(), ys.mean()], dtype=np.float32), True


def _predicted_center_pixels(
    normalized_xy: np.ndarray, *, height: int, width: int
) -> np.ndarray:
    value = np.asarray(normalized_xy, dtype=np.float32)
    return np.asarray(
        [
            (float(value[0]) + 1.0) * max(0, width - 1) / 2.0,
            (float(value[1]) + 1.0) * max(0, height - 1) / 2.0,
        ],
        dtype=np.float32,
    )


def _clause_statuses(
    *,
    clause_valid: np.ndarray,
    predicate_truth: np.ndarray,
    subject_grasped: np.ndarray,
    subject_ever_grasped: np.ndarray,
) -> tuple[list[str], np.ndarray, str]:
    """Build non-sticky per-clause phases for compound-task diagnosis.

    The legacy `postgrasp` label remains in the output for comparison, but it
    becomes sticky after the first grasp.  These labels distinguish holding,
    an unsuccessful release, and the search for a later conjunction member.
    """
    valid_indices = list(np.flatnonzero(clause_valid))
    any_prior_progress = bool(
        predicate_truth[clause_valid].any() or subject_ever_grasped[clause_valid].any()
    )
    statuses: list[str] = []
    phase_targets = np.zeros_like(predicate_truth, dtype=np.int64)
    for index in valid_indices:
        if bool(predicate_truth[index]):
            status = "completed"
            phase_targets[index] = 2
        elif bool(subject_grasped[index]):
            status = "holding"
            phase_targets[index] = 1
        elif bool(subject_ever_grasped[index]):
            status = "released_unfinished"
            phase_targets[index] = 1
        elif any_prior_progress:
            status = "next_clause_search"
            phase_targets[index] = 0
        else:
            status = "initial_search"
            phase_targets[index] = 0
        statuses.append(status)
    status_set = set(statuses)
    if statuses and status_set == {"completed"}:
        online_stage = "complete"
    elif "holding" in status_set:
        online_stage = "holding"
    elif "released_unfinished" in status_set:
        online_stage = "released_unfinished"
    elif "next_clause_search" in status_set:
        online_stage = "next_clause_search"
    else:
        online_stage = "initial_search"
    return statuses, phase_targets, online_stage


def _extended_clause_diagnostics(
    *,
    diagnostics: Mapping[str, Any],
    sample: Mapping[str, Any],
    clauses: Sequence[Mapping[str, Any]],
    clause_statuses: Sequence[str],
    phase_targets: np.ndarray,
    predicate_truth: np.ndarray,
    subject_grasped: np.ndarray,
    subject_ever_grasped: np.ndarray,
    subject_positions: np.ndarray,
    reference_positions: np.ndarray,
    subject_position_valid: np.ndarray,
    reference_position_valid: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> dict[str, Any]:
    """Decompose online failures with privileged labels, never policy inputs."""
    required = {
        "active_logits",
        "predicate_logits",
        "subject_attention",
        "reference_attention",
        "subject_position",
        "reference_position",
        "goal_anchor",
        "predicate_truth_logits",
        "phase_logits",
        "camera_ids",
        "subject_view_visibility_logits",
        "reference_view_visibility_logits",
        "subject_view_centers",
        "reference_view_centers",
        "subject_view_attention_mass",
        "reference_view_attention_mass",
    }
    missing = sorted(required.difference(diagnostics))
    if missing:
        return {"available": False, "missing_diagnostics": missing, "clauses": []}

    from scripts.eval_pgc_v9_grounding_gate import _patch_targets

    clause_valid = np.asarray(sample["pgc_eraf_clause_valid"], dtype=bool)
    predicate_ids = np.asarray(sample["pgc_eraf_predicate_ids"], dtype=np.int64)
    subject_ids = np.asarray(sample["pgc_eraf_subject_entity_ids"], dtype=np.int64)
    reference_ids = np.asarray(sample["pgc_eraf_reference_entity_ids"], dtype=np.int64)
    subject_masks = np.asarray(sample["pgc_eraf_subject_masks"], dtype=bool)
    reference_masks = np.asarray(sample["pgc_eraf_reference_masks"], dtype=bool)
    subject_valid = clause_valid & np.asarray(
        sample["pgc_eraf_subject_mask_valid"], dtype=bool
    )
    reference_valid = clause_valid & np.asarray(
        sample["pgc_eraf_reference_mask_valid"], dtype=bool
    )
    subject_attention = _diagnostic_array(diagnostics, "subject_attention")
    reference_attention = _diagnostic_array(diagnostics, "reference_attention")
    token_count = int(subject_attention.shape[-1])
    subject_targets = _patch_targets(subject_masks, token_count)
    reference_targets = _patch_targets(reference_masks, token_count)
    camera_ids = _diagnostic_array(diagnostics, "camera_ids").astype(np.int64)
    active_prediction = _diagnostic_array(diagnostics, "active_logits") > 0
    predicate_prediction = _diagnostic_array(diagnostics, "predicate_logits").argmax(
        axis=-1
    )
    truth_probability = _sigmoid(
        _diagnostic_array(diagnostics, "predicate_truth_logits")
    )
    truth_prediction = truth_probability >= 0.5
    phase_prediction = _diagnostic_array(diagnostics, "phase_logits").argmax(axis=-1)
    predicted_subject_positions = _diagnostic_array(diagnostics, "subject_position")
    predicted_reference_positions = _diagnostic_array(diagnostics, "reference_position")
    predicted_goal_anchor = _diagnostic_array(diagnostics, "goal_anchor")
    v99_view_names = {
        f"{role}_{name}"
        for role in ("subject", "reference")
        for name in (
            "base_view_attention_mass",
            "view_gate_residual_logits",
        )
    }
    v99_scheduler_names = {
        "clause_execution_probability",
        "clause_routing_residual",
        "clause_routing_multiplier",
    }
    v99_enabled = (
        "view_scheduler_enabled" in diagnostics
        and bool(
            _diagnostic_array(
                diagnostics, "view_scheduler_enabled"
            ).reshape(-1)[0]
        )
    )
    v99_view_available = v99_enabled and not v99_view_names.difference(
        diagnostics
    )
    v99_scheduler_available = v99_enabled and not v99_scheduler_names.difference(
        diagnostics
    )
    if v99_scheduler_available:
        execution_probability = _diagnostic_array(
            diagnostics, "clause_execution_probability"
        )
        routing_residual = _diagnostic_array(
            diagnostics, "clause_routing_residual"
        )
        routing_multiplier = _diagnostic_array(
            diagnostics, "clause_routing_multiplier"
        )
        execution_selected_index = int(
            np.where(clause_valid, execution_probability, -np.inf).argmax()
        )
        unfinished_indices = np.flatnonzero(clause_valid & ~predicate_truth)
        execution_target_index = (
            int(unfinished_indices[0]) if len(unfinished_indices) else None
        )
        execution_selection_correct = (
            execution_selected_index == execution_target_index
            if execution_target_index is not None
            else None
        )
    else:
        execution_probability = None
        routing_residual = None
        routing_multiplier = None
        execution_selected_index = None
        execution_target_index = None
        execution_selection_correct = None
    phase_safe_memory_available = bool(
        np.asarray(diagnostics.get("phase_safe_memory_enabled", False))
        .reshape(-1)[0]
    ) and all(
        name in diagnostics
        for name in (
            "phase_safe_memory_previous_state_ids",
            "phase_safe_memory_previous_state_valid",
            "phase_safe_memory_next_state_ids",
            "phase_safe_memory_next_state_valid",
            "phase_safe_memory_completed_sticky",
            "phase_safe_memory_released_unsatisfied_retry",
        )
    )
    if phase_safe_memory_available:
        memory_previous_ids = _diagnostic_array(
            diagnostics, "phase_safe_memory_previous_state_ids"
        ).astype(np.int64)
        memory_previous_valid = _diagnostic_array(
            diagnostics, "phase_safe_memory_previous_state_valid"
        ).astype(bool)
        memory_next_ids = _diagnostic_array(
            diagnostics, "phase_safe_memory_next_state_ids"
        ).astype(np.int64)
        memory_next_valid = _diagnostic_array(
            diagnostics, "phase_safe_memory_next_state_valid"
        ).astype(bool)
        memory_completed_sticky = _diagnostic_array(
            diagnostics, "phase_safe_memory_completed_sticky"
        ).astype(bool)
        memory_released_retry = _diagnostic_array(
            diagnostics, "phase_safe_memory_released_unsatisfied_retry"
        ).astype(bool)
    else:
        memory_previous_ids = memory_previous_valid = None
        memory_next_ids = memory_next_valid = None
        memory_completed_sticky = memory_released_retry = None
    scale = (workspace_max - workspace_min) / 2.0
    half_width = int(subject_masks.shape[-1] // len(CAMERA_NAMES))
    mask_height = int(subject_masks.shape[-2])
    status_by_index = {
        int(index): str(status)
        for index, status in zip(np.flatnonzero(clause_valid), clause_statuses)
    }
    prompt = str(sample.get("prompt", "unknown"))
    prompt_prefix = DEFAULT_PROMPT.split("{task}", 1)[0]
    task = prompt[len(prompt_prefix) :] if prompt.startswith(prompt_prefix) else prompt

    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(clause_valid):
        status = status_by_index[int(index)]
        expected_memory_state = {
            "initial_search": 0,
            "holding": 1,
            "released_unfinished": 2,
            "next_clause_search": 0,
            "completed": 3,
        }[status]
        subject_target = subject_targets[index]
        reference_target = reference_targets[index]
        subject_hit = (
            bool(subject_target[subject_attention[index].argmax()] > 0)
            if subject_valid[index]
            else None
        )
        reference_hit = (
            bool(reference_target[reference_attention[index].argmax()] > 0)
            if reference_valid[index]
            else None
        )
        role_applicable = bool(
            subject_valid[index]
            and reference_valid[index]
            and subject_ids[index] != reference_ids[index]
        )
        full_role_correct = None
        exclusive_role_valid = False
        exclusive_role_correct = None
        overlap_iou = None
        if role_applicable:
            subject_own = float((subject_attention[index] * subject_target).sum())
            subject_wrong = float((subject_attention[index] * reference_target).sum())
            reference_own = float((reference_attention[index] * reference_target).sum())
            reference_wrong = float((reference_attention[index] * subject_target).sum())
            full_role_correct = bool(
                subject_own > subject_wrong and reference_own > reference_wrong
            )
            shared = np.minimum(subject_target, reference_target)
            union = np.maximum(subject_target, reference_target)
            subject_exclusive = np.clip(subject_target - shared, 0.0, None)
            reference_exclusive = np.clip(reference_target - shared, 0.0, None)
            exclusive_role_valid = bool(
                float(subject_exclusive.sum()) > 1.0e-8
                and float(reference_exclusive.sum()) > 1.0e-8
            )
            if exclusive_role_valid:
                exclusive_role_correct = bool(
                    float((subject_attention[index] * subject_exclusive).sum())
                    > float((subject_attention[index] * reference_exclusive).sum())
                    and float((reference_attention[index] * reference_exclusive).sum())
                    > float((reference_attention[index] * subject_exclusive).sum())
                )
            overlap_iou = (
                float(shared.sum() / union.sum()) if float(union.sum()) > 0 else 0.0
            )

        if subject_ids[index] == reference_ids[index]:
            oracle_partition = "unary_role_not_applicable"
        elif not subject_valid[index] or not reference_valid[index]:
            oracle_partition = "gt_not_jointly_visible"
        elif not exclusive_role_valid:
            oracle_partition = "mask_overlap_ambiguous"
        elif not bool(exclusive_role_correct):
            oracle_partition = "visible_binding_error"
        else:
            oracle_partition = "role_pass"

        views: dict[str, Any] = {}
        for camera_index, camera_name in enumerate(CAMERA_NAMES):
            token_indices = np.flatnonzero(camera_ids == camera_index)
            if not len(token_indices):
                raise ValueError(
                    f"ERAF diagnostics expose no patches for camera {camera_name}."
                )
            left = camera_index * half_width
            right = left + half_width
            role_views: dict[str, Any] = {}
            for role, mask, attention in (
                ("subject", subject_masks[index, :, left:right], subject_attention),
                (
                    "reference",
                    reference_masks[index, :, left:right],
                    reference_attention,
                ),
            ):
                gt_center, gt_visible = _mask_center_pixels(mask)
                local_index = int(
                    token_indices[np.argmax(attention[index, token_indices])]
                )
                target = subject_target if role == "subject" else reference_target
                predicted_visibility = bool(
                    _diagnostic_array(diagnostics, f"{role}_view_visibility_logits")[
                        index, camera_index
                    ]
                    > 0
                )
                visibility_probability = float(
                    _sigmoid(
                        _diagnostic_array(
                            diagnostics, f"{role}_view_visibility_logits"
                        )[index, camera_index]
                    )
                )
                predicted_center = _predicted_center_pixels(
                    _diagnostic_array(diagnostics, f"{role}_view_centers")[
                        index, camera_index
                    ],
                    height=mask_height,
                    width=half_width,
                )
                final_attention_mass = float(
                    _diagnostic_array(
                        diagnostics, f"{role}_view_attention_mass"
                    )[index, camera_index]
                )
                role_views[role] = {
                    "gt_visible": bool(gt_visible),
                    "predicted_visible": predicted_visibility,
                    "visibility_probability": visibility_probability,
                    "visibility_correct": bool(predicted_visibility == gt_visible),
                    "top1_hit": (bool(target[local_index] > 0) if gt_visible else None),
                    "attention_mass": final_attention_mass,
                    "center_error_px": (
                        float(np.linalg.norm(predicted_center - gt_center))
                        if gt_visible
                        else None
                    ),
                }
                if v99_view_available:
                    base_attention_mass = float(
                        _diagnostic_array(
                            diagnostics, f"{role}_base_view_attention_mass"
                        )[index, camera_index]
                    )
                    role_views[role].update(
                        {
                            "base_attention_mass": base_attention_mass,
                            "attention_mass_delta": (
                                final_attention_mass - base_attention_mass
                            ),
                            "view_gate_residual_logit": float(
                                _diagnostic_array(
                                    diagnostics,
                                    f"{role}_view_gate_residual_logits",
                                )[index, camera_index]
                            ),
                        }
                    )
            views[camera_name] = role_views

        goal_anchor_valid = bool(sample["pgc_eraf_goal_anchor_valid"][index])
        goal_anchor_error_m = (
            float(
                np.linalg.norm(
                    (
                        predicted_goal_anchor[index]
                        - np.asarray(sample["pgc_eraf_goal_anchors"])[index]
                    )
                    * scale
                )
            )
            if goal_anchor_valid
            else None
        )
        subject_position_error_m = (
            float(
                np.linalg.norm(
                    (predicted_subject_positions[index] - subject_positions[index])
                    * scale
                )
            )
            if subject_position_valid[index]
            else None
        )
        reference_position_error_m = (
            float(
                np.linalg.norm(
                    (predicted_reference_positions[index] - reference_positions[index])
                    * scale
                )
            )
            if reference_position_valid[index]
            else None
        )
        rows.append(
            {
                "clause_index": int(index),
                "task": task,
                "predicate": str(clauses[index]["predicate"]),
                "status": status,
                "active_correct": bool(active_prediction[index]),
                "predicate_correct": bool(
                    predicate_prediction[index] == predicate_ids[index]
                ),
                "predicate_truth": bool(predicate_truth[index]),
                "predicate_truth_probability": float(truth_probability[index]),
                "predicate_truth_correct": bool(
                    truth_prediction[index] == predicate_truth[index]
                ),
                "phase_target": int(phase_targets[index]),
                "phase_target_kind": "online_state_proxy",
                "phase_prediction": int(phase_prediction[index]),
                "phase_correct": bool(phase_prediction[index] == phase_targets[index]),
                "execution_probability": (
                    float(execution_probability[index])
                    if execution_probability is not None
                    else None
                ),
                "execution_selected": (
                    bool(index == execution_selected_index)
                    if execution_selected_index is not None
                    else None
                ),
                "execution_target": (
                    bool(index == execution_target_index)
                    if execution_target_index is not None
                    else None
                ),
                "execution_selection_correct": execution_selection_correct,
                "routing_residual": (
                    float(routing_residual[index])
                    if routing_residual is not None
                    else None
                ),
                "routing_multiplier": (
                    float(routing_multiplier[index])
                    if routing_multiplier is not None
                    else None
                ),
                "phase_safe_memory_available": phase_safe_memory_available,
                "phase_safe_memory_expected_state": expected_memory_state,
                "phase_safe_memory_previous_state": (
                    int(memory_previous_ids[index])
                    if phase_safe_memory_available
                    and bool(memory_previous_valid[index])
                    else None
                ),
                "phase_safe_memory_next_state": (
                    int(memory_next_ids[index])
                    if phase_safe_memory_available and bool(memory_next_valid[index])
                    else None
                ),
                "phase_safe_memory_state_valid": (
                    bool(memory_next_valid[index])
                    if phase_safe_memory_available
                    else None
                ),
                "phase_safe_memory_state_correct": (
                    bool(memory_next_ids[index] == expected_memory_state)
                    if phase_safe_memory_available and bool(memory_next_valid[index])
                    else None
                ),
                "phase_safe_memory_completed_sticky_violation": (
                    bool(
                        memory_previous_valid[index]
                        and memory_previous_ids[index] == 3
                        and memory_next_ids[index] != 3
                    )
                    if phase_safe_memory_available
                    else None
                ),
                "phase_safe_memory_completed_sticky_applied": (
                    bool(memory_completed_sticky[index])
                    if phase_safe_memory_available
                    else None
                ),
                "phase_safe_memory_retry_transition": (
                    bool(memory_released_retry[index])
                    if phase_safe_memory_available
                    else None
                ),
                "subject_grasped": bool(subject_grasped[index]),
                "subject_ever_grasped": bool(subject_ever_grasped[index]),
                "subject_visible": bool(subject_valid[index]),
                "reference_visible": bool(reference_valid[index]),
                "subject_top1_hit": subject_hit,
                "reference_top1_hit": reference_hit,
                "full_role_correct": full_role_correct,
                "exclusive_role_valid": exclusive_role_valid,
                "exclusive_role_correct": exclusive_role_correct,
                "mask_overlap_iou": overlap_iou,
                "oracle_partition": oracle_partition,
                "subject_position_error_m": subject_position_error_m,
                "reference_position_error_m": reference_position_error_m,
                "goal_anchor_error_m": goal_anchor_error_m,
                "views": views,
            }
        )
    return {
        "available": True,
        "missing_diagnostics": [],
        "v99_view_fusion_available": v99_view_available,
        "v99_clause_scheduler_available": v99_scheduler_available,
        "phase_safe_memory_available": phase_safe_memory_available,
        "clauses": rows,
    }


@dataclass(frozen=True)
class ERAFShadowContract:
    """Small deployment-audit subset of a hash-audited ERAF sidecar index."""

    index_path: Path
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    mask_height: int
    mask_width: int
    max_clauses: int
    predicate_vocabulary: tuple[str, ...]

    @classmethod
    def load(cls, sidecar: str | Path) -> "ERAFShadowContract":
        path = Path(sidecar).expanduser().resolve()
        index_path = path if path.is_file() else path / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"ERAF shadow sidecar index not found: {index_path}"
            )
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("format") != PGC_ENTITY_RELATION_FORMAT:
            raise ValueError(
                f"ERAF shadow audit requires {PGC_ENTITY_RELATION_FORMAT}, "
                f"got {payload.get('format')!r}."
            )
        if payload.get("privileged_supervision") != "training_only":
            raise ValueError(
                "ERAF privileged labels must remain training/evaluation only."
            )
        if payload.get("deployment_inputs") != "rgb_language_proprio":
            raise ValueError(
                "ERAF checkpoint does not declare the deployment input contract."
            )
        if tuple(payload.get("camera_names", ())) != CAMERA_NAMES:
            raise ValueError("ERAF shadow camera order does not match FastWAM.")
        vocabulary = tuple(
            str(value) for value in payload.get("predicate_vocabulary", ())
        )
        if vocabulary != PGC_ENTITY_RELATION_PREDICATES:
            raise ValueError("ERAF shadow predicate vocabulary is incompatible.")
        mask_size = tuple(int(value) for value in payload.get("mask_size", ()))
        if len(mask_size) != 2 or min(mask_size) <= 0 or mask_size[1] % 2:
            raise ValueError("ERAF shadow mask size must be positive with even width.")
        workspace_min = np.asarray(payload.get("workspace_min"), dtype=np.float32)
        workspace_max = np.asarray(payload.get("workspace_max"), dtype=np.float32)
        if (
            workspace_min.shape != (3,)
            or workspace_max.shape != (3,)
            or not np.isfinite(workspace_min).all()
            or not np.isfinite(workspace_max).all()
            or np.any(workspace_max <= workspace_min)
        ):
            raise ValueError("ERAF shadow workspace bounds are invalid.")
        max_clauses = int(payload.get("max_clauses", 0))
        if max_clauses != 4:
            raise ValueError("ERAF shadow audit requires exactly four clause slots.")
        return cls(
            index_path=index_path,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            mask_height=mask_size[0],
            mask_width=mask_size[1],
            max_clauses=max_clauses,
            predicate_vocabulary=vocabulary,
        )


def verify_shadow_action_integrity(
    selected_action: Any,
    base_action: Any,
    *,
    gate_mode: str,
) -> dict[str, Any]:
    """Prove that the action returned by a shadow inference is exactly Base."""
    import torch

    selected = torch.as_tensor(selected_action).detach().cpu()
    base = torch.as_tensor(base_action).detach().cpu()
    if selected.shape != base.shape:
        raise ValueError(
            "ERAF shadow selected/Base action shapes differ: "
            f"{tuple(selected.shape)} != {tuple(base.shape)}."
        )
    difference = selected.float() - base.float()
    exact = bool(torch.equal(selected, base))
    result = {
        "gate_mode": str(gate_mode),
        "exact": exact,
        "max_abs_error": (
            float(difference.abs().max().item()) if difference.numel() else 0.0
        ),
        "rms_error": (
            float(difference.square().mean().sqrt().item())
            if difference.numel()
            else 0.0
        ),
    }
    if gate_mode != "base":
        raise RuntimeError("ERAF shadow audit requires policy_guard.gate_mode=base.")
    if not exact:
        raise RuntimeError(
            "ERAF shadow observer changed the deployed Base action: "
            f"max_abs_error={result['max_abs_error']:.8g}."
        )
    return result


def _inner_env(env: Any) -> Any:
    inner = getattr(env, "env", None)
    if inner is None:
        raise TypeError("ERAF shadow audit requires a LIBERO wrapper with `.env`.")
    return inner


def _entity_id(name: str) -> int:
    digest = hashlib.sha256(str(name).strip().casefold().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _instance_geom_ids(env: Any, entity_names: Sequence[str]) -> dict[str, np.ndarray]:
    mapping = getattr(_inner_env(env).model, "instances_to_ids", None)
    if not isinstance(mapping, Mapping):
        raise RuntimeError("robosuite model has no instances_to_ids mapping.")
    result: dict[str, np.ndarray] = {}
    for name in entity_names:
        instance = mapping.get(name)
        if not isinstance(instance, Mapping):
            stem = name.rsplit("_", 1)[0]
            candidates = [
                value
                for key, value in mapping.items()
                if key == stem or key.startswith(stem + "_")
            ]
            if len(candidates) == 1:
                instance = candidates[0]
        ids = [] if not isinstance(instance, Mapping) else instance.get("geom", [])
        result[name] = np.asarray(ids, dtype=np.int32).reshape(-1)
    return result


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return (
        np.asarray(
            image.resize((width, height), resample=Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        > 0
    )


def _entity_masks(
    obs: Mapping[str, Any],
    geom_ids: Mapping[str, np.ndarray],
    *,
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    half_width = width // 2
    result: dict[str, list[np.ndarray]] = {name: [] for name in geom_ids}
    for camera in CAMERA_NAMES:
        key = f"{camera}_segmentation_element"
        if key not in obs:
            raise KeyError(
                f"ERAF shadow observation lacks {key!r}; construct LIBERO with "
                "camera_segmentations='element'."
            )
        segmentation = np.asarray(obs[key])
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[..., 0]
        if segmentation.ndim != 2:
            raise ValueError(f"ERAF element segmentation {key!r} must be 2-D.")
        segmentation = np.ascontiguousarray(segmentation[::-1, ::-1])
        for name, ids in geom_ids.items():
            result[name].append(
                _resize_mask(np.isin(segmentation, ids), height, half_width)
            )
    return {
        name: np.concatenate(camera_masks, axis=-1)
        for name, camera_masks in result.items()
    }


def _body_position(env: Any, name: str) -> tuple[np.ndarray, bool]:
    inner = _inner_env(env)
    body_ids = getattr(inner, "obj_body_id", {})
    body_id = body_ids.get(name) if isinstance(body_ids, Mapping) else None
    if body_id is None:
        try:
            body_id = inner.sim.model.body_name2id(name)
        except Exception:
            return np.zeros(3, dtype=np.float32), False
    return np.asarray(inner.sim.data.body_xpos[int(body_id)], dtype=np.float32), True


def _site_position(env: Any, name: str) -> tuple[np.ndarray, bool]:
    inner = _inner_env(env)
    model = inner.sim.model
    candidates = [str(name)]
    object_sites = getattr(inner, "object_sites_dict", {})
    site = object_sites.get(str(name)) if isinstance(object_sites, Mapping) else None
    if site is not None:
        if isinstance(site, str):
            candidates.append(site)
        for attribute in ("name", "site_name"):
            value = getattr(site, attribute, None)
            if value:
                candidates.append(str(value))
    site_names = [str(value) for value in getattr(model, "site_names", ())]
    suffix_matches = [
        value
        for value in site_names
        if value == str(name) or value.endswith("_" + str(name))
    ]
    if len(suffix_matches) == 1:
        candidates.append(suffix_matches[0])
    for candidate in dict.fromkeys(candidates):
        try:
            site_id = int(model.site_name2id(candidate))
            if site_id >= 0:
                return (
                    np.asarray(inner.sim.data.site_xpos[site_id], dtype=np.float32),
                    True,
                )
        except Exception:
            continue
    return np.zeros(3, dtype=np.float32), False


def _workspace_offset(env: Any) -> tuple[np.ndarray, bool]:
    value = np.asarray(
        getattr(_inner_env(env), "workspace_offset", ()), dtype=np.float32
    ).reshape(-1)
    if value.size < 3 or not np.isfinite(value[:3]).all():
        return np.zeros(3, dtype=np.float32), False
    return value[:3].copy(), True


def _region_anchor(
    env: Any,
    clause: Mapping[str, Any],
    problem: Mapping[str, Any],
) -> tuple[np.ndarray, bool]:
    region_name = clause.get("reference_region") or clause.get("subject_region")
    if region_name:
        site_position, site_valid = _site_position(env, str(region_name))
        if site_valid:
            return site_position, True
        region = (problem.get("regions") or {}).get(str(region_name), {})
        ranges = region.get("ranges") if isinstance(region, Mapping) else None
        if isinstance(ranges, Sequence) and ranges:
            values = np.asarray(ranges[0], dtype=np.float32).reshape(-1)
            offset, offset_valid = _workspace_offset(env)
            if values.size >= 4 and offset_valid and np.isfinite(values[:4]).all():
                return (
                    offset
                    + np.asarray(
                        [
                            (values[0] + values[2]) / 2,
                            (values[1] + values[3]) / 2,
                            0.0,
                        ],
                        dtype=np.float32,
                    ),
                    True,
                )
    return _body_position(env, str(clause["reference"]))


def _normalize_position(
    position: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return np.clip(2.0 * (position - lower) / (upper - lower) - 1.0, -1.0, 1.0)


def _clause_truth(env: Any, raw_clause: Sequence[Any]) -> bool:
    inner = _inner_env(env)
    original = inner.parsed_problem.get("goal_state")
    try:
        inner.parsed_problem["goal_state"] = [list(raw_clause)]
        for method_name in ("_check_success", "check_success"):
            method = getattr(inner, method_name, None)
            if callable(method):
                return bool(method())
    finally:
        inner.parsed_problem["goal_state"] = original
    raise RuntimeError("LIBERO environment exposes no predicate success check.")


def _is_grasped(env: Any, entity: str) -> bool:
    inner = _inner_env(env)
    objects = getattr(inner, "objects_dict", {})
    if entity not in objects:
        return False
    gripper = inner.robots[0].gripper
    grippers = list(gripper.values()) if isinstance(gripper, dict) else [gripper]
    return any(
        bool(inner._check_grasp(gripper=item, object_geoms=objects[entity]))
        for item in grippers
    )


class ERAFShadowAuditor:
    """Construct same-state privileged labels and score one online replan."""

    def __init__(
        self,
        *,
        env: Any,
        policy_instruction: str,
        instruction_condition: str,
        contract: ERAFShadowContract,
        counterfactual_metadata: Mapping[str, Any] | None,
        all_entity_role_gate: bool = False,
    ) -> None:
        if instruction_condition not in {"correct", "counterfactual"}:
            raise ValueError(
                "ERAF shadow audit supports only correct or counterfactual instructions."
            )
        self.env = env
        self.policy_instruction = str(policy_instruction)
        self.instruction_condition = str(instruction_condition)
        self.contract = contract
        self.all_entity_role_gate = bool(all_entity_role_gate)
        inner = _inner_env(env)
        if instruction_condition == "counterfactual":
            if counterfactual_metadata is None:
                raise ValueError("Counterfactual ERAF shadow audit requires metadata.")
            from libero.libero.envs import bddl_utils as BDDLUtils

            problem = BDDLUtils.robosuite_parse_problem(
                str(counterfactual_metadata["counterfactual_bddl_file"])
            )
            goal_state = counterfactual_metadata["counterfactual_goal_state"]
        else:
            problem = inner.parsed_problem
            goal_state = problem["goal_state"]
        self.problem = problem
        self.clauses = parse_libero_goal_clauses(
            goal_state,
            regions=problem.get("regions", {}),
            max_clauses=contract.max_clauses,
            instruction=self.policy_instruction,
            entity_catalog=libero_problem_entity_catalog(problem),
        )
        entities = sorted(
            {
                str(clause[role])
                for clause in self.clauses
                for role in ("subject", "reference")
            }
        )
        self.geom_ids = _instance_geom_ids(env, entities)
        self._episode_idx: int | None = None
        self._ever_grasped = np.zeros(contract.max_clauses, dtype=np.bool_)

    def observe(
        self,
        *,
        obs: Mapping[str, Any],
        diagnostics: Mapping[str, Any],
        episode_idx: int,
        replan_idx: int,
        policy_step: int,
    ) -> dict[str, Any]:
        from scripts.eval_pgc_v9_grounding_gate import _sample_record

        masks = _entity_masks(
            obs,
            self.geom_ids,
            height=self.contract.mask_height,
            width=self.contract.mask_width,
        )
        count = self.contract.max_clauses
        clause_valid = np.zeros(count, dtype=np.bool_)
        predicate_ids = np.zeros(count, dtype=np.int64)
        subject_ids = np.full(count, -1, dtype=np.int64)
        reference_ids = np.full(count, -1, dtype=np.int64)
        subject_masks = np.zeros(
            (count, self.contract.mask_height, self.contract.mask_width),
            dtype=np.bool_,
        )
        reference_masks = np.zeros_like(subject_masks)
        subject_mask_valid = np.zeros(count, dtype=np.bool_)
        reference_mask_valid = np.zeros(count, dtype=np.bool_)
        goal_anchors = np.zeros((count, 3), dtype=np.float32)
        goal_anchor_valid = np.zeros(count, dtype=np.bool_)
        predicate_truth = np.zeros(count, dtype=np.bool_)
        subject_grasped = np.zeros(count, dtype=np.bool_)
        subject_positions = np.zeros((count, 3), dtype=np.float32)
        reference_positions = np.zeros((count, 3), dtype=np.float32)
        subject_position_valid = np.zeros(count, dtype=np.bool_)
        reference_position_valid = np.zeros(count, dtype=np.bool_)
        if self._episode_idx != int(episode_idx):
            self._episode_idx = int(episode_idx)
            self._ever_grasped.fill(False)
        for index, clause in enumerate(self.clauses):
            clause_valid[index] = True
            predicate_ids[index] = int(clause["predicate_id"])
            subject = str(clause["subject"])
            reference = str(clause["reference"])
            subject_ids[index] = _entity_id(subject)
            reference_ids[index] = _entity_id(reference)
            subject_masks[index] = masks[subject]
            reference_masks[index] = masks[reference]
            subject_mask_valid[index] = bool(masks[subject].any())
            reference_mask_valid[index] = bool(masks[reference].any())
            anchor, valid = _region_anchor(self.env, clause, self.problem)
            goal_anchor_valid[index] = bool(valid)
            if valid:
                goal_anchors[index] = _normalize_position(
                    anchor,
                    self.contract.workspace_min,
                    self.contract.workspace_max,
                )
            predicate_truth[index] = _clause_truth(self.env, clause["raw"])
            subject_grasped[index] = _is_grasped(self.env, subject)
            position, valid = _body_position(self.env, subject)
            subject_position_valid[index] = bool(valid)
            if valid:
                subject_positions[index] = _normalize_position(
                    position,
                    self.contract.workspace_min,
                    self.contract.workspace_max,
                )
            position, valid = _body_position(self.env, reference)
            reference_position_valid[index] = bool(valid)
            if valid:
                reference_positions[index] = _normalize_position(
                    position,
                    self.contract.workspace_min,
                    self.contract.workspace_max,
                )
        self._ever_grasped |= subject_grasped
        clause_statuses, phase_targets, stage_v2 = _clause_statuses(
            clause_valid=clause_valid,
            predicate_truth=predicate_truth,
            subject_grasped=subject_grasped,
            subject_ever_grasped=self._ever_grasped,
        )
        if bool(predicate_truth[clause_valid].all()):
            stage = "complete"
        elif bool(self._ever_grasped[clause_valid].any()):
            stage = "postgrasp"
        else:
            stage = "pregrasp"
        sample = {
            "prompt": DEFAULT_PROMPT.format(task=self.policy_instruction),
            "pgc_eraf_clause_valid": clause_valid,
            "pgc_eraf_predicate_ids": predicate_ids,
            "pgc_eraf_subject_entity_ids": subject_ids,
            "pgc_eraf_reference_entity_ids": reference_ids,
            "pgc_eraf_subject_masks": subject_masks,
            "pgc_eraf_reference_masks": reference_masks,
            "pgc_eraf_subject_mask_valid": subject_mask_valid,
            "pgc_eraf_reference_mask_valid": reference_mask_valid,
            "pgc_eraf_subject_view_visible": np.stack(
                (
                    subject_masks[:, :, : self.contract.mask_width // 2].any(
                        axis=(1, 2)
                    ),
                    subject_masks[:, :, self.contract.mask_width // 2 :].any(
                        axis=(1, 2)
                    ),
                ),
                axis=-1,
            ),
            "pgc_eraf_reference_view_visible": np.stack(
                (
                    reference_masks[:, :, : self.contract.mask_width // 2].any(
                        axis=(1, 2)
                    ),
                    reference_masks[:, :, self.contract.mask_width // 2 :].any(
                        axis=(1, 2)
                    ),
                ),
                axis=-1,
            ),
            "pgc_eraf_goal_anchors": goal_anchors,
            "pgc_eraf_goal_anchor_valid": goal_anchor_valid,
            "pgc_eraf_predicate_truth": predicate_truth.astype(np.float32),
            "pgc_eraf_predicate_truth_valid": clause_valid.copy(),
        }
        record = _sample_record(
            diagnostics,
            sample,
            self.contract.workspace_min,
            self.contract.workspace_max,
            dataset_kind="online_closed_loop",
            dataset_label=self.instruction_condition,
            predicate_vocabulary=self.contract.predicate_vocabulary,
            all_entity_role_gate=self.all_entity_role_gate,
        )
        phase_safe_memory_enabled = bool(
            np.asarray(diagnostics.get("phase_safe_memory_enabled", False))
            .reshape(-1)[0]
        )
        memory_geometry_max_abs: float | None = None
        if phase_safe_memory_enabled:
            geometry_pairs = (
                ("subject_attention", "pre_memory_subject_attention"),
                ("reference_attention", "pre_memory_reference_attention"),
                ("subject_position", "pre_memory_subject_position"),
                ("reference_position", "pre_memory_reference_position"),
                ("goal_anchor", "pre_memory_goal_anchor"),
            )
            missing = sorted(
                name
                for pair in geometry_pairs
                for name in pair
                if name not in diagnostics
            )
            if missing:
                raise ValueError(
                    "V9.13 shadow diagnostics are missing frozen-geometry "
                    f"comparison outputs: {missing}."
                )
            memory_geometry_max_abs = max(
                float(
                    np.max(
                        np.abs(
                            _diagnostic_array(diagnostics, post).astype(np.float64)
                            - _diagnostic_array(diagnostics, pre).astype(np.float64)
                        )
                    )
                )
                for post, pre in geometry_pairs
            )
        rebinding_enabled = bool(
            np.asarray(
                diagnostics.get("closed_loop_rebinding_enabled", False)
            ).reshape(-1)[0]
        )
        pre_rebinding_record: dict[str, Any] | None = None
        if rebinding_enabled:
            pre_rebinding_names = {
                "subject_attention": "pre_rebinding_subject_attention",
                "reference_attention": "pre_rebinding_reference_attention",
                "subject_position": "pre_rebinding_subject_position",
                "reference_position": "pre_rebinding_reference_position",
                "subject_view_attention_mass": (
                    "pre_rebinding_subject_view_attention_mass"
                ),
                "reference_view_attention_mass": (
                    "pre_rebinding_reference_view_attention_mass"
                ),
                "goal_anchor": "pre_rebinding_goal_anchor",
                "predicate_truth_logits": (
                    "pre_rebinding_predicate_truth_logits"
                ),
                "phase_logits": "pre_rebinding_phase_logits",
                "clause_execution_probability": (
                    "pre_rebinding_clause_execution_probability"
                ),
                "clause_routing_residual": (
                    "pre_rebinding_clause_routing_residual"
                ),
            }
            missing = sorted(
                source
                for source in pre_rebinding_names.values()
                if source not in diagnostics
            )
            if missing:
                raise ValueError(
                    "V9.12 shadow diagnostics are missing pre-rebinding "
                    f"same-state outputs: {missing}."
                )
            pre_rebinding_diagnostics = dict(diagnostics)
            pre_rebinding_diagnostics.update(
                {
                    target: diagnostics[source]
                    for target, source in pre_rebinding_names.items()
                }
            )
            pre_rebinding_record = _sample_record(
                pre_rebinding_diagnostics,
                sample,
                self.contract.workspace_min,
                self.contract.workspace_max,
                dataset_kind="online_closed_loop_pre_rebinding",
                dataset_label=self.instruction_condition,
                predicate_vocabulary=self.contract.predicate_vocabulary,
                all_entity_role_gate=self.all_entity_role_gate,
            )
        extended = _extended_clause_diagnostics(
            diagnostics=diagnostics,
            sample=sample,
            clauses=self.clauses,
            clause_statuses=clause_statuses,
            phase_targets=phase_targets,
            predicate_truth=predicate_truth,
            subject_grasped=subject_grasped,
            subject_ever_grasped=self._ever_grasped,
            subject_positions=subject_positions,
            reference_positions=reference_positions,
            subject_position_valid=subject_position_valid,
            reference_position_valid=reference_position_valid,
            workspace_min=self.contract.workspace_min,
            workspace_max=self.contract.workspace_max,
        )
        record.update(
            {
                "episode": int(episode_idx),
                "replan_index": int(replan_idx),
                "policy_step": int(policy_step),
                "online_stage": stage,
                "online_stage_v2": stage_v2,
                "clause_statuses": clause_statuses,
                "phase_targets": phase_targets[clause_valid].tolist(),
                "instruction_condition": self.instruction_condition,
                "policy_instruction": self.policy_instruction,
                "predicate_truth": predicate_truth[clause_valid].tolist(),
                "subject_grasped": subject_grasped[clause_valid].tolist(),
                "subject_ever_grasped": self._ever_grasped[clause_valid].tolist(),
                "extended_diagnostics": extended,
                "closed_loop_rebinding_enabled": rebinding_enabled,
                "phase_safe_memory_enabled": phase_safe_memory_enabled,
                "phase_safe_memory_geometry_max_abs": memory_geometry_max_abs,
                "pre_rebinding_record": pre_rebinding_record,
            }
        )
        return record


def _optional_rate(values: Sequence[Any]) -> float | None:
    valid = [bool(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def _optional_mean(values: Sequence[Any]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def _optional_median(values: Sequence[Any]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return float(np.median(valid)) if valid else None


def _summarize_camera_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for camera_name in CAMERA_NAMES:
        camera: dict[str, Any] = {}
        for role in ("subject", "reference"):
            role_rows = [
                row["views"][camera_name][role]
                for row in rows
                if camera_name in row.get("views", {})
                and role in row["views"][camera_name]
            ]
            visible_rows = [row for row in role_rows if row.get("gt_visible")]
            hidden_rows = [row for row in role_rows if not row.get("gt_visible")]
            camera[role] = {
                "samples": len(role_rows),
                "gt_visible_samples": len(visible_rows),
                "gt_visibility_rate": _optional_rate(
                    [row.get("gt_visible") for row in role_rows]
                ),
                "visibility_accuracy": _optional_rate(
                    [row.get("visibility_correct") for row in role_rows]
                ),
                "visible_top1_in_gt_mask": _optional_rate(
                    [row.get("top1_hit") for row in visible_rows]
                ),
                "visible_center_median_error_px": _optional_median(
                    [row.get("center_error_px") for row in visible_rows]
                ),
                "attention_mass_when_visible_mean": _optional_mean(
                    [row.get("attention_mass") for row in visible_rows]
                ),
                "attention_mass_when_hidden_mean": _optional_mean(
                    [row.get("attention_mass") for row in hidden_rows]
                ),
                "base_attention_mass_when_visible_mean": _optional_mean(
                    [row.get("base_attention_mass") for row in visible_rows]
                ),
                "base_attention_mass_when_hidden_mean": _optional_mean(
                    [row.get("base_attention_mass") for row in hidden_rows]
                ),
                "attention_mass_delta_when_visible_mean": _optional_mean(
                    [row.get("attention_mass_delta") for row in visible_rows]
                ),
                "attention_mass_delta_when_hidden_mean": _optional_mean(
                    [row.get("attention_mass_delta") for row in hidden_rows]
                ),
                "view_gate_residual_abs_mean": _optional_mean(
                    [
                        abs(row["view_gate_residual_logit"])
                        for row in role_rows
                        if row.get("view_gate_residual_logit") is not None
                    ]
                ),
            }
        result[camera_name] = camera
    return result


def _summarize_clause_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    if not rows:
        return {"clauses": 0, "metrics": {}, "status_counts": {}}
    exclusive_rows = [row for row in rows if row.get("exclusive_role_valid")]
    applicable_rows = [row for row in rows if row.get("full_role_correct") is not None]
    phase_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        phase_confusion[str(row.get("phase_target"))][
            str(row.get("phase_prediction"))
        ] += 1
    oracle_counts = Counter(str(row.get("oracle_partition")) for row in rows)
    oracle_total = sum(
        value
        for key, value in oracle_counts.items()
        if key != "unary_role_not_applicable"
    )
    metrics = {
        "active_accuracy": _optional_rate([row.get("active_correct") for row in rows]),
        "predicate_accuracy": _optional_rate(
            [row.get("predicate_correct") for row in rows]
        ),
        "predicate_truth_accuracy": _optional_rate(
            [row.get("predicate_truth_correct") for row in rows]
        ),
        "phase_proxy_accuracy": _optional_rate(
            [row.get("phase_correct") for row in rows]
        ),
        "clause_scheduler_accuracy": _optional_rate(
            [row.get("execution_selection_correct") for row in rows]
        ),
        "clause_scheduler_top1_probability": _optional_mean(
            [
                row.get("execution_probability")
                for row in rows
                if row.get("execution_selected")
            ]
        ),
        "clause_routing_residual_abs_mean": _optional_mean(
            [
                abs(row["routing_residual"])
                for row in rows
                if row.get("routing_residual") is not None
            ]
        ),
        "phase_safe_memory_state_accuracy": _optional_rate(
            [row.get("phase_safe_memory_state_correct") for row in rows]
        ),
        "phase_safe_memory_completed_sticky_violation_rate": _optional_rate(
            [
                row.get("phase_safe_memory_completed_sticky_violation")
                for row in rows
            ]
        ),
        "phase_safe_memory_retry_transition_rate": _optional_rate(
            [
                row.get("phase_safe_memory_retry_transition")
                for row in rows
                if row.get("status") == "released_unfinished"
            ]
        ),
        "subject_top1_in_gt_mask": _optional_rate(
            [row.get("subject_top1_hit") for row in rows]
        ),
        "reference_top1_in_gt_mask": _optional_rate(
            [row.get("reference_top1_hit") for row in rows]
        ),
        "full_role_accuracy": _optional_rate(
            [row.get("full_role_correct") for row in applicable_rows]
        ),
        "exclusive_role_coverage": (
            float(len(exclusive_rows) / len(applicable_rows))
            if applicable_rows
            else None
        ),
        "exclusive_role_accuracy": _optional_rate(
            [row.get("exclusive_role_correct") for row in exclusive_rows]
        ),
        "subject_position_median_error_cm": (
            100.0
            * _optional_median([row.get("subject_position_error_m") for row in rows])
            if any(row.get("subject_position_error_m") is not None for row in rows)
            else None
        ),
        "reference_position_median_error_cm": (
            100.0
            * _optional_median([row.get("reference_position_error_m") for row in rows])
            if any(row.get("reference_position_error_m") is not None for row in rows)
            else None
        ),
        "goal_anchor_median_error_cm": (
            100.0 * _optional_median([row.get("goal_anchor_error_m") for row in rows])
            if any(row.get("goal_anchor_error_m") is not None for row in rows)
            else None
        ),
    }
    return {
        "clauses": len(rows),
        "metrics": metrics,
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "phase_confusion": {
            target: dict(predictions)
            for target, predictions in sorted(phase_confusion.items())
        },
        "phase_target_kind": "online_state_proxy",
        "privileged_gt_mask_oracle_partition": {
            "counts": dict(oracle_counts),
            "rates_excluding_unary": {
                key: (float(value / oracle_total) if oracle_total else None)
                for key, value in oracle_counts.items()
                if key != "unary_role_not_applicable"
            },
            "audited_non_unary_clauses": oracle_total,
        },
        "per_camera": _summarize_camera_rows(rows),
    }


def _extended_shadow_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available_records = [
        record
        for record in records
        if record.get("extended_diagnostics", {}).get("available")
    ]
    missing = Counter(
        name
        for record in records
        for name in record.get("extended_diagnostics", {}).get(
            "missing_diagnostics", []
        )
    )
    clause_rows = [
        dict(clause)
        for record in available_records
        for clause in record["extended_diagnostics"].get("clauses", [])
    ]
    if not clause_rows:
        return {
            "format": "pgc_v9_eraf_shadow_diagnostic_v2",
            "available": False,
            "records": 0,
            "missing_diagnostics": dict(missing),
        }
    by_status = {
        status: _summarize_clause_rows(
            [row for row in clause_rows if row.get("status") == status]
        )
        for status in sorted({str(row.get("status")) for row in clause_rows})
    }
    by_predicate = {
        predicate: _summarize_clause_rows(
            [row for row in clause_rows if row.get("predicate") == predicate]
        )
        for predicate in sorted({str(row.get("predicate")) for row in clause_rows})
    }
    by_task = {
        task: _summarize_clause_rows(
            [row for row in clause_rows if row.get("task") == task]
        )
        for task in sorted({str(row.get("task")) for row in clause_rows})
    }
    unfinished = [row for row in clause_rows if row.get("status") != "completed"]
    overall = _summarize_clause_rows(clause_rows)
    oracle_rates = overall["privileged_gt_mask_oracle_partition"][
        "rates_excluding_unary"
    ]
    occlusion_or_missing_rate = float(oracle_rates.get("gt_not_jointly_visible") or 0.0)
    overlap_ambiguity_rate = float(oracle_rates.get("mask_overlap_ambiguous") or 0.0)
    visible_binding_error_rate = float(oracle_rates.get("visible_binding_error") or 0.0)
    phase_proxy_accuracy = overall["metrics"].get("phase_proxy_accuracy")
    truth_accuracy = overall["metrics"].get("predicate_truth_accuracy")
    evidence = {
        "gt_not_jointly_visible_rate": occlusion_or_missing_rate,
        "mask_overlap_ambiguous_rate": overlap_ambiguity_rate,
        "visible_binding_error_rate": visible_binding_error_rate,
        "phase_proxy_error_rate": (
            None if phase_proxy_accuracy is None else 1.0 - float(phase_proxy_accuracy)
        ),
        "predicate_truth_error_rate": (
            None if truth_accuracy is None else 1.0 - float(truth_accuracy)
        ),
    }
    ranked = sorted(
        ((name, value) for name, value in evidence.items() if value is not None),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "format": "pgc_v9_eraf_shadow_diagnostic_v2",
        "available": True,
        "records": len(available_records),
        "record_coverage": float(len(available_records) / len(records)),
        "missing_diagnostics": dict(missing),
        "phase_target_note": (
            "Online phase targets are state-derived proxies: search=0, holding or "
            "released-unfinished=1, predicate-complete=2. They are not replay "
            "interaction-step labels."
        ),
        "overall": overall,
        "unfinished_clauses": _summarize_clause_rows(unfinished),
        "by_clause_status": by_status,
        "by_predicate": by_predicate,
        "by_task": by_task,
        "diagnosis": {
            "dominant_observed_error": ranked[0][0] if ranked else None,
            "evidence_rates": evidence,
            "interpretation": (
                "This is a passive information and prediction decomposition, "
                "not a causal intervention. GT masks determine whether the error "
                "is compatible with camera visibility, mask overlap, or visible "
                "role binding; phase uses the documented online-state proxy."
            ),
        },
    }


def summarize_eraf_shadow_records(
    records: Sequence[Mapping[str, Any]],
    *,
    action_integrity: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Aggregate online records with the exact offline ERAF gate metric code."""
    from scripts.eval_pgc_v9_grounding_gate import compute_grounding_gate_report

    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("ERAF shadow summary received no online records.")

    def gate_report(items: list[dict[str, Any]]) -> dict[str, Any]:
        require_view_scheduler = bool(items) and all(
            bool(item.get("v99_view_scheduler_available")) for item in items
        )
        return compute_grounding_gate_report(
            items,
            require_view_scheduler=require_view_scheduler,
        )

    delta_metric_names = (
        "subject_top1_in_gt_mask",
        "reference_top1_in_gt_mask",
        "relation_macro_f1",
        "all_entity_exclusive_role_accuracy",
        "visible_goal_anchor_median_error_cm",
        "clause_exact_match",
        "multi_clause_exact_match",
        "single_visible_view_selection_accuracy",
        "clause_scheduler_accuracy",
    )

    def metric_deltas(
        post_metrics: Mapping[str, Any], pre_metrics: Mapping[str, Any]
    ) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for name in delta_metric_names:
            post_value = post_metrics.get(name)
            pre_value = pre_metrics.get(name)
            if post_value is None or pre_value is None:
                result[name] = None
                continue
            post_value = float(post_value)
            pre_value = float(pre_value)
            result[name] = (
                post_value - pre_value
                if np.isfinite(post_value) and np.isfinite(pre_value)
                else None
            )
        return result

    gate = gate_report(rows)
    require_view_scheduler = bool(rows) and all(
        bool(row.get("v99_view_scheduler_available")) for row in rows
    )
    integrity = [dict(item) for item in action_integrity]
    action_summary = {
        "chunks": len(integrity),
        "exact_chunks": sum(bool(item.get("exact", False)) for item in integrity),
        "exact_rate": (
            float(np.mean([bool(item.get("exact", False)) for item in integrity]))
            if integrity
            else 0.0
        ),
        "max_abs_error": (
            max(float(item["max_abs_error"]) for item in integrity)
            if integrity
            else None
        ),
        "rms_error_max": (
            max(float(item["rms_error"]) for item in integrity) if integrity else None
        ),
    }
    by_stage: dict[str, Any] = {}
    for stage in ("pregrasp", "postgrasp", "complete"):
        subset = [row for row in rows if row.get("online_stage") == stage]
        if subset:
            report = gate_report(subset)
            by_stage[stage] = {
                "decisions": len(subset),
                "metrics": report["metrics"],
            }
    by_stage_v2: dict[str, Any] = {}
    for stage in (
        "initial_search",
        "holding",
        "released_unfinished",
        "next_clause_search",
        "complete",
    ):
        subset = [row for row in rows if row.get("online_stage_v2") == stage]
        if subset:
            report = gate_report(subset)
            by_stage_v2[stage] = {
                "decisions": len(subset),
                "metrics": report["metrics"],
                "extended": _extended_shadow_summary(subset),
            }
            pre_subset = [
                dict(row["pre_rebinding_record"])
                for row in subset
                if isinstance(row.get("pre_rebinding_record"), Mapping)
            ]
            if len(pre_subset) == len(subset):
                pre_report = gate_report(pre_subset)
                by_stage_v2[stage].update(
                    {
                        "pre_rebinding_metrics": pre_report["metrics"],
                        "post_minus_pre_rebinding": metric_deltas(
                            report["metrics"], pre_report["metrics"]
                        ),
                    }
                )
    replan_windows = {
        "initial_0": lambda value: value == 0,
        "early_1_4": lambda value: 1 <= value <= 4,
        "middle_5_19": lambda value: 5 <= value <= 19,
        "late_20_plus": lambda value: value >= 20,
    }
    by_replan_window: dict[str, Any] = {}
    for name, predicate in replan_windows.items():
        subset = [row for row in rows if predicate(int(row.get("replan_index", -1)))]
        if subset:
            report = gate_report(subset)
            by_replan_window[name] = {
                "decisions": len(subset),
                "metrics": report["metrics"],
            }
    extended = _extended_shadow_summary(rows)
    phase_safe_memory_records = [
        row for row in rows if bool(row.get("phase_safe_memory_enabled", False))
    ]
    phase_safe_memory_clause_rows = [
        clause
        for row in phase_safe_memory_records
        for clause in row.get("extended_diagnostics", {}).get("clauses", [])
    ]
    phase_safe_memory_summary = {
        "available": bool(phase_safe_memory_records),
        "admission_thresholds": {
            "state_coverage": 1.0,
            "state_accuracy": 0.90,
            "released_unfinished_retry_rate": 0.90,
            "postgrasp_clause_scheduler_accuracy": 0.90,
            "completed_sticky_violation_rate": 0.0,
            "geometry_max_abs": 0.0,
        },
        "record_coverage": (
            float(len(phase_safe_memory_records) / len(rows)) if rows else 0.0
        ),
        "state_accuracy": _optional_rate(
            [
                clause.get("phase_safe_memory_state_correct")
                for clause in phase_safe_memory_clause_rows
            ]
        ),
        "state_coverage": _optional_rate(
            [
                clause.get("phase_safe_memory_state_valid")
                for clause in phase_safe_memory_clause_rows
            ]
        ),
        "completed_sticky_violation_rate": _optional_rate(
            [
                clause.get("phase_safe_memory_completed_sticky_violation")
                for clause in phase_safe_memory_clause_rows
            ]
        ),
        "released_unfinished_retry_rate": _optional_rate(
            [
                clause.get("phase_safe_memory_retry_transition")
                for clause in phase_safe_memory_clause_rows
                if clause.get("status") == "released_unfinished"
            ]
        ),
        "clause_scheduler_accuracy": _optional_rate(
            [
                clause.get("execution_selection_correct")
                for clause in phase_safe_memory_clause_rows
            ]
        ),
        "postgrasp_clause_scheduler_accuracy": _optional_rate(
            [
                clause.get("execution_selection_correct")
                for clause in phase_safe_memory_clause_rows
                if clause.get("status")
                in {"holding", "released_unfinished", "next_clause_search"}
            ]
        ),
        "geometry_max_abs": (
            max(
                float(row["phase_safe_memory_geometry_max_abs"])
                for row in phase_safe_memory_records
                if row.get("phase_safe_memory_geometry_max_abs") is not None
            )
            if any(
                row.get("phase_safe_memory_geometry_max_abs") is not None
                for row in phase_safe_memory_records
            )
            else None
        ),
    }
    pre_rebinding_rows = [
        dict(row["pre_rebinding_record"])
        for row in rows
        if isinstance(row.get("pre_rebinding_record"), Mapping)
    ]
    same_state_rebinding: dict[str, Any] = {
        "available": bool(pre_rebinding_rows),
        "record_coverage": (
            float(len(pre_rebinding_rows) / len(rows)) if rows else 0.0
        ),
    }
    if len(pre_rebinding_rows) == len(rows):
        pre_rebinding_gate = gate_report(pre_rebinding_rows)
        same_state_rebinding.update(
            {
                "pre_rebinding_metrics": pre_rebinding_gate["metrics"],
                "post_rebinding_metrics": gate["metrics"],
                "post_minus_pre_rebinding": metric_deltas(
                    gate["metrics"], pre_rebinding_gate["metrics"]
                ),
            }
        )
    phase_safe_memory_admission = bool(
        phase_safe_memory_summary["available"]
        and phase_safe_memory_summary["record_coverage"] == 1.0
        and phase_safe_memory_summary["state_coverage"] == 1.0
        and phase_safe_memory_summary["state_accuracy"] is not None
        and phase_safe_memory_summary["state_accuracy"] >= 0.90
        and phase_safe_memory_summary["released_unfinished_retry_rate"] is not None
        and phase_safe_memory_summary["released_unfinished_retry_rate"] >= 0.90
        and phase_safe_memory_summary["postgrasp_clause_scheduler_accuracy"]
        is not None
        and phase_safe_memory_summary["postgrasp_clause_scheduler_accuracy"] >= 0.90
        and phase_safe_memory_summary["geometry_max_abs"] == 0.0
        and phase_safe_memory_summary["completed_sticky_violation_rate"] == 0.0
        and action_summary["chunks"] == len(rows)
        and action_summary["exact_rate"] == 1.0
        and action_summary["max_abs_error"] == 0.0
    )
    return {
        "format": "pgc_v9_eraf_shadow_audit_v2",
        "decisions": len(rows),
        "action_integrity": action_summary,
        "grounding_gate": gate,
        "v99_view_scheduler_gate_required": require_view_scheduler,
        "by_online_stage": by_stage,
        "by_online_stage_v2": by_stage_v2,
        "by_replan_window": by_replan_window,
        "extended_diagnostics": extended,
        "same_state_rebinding": same_state_rebinding,
        "phase_safe_clause_memory": phase_safe_memory_summary,
        "phase_safe_memory_admission_passed": phase_safe_memory_admission,
        "passed": bool(
            gate["passed"]
            and action_summary["chunks"] == len(rows)
            and action_summary["exact_rate"] == 1.0
            and action_summary["max_abs_error"] == 0.0
            and (
                not phase_safe_memory_summary["available"]
                or phase_safe_memory_summary["geometry_max_abs"] == 0.0
            )
        ),
    }
