"""Passive closed-loop ERAF audit for LIBERO.

The observer in this module is deliberately privileged and evaluation-only.
It reads MuJoCo element segmentation and BDDL predicates to score ERAF while
the deployed action remains the immutable Base action.  None of the labels
constructed here are passed back into the policy.
"""

from __future__ import annotations

import hashlib
import json
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
    ) -> None:
        if instruction_condition not in {"correct", "counterfactual"}:
            raise ValueError(
                "ERAF shadow audit supports only correct or counterfactual instructions."
            )
        self.env = env
        self.policy_instruction = str(policy_instruction)
        self.instruction_condition = str(instruction_condition)
        self.contract = contract
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
        self._ever_grasped |= subject_grasped
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
            "pgc_eraf_goal_anchors": goal_anchors,
            "pgc_eraf_goal_anchor_valid": goal_anchor_valid,
        }
        record = _sample_record(
            diagnostics,
            sample,
            self.contract.workspace_min,
            self.contract.workspace_max,
            dataset_kind="online_closed_loop",
            dataset_label=self.instruction_condition,
            predicate_vocabulary=self.contract.predicate_vocabulary,
        )
        record.update(
            {
                "episode": int(episode_idx),
                "replan_index": int(replan_idx),
                "policy_step": int(policy_step),
                "online_stage": stage,
                "instruction_condition": self.instruction_condition,
                "policy_instruction": self.policy_instruction,
                "predicate_truth": predicate_truth[clause_valid].tolist(),
                "subject_grasped": subject_grasped[clause_valid].tolist(),
                "subject_ever_grasped": self._ever_grasped[clause_valid].tolist(),
            }
        )
        return record


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
    gate = compute_grounding_gate_report(rows)
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
            report = compute_grounding_gate_report(subset)
            by_stage[stage] = {
                "decisions": len(subset),
                "metrics": report["metrics"],
            }
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
            report = compute_grounding_gate_report(subset)
            by_replan_window[name] = {
                "decisions": len(subset),
                "metrics": report["metrics"],
            }
    return {
        "format": "pgc_v9_eraf_shadow_audit_v1",
        "decisions": len(rows),
        "action_integrity": action_summary,
        "grounding_gate": gate,
        "by_online_stage": by_stage,
        "by_replan_window": by_replan_window,
        "passed": bool(
            gate["passed"]
            and action_summary["chunks"] == len(rows)
            and action_summary["exact_rate"] == 1.0
            and action_summary["max_abs_error"] == 0.0
        ),
    }
