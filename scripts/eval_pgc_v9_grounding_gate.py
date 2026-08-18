#!/usr/bin/env python3
"""Evaluate the PGC V9 ERAF gate through the deployed RGB-language path."""

from __future__ import annotations

import argparse
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


def compute_grounding_gate_report(
    records: list[Mapping[str, Any]],
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
    anchor_median_cm = (
        float(np.median(goal_anchor_errors_m) * 100.0)
        if goal_anchor_errors_m
        else float("inf")
    )
    metrics = {
        "samples": len(records),
        "subject_top1_in_gt_mask": _safe_rate(subject_hits),
        "reference_top1_in_gt_mask": _safe_rate(reference_hits),
        "relation_macro_f1": _macro_f1(
            relation_targets, relation_predictions
        ),
        "role_swap_accuracy": _safe_rate(role_swap),
        "visible_goal_anchor_median_error_cm": anchor_median_cm,
        "clause_exact_match": _safe_rate(clause_exact),
        "multi_clause_exact_match": _safe_rate(multi_clause_exact),
        "multi_clause_samples": len(multi_clause_exact),
    }
    checks = {
        "subject_top1_at_least_80pct": (
            metrics["subject_top1_in_gt_mask"] >= 0.80
        ),
        "reference_top1_at_least_80pct": (
            metrics["reference_top1_in_gt_mask"] >= 0.80
        ),
        "relation_macro_f1_at_least_90pct": (
            metrics["relation_macro_f1"] >= 0.90
        ),
        "role_swap_at_least_90pct": metrics["role_swap_accuracy"] >= 0.90,
        "visible_goal_anchor_median_at_most_5cm": anchor_median_cm <= 5.0,
        "multi_clause_exact_at_least_80pct": (
            bool(multi_clause_exact)
            and metrics["multi_clause_exact_match"] >= 0.80
        ),
    }
    return {
        "format": "pgc_v9_eraf_grounding_gate_v1",
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
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
        subject_valid
        & reference_valid
        & (subject_entity_ids != reference_entity_ids)
    )
    role_swap_correct = []
    for index in np.flatnonzero(role_valid):
        subject_own = float(
            (subject_attention[index] * subject_target[index]).sum()
        )
        subject_wrong = float(
            (subject_attention[index] * reference_target[index]).sum()
        )
        reference_own = float(
            (reference_attention[index] * reference_target[index]).sum()
        )
        reference_wrong = float(
            (reference_attention[index] * subject_target[index]).sum()
        )
        role_swap_correct.append(
            subject_own > subject_wrong and reference_own > reference_wrong
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
    semantic_exact = bool(
        np.array_equal(active, clause_valid)
        and np.array_equal(
            predicate_prediction[clause_valid], predicate_ids[clause_valid]
        )
    )
    # Clause exact match is intentionally stricter than predicate decoding:
    # every visible semantic role must also land in its own mask and beat its
    # same-state role-swap negative.
    exact = bool(
        semantic_exact
        and all(subject_hits)
        and all(reference_hits)
        and all(role_swap_correct)
    )
    return {
        "subject_top1_hits": subject_hits,
        "reference_top1_hits": reference_hits,
        "role_swap_correct": role_swap_correct,
        "relation_targets": predicate_ids[clause_valid].tolist(),
        "relation_predictions": predicate_prediction[clause_valid].tolist(),
        "goal_anchor_errors_m": anchor_errors[goal_valid].tolist(),
        "clause_exact": exact,
        "clause_count": int(clause_valid.sum()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict

    from fastwam.utils import misc
    from fastwam.utils.pytorch_utils import set_global_seed

    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = (
        args.training_config.expanduser().resolve()
        if args.training_config
        else _find_training_config(checkpoint)
    )
    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        cfg.model.load_text_encoder = True
        cfg.model.skip_dit_load_from_pretrain = True
        cfg.model.action_dit_pretrained_path = None
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
    if int(payload.get("step", -1)) != 1500 or metadata.get(
        "eraf_training_stage"
    ) != "grounding":
        raise ValueError(
            "The pre-action grounding gate requires the V9 grounding-stage "
            "checkpoint at cumulative step 1500."
        )
    model = model.to(args.device).eval()
    sidecars = list(dataset.pgc_entity_relation_indices.values())
    workspace_bounds = {
        (
            tuple(float(value) for value in index["workspace_min"]),
            tuple(float(value) for value in index["workspace_max"]),
        )
        for index in sidecars
    }
    if len(workspace_bounds) != 1:
        raise ValueError("PGC v9 sidecars disagree on workspace bounds.")
    lower, upper = next(iter(workspace_bounds))
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
        records.append(
            _sample_record(
                prediction["policy_guard_eraf_diagnostics"],
                sample,
                workspace_min,
                workspace_max,
            )
        )
        print(f"ERAF_GATE {ordinal + 1}/{len(selected)}", flush=True)
    report = compute_grounding_gate_report(records)
    report.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_step": payload.get("step"),
            "training_config": str(config_path),
            "seed": args.seed,
        }
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
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
