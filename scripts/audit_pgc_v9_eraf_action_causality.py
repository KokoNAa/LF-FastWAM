#!/usr/bin/env python3
"""Same-state causal audit of the PGC V9 ERAF-to-Proposal interface.

The audit runs one frozen Base diffusion per row, then applies the same learned
ActionChunkProposal to learned, privileged, bypassed, and deliberately corrupted
ERAF routes.  Consequently action differences cannot be attributed to diffusion
noise or a different visual state.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ORACLE_FIELDS = (
    "clause_valid",
    "predicate_ids",
    "subject_masks",
    "reference_masks",
    "subject_mask_valid",
    "reference_mask_valid",
    "subject_positions",
    "reference_positions",
    "subject_position_valid",
    "reference_position_valid",
    "goal_anchors",
    "goal_anchor_valid",
    "predicate_truth",
    "phase_ids",
    "phase_valid",
)
SUBJECT_FIELDS = (
    "subject_masks",
    "subject_mask_valid",
    "subject_positions",
    "subject_position_valid",
)
REFERENCE_FIELDS = (
    "reference_masks",
    "reference_mask_valid",
    "reference_positions",
    "reference_position_valid",
)
GOAL_ANCHOR_FIELDS = (
    "goal_anchors",
    "goal_anchor_valid",
)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _copy_value(value: Any) -> Any:
    if hasattr(value, "clone"):
        return value.clone()
    return np.asarray(value).copy()


def oracle_from_sample(
    sample: Mapping[str, Any], *, source: bool = False
) -> dict[str, Any]:
    prefix = "pgc_eraf_source_" if source else "pgc_eraf_"
    missing = [name for name in ORACLE_FIELDS if prefix + name not in sample]
    if missing:
        raise KeyError(f"Sample is missing ERAF Oracle fields: {missing}.")
    return {name: _copy_value(sample[prefix + name]) for name in ORACLE_FIELDS}


def build_causal_variants(
    sample: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any] | None], dict[str, bool]]:
    target = oracle_from_sample(sample)
    source = oracle_from_sample(sample, source=True)
    wrong_subject = {name: _copy_value(value) for name, value in target.items()}
    for name in SUBJECT_FIELDS:
        wrong_subject[name] = _copy_value(source[name])
    wrong_reference = {
        name: _copy_value(value) for name, value in target.items()
    }
    for name in REFERENCE_FIELDS:
        wrong_reference[name] = _copy_value(source[name])
    wrong_goal_anchor = {
        name: _copy_value(value) for name, value in target.items()
    }
    for name in GOAL_ANCHOR_FIELDS:
        wrong_goal_anchor[name] = _copy_value(source[name])

    clause_swap = {name: _copy_value(value) for name, value in target.items()}
    active = np.flatnonzero(_as_numpy(target["clause_valid"]).astype(bool))
    clause_swap_eligible = bool(active.size >= 2)
    if clause_swap_eligible:
        first, second = map(int, active[:2])
        for name, value in clause_swap.items():
            if hasattr(value, "clone"):
                original = value.clone()
                value[first] = original[second]
                value[second] = original[first]
            else:
                original = np.asarray(value).copy()
                value[first] = original[second]
                value[second] = original[first]

    def changed(fields: Iterable[str], left: Mapping[str, Any]) -> bool:
        return any(
            not np.array_equal(_as_numpy(left[name]), _as_numpy(target[name]))
            for name in fields
        )

    eligibility = {
        "wrong_subject": changed(SUBJECT_FIELDS, wrong_subject),
        "wrong_reference": changed(REFERENCE_FIELDS, wrong_reference),
        "wrong_goal_anchor": changed(GOAL_ANCHOR_FIELDS, wrong_goal_anchor),
        "clause_swap": clause_swap_eligible,
    }
    variants: dict[str, Mapping[str, Any] | None] = {
        "learned": None,
        "oracle": target,
        "bypass": {"_audit_bypass_bridge": True},
        "wrong_subject": wrong_subject,
        "wrong_reference": wrong_reference,
        "wrong_goal_anchor": wrong_goal_anchor,
        "clause_swap": clause_swap,
    }
    return variants, eligibility


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


def _optional_rate(values: Iterable[bool]) -> float | None:
    array = np.asarray(list(values), dtype=bool)
    return None if array.size == 0 else float(array.mean())


def compute_causal_report(
    records: list[Mapping[str, Any]],
    *,
    minimum_action_effect_rms: float = 0.002,
    minimum_directional_rate: float = 0.55,
) -> dict[str, Any]:
    if not records:
        raise ValueError("ERAF causal audit requires at least one record.")
    variant_names = tuple(records[0]["variants"])
    variants: dict[str, Any] = {}
    for name in variant_names:
        variants[name] = {
            "expert_mse": _distribution(
                float(record["variants"][name]["expert_mse"])
                for record in records
            ),
            "base_mse_improvement": _distribution(
                float(record["variants"][name]["base_mse_improvement"])
                for record in records
            ),
            "improves_base_rate": _optional_rate(
                float(record["variants"][name]["base_mse_improvement"]) > 0
                for record in records
            ),
            "residual_rms": _distribution(
                float(record["variants"][name]["residual_rms"])
                for record in records
            ),
            "attention_entropy": _distribution(
                float(record["variants"][name]["attention_entropy"])
                for record in records
                if record["variants"][name]["attention_entropy"] is not None
            ),
            "expert_loss_goal_query_gradient_rms": _distribution(
                float(
                    record["variants"][name][
                        "expert_loss_goal_query_gradient_rms"
                    ]
                )
                for record in records
            ),
        }

    pair_specs = (
        ("oracle_vs_bypass", "oracle", "bypass", None),
        ("learned_vs_oracle", "learned", "oracle", None),
        ("wrong_subject_vs_oracle", "wrong_subject", "oracle", "wrong_subject"),
        (
            "wrong_reference_vs_oracle",
            "wrong_reference",
            "oracle",
            "wrong_reference",
        ),
        (
            "wrong_goal_anchor_vs_oracle",
            "wrong_goal_anchor",
            "oracle",
            "wrong_goal_anchor",
        ),
        ("clause_swap_vs_oracle", "clause_swap", "oracle", "clause_swap"),
    )
    effects: dict[str, Any] = {}
    for label, left, right, eligibility_name in pair_specs:
        eligible = [
            record
            for record in records
            if eligibility_name is None
            or bool(record["eligibility"][eligibility_name])
        ]
        action_delta = [
            float(record["pairwise_action_delta_rms"][f"{left}__{right}"])
            for record in eligible
        ]
        query_delta = [
            float(record["pairwise_goal_query_delta_rms"][f"{left}__{right}"])
            for record in eligible
        ]
        query_cosine = [
            float(record["pairwise_goal_query_cosine"][f"{left}__{right}"])
            for record in eligible
        ]
        excess_mse = [
            float(record["variants"][left]["expert_mse"])
            - float(record["variants"][right]["expert_mse"])
            for record in eligible
        ]
        effects[label] = {
            "eligible_samples": len(eligible),
            "action_delta_rms": _distribution(action_delta),
            "goal_query_delta_rms": _distribution(query_delta),
            "goal_query_cosine": _distribution(query_cosine),
            "action_effect_rate": _optional_rate(
                value >= minimum_action_effect_rms for value in action_delta
            ),
            "left_minus_right_expert_mse": _distribution(excess_mse),
            "right_is_better_rate": _optional_rate(
                value > 0 for value in excess_mse
            ),
        }

    finite_bridge_response = bool(
        effects["oracle_vs_bypass"]["action_delta_rms"]["mean"] is not None
        and effects["oracle_vs_bypass"]["action_delta_rms"]["mean"]
        >= minimum_action_effect_rms
    )
    oracle_gradient = variants["oracle"][
        "expert_loss_goal_query_gradient_rms"
    ]["mean"]
    locally_connected = bool(
        oracle_gradient is not None
        and np.isfinite(oracle_gradient)
        and oracle_gradient > 1.0e-8
    )
    bridge_response = bool(finite_bridge_response and locally_connected)
    oracle_alignment = bool(
        variants["oracle"]["improves_base_rate"] is not None
        and variants["oracle"]["improves_base_rate"] >= 0.5
        and variants["oracle"]["base_mse_improvement"]["mean"] > 0
    )
    subject_directional = effects["wrong_subject_vs_oracle"][
        "right_is_better_rate"
    ]
    reference_directional = effects["wrong_reference_vs_oracle"][
        "right_is_better_rate"
    ]
    semantic_ordering = bool(
        subject_directional is not None
        and reference_directional is not None
        and subject_directional >= minimum_directional_rate
        and reference_directional >= minimum_directional_rate
    )
    anchor_effect = effects["wrong_goal_anchor_vs_oracle"][
        "action_delta_rms"
    ]["mean"]
    anchor_connected = bool(
        anchor_effect is not None and anchor_effect >= minimum_action_effect_rms
    )
    if not finite_bridge_response:
        diagnosis = "eraf_action_bridge_bypassed_or_insensitive"
        recommendation = (
            "Expose ERAF tokens/anchors through a stronger phase-conditioned "
            "action interface before further grounding training."
        )
    elif not anchor_connected:
        diagnosis = "eraf_goal_anchor_not_connected_to_action"
        recommendation = (
            "Inject normalized grasp/goal/interaction anchors directly into "
            "phase-conditioned Proposal tokens and train anchor-only causal "
            "negatives."
        )
    elif not locally_connected:
        diagnosis = "eraf_action_proposal_locally_saturated"
        recommendation = (
            "Reduce Proposal saturation and restore a nonzero expert-loss "
            "Jacobian with respect to ERAF-routed goal queries."
        )
    elif not semantic_ordering:
        diagnosis = "eraf_semantics_not_directionally_used_by_proposal"
        recommendation = (
            "Train same-state wrong-subject and wrong-reference ranking losses "
            "at the Proposal output, not only inside ERAF."
        )
    elif not oracle_alignment:
        diagnosis = "eraf_action_bridge_active_but_expert_misaligned"
        recommendation = (
            "Supervise phase-specific transport/release residuals and directly "
            "align privileged anchors with expert action prefixes."
        )
    else:
        diagnosis = "eraf_action_causal_interface_pass"
        recommendation = (
            "Proceed to closed-loop validation; the same-state bridge responds "
            "directionally to privileged entity and relation interventions."
        )

    return {
        "format": "pgc_v9_eraf_action_causal_audit_v1",
        "samples": len(records),
        "action_integrity": {
            "single_base_diffusion_per_sample": True,
            "shared_base_action_across_all_variants": True,
            "diffusion_noise_is_a_confounder": False,
        },
        "passed": bool(
            bridge_response
            and anchor_connected
            and semantic_ordering
            and oracle_alignment
        ),
        "checks": {
            "oracle_changes_action_vs_bypass": finite_bridge_response,
            "oracle_expert_loss_has_nonzero_query_gradient": locally_connected,
            "goal_anchor_changes_action": anchor_connected,
            "oracle_improves_expert_alignment": oracle_alignment,
            "wrong_semantics_are_worse_than_oracle": semantic_ordering,
        },
        "thresholds": {
            "minimum_action_effect_rms": minimum_action_effect_rms,
            "minimum_directional_rate": minimum_directional_rate,
            "minimum_oracle_improves_base_rate": 0.5,
        },
        "base_expert_mse": _distribution(
            float(record["base_expert_mse"]) for record in records
        ),
        "variants": variants,
        "causal_effects": effects,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
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


def _bridge_parameter_norms(model: Any) -> dict[str, float]:
    module = model.policy_guard_modules["entity_relation_affordance"]
    names = (
        "base_query_projection",
        "relation_attention",
        "query_delta_projection",
        "embedding_delta_projection",
    )
    result: dict[str, float] = {}
    for name in names:
        squared = 0.0
        for parameter in getattr(module, name).parameters():
            squared += float(parameter.detach().float().square().sum().item())
        result[name] = float(squared**0.5)
    return result


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
    metadata = payload.get("architecture_metadata") or {}
    if (
        payload.get("format") != "fastwam_policy_guard_v9"
        or metadata.get("eraf_training_stage") != "action"
        or int(metadata.get("eraf_grounding_objective_version", 0)) < 14
    ):
        raise ValueError(
            "The causal audit requires a completed V9.14+ ERAF-Proposal "
            "action-stage checkpoint."
        )
    model = model.to(args.device).eval()

    allowed_datasets = set()
    for index, sidecar in dataset.pgc_entity_relation_indices.items():
        label = Path(str(sidecar["dataset"])).name.casefold()
        if args.counterfactual_scope == "all" or "strict" in label:
            if str(sidecar["dataset_kind"]) == "counterfactual":
                allowed_datasets.add(int(index))
    if not allowed_datasets:
        raise ValueError(
            f"No {args.counterfactual_scope} counterfactual ERAF datasets found."
        )

    unique_positions = []
    seen = set()
    for position, raw_index in enumerate(dataset._sample_indices):
        if int(raw_index) not in seen:
            seen.add(int(raw_index))
            unique_positions.append(position)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_positions)
    selected: list[tuple[int, Mapping[str, Any]]] = []
    per_task: dict[str, int] = defaultdict(int)
    per_task_limit = max(1, int(np.ceil(args.num_samples / 10)))
    for position in unique_positions:
        sample = dataset[position]
        dataset_index = int(torch.as_tensor(sample["pgc_dataset_index"]).item())
        if dataset_index not in allowed_datasets:
            continue
        if not bool(torch.as_tensor(sample["pgc_is_counterfactual"]).item()):
            continue
        if not bool(torch.as_tensor(sample["pgc_direct_action_valid"]).item()):
            continue
        if not bool(torch.as_tensor(sample["pgc_paired_language_valid"]).item()):
            continue
        task = str(sample.get("prompt", "unknown"))
        if per_task[task] >= per_task_limit:
            continue
        per_task[task] += 1
        selected.append((position, sample))
        if len(selected) >= args.num_samples:
            break
    if len(selected) < args.num_samples:
        raise ValueError(
            f"Requested {args.num_samples} balanced causal rows, found only "
            f"{len(selected)} eligible rows."
        )

    records = []
    for ordinal, (position, sample) in enumerate(selected):
        variants, eligibility = build_causal_variants(sample)
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
            policy_guard_eraf_audit_variants=variants,
        )
        base = _as_numpy(prediction["policy_guard_base_action"]).astype(np.float64)
        expert = _as_numpy(sample["action"]).astype(np.float64)
        padding = _as_numpy(sample["action_is_pad"]).astype(bool)
        valid = np.flatnonzero(~padding)[: args.execution_prefix_steps]
        if valid.size == 0:
            raise RuntimeError("Selected causal row has no valid expert action.")
        base_mse = float(np.square(base[valid] - expert[valid]).mean())
        variant_records: dict[str, Any] = {}
        variant_actions: dict[str, np.ndarray] = {}
        variant_queries: dict[str, np.ndarray] = {}
        for name, values in prediction[
            "policy_guard_eraf_causal_variants"
        ].items():
            action = _as_numpy(values["action"]).astype(np.float64)
            variant_actions[name] = action
            variant_queries[name] = _as_numpy(values["goal_queries"]).astype(
                np.float64
            )
            expert_mse = float(np.square(action[valid] - expert[valid]).mean())
            # This backward pass touches only the 3M-parameter Proposal: Base
            # action and ERAF queries are detached.  It diagnoses a locally
            # disconnected/saturated Proposal without retaining the 5B Base
            # diffusion graph.
            with torch.enable_grad():
                query = values["goal_queries"].to(
                    device=args.device, dtype=model.torch_dtype
                ).detach().requires_grad_(True)
                base_tensor = prediction["policy_guard_base_action"].to(
                    device=args.device, dtype=model.torch_dtype
                ).unsqueeze(0)
                expert_tensor = sample["action"].to(
                    device=args.device, dtype=model.torch_dtype
                ).unsqueeze(0)
                gradient_action, _, _ = model.policy_guard_modules[
                    "action_chunk_proposal"
                ](
                    base_action=base_tensor,
                    goal_queries=query,
                    action_is_pad=None,
                )
                valid_tensor = torch.as_tensor(
                    valid, device=args.device, dtype=torch.long
                )
                gradient_loss = (
                    gradient_action[:, valid_tensor]
                    - expert_tensor[:, valid_tensor]
                ).float().square().mean()
                query_gradient = torch.autograd.grad(
                    gradient_loss, query, retain_graph=False
                )[0]
            variant_records[name] = {
                "expert_mse": expert_mse,
                "base_mse_improvement": base_mse - expert_mse,
                "residual_rms": float(values["residual_rms"]),
                "attention_entropy": values["proposal_attention_entropy"],
                "expert_loss_goal_query_gradient_rms": float(
                    query_gradient.float().square().mean().sqrt().item()
                ),
                "goal_query_rms": float(values["goal_query_rms"]),
                "goal_query_delta_from_learned_rms": float(
                    values["goal_query_delta_from_learned_rms"]
                ),
            }
        pairwise = {}
        pairwise_queries = {}
        pairwise_query_cosine = {}
        for left, right in (
            ("oracle", "bypass"),
            ("learned", "oracle"),
            ("wrong_subject", "oracle"),
            ("wrong_reference", "oracle"),
            ("wrong_goal_anchor", "oracle"),
            ("clause_swap", "oracle"),
        ):
            pairwise[f"{left}__{right}"] = float(
                np.square(
                    variant_actions[left][valid] - variant_actions[right][valid]
                )
                .mean() ** 0.5
            )
            left_query = variant_queries[left].reshape(-1)
            right_query = variant_queries[right].reshape(-1)
            pairwise_queries[f"{left}__{right}"] = float(
                np.square(left_query - right_query).mean() ** 0.5
            )
            denominator = float(
                np.linalg.norm(left_query) * np.linalg.norm(right_query)
            )
            pairwise_query_cosine[f"{left}__{right}"] = (
                1.0
                if denominator == 0 and np.array_equal(left_query, right_query)
                else (
                    0.0
                    if denominator == 0
                    else float(np.dot(left_query, right_query) / denominator)
                )
            )
        records.append(
            {
                "position": int(position),
                "raw_index": int(dataset._sample_indices[position]),
                "task": str(sample.get("prompt", "unknown")),
                "dataset_index": int(
                    torch.as_tensor(sample["pgc_dataset_index"]).item()
                ),
                "valid_prefix_steps": int(valid.size),
                "base_expert_mse": base_mse,
                "eligibility": eligibility,
                "variants": variant_records,
                "pairwise_action_delta_rms": pairwise,
                "pairwise_goal_query_delta_rms": pairwise_queries,
                "pairwise_goal_query_cosine": pairwise_query_cosine,
            }
        )
        print(f"ERAF_CAUSAL_AUDIT {ordinal + 1}/{len(selected)}", flush=True)

    report = compute_causal_report(
        records,
        minimum_action_effect_rms=args.minimum_action_effect_rms,
        minimum_directional_rate=args.minimum_directional_rate,
    )
    report.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(payload.get("step", -1)),
            "training_config": str(config_path),
            "counterfactual_scope": args.counterfactual_scope,
            "execution_prefix_steps": args.execution_prefix_steps,
            "seed": args.seed,
            "bridge_parameter_l2_norms": _bridge_parameter_norms(model),
            "records": records,
        }
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "passed", "checks", "diagnosis", "recommendation"
    )}, indent=2))
    if not report["passed"]:
        raise SystemExit(2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--execution-prefix-steps", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--counterfactual-scope", choices=("strict", "all"), default="strict"
    )
    parser.add_argument(
        "--minimum-action-effect-rms", type=float, default=0.002
    )
    parser.add_argument(
        "--minimum-directional-rate", type=float, default=0.55
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    args = parser.parse_args()
    if min(
        args.num_samples,
        args.execution_prefix_steps,
        args.num_inference_steps,
    ) <= 0:
        parser.error("sample count, prefix length, and inference steps must be positive")
    if args.minimum_action_effect_rms < 0:
        parser.error("minimum action effect must be non-negative")
    if not 0 <= args.minimum_directional_rate <= 1:
        parser.error("minimum directional rate must be in [0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
