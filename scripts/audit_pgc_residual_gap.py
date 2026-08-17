#!/usr/bin/env python3
"""Audit the PGC Proposal residual from training states to deployment.

The audit deliberately runs ``FastWAM.infer_action`` on counterfactual
demonstration states instead of calling the training-loss implementation.  It
therefore measures the exact post-checkpoint-load Proposal path used by LIBERO
evaluation.  Each state is evaluated twice with identical action noise:

* the cached text embedding used during training;
* a freshly encoded prompt, matching deployment.

Existing rollout result JSON files provide the third (closed-loop deployment)
distribution.  The resulting report distinguishes checkpoint corruption,
cached/live text drift, and state-distribution shift.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def summarize_values(values: Iterable[float]) -> dict[str, Any]:
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
    if not np.isfinite(array).all():
        raise ValueError("Audit metrics must be finite.")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    candidates = sorted(path.rglob("*_results.json"))
    result: list[Path] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload.get("policy_guard_episode_diagnostics"), list):
            result.append(candidate)
    return result


def summarize_rollout_results(path: str | Path) -> dict[str, Any]:
    """Aggregate the exact per-replan PGC diagnostics from LIBERO results."""
    path = Path(path).expanduser().resolve()
    files = _result_files(path)
    if not files:
        raise FileNotFoundError(
            f"No policy-guard evaluation result JSON files found under {path}."
        )

    decisions: list[Mapping[str, Any]] = []
    episodes = 0
    successes = 0
    per_task: dict[int, dict[str, Any]] = {}
    for result_path in files:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        task_id = int(payload.get("task_id", -1))
        task_decisions: list[Mapping[str, Any]] = []
        for episode in payload["policy_guard_episode_diagnostics"]:
            episode_decisions = episode.get("decisions", [])
            if not isinstance(episode_decisions, list):
                raise ValueError(
                    f"Malformed policy-guard decisions in {result_path}."
                )
            task_decisions.extend(episode_decisions)
        decisions.extend(task_decisions)
        task_episodes = int(payload.get("total_episodes", 0))
        task_successes = int(payload.get("successes", 0))
        episodes += task_episodes
        successes += task_successes
        per_task[task_id] = {
            "task_description": str(payload.get("task_description", "")),
            "episodes": task_episodes,
            "successes": task_successes,
            "decisions": len(task_decisions),
            "candidate_delta_rms": summarize_values(
                item["candidate_delta_rms"]
                for item in task_decisions
                if "candidate_delta_rms" in item
            ),
        }

    if not decisions:
        raise ValueError(f"No policy-guard decisions found under {path}.")

    def decision_values(name: str) -> list[float]:
        return [float(item[name]) for item in decisions if name in item]

    overrides = sum(bool(item.get("selected_counterfactual", False)) for item in decisions)
    return {
        "path": str(path),
        "result_files": len(files),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / max(episodes, 1),
        "decisions": len(decisions),
        "overrides": overrides,
        "override_rate": overrides / max(len(decisions), 1),
        "candidate_delta_rms": summarize_values(
            decision_values("candidate_delta_rms")
        ),
        "candidate_saturation_fraction": summarize_values(
            decision_values("candidate_saturation_fraction")
        ),
        "target_binding_top1_mass": summarize_values(
            decision_values("target_binding_top1_mass")
        ),
        "target_binding_entropy": summarize_values(
            decision_values("target_binding_entropy")
        ),
        "target_binding_similarity_max": summarize_values(
            decision_values("target_binding_similarity_max")
        ),
        "per_task": {
            str(task_id): value for task_id, value in sorted(per_task.items())
        },
    }


def summarize_offline_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("The offline audit produced no records.")

    def values(mode: str, name: str) -> list[float]:
        result = []
        for record in records:
            value = record[mode].get(name)
            if value is not None:
                result.append(float(value))
        return result

    summary: dict[str, Any] = {"samples": len(records)}
    for mode in ("cached_text", "live_text"):
        summary[mode] = {
            "candidate_delta_rms": summarize_values(
                values(mode, "candidate_delta_rms")
            ),
            "candidate_saturation_fraction": summarize_values(
                values(mode, "candidate_saturation_fraction")
            ),
            "base_target_prefix_mse": summarize_values(
                values(mode, "base_target_prefix_mse")
            ),
            "candidate_target_prefix_mse": summarize_values(
                values(mode, "candidate_target_prefix_mse")
            ),
            "candidate_mse_improvement": summarize_values(
                values(mode, "candidate_mse_improvement")
            ),
            "target_binding_top1_mass": summarize_values(
                values(mode, "target_binding_top1_mass")
            ),
            "target_binding_entropy": summarize_values(
                values(mode, "target_binding_entropy")
            ),
            "target_binding_similarity_max": summarize_values(
                values(mode, "target_binding_similarity_max")
            ),
        }
        per_dim = [record[mode]["delta_rms_by_action_dim"] for record in records]
        summary[mode]["delta_rms_by_action_dim_mean"] = (
            np.asarray(per_dim, dtype=np.float64).mean(axis=0).tolist()
        )

    by_instruction: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_instruction[str(record["prompt"])].append(record)
    summary["per_instruction"] = {}
    for prompt, prompt_records in sorted(by_instruction.items()):
        summary["per_instruction"][prompt] = {
            "samples": len(prompt_records),
            "cached_delta_rms": summarize_values(
                row["cached_text"]["candidate_delta_rms"]
                for row in prompt_records
            ),
            "live_delta_rms": summarize_values(
                row["live_text"]["candidate_delta_rms"]
                for row in prompt_records
            ),
        }
    return summary


def diagnose_residual_gap(
    *,
    checkpoint_state: Mapping[str, Any],
    offline: Mapping[str, Any],
    rollout: Mapping[str, Any],
    text_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify the first failing boundary using measured scale ratios."""
    cached = float(offline["cached_text"]["candidate_delta_rms"]["mean"])
    live = float(offline["live_text"]["candidate_delta_rms"]["mean"])
    deployed = float(rollout["candidate_delta_rms"]["mean"])
    live_to_cached = live / max(cached, 1.0e-12)
    deployed_to_live = deployed / max(live, 1.0e-12)
    live_improvement = float(
        offline["live_text"]["candidate_mse_improvement"]["mean"]
    )
    context_cosine = text_context.get("cosine_similarity", {}).get("mean")

    diagnosis: str
    conclusion: str
    recommendation: str
    if not bool(checkpoint_state.get("exact_match", False)):
        diagnosis = "checkpoint_state_mismatch"
        conclusion = (
            "Loaded policy-guard tensors do not exactly match the checkpoint; "
            "the deployment audit is invalid until loading is fixed."
        )
        recommendation = "Fix checkpoint restoration before changing PGC training."
    elif cached < 0.02 and live < 0.02:
        diagnosis = "post_load_training_state_residual_collapse"
        conclusion = (
            "The Proposal is already near zero on counterfactual demonstration "
            "states after checkpoint reload."
        )
        recommendation = (
            "Compare the saved final training batch against this checkpoint and "
            "inspect Proposal state saving, loading, and model-mode parity."
        )
    elif live_to_cached < 0.35:
        diagnosis = "cached_live_text_path_mismatch"
        conclusion = (
            "The Proposal is active with cached training text but collapses with "
            "the live text-encoder representation used at deployment."
        )
        recommendation = (
            "Regenerate/verify the text cache with the deployed tokenizer and T5, "
            "then train and evaluate through one canonical text path."
        )
    elif live_improvement <= 0.0:
        diagnosis = "offline_proposal_not_action_aligned"
        conclusion = (
            "Even on demonstration states, the deployed Proposal does not reduce "
            "prefix action MSE relative to Base."
        )
        recommendation = (
            "Fix the action target, normalization, or Proposal objective before "
            "collecting more rollout data."
        )
    elif live >= 0.02 and deployed_to_live < 0.35:
        diagnosis = "closed_loop_state_distribution_shift"
        conclusion = (
            "The post-load Proposal is active on demonstration states but becomes "
            "near-Base on closed-loop LIBERO states."
        )
        recommendation = (
            "Build V8 around closed-loop corrective supervision (DAgger-style "
            "states or perturbation rollouts), not additional offline mask losses."
        )
    else:
        diagnosis = "residual_scale_is_not_primary_failure"
        conclusion = (
            "Training-state and deployment residual scales are comparable; the "
            "remaining failure is more likely target identity or action quality."
        )
        recommendation = (
            "Inspect per-task target binding and candidate-vs-demonstration action "
            "errors before modifying the gate."
        )

    return {
        "diagnosis": diagnosis,
        "conclusion": conclusion,
        "recommendation": recommendation,
        "cached_training_state_delta_rms": cached,
        "live_training_state_delta_rms": live,
        "closed_loop_delta_rms": deployed,
        "live_to_cached_delta_ratio": live_to_cached,
        "closed_loop_to_live_delta_ratio": deployed_to_live,
        "live_candidate_mse_improvement": live_improvement,
        "cached_live_context_cosine": context_cosine,
    }


def _find_training_config(checkpoint: Path) -> Path:
    for parent in checkpoint.parents:
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find the run config.yaml above checkpoint {checkpoint}."
    )


def _model_dtype(name: str):
    import torch

    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _compare_loaded_guard_state(model, checkpoint_payload) -> dict[str, Any]:
    import torch

    saved = checkpoint_payload.get("policy_guard")
    if not isinstance(saved, dict) or not saved:
        raise ValueError("Checkpoint has no policy_guard state dictionary.")
    loaded = model.policy_guard_modules.state_dict()
    missing = sorted(set(saved) - set(loaded))
    unexpected = sorted(set(loaded) - set(saved))
    shape_mismatches: dict[str, Any] = {}
    max_abs_error = 0.0
    unequal_tensors = 0
    for name in sorted(set(saved) & set(loaded)):
        saved_value = saved[name].detach().to(device="cpu")
        loaded_value = loaded[name].detach().to(device="cpu")
        if tuple(saved_value.shape) != tuple(loaded_value.shape):
            shape_mismatches[name] = {
                "checkpoint": list(saved_value.shape),
                "loaded": list(loaded_value.shape),
            }
            continue
        if saved_value.is_floating_point() or loaded_value.is_floating_point():
            difference = (
                saved_value.to(dtype=torch.float32)
                - loaded_value.to(dtype=torch.float32)
            ).abs()
            tensor_error = float(difference.max().item()) if difference.numel() else 0.0
            max_abs_error = max(max_abs_error, tensor_error)
            unequal_tensors += int(tensor_error != 0.0)
        elif not torch.equal(saved_value, loaded_value):
            unequal_tensors += 1
    exact = not missing and not unexpected and not shape_mismatches and unequal_tensors == 0
    return {
        "checkpoint_tensors": len(saved),
        "loaded_tensors": len(loaded),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
        "unequal_tensors": unequal_tensors,
        "max_abs_error": max_abs_error,
        "exact_match": exact,
    }


def _context_comparison(cached, live, cached_mask, live_mask) -> dict[str, float | bool]:
    import torch

    cached = cached.detach().to(device="cpu", dtype=torch.float32)
    live = live.detach().to(device="cpu", dtype=torch.float32)
    if cached.ndim == 3:
        cached = cached[0]
    if live.ndim == 3:
        live = live[0]
    if tuple(cached.shape) != tuple(live.shape):
        raise ValueError(
            f"Cached/live text shapes differ: {tuple(cached.shape)} vs {tuple(live.shape)}."
        )
    difference = cached - live
    cosine = torch.nn.functional.cosine_similarity(
        cached.reshape(1, -1), live.reshape(1, -1), dim=-1
    )[0]
    cached_mask = cached_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
    live_mask = live_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
    return {
        "rms": float(difference.square().mean().sqrt().item()),
        "max_abs": float(difference.abs().max().item()),
        "cosine_similarity": float(cosine.item()),
        "mask_equal": bool(torch.equal(cached_mask, live_mask)),
    }


def _prediction_metrics(pred, target_action, action_is_pad, prefix: int) -> dict[str, Any]:
    import torch

    required = (
        "policy_guard_candidate_delta_rms",
        "policy_guard_candidate_saturation_fraction",
        "policy_guard_base_action",
        "policy_guard_counterfactual_action",
    )
    missing = [name for name in required if name not in pred]
    if missing:
        raise ValueError(f"PGC inference did not return audit fields: {missing}.")
    base = pred["policy_guard_base_action"].detach().to(dtype=torch.float32)
    candidate = pred["policy_guard_counterfactual_action"].detach().to(
        dtype=torch.float32
    )
    target = target_action.detach().to(device="cpu", dtype=torch.float32)
    horizon = min(prefix, int(base.shape[0]), int(candidate.shape[0]), int(target.shape[0]))
    valid = torch.ones(horizon, dtype=torch.bool)
    if action_is_pad is not None:
        valid = ~action_is_pad.detach().to(device="cpu", dtype=torch.bool)[:horizon]
    if not bool(valid.any()):
        raise ValueError("Selected audit sample has no valid execution-prefix actions.")
    base_prefix = base[:horizon][valid]
    candidate_prefix = candidate[:horizon][valid]
    target_prefix = target[:horizon][valid]
    delta = candidate_prefix - base_prefix
    base_mse = float((base_prefix - target_prefix).square().mean().item())
    candidate_mse = float(
        (candidate_prefix - target_prefix).square().mean().item()
    )
    return {
        "candidate_delta_rms": float(pred["policy_guard_candidate_delta_rms"]),
        "candidate_saturation_fraction": float(
            pred["policy_guard_candidate_saturation_fraction"]
        ),
        "base_target_prefix_mse": base_mse,
        "candidate_target_prefix_mse": candidate_mse,
        "candidate_mse_improvement": base_mse - candidate_mse,
        "delta_rms_by_action_dim": delta.square().mean(dim=0).sqrt().tolist(),
        "target_binding_top1_mass": pred.get(
            "policy_guard_target_binding_top1_mass"
        ),
        "target_binding_entropy": pred.get(
            "policy_guard_target_binding_entropy"
        ),
        "target_binding_similarity_max": pred.get(
            "policy_guard_target_binding_similarity_max"
        ),
    }


def _select_counterfactual_indices(dataset, count: int, seed: int) -> list[int]:
    sample_indices = getattr(dataset, "_sample_indices", None)
    native_count = getattr(dataset, "pgc_native_frame_count", None)
    if sample_indices is None or native_count is None:
        raise TypeError(
            "Residual audit requires RobotVideoDataset PGC sample-index metadata."
        )
    candidates = np.asarray(
        [
            position
            for position, raw_index in enumerate(sample_indices)
            if int(raw_index) >= int(native_count)
        ],
        dtype=np.int64,
    )
    if candidates.size == 0:
        raise ValueError("Training config contains no counterfactual samples.")
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        candidates, size=min(int(count), int(candidates.size)), replace=False
    )
    return sorted(int(value) for value in selected.tolist())


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict

    from fastwam.utils import misc
    from fastwam.utils.pytorch_utils import set_global_seed

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    config_path = (
        args.training_config.expanduser().resolve()
        if args.training_config is not None
        else _find_training_config(checkpoint)
    )
    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        cfg.model.load_text_encoder = True
        cfg.model.skip_dit_load_from_pretrain = True
        cfg.model.action_dit_pretrained_path = None
        cfg.model.lora.enabled = False
        cfg.model.policy_guard.gate_mode = "counterfactual"
        if args.native_dataset_dir is not None:
            cfg.data.train.dataset_dirs = [str(args.native_dataset_dir.resolve())]
        if args.counterfactual_dataset_dir is not None:
            cfg.data.train.pgc_counterfactual_dataset_dirs = [
                str(args.counterfactual_dataset_dir.resolve())
            ]
        if args.text_cache_dir is not None:
            cfg.data.train.text_embedding_cache_dir = str(
                args.text_cache_dir.resolve()
            )
        if args.dataset_stats_path is not None:
            cfg.data.train.pretrained_norm_stats = str(
                args.dataset_stats_path.resolve()
            )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output.parent))
    set_global_seed(args.seed, get_worker_init_fn=False)

    dataset = instantiate(cfg.data.train)
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint_payload.get("format") != "fastwam_policy_guard_v7":
        raise ValueError(
            "This audit currently requires a final PGC v7 checkpoint, got "
            f"{checkpoint_payload.get('format')!r}."
        )
    model = instantiate(
        cfg.model,
        model_dtype=_model_dtype(args.dtype),
        device=args.device,
    )
    model.load_checkpoint(str(checkpoint))
    model = model.to(args.device).eval()
    checkpoint_state = _compare_loaded_guard_state(model, checkpoint_payload)

    metadata = checkpoint_payload.get("architecture_metadata") or {}
    prefix = int(metadata.get("execution_prefix_steps", 10))
    inference_steps = int(metadata.get("rollout_num_inference_steps", 10))
    selected_indices = _select_counterfactual_indices(
        dataset, args.num_samples, args.seed
    )
    live_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    text_checks: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for ordinal, dataset_index in enumerate(selected_indices):
        sample = dataset[dataset_index]
        if not bool(sample["pgc_is_counterfactual"].item()):
            raise RuntimeError(
                f"Selected PGC sample {dataset_index} was not counterfactual."
            )
        prompt = str(sample["prompt"])
        if prompt not in live_context_cache:
            with torch.no_grad():
                live_context, live_mask = model.encode_prompt(prompt)
            live_context_cache[prompt] = (
                live_context.detach().to(device="cpu"),
                live_mask.detach().to(device="cpu"),
            )
            text_checks[prompt] = _context_comparison(
                sample["context"],
                live_context_cache[prompt][0],
                sample["context_mask"],
                live_context_cache[prompt][1],
            )
        live_context, live_mask = live_context_cache[prompt]
        inference_seed = int(args.seed + ordinal)
        common = {
            "prompt": None,
            "input_image": sample["video"][:, 0],
            "action_horizon": int(sample["action"].shape[0]),
            "proprio": sample["proprio"][0],
            "num_inference_steps": inference_steps,
            "seed": inference_seed,
            "rand_device": "cpu",
            "tiled": False,
        }
        with torch.no_grad():
            cached_pred = model.infer_action(
                **common,
                context=sample["context"],
                context_mask=sample["context_mask"],
            )
            live_pred = model.infer_action(
                **common,
                context=live_context,
                context_mask=live_mask,
            )
        records.append(
            {
                "dataset_index": dataset_index,
                "seed": inference_seed,
                "prompt": prompt,
                "cached_text": _prediction_metrics(
                    cached_pred,
                    sample["action"],
                    sample.get("action_is_pad"),
                    prefix,
                ),
                "live_text": _prediction_metrics(
                    live_pred,
                    sample["action"],
                    sample.get("action_is_pad"),
                    prefix,
                ),
            }
        )
        print(
            f"AUDIT {ordinal + 1}/{len(selected_indices)} "
            f"cached_delta={records[-1]['cached_text']['candidate_delta_rms']:.6f} "
            f"live_delta={records[-1]['live_text']['candidate_delta_rms']:.6f}"
        )

    text_context = {
        "unique_prompts": len(text_checks),
        "rms": summarize_values(item["rms"] for item in text_checks.values()),
        "max_abs": summarize_values(
            item["max_abs"] for item in text_checks.values()
        ),
        "cosine_similarity": summarize_values(
            item["cosine_similarity"] for item in text_checks.values()
        ),
        "mask_equal_rate": float(
            np.mean([item["mask_equal"] for item in text_checks.values()])
        ),
        "per_prompt": text_checks,
    }
    offline = summarize_offline_records(records)
    rollout = summarize_rollout_results(args.rollout_results)
    diagnosis = diagnose_residual_gap(
        checkpoint_state=checkpoint_state,
        offline=offline,
        rollout=rollout,
        text_context=text_context,
    )
    report = {
        "format": "pgc_training_deployment_residual_audit_v1",
        "checkpoint": str(checkpoint),
        "training_config": str(config_path),
        "checkpoint_format": checkpoint_payload.get("format"),
        "checkpoint_step": checkpoint_payload.get("step"),
        "architecture_metadata": metadata,
        "checkpoint_state": checkpoint_state,
        "audit_settings": {
            "seed": args.seed,
            "num_samples": len(records),
            "execution_prefix_steps": prefix,
            "num_inference_steps": inference_steps,
            "dtype": args.dtype,
            "device": args.device,
        },
        "text_context": text_context,
        "offline_demonstration_states": offline,
        "closed_loop_rollout": rollout,
        "diagnosis": diagnosis,
        "offline_records": records,
    }
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(diagnosis, indent=2, ensure_ascii=False))
    print(f"Wrote residual audit: {output}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--native-dataset-dir", type=Path)
    parser.add_argument("--counterfactual-dataset-dir", type=Path)
    parser.add_argument("--text-cache-dir", type=Path)
    parser.add_argument("--dataset-stats-path", type=Path)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    return args


if __name__ == "__main__":
    run_audit(parse_args())
