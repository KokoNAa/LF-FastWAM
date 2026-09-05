"""Small, deployment-cache Action-LoRA repair experiments.

The two correct expert targets are supervised together. Video/context caches
are frozen, dropout is disabled, and only Action LoRA A/B may receive updates.
This is a learnability intervention, not the ordinary four-pool training loss.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from experiments.robotwin.no_eraf_probe import difference, paired_action_details


FORMAT = "robotwin_same_state_action_repair_v1"
ARMS = ("paired_flow", "paired_flow_anchor", "paired_flow_anchor_delta")


def move_cache(value, device, *, clone=False):
    import torch

    if isinstance(value, torch.Tensor):
        result = value.detach().to(device=device)
        return result.clone() if clone else result
    if isinstance(value, dict):
        return {key: move_cache(item, device, clone=clone) for key, item in value.items()}
    if isinstance(value, list):
        return [move_cache(item, device, clone=clone) for item in value]
    if isinstance(value, tuple):
        return tuple(move_cache(item, device, clone=clone) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported cache value: {type(value)}")


def configure_action_lora(model):
    """Freeze everything, then select canonical MoT Action adapter names only."""
    import torch

    model.eval()
    model.requires_grad_(False)
    selected = {}
    for name, parameter in model.mot.named_parameters():
        if name.startswith("mixtures.action.") and name.endswith((".lora_A", ".lora_B")):
            parameter.data = parameter.data.to(dtype=torch.float32)
            parameter.requires_grad_(True)
            selected[name] = parameter
    if not selected or not any(name.endswith(".lora_B") for name in selected):
        raise ValueError("No Action LoRA A/B parameters were found.")
    selected_ids = {id(parameter) for parameter in selected.values()}
    if {id(p) for p in model.parameters() if p.requires_grad} != selected_ids:
        raise ValueError("Trainable scope escaped Action LoRA.")
    return selected


def frozen_versions(model):
    return {name: (id(p), p._version, p.data_ptr(), p.dtype)
            for name, p in model.named_parameters() if not p.requires_grad}


def audit_frozen(model, expected):
    if frozen_versions(model) != expected:
        raise RuntimeError("A frozen parameter changed identity, version, storage, or dtype.")
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError("A frozen parameter received a gradient.")


def predict_with_grad(model, cache, noisy, timestep, *, checkpoint=True):
    """Use the exact production predictor body with its no_grad wrapper removed.

    The original method is never replaced. Non-reentrant activation checkpointing
    works even though the cached observations and noisy action require no grad.
    """
    import torch
    from torch.utils.checkpoint import checkpoint as activation_checkpoint

    method = type(model)._predict_action_noise_with_cache
    body = getattr(method, "__wrapped__", None)
    if body is None:
        raise ValueError("Expected the audited no_grad-decorated production predictor.")

    def forward(action, time):
        return body(model, latents_action=action, timestep_action=time, **cache)

    with torch.enable_grad():
        prediction = (activation_checkpoint(forward, noisy, timestep, use_reentrant=False)
                      if checkpoint else forward(noisy, timestep))
    if not prediction.requires_grad:
        raise RuntimeError("Repair prediction is detached from Action LoRA.")
    return prediction


def noise_tensor(shape, seed, model):
    import torch

    return torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed),
                       dtype=torch.float32).to(device=model.device, dtype=model.torch_dtype)


def sample_cached_actions(model, cache, seed, steps, horizon=32):
    """Same CPU noise, inference scheduler, predictor and Euler steps as infer_action."""
    import torch

    with torch.no_grad():
        action = noise_tensor((1, horizon, 14), seed, model)
        scheduler = model.infer_action_scheduler
        times, deltas = scheduler.build_inference_schedule(
            num_inference_steps=steps, device=model.device, dtype=action.dtype, shift_override=None)
        for time, delta in zip(times, deltas):
            velocity = model._predict_action_noise_with_cache(
                latents_action=action, timestep_action=time.unsqueeze(0), **cache)
            action = scheduler.step(velocity, delta, action)
        return action[0].float().cpu().numpy()


def backward_paired_flow(model, caches, references, noise, timestep, coefficient,
                         *, checkpoint=True, include_components=False):
    """Backpropagate equal source/target positives without keeping two graphs.

    coefficient is explicit: the sigma=1 anchor MUST NOT inherit the scheduler's
    zero endpoint weight. Targets always come from the actual training scheduler.
    """
    import torch

    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("Flow coefficient must be finite and nonnegative.")
    scheduler = model.train_action_scheduler
    errors, saved_predictions, saved_targets = {}, {}, {}
    for language in ("source", "target"):
        clean = references[language]
        noisy = scheduler.add_noise(clean, noise, timestep)
        target = scheduler.training_target(clean, noise, timestep)
        prediction = predict_with_grad(model, caches[language], noisy, timestep, checkpoint=checkpoint)
        error = (prediction.float() - target.float()).square().mean()
        if not bool(torch.isfinite(error)):
            raise RuntimeError("Nonfinite paired flow loss.")
        (error * (coefficient / 2)).backward()
        errors[language] = float(error.detach())
        if include_components:
            saved_predictions[language], saved_targets[language] = prediction.detach(), target.detach()
    if include_components:
        parts = paired_velocity_losses(saved_predictions, saved_targets)
        errors.update({key: float(parts[key]) for key in ("common_mse", "conditional_mse")})
    return errors


def paired_velocity_losses(predictions, targets):
    """Pair MSE = common MSE + half-difference MSE, with gradients on BOTH branches."""
    s, t = (predictions[k].float() for k in ("source", "target"))
    r, q = (targets[k].float() for k in ("source", "target"))
    common = ((s + t - r - q) / 2).square().mean()
    conditional = ((t - s - (q - r)) / 2).square().mean()
    return {"source": (s - r).square().mean(), "target": (t - q).square().mean(),
            "common_mse": common, "conditional_mse": conditional}


def pure_anchor_losses(model, caches, references, noise, *, checkpoint=True):
    import torch

    scheduler = model.train_action_scheduler
    time = torch.tensor([scheduler.num_train_timesteps], device=model.device, dtype=model.torch_dtype)
    predictions, targets = {}, {}
    for kind in ("source", "target"):
        noisy = scheduler.add_noise(references[kind], noise, time)
        if not torch.equal(noisy, noise):
            raise ValueError("Pure-noise anchor input still contains expert actions.")
        predictions[kind] = predict_with_grad(model, caches[kind], noisy, time, checkpoint=checkpoint)
        targets[kind] = scheduler.training_target(references[kind], noise, time)
    losses = paired_velocity_losses(predictions, targets)
    if any(not bool(torch.isfinite(value)) for value in losses.values()):
        raise RuntimeError("Nonfinite pure-noise paired loss.")
    return losses


def backward_paired_anchor(model, caches, references, noise, coefficient, *, conditional_gain=1., checkpoint=True):
    """Reweight an existing endpoint loss component; no new labels or negatives.

    gain=1 exactly recovers equal source/target positive MSE. gain>1 adds
    (gain-1)*MSE((v_target-v_source)-(y_target-y_source))/4. At sigma=1 the
    two inputs share noise, and y_target-y_source = -(a_target-a_source)
    up to scheduler dtype rounding. Whole-forward checkpointing retains
    only the two graph boundaries until their joint backward.
    """
    if (not math.isfinite(coefficient) or coefficient < 0
            or not math.isfinite(conditional_gain) or conditional_gain < 1):
        raise ValueError("Invalid anchor coefficient or conditional gain.")
    if conditional_gain == 1:
        # Preserve the old control's exact arithmetic/backward order, including
        # BF16 gradient rounding; detached diagnostics add no model calls.
        import torch
        scheduler = model.train_action_scheduler
        time = torch.tensor([scheduler.num_train_timesteps], device=model.device, dtype=model.torch_dtype)
        if any(not torch.equal(scheduler.add_noise(ref, noise, time), noise) for ref in references.values()):
            raise ValueError("Pure-noise anchor input still contains expert actions.")
        return backward_paired_flow(model, caches, references, noise, time, coefficient,
                                    checkpoint=checkpoint, include_components=True)
    losses = pure_anchor_losses(model, caches, references, noise, checkpoint=checkpoint)
    (coefficient * (losses["common_mse"] + conditional_gain * losses["conditional_mse"])).backward()
    return {key: float(value.detach()) for key, value in losses.items()}


def anchor_gradient_audit(model, caches, references, noise):
    """Local common/difference gradients at one fixed sigma=1 input; no update.

    These are unweighted endpoint-component gradients, not gradients of the
    whole training objective or evidence of interference between tasks.
    """
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    losses = pure_anchor_losses(model, caches, references, noise)
    common = torch.autograd.grad(losses["common_mse"], params, retain_graph=True, allow_unused=True)
    common = [g.detach().float().cpu() if g is not None else None for g in common]
    conditional = torch.autograd.grad(losses["conditional_mse"], params, allow_unused=True)
    c2 = d2 = dot = 0.
    for c, d in zip(common, conditional):
        if c is not None:
            c2 += float(c.double().square().sum())
        if d is not None:
            d = d.detach().float().cpu()
            d2 += float(d.double().square().sum())
            if c is not None:
                dot += float((c.double() * d.double()).sum())
    if not all(math.isfinite(value) for value in (c2, d2, dot)):
        raise RuntimeError("Nonfinite anchor component gradient.")
    return {"common_mse": float(losses["common_mse"].detach()),
            "conditional_mse": float(losses["conditional_mse"].detach()),
            "common_grad_norm": math.sqrt(c2), "conditional_grad_norm": math.sqrt(d2),
            "conditional_to_common_grad_norm": math.sqrt(d2 / c2) if c2 else None,
            "gradient_cosine": dot / math.sqrt(c2 * d2) if c2 and d2 else None}


def fixed_flow_rows(model, caches, references, seeds, sigmas):
    """Fixed-input positive-flow evaluation, separate from ten-step generation.

    Only sigma=1 compares languages at identical noisy actions. At lower
    sigma the noisy inputs contain their own expert actions.
    """
    import torch

    rows = []
    scheduler = model.train_action_scheduler
    with torch.no_grad():
        for seed in seeds:
            noise = noise_tensor((1, 32, 14), seed, model)
            for sigma in sigmas:
                if not 0 < sigma <= 1:
                    raise ValueError("Fixed-flow sigma must be in (0,1].")
                time = torch.tensor([sigma * scheduler.num_train_timesteps],
                                    device=model.device, dtype=model.torch_dtype)
                predictions, targets = {}, {}
                for kind in ("source", "target"):
                    noisy = scheduler.add_noise(references[kind], noise, time)
                    predictions[kind] = model._predict_action_noise_with_cache(
                        latents_action=noisy, timestep_action=time, **caches[kind])
                    targets[kind] = scheduler.training_target(references[kind], noise, time)
                for horizon in (24, 32):
                    pred = {k: p[:, :horizon] for k, p in predictions.items()}
                    truth = {k: p[:, :horizon] for k, p in targets.items()}
                    losses = paired_velocity_losses(pred, truth)
                    if any(not bool(torch.isfinite(value)) for value in losses.values()):
                        raise RuntimeError("Nonfinite fixed-flow evaluation.")
                    delta = pred["target"].float() - pred["source"].float()
                    desired = truth["target"].float() - truth["source"].float()
                    energy = float(desired.square().sum())
                    rows.append({"noise_seed": seed, "sigma": sigma, "horizon": horizon,
                        "actual_time": float(time.float().item()),
                        "source_mse": float(losses["source"]), "target_mse": float(losses["target"]),
                        "common_mse": float(losses["common_mse"]),
                        "conditional_mse": float(losses["conditional_mse"]),
                        "delta_projection": float((delta * desired).sum()) / energy if energy > 1e-12 else None})
    return rows


def action_rows(source, target, source_ref, target_ref, initial_source, initial_target):
    rows = []
    for detail in paired_action_details(source, target, source_ref, target_ref,
                                        initial_source, initial_target):
        h = detail["horizon"]
        if h not in (24, 32):
            continue
        source_error = detail["source_prediction_source_rmse"]
        target_error = detail["target_prediction_target_rmse"]
        # Language margins compare two predictions against the SAME expert.
        source_language_margin = difference(target[:h], source_ref[:h])["rms"] - source_error
        target_language_margin = difference(source[:h], target_ref[:h])["rms"] - target_error
        rows.append({"horizon": h, "preference": detail["preference"],
            "source_correct_rmse": source_error, "target_correct_rmse": target_error,
            "initial_source_rmse": difference(initial_source[:h], source_ref[:h])["rms"],
            "initial_target_rmse": difference(initial_target[:h], target_ref[:h])["rms"],
            "source_expert_margin": detail["source_reference_margin"],
            "target_expert_margin": detail["target_reference_margin"],
            "source_language_margin": source_language_margin,
            "target_language_margin": target_language_margin,
            "both_correct": bool(detail["both_languages_prefer_own_expert"]
                                 and source_language_margin > 0 and target_language_margin > 0),
            "expert_separation_rms": detail["expert_separation_rms"],
            "language_delta_rms": detail["language_delta_rms"],
            "language_delta_cosine": detail["language_delta_cosine_with_expert_delta"]})
    return rows


def checkpoint_score(rows, *, minimum_improvement=.05, guard_relative=.10, guard_absolute=.005):
    """Fit on repair states; guard on states excluded from these repair updates.

    Guard states are NOT claimed to be unseen during earlier model training.
    No aggregate improvement may hide a regressing language on a repair state.
    """
    fit = [row for row in rows if row["split"] == "repair"]
    guard = [row for row in rows if row["split"] == "guard"]
    if not fit:
        raise ValueError("No repair evaluations.")
    fit_pass = all(row["both_correct"] and all(
        row[f"{kind}_correct_rmse"] <= row[f"initial_{kind}_rmse"] * (1 - minimum_improvement)
        for kind in ("source", "target")) for row in fit)
    groups = {}
    for row in guard:
        groups.setdefault((row["id"], row["horizon"]), []).append(row)
    regressions = []
    for (state_id, horizon), group in groups.items():
        for kind in ("source", "target"):
            current = float(np.mean([r[f"{kind}_correct_rmse"] for r in group]))
            initial = float(np.mean([r[f"initial_{kind}_rmse"] for r in group]))
            if current > initial * (1 + guard_relative) + guard_absolute:
                regressions.append({"id": state_id, "horizon": horizon, "language": kind,
                                    "initial_rmse": initial, "current_rmse": current})
    guard_pass = bool(guard) and not regressions
    return {"fit_pass": fit_pass, "guard_pass": guard_pass,
            "eligible_for_closed_loop_check": fit_pass and guard_pass,
            "fit_both_correct": sum(r["both_correct"] for r in fit), "fit_rows": len(fit),
            "fit_mean_correct_rmse": float(np.mean([
                r[f"{kind}_correct_rmse"] for r in fit for kind in ("source", "target")])),
            "guard_regressions": regressions,
            "interpretation": "Offline joint-space fit and regression checks; not task success or scene generalization."}


def repair_payload(model, initial_payload, step, metadata):
    """Keep BOTH Video and Action adapters so ordinary inference can reload it."""
    import torch

    current = model._lora_adapter_state_dict()
    if set(current) != set(initial_payload["mot_trainable"]):
        raise ValueError("Repair checkpoint lost inherited adapter tensors.")
    for name, tensor in current.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Nonfinite adapter: {name}")
        if not name.startswith("mixtures.action."):
            expected = initial_payload["mot_trainable"][name].to(dtype=tensor.dtype)
            if not torch.equal(tensor.cpu(), expected.cpu()):
                raise ValueError(f"Frozen inherited adapter changed: {name}")
    return {"format": "fastwam_lora_adapter_v1", "mot_trainable": current,
            "lora_config": copy.deepcopy(model.lora_config),
            "base_checkpoint": model.lora_base_checkpoint,
            "step": int(initial_payload["step"]) + step, "torch_dtype": str(model.torch_dtype),
            "robotwin_same_state_repair": {"format": FORMAT, "optimizer_steps": step, **metadata}}
