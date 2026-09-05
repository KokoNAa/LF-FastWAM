"""Differentiable production first-frame Video cache for scoped repair controls."""
from __future__ import annotations

from experiments.robotwin.same_state_repair import move_cache


def configure_adapters(model, experts):
    """Use identical fp32 adapter arithmetic in both arms; vary requires_grad only."""
    import torch
    if not experts or not set(experts) <= {"video", "action"}:
        raise ValueError("Invalid trainable experts.")
    model.eval().requires_grad_(False)
    selected = {}
    for name, p in model.mot.named_parameters():
        if name.startswith(("mixtures.video.", "mixtures.action.")) and name.endswith((".lora_A", ".lora_B")):
            p.data = p.data.to(torch.float32)
            if name.split(".")[1] in experts:
                p.requires_grad_(True)
                selected[name] = p
    if {n.split(".")[1] for n in selected} != set(experts):
        raise ValueError("Missing adapter expert.")
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in selected.values()}:
        raise ValueError("Unexpected trainable parameter.")
    return selected


def capture_inputs(model, run):
    """Capture frozen VAE/T5 outputs before all trainable Video operations."""
    from experiments.robotwin.denoising_probe import capture_action_cache
    obj, name = model.video_expert, "pre_dit"
    original = getattr(obj, name)
    existed, old = name in vars(obj), vars(obj).get(name)
    calls, inputs = [], {}

    def observe(*args, **kwargs):
        if args:
            raise ValueError("Expected production keyword-only pre_dit call.")
        calls.append(1)
        inputs.update(move_cache(kwargs, "cpu", clone=True))
        return original(**kwargs)

    setattr(obj, name, observe)
    try:
        result, cache, action_calls = capture_action_cache(model, run)
    finally:
        if existed:
            setattr(obj, name, old)
        else:
            delattr(obj, name)
    if len(calls) != 1 or inputs["x"].shape[2] != 1 or inputs["action"] is not None:
        raise ValueError("Expected exactly one action-free initial Video frame.")
    cache = {k: v for k, v in cache.items() if k != "video_kv_cache"}
    return result, {"video_inputs": inputs, "action_inputs": move_cache(cache, "cpu", clone=True)}, action_calls


def build_cache(model, captured):
    """Recompute every Video LoRA operation; never reuse detached Video K/V."""
    pre = model.video_expert.pre_dit(**captured["video_inputs"])
    action = dict(captured["action_inputs"])
    length = action["video_seq_len"]
    if pre["tokens"].shape[1] != length:
        raise ValueError("Video token count changed.")
    action["video_kv_cache"] = model.mot.prefill_video_cache(
        video_tokens=pre["tokens"], video_freqs=pre["freqs"], video_t_mod=pre["t_mod"],
        video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
        video_attention_mask=action["attention_mask"][:length, :length], return_final_hidden=False)
    return action


def predict(model, captured, noisy, time, checkpoint=True):
    import torch
    from torch.utils.checkpoint import checkpoint as ckpt
    body = getattr(type(model)._predict_action_noise_with_cache, "__wrapped__", None)
    if body is None:
        raise ValueError("Expected production no_grad predictor.")

    def forward(x, t):
        return body(model, latents_action=x, timestep_action=t, **build_cache(model, captured))

    with torch.enable_grad():
        value = ckpt(forward, noisy, time, use_reentrant=False) if checkpoint else forward(noisy, time)
    if not value.requires_grad:
        raise ValueError("Prediction detached from all adapters.")
    return value


def paired_backward(model, captured, refs, noise, time, flow_weight, anchor_weight, gain):
    """Identical loss, shared noise/time and backward order for both scope arms."""
    import torch
    from experiments.robotwin.same_state_repair import paired_velocity_losses
    scheduler = model.train_action_scheduler
    terms = {}
    for language in ("source", "target"):
        noisy = scheduler.add_noise(refs[language], noise, time)
        target = scheduler.training_target(refs[language], noise, time)
        pred = predict(model, captured[language], noisy, time)
        loss = (pred.float() - target.float()).square().mean()
        (flow_weight * loss / 2).backward()
        terms["flow_" + language] = float(loss.detach())
    end = torch.tensor([scheduler.num_train_timesteps], device=model.device, dtype=model.torch_dtype)
    predictions, targets = {}, {}
    for language in ("source", "target"):
        noisy = scheduler.add_noise(refs[language], noise, end)
        if not torch.equal(noisy, noise):
            raise ValueError("Endpoint contains expert actions.")
        predictions[language] = predict(model, captured[language], noisy, end)
        targets[language] = scheduler.training_target(refs[language], noise, end)
    parts = paired_velocity_losses(predictions, targets)
    loss = anchor_weight * (parts["common_mse"] + gain * parts["conditional_mse"])
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite anchor loss.")
    loss.backward()
    return {**terms, **{k: float(v.detach()) for k, v in parts.items()}}
