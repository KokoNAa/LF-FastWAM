"""Numerical and cache-capture contracts for an inference-only noise probe."""

import numpy as np

from experiments.robotwin.no_eraf_probe import _actions, difference


def capture_action_cache(model, run_inference):
    """Observe the production predictor without changing its inputs or output."""
    name = "_predict_action_noise_with_cache"
    original = getattr(model, name)
    had_override = name in vars(model)
    previous_override = vars(model).get(name)
    capture = {}
    calls = 0

    def observe(*args, **kwargs):
        nonlocal calls
        if args:
            raise ValueError("Expected keyword-only calls from production infer_action.")
        calls += 1
        if calls == 1:
            capture.update({key: value for key, value in kwargs.items()
                            if key not in {"latents_action", "timestep_action"}})
        return original(**kwargs)

    setattr(model, name, observe)
    try:
        result = run_inference()
    finally:
        if had_override:
            setattr(model, name, previous_override)
        else:
            delattr(model, name)
    if not {"context", "context_mask", "video_kv_cache", "attention_mask", "video_seq_len"}.issubset(capture):
        raise ValueError("Production inference did not expose the expected no-ERAF action cache.")
    return result, capture, calls


def denoising_metrics(clean, noisy, velocity_target, correct_velocity, wrong_velocity, sigma):
    """Report velocity error and x0 estimate; low noise reveals expert actions.

    noisy/target are the actual dtype-rounded tensors supplied to the model.
    x0 is computed in float64 for diagnostics, not used to alter the rollout.
    """
    if not 0 < sigma <= 1:
        raise ValueError("sigma must be in (0,1].")
    clean, noisy, vt, cv, wv = map(_actions, (
        clean, noisy, velocity_target, correct_velocity, wrong_velocity))
    if len({x.shape for x in (clean, noisy, vt, cv, wv)}) != 1:
        raise ValueError("Denoising arrays must have identical shapes.")
    outputs = []
    for horizon in sorted({min(24, len(clean)), len(clean)}):
        sl = slice(0, horizon)
        cf = difference(cv[sl], vt[sl])["rms"]
        wf = difference(wv[sl], vt[sl])["rms"]
        cx = difference((noisy - sigma * cv)[sl], clean[sl])["rms"]
        wx = difference((noisy - sigma * wv)[sl], clean[sl])["rms"]
        outputs.append({"horizon": horizon, "correct_flow_rmse": cf,
            "wrong_flow_rmse": wf, "wrong_minus_correct_flow_rmse": wf - cf,
            "correct_x0_rmse": cx, "wrong_x0_rmse": wx,
            "wrong_minus_correct_x0_rmse": wx - cx,
            "x0_dtype_rounding_floor_rmse": difference((noisy - sigma * vt)[sl], clean[sl])["rms"],
            "language_velocity_delta_rms": difference(cv[sl], wv[sl])["rms"]})
    return outputs
