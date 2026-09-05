"""Numerical contracts for deployment-sampled, same-observation language probes.

No simulator or torch imports: CPU tests exercise the scientific comparisons.
Joint-space reference alignment is a diagnostic, never a goal-success metric.
"""

from __future__ import annotations

import hashlib
import numpy as np


FORMAT = "robotwin_no_eraf_same_state_probe_v1"
CAMERAS = ("head_camera", "left_camera", "right_camera")


def typed_hash(value):
    value = np.ascontiguousarray(value)
    header = f"{value.dtype.str}|{value.shape}".encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def training_episode_ids(total, val_proportion, seed=42):
    """Match BaseLerobotDataset's episode split (its seed defaults to 42)."""
    if total < 1 or not 0 <= val_proportion < 1:
        raise ValueError("Invalid episode count or validation proportion.")
    indices = list(range(total))
    if val_proportion < 1e-6:
        return set(indices)
    np.random.default_rng(seed).shuffle(indices)
    return set(indices[:int(total * (1 - val_proportion))])


def frame_positions(length, horizon, fractions):
    """Select full-horizon windows; fractions refer to the valid-start range."""
    if horizon < 1 or length < horizon:
        raise ValueError(f"Need a full {horizon}-action window, got {length}.")
    if not fractions or any(not 0 <= f <= 1 for f in fractions):
        raise ValueError("Fractions must be in [0,1].")
    return sorted({0, *(int(round(f * (length - horizon))) for f in fractions)})


def last_equal_qpos_prefix(native, target, horizon):
    """Candidate just before command trajectories diverge; RGB is checked later."""
    native, target = _actions(native), _actions(target)
    length = min(len(native), len(target))
    if length < horizon:
        raise ValueError("No common full-horizon window.")
    differences = np.flatnonzero(np.any(native[:length] != target[:length], axis=1))
    first_difference = int(differences[0]) if len(differences) else length
    if first_difference == 0:
        return None
    return min(first_difference - 1, length - horizon)


def require_pair(native, target):
    for key in ("pair_id", "scene_seed", "initial_state_sha256",
                "source_instruction", "counterfactual_instruction"):
        if native.get(key) in (None, "") or native.get(key) != target.get(key):
            raise ValueError(f"Native/CF pair mismatch or missing field: {key}")
    if (native.get("dataset_kind") != "native"
            or target.get("dataset_kind") != "counterfactual"
            or native.get("source_goal_verified") is not True
            or target.get("counterfactual_goal_verified") is not True):
        raise ValueError("Need replay-verified native and CF expert trajectories.")
    if any(r.get("full_goal_verified") is True for r in (native, target)):
        raise ValueError("This probe uses non-final expert data only.")
    if native["source_instruction"].strip() == native["counterfactual_instruction"].strip():
        raise ValueError("Source and target language must differ.")


def observations_equal(a, b):
    return np.array_equal(a["state"], b["state"]) and all(
        np.array_equal(a[camera], b[camera]) for camera in CAMERAS
    )


def observation_hash(obs):
    return hashlib.sha256("|".join(
        typed_hash(obs[key]) for key in ("state", *CAMERAS)
    ).encode()).hexdigest()


def _actions(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 14 or value.shape[0] < 1:
        raise ValueError(f"Expected [T,14] actions, got {value.shape}.")
    if not np.isfinite(value).all():
        raise ValueError("Action array contains nonfinite values.")
    return value


def difference(a, b):
    a, b = _actions(a), _actions(b)
    if a.shape != b.shape:
        raise ValueError(f"Action shapes differ: {a.shape}, {b.shape}")
    delta = a - b
    return {"rms": float(np.sqrt(np.mean(delta ** 2))),
            "max_abs": float(np.max(np.abs(delta))),
            "first_action_rms": float(np.sqrt(np.mean(delta[0] ** 2))),
            "left_arm_rms": float(np.sqrt(np.mean(delta[:, :6] ** 2))),
            "right_arm_rms": float(np.sqrt(np.mean(delta[:, 7:13] ** 2))),
            "grippers_rms": float(np.sqrt(np.mean(delta[:, [6, 13]] ** 2)))}


def reference_metrics(source_prediction, target_prediction, own_reference, own_kind,
                      source_reference=None, target_reference=None):
    """Inputs must all use the *same released action normalizer*.

    The single-reference test is legal on every state. The two-reference test
    is legal only if both experts start at the exact recorded observation.
    The caller is responsible for enforcing that observation contract.
    """
    if own_kind not in ("native", "counterfactual"):
        raise ValueError("Unknown expert kind.")
    predictions = {"source": source_prediction, "target": target_prediction}
    correct = "source" if own_kind == "native" else "target"
    wrong = "target" if correct == "source" else "source"
    correct_error = difference(predictions[correct], own_reference)["rms"]
    wrong_error = difference(predictions[wrong], own_reference)["rms"]
    out = {
        "correct_language_reference_rmse": correct_error,
        "wrong_language_reference_rmse": wrong_error,
        "wrong_minus_correct_reference_rmse": wrong_error - correct_error,
        "language_delta": difference(target_prediction, source_prediction),
        "dual_reference": None,
    }
    if source_reference is None and target_reference is None:
        return out
    if source_reference is None or target_reference is None:
        raise ValueError("Supply both expert references or neither.")
    sr, tr = _actions(source_reference), _actions(target_reference)
    sp, tp = _actions(source_prediction), _actions(target_prediction)
    if len({x.shape for x in (sr, tr, sp, tp)}) != 1:
        raise ValueError("Two-reference action shapes differ.")
    reference_delta, language_delta = tr - sr, tp - sp
    energy = float(np.sum(reference_delta ** 2))
    predicted_energy = float(np.sum(language_delta ** 2))
    dot = float(np.sum(reference_delta * language_delta))
    separated = energy / reference_delta.size > 1e-12
    out["dual_reference"] = {
        "expert_separation_rms": float(np.sqrt(energy / reference_delta.size)),
        "expert_references_distinguishable": separated,
        "language_delta_projection_on_expert_delta": dot / energy if separated else None,
        "language_delta_cosine_with_expert_delta": (
            dot / np.sqrt(energy * predicted_energy)
            if separated and predicted_energy > 1e-12 else None
        ),
        "source_language_prefers_source_expert": (
            difference(sp, sr)["rms"] < difference(sp, tr)["rms"] if separated else None
        ),
        "target_language_prefers_target_expert": (
            difference(tp, tr)["rms"] < difference(tp, sr)["rms"] if separated else None
        ),
        "source_prediction_source_rmse": difference(sp, sr)["rms"],
        "source_prediction_target_rmse": difference(sp, tr)["rms"],
        "target_prediction_source_rmse": difference(tp, sr)["rms"],
        "target_prediction_target_rmse": difference(tp, tr)["rms"],
    }
    return out


def paired_action_details(source, target, source_reference, target_reference, base_source, base_target):
    """Inspect saved normalized actions; no additional model calls.

    Both language choices must prefer their respective reference. A target-only
    preference can instead reflect both predictions moving to the same expert.
    Coordinates are along a joint-space reference axis, not Cartesian targets.
    """
    sp, tp, sr, tr, bs, bt = map(_actions, (
        source, target, source_reference, target_reference, base_source, base_target))
    if len({x.shape for x in (sp, tp, sr, tr, bs, bt)}) != 1:
        raise ValueError("Paired action audit shapes differ.")
    names = [*(f"left_joint_{i}" for i in range(1, 7)), "left_gripper",
             *(f"right_joint_{i}" for i in range(1, 7)), "right_gripper"]
    windows = []
    for horizon in sorted({min(h, len(sp)) for h in (8, 16, 24, 32)}):
        s, t, r, q = (x[:horizon] for x in (sp, tp, sr, tr))
        dual = reference_metrics(s, t, r, "native", r, q)["dual_reference"]
        reference_delta, language_delta = q - r, t - s
        energy = float(np.sum(reference_delta ** 2))
        valid = dual["expert_references_distinguishable"]
        source_margin = dual["source_prediction_target_rmse"] - dual["source_prediction_source_rmse"]
        target_margin = dual["target_prediction_source_rmse"] - dual["target_prediction_target_rmse"]
        if not valid:
            choice = "indistinguishable_references"
        elif source_margin == 0 or target_margin == 0:
            choice = "tie"
        else:
            choice = {(True, True): "both_correct", (True, False): "both_source",
                      (False, True): "both_target", (False, False): "reversed"}[
                          source_margin > 0, target_margin > 0]
        dims = []
        for delta in (reference_delta, language_delta):
            per_dim = np.sum(delta ** 2, axis=0)
            total = float(per_dim.sum())
            dims.append([{"dimension": names[int(i)], "rms": float(np.sqrt(per_dim[i] / horizon)),
                          "energy_fraction": float(per_dim[i] / total) if total else 0.}
                         for i in np.argsort(-per_dim, kind="stable")[:3]])
        source_update, target_update = s - bs[:horizon], t - bt[:horizon]
        common_update = (source_update + target_update) / 2
        conditional_update = target_update - source_update
        windows.append({"horizon": horizon, "preference": choice, **dual,
            "both_languages_prefer_own_expert": choice == "both_correct" if valid else None,
            "source_reference_margin": source_margin, "target_reference_margin": target_margin,
            "source_coordinate_on_reference_axis": float(np.sum((s - r) * reference_delta) / energy) if valid else None,
            "target_coordinate_on_reference_axis": float(np.sum((t - r) * reference_delta) / energy) if valid else None,
            "language_delta_rms": difference(t, s)["rms"],
            "common_update_vs_base_rms": float(np.sqrt(np.mean(common_update ** 2))),
            "language_delta_update_vs_base_rms": float(np.sqrt(np.mean(conditional_update ** 2))),
            "reference_delta_top_dimensions": dims[0], "language_delta_top_dimensions": dims[1]})
    return windows
