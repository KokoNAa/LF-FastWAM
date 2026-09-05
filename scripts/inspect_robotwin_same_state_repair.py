#!/usr/bin/env python3
"""Inspect a completed RoboTwin repair using saved logs/actions, without a model.

Training-window losses use different random inputs and are descriptive only.
Saved generated actions use the same evaluation seeds at each checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import fmean

import numpy as np

FORMAT = "robotwin_same_state_action_repair_v1"
REPORT_FORMAT = "robotwin_same_state_repair_diagnostics_v1"
LANGUAGES = ("source", "target")
NOTES = [
    "Training-window losses use changing noise/timesteps; they are not fixed-input denoising evaluations.",
    "Loss shares measure objective values, not gradient shares or gradient conflict.",
    "pair_mse = common_mse + conditional_mse; conditional_mse includes the factor 1/4.",
    "Generated-action metrics use matched seeds. Seed/horizon rows are not independent scenes or task successes.",
    "Recorded runtime checks are read from complete.json; checkpoints/models are not reloaded.",
]


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value in {path}: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def finite(value, label, *, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or (nonnegative and value < 0)):
        raise ValueError(f"Invalid numeric value: {label}={value}")
    return float(value)


def by_id(rows, expected, label):
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows) or set(result) != set(expected):
        raise ValueError(f"Missing/duplicate/extra states in {label}")
    return result


def paired_components(source, target, source_ref, target_ref, initial_source, initial_target):
    """Orthogonal common-action / language-difference decomposition in joint space."""
    s, t, r, q, bs, bt = [np.asarray(x, dtype=np.float64) for x in (
        source, target, source_ref, target_ref, initial_source, initial_target)]
    if len({x.shape for x in (s, t, r, q, bs, bt)}) != 1 or not all(
            np.isfinite(x).all() for x in (s, t, r, q, bs, bt)):
        raise ValueError("Nonfinite or mismatched saved action arrays.")
    mean_error = (s + t - r - q) / 2
    delta, reference_delta = t - s, q - r
    common = float(np.mean(mean_error ** 2))
    conditional = float(np.mean((delta - reference_delta) ** 2) / 4)
    pair = float((np.mean((s - r) ** 2) + np.mean((t - q) ** 2)) / 2)
    if not np.isclose(pair, common + conditional, rtol=1e-10, atol=1e-12):
        raise ValueError("Paired MSE decomposition failed.")
    energy = float(np.sum(reference_delta ** 2))
    predicted_energy = float(np.sum(delta ** 2))
    dot = float(np.sum(delta * reference_delta))
    return {
        "pair_mse": pair, "common_mse": common, "conditional_mse": conditional,
        "common_update_mse": float(np.mean(((s - bs + t - bt) / 2) ** 2)),
        "conditional_update_mse": float(np.mean(((t - bt) - (s - bs)) ** 2) / 4),
        "delta_projection": dot / energy if energy > 1e-12 else None,
        "delta_cosine": dot / np.sqrt(energy * predicted_energy)
            if energy > 1e-12 and predicted_energy > 1e-12 else None,
    }


def training_windows(training, fit_ids, arm, anchor_weight, window):
    width = min(window, max(1, len(training) // 2))
    groups = (("early", training[:width]), ("late", training[-width:]))
    rows = []
    for label, group in groups:
        for state_id in fit_ids:
            draws = [next(d for d in row["draws"] if d["id"] == state_id) for row in group]
            terms = [next(t for t in row["terms"] if t["id"] == state_id) for row in group]
            result = {"arm": arm, "id": state_id, "window": label,
                      "first_step": group[0]["step"], "last_step": group[-1]["step"],
                      "samples": len(group), "mean_time": fmean(d["time"] for d in draws),
                      "mean_scheduler_weight": fmean(d["scheduler_weight"] for d in draws)}
            objective = {}
            for kind in LANGUAGES:
                result[f"flow_{kind}_mse"] = fmean(t[f"flow_{kind}_mse"] for t in terms)
                weighted = fmean(d["scheduler_weight"] * t[f"flow_{kind}_mse"]
                                 for d, t in zip(draws, terms))
                result[f"weighted_flow_{kind}_mse"] = weighted
                anchor = (fmean(t[f"anchor_{kind}_mse"] for t in terms)
                          if arm == "paired_flow_anchor" else None)
                result[f"anchor_{kind}_mse"] = anchor
                # Each language contributes 1/(2 * number_of_repair_states).
                objective[kind] = (weighted + anchor_weight * (anchor or 0)) / (2 * len(fit_ids))
                result[f"{kind}_objective_contribution"] = objective[kind]
            total = sum(objective.values())
            result["source_objective_share"] = objective["source"] / total if total else None
            rows.append(result)
    return rows


def inspect(root, *, window=50):
    if window <= 0:
        raise ValueError("window must be positive.")
    root = Path(root).expanduser().resolve()
    inputs = {}

    def record(path):
        inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(path):
        return read_json(record(path))

    plan, summary = read(root / "plan.json"), read(root / "summary.json")
    if (plan.get("format") != FORMAT or summary.get("format") != FORMAT
            or summary.get("complete") is not True):
        raise ValueError("Expected a completed same-state Action-LoRA repair experiment.")
    states = by_id(plan["states"], [s["id"] for s in plan["states"]], "plan")
    fit_ids = [key for key, state in states.items() if state["split"] == "repair"]
    if not fit_ids or any(s["split"] not in ("repair", "guard") for s in states.values()):
        raise ValueError("Invalid repair/guard split.")
    arms = plan["arms"]
    if (not arms or len(set(arms)) != len(arms)
            or not set(arms) <= {"paired_flow", "paired_flow_anchor"}):
        raise ValueError("Invalid repair arms.")
    finite(plan["anchor_weight"], "anchor_weight", nonnegative=True)
    steps, seeds = plan["evaluation_steps"], plan["eval_seeds"]
    if (not steps or steps != sorted(set(steps)) or steps[0] != 0 or steps[-1] != plan["steps"]
            or plan["steps"] <= 0 or not seeds or len(set(seeds)) != len(seeds)):
        raise ValueError("Invalid evaluation steps/seeds.")
    expected = {(key, seed, h) for key in states for seed in seeds for h in (24, 32)}
    losses, gradients, per_state, endpoints, arms_info = [], [], [], [], {}
    matched_draws, matched_initial = {}, {}
    for arm in arms:
        folder = root / arm
        complete = read(folder / "complete.json")
        if (complete.get("format") != FORMAT or complete.get("complete") is not True
                or complete.get("arm") != arm or complete.get("steps") != plan["steps"]
                or complete.get("final_production_replay_passed") is not True
                or complete.get("frozen_parameter_versions_unchanged") is not True):
            raise ValueError(f"Incomplete repair arm: {arm}")
        path = record(folder / "training.jsonl")
        training = [json.loads(line) for line in path.read_text().splitlines()]
        if [row["step"] for row in training] != list(range(1, plan["steps"] + 1)):
            raise ValueError("Incomplete/duplicate training steps.")
        for row in training:
            draws = by_id(row["draws"], fit_ids, "training draws")
            terms = by_id(row["terms"], fit_ids, "training terms")
            for key in fit_ids:
                for field in ("time", "scheduler_weight", "noise_seed"):
                    finite(draws[key][field], field, nonnegative=True)
                for field in ("flow_source_mse", "flow_target_mse"):
                    finite(terms[key][field], field, nonnegative=True)
                if arm == "paired_flow_anchor":
                    for field in ("anchor_source_mse", "anchor_target_mse"):
                        finite(terms[key][field], field, nonnegative=True)
            if matched_draws.setdefault(row["step"], draws) != draws:
                raise ValueError("Training noise/timestep draws differ across arms.")
            if (finite(row["gradient_norm_before_clip"], "gradient norm", nonnegative=True) == 0
                    or finite(row["parameters_with_grad"], "gradient count", nonnegative=True) == 0):
                raise ValueError("Missing/zero training gradients.")
        norms = [r["gradient_norm_before_clip"] for r in training]
        gradients.append({"arm": arm, "steps": len(training), "min": min(norms),
                          "median": float(np.median(norms)), "p95": float(np.quantile(norms, .95)),
                          "max": max(norms), "fraction_clipped_at_1": fmean(n > 1 for n in norms),
                          "min_parameters_with_grad": min(r["parameters_with_grad"] for r in training),
                          "max_parameters_with_grad": max(r["parameters_with_grad"] for r in training)})
        losses.extend(training_windows(training, fit_ids, arm, plan["anchor_weight"], window))
        baseline = {}
        for step in steps:
            evaluation = read(folder / f"evaluation_{step:06d}.json")
            rows = evaluation["rows"]
            keys = [(r["id"], r["noise_seed"], r["horizon"]) for r in rows]
            if len(keys) != len(expected) or set(keys) != expected:
                raise ValueError("Incomplete/duplicate evaluations.")
            for row, key in zip(rows, keys):
                state = states[row["id"]]
                if (row["arm"] != arm or row["repair_step"] != step
                        or any(row[f] != state[f] for f in ("pair_id", "profile", "split"))
                        or not isinstance(row["both_correct"], bool)):
                    raise ValueError("Evaluation identity mismatch.")
                fields = [f"{lang}_{field}" for lang in LANGUAGES for field in (
                    "correct_rmse", "expert_margin", "language_margin")]
                for field in fields:
                    finite(row[field], field, nonnegative=field.endswith("rmse"))
                if step == 0:
                    baseline[key] = row
                    other = matched_initial.setdefault(key, row)
                    if any(abs(row[f] - other[f]) > 1e-5 for f in fields):
                        raise ValueError("Initial predictions differ across arms.")
                for kind in LANGUAGES:
                    if abs(finite(row[f"initial_{kind}_rmse"], "initial RMSE", nonnegative=True)
                           - baseline[key][f"{kind}_correct_rmse"]) > 1e-5:
                        raise ValueError("Evaluation initial RMSE changed.")
            for state_id in states:
                for horizon in (24, 32):
                    group = [r for r in rows if r["id"] == state_id and r["horizon"] == horizon]
                    metrics = {"arm": arm, "repair_step": step, "id": state_id,
                               "split": states[state_id]["split"], "horizon": horizon,
                               "seeds": len(group), "both_correct": sum(r["both_correct"] for r in group),
                               "preferences": dict(Counter(r["preference"] for r in group)),
                               **{field: fmean(r[field] for r in group) for field in fields}}
                    per_state.append(metrics)
                    if state_id not in fit_ids or step not in (0, plan["steps"]):
                        continue
                    components = []
                    for row in group:
                        seed = row["noise_seed"]

                        def actions(at_step):
                            path = record(folder / "actions" / f"repair{at_step:06d}_{state_id}_seed{seed}.npz")
                            with np.load(path, allow_pickle=False) as data:
                                result = {k: np.asarray(data[k], dtype=np.float64) for k in (
                                    "source", "target", "source_reference", "target_reference")}
                            if any(x.shape != (32, 14) or not np.isfinite(x).all() for x in result.values()):
                                raise ValueError("Invalid saved action array.")
                            return result

                        initial, current = actions(0), actions(step)
                        for kind in LANGUAGES:
                            ref = kind + "_reference"
                            if not np.array_equal(initial[ref], current[ref]):
                                raise ValueError("Saved expert reference changed.")
                            for arrays, field in ((current, f"{kind}_correct_rmse"),
                                                  (initial, f"initial_{kind}_rmse")):
                                rmse = float(np.sqrt(np.mean((arrays[kind][:horizon] - arrays[ref][:horizon]) ** 2)))
                                if abs(rmse - row[field]) > 1e-5:
                                    raise ValueError("Saved actions disagree with evaluation RMSE.")
                        components.append(paired_components(
                            current["source"][:horizon], current["target"][:horizon],
                            current["source_reference"][:horizon], current["target_reference"][:horizon],
                            initial["source"][:horizon], initial["target"][:horizon]))
                    aggregate = {field: (fmean(c[field] for c in components)
                                          if all(c[field] is not None for c in components) else None)
                                 for field in components[0]}
                    update = aggregate["common_update_mse"] + aggregate["conditional_update_mse"]
                    aggregate["conditional_update_fraction"] = aggregate["conditional_update_mse"] / update if update else None
                    endpoints.append({**metrics, **aggregate})
        arms_info[arm] = {"best": complete["best"], "last_score": complete["scores"][-1],
                          "recorded_frozen_check": complete["frozen_parameter_versions_unchanged"],
                          "recorded_production_replay_check": complete["final_production_replay_passed"]}
    return {"format": REPORT_FORMAT, "complete": True, "root": str(root),
            "initial_model": plan["initial_model"], "training_steps": plan["steps"],
            "repair_states": len(fit_ids), "guard_states": len(states) - len(fit_ids),
            "eval_seeds": seeds, "evaluation_steps": steps, "arms": arms_info,
            "training_inputs_match_across_arms": True if len(arms) > 1 else None,
            "training_windows_overlap": plan["steps"] < 2,
            "gradient_summary": gradients, "training_windows": losses,
            "per_state_evaluation": per_state, "repair_endpoint_components": endpoints,
            "input_sha256": inputs, "notes": NOTES}


def render(report):
    out = io.StringIO()
    print(f"[inspect] {report['root']}", file=out)
    print(json.dumps({key: report[key] for key in (
        "format", "complete", "initial_model", "training_steps", "repair_states", "guard_states",
        "eval_seeds", "training_inputs_match_across_arms", "training_windows_overlap")}), file=out)
    for note in NOTES:
        print(f"[note] {note}", file=out)

    def table(name, rows, fields):
        print(f"\n[{name}]", file=out)
        writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: (format(row[field], ".7g") if isinstance(row[field], float)
                                     else json.dumps(row[field], sort_keys=True) if isinstance(row[field], dict)
                                     else row[field]) for field in fields})

    table("gradient_summary", report["gradient_summary"], list(report["gradient_summary"][0]))
    table("training_windows", report["training_windows"], list(report["training_windows"][0]))
    table("repair_endpoints", report["repair_endpoint_components"], [
        "arm", "repair_step", "id", "horizon", "seeds", "both_correct", "preferences",
        "source_correct_rmse", "target_correct_rmse", "source_language_margin", "target_language_margin",
        "pair_mse", "common_mse", "conditional_mse", "delta_projection", "delta_cosine",
        "common_update_mse", "conditional_update_mse", "conditional_update_fraction"])
    regressions = [{"arm": arm, **row} for arm, info in report["arms"].items()
                   for row in info["last_score"]["guard_regressions"]]
    table("final_guard_regressions", regressions,
          ["arm", "id", "horizon", "language", "initial_rmse", "current_rmse"])
    return out.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Explicit path to a completed repair run (not a checkpoint).")
    parser.add_argument("--window", type=int, default=50, help="First/last training-window size (default: 50).")
    args = parser.parse_args()
    result = inspect(args.output, window=args.window)
    root = Path(result["root"])
    text = render(result)
    # Validate everything before writing derived reports. Source artifacts stay intact.
    (root / "repair_diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (root / "repair_diagnostics.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
