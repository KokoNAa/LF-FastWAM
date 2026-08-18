#!/usr/bin/env python3
"""Paired, three-seed evaluation report and admission gates for PGC V9.

Each ``--run`` value has the form ``MODEL:CONDITION:SEED:PATH``.  MODEL must
be ``Base`` or ``V9`` and CONDITION must be ``correct``, ``raw_cis``, or
``strict_cis``.  PATH can be one task-result JSON or a manager output tree.
The script pairs outcomes by seed, LIBERO task id, and trial index, computes
Wilson confidence intervals and an exact paired McNemar test, and evaluates
the V9 release gates defined by the implementation plan.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


MODELS = ("Base", "V9")
CONDITIONS = ("correct", "raw_cis", "strict_cis")


@dataclass(frozen=True)
class RunSpec:
    model: str
    condition: str
    seed: int
    path: Path


def _parse_run_spec(value: str) -> RunSpec:
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "run must be MODEL:CONDITION:SEED:PATH"
        )
    model, condition, seed_text, path_text = parts
    if model not in MODELS:
        raise argparse.ArgumentTypeError(f"MODEL must be one of {MODELS}")
    if condition not in CONDITIONS:
        raise argparse.ArgumentTypeError(
            f"CONDITION must be one of {CONDITIONS}"
        )
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("SEED must be an integer") from error
    return RunSpec(model=model, condition=condition, seed=seed, path=Path(path_text))


def _result_files(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    files = sorted(path.rglob("gpu*_task*_results.json"))
    if not files:
        raise FileNotFoundError(f"No LIBERO task result JSON files under {path}")
    return files


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a Bernoulli rate."""
    successes = int(successes)
    total = int(total)
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def mcnemar_exact_p(base_only: int, v9_only: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant pairs."""
    base_only = int(base_only)
    v9_only = int(v9_only)
    if min(base_only, v9_only) < 0:
        raise ValueError("McNemar counts must be non-negative")
    discordant = base_only + v9_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(base_only, v9_only) + 1)
    ) / (2.0 ** discordant)
    return min(1.0, 2.0 * tail)


def _graspable_targets(episode: Mapping[str, Any]) -> set[str]:
    """Return manipulable CF subjects, with old-result compatibility."""
    if "counterfactual_graspable_target_objects" in episode:
        return set(episode["counterfactual_graspable_target_objects"])
    return set(episode.get("counterfactual_target_objects", []))


def _target_event(episode: Mapping[str, Any], field: str) -> bool:
    targets = _graspable_targets(episode)
    observed = set(episode.get(field, []))
    return bool(targets & observed)


def _load_run(spec: RunSpec) -> dict[tuple[int, int, int], dict[str, Any]]:
    episodes: dict[tuple[int, int, int], dict[str, Any]] = {}
    seen_tasks: set[int] = set()
    for result_path in _result_files(spec.path):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        task_id = int(payload["task_id"])
        if task_id in seen_tasks:
            raise ValueError(
                f"Duplicate task {task_id} in {spec.model}/{spec.condition}/seed{spec.seed}"
            )
        seen_tasks.add(task_id)
        total = int(payload["total_episodes"])
        success_trials = {int(value) for value in payload.get("success_episodes", [])}
        diagnostics = payload.get("counterfactual_episode_diagnostics")
        diagnostics_by_trial: dict[int, Mapping[str, Any]] = {}
        if spec.condition != "correct":
            if not isinstance(diagnostics, list) or len(diagnostics) != total:
                raise ValueError(
                    f"{spec.condition} requires one counterfactual diagnostic per "
                    f"episode in {result_path}"
                )
            diagnostics_by_trial = {
                int(item["episode"]): item for item in diagnostics
            }
        for trial in range(total):
            diagnostic = diagnostics_by_trial.get(trial)
            success = trial in success_trials
            if diagnostic is not None:
                diagnostic_success = bool(
                    diagnostic.get("counterfactual_goal_achieved", False)
                )
                if diagnostic_success != success:
                    raise ValueError(
                        "Simulator success and counterfactual diagnostic disagree "
                        f"for seed={spec.seed} task={task_id} trial={trial}."
                    )
            key = (spec.seed, task_id, trial)
            episodes[key] = {
                "success": success,
                "target_eligible": bool(
                    diagnostic and _graspable_targets(diagnostic)
                ),
                "target_grasped": bool(
                    diagnostic and _target_event(diagnostic, "grasped_objects")
                ),
                "target_lifted": bool(
                    diagnostic and _target_event(diagnostic, "lifted_objects")
                ),
                "result_file": str(result_path),
            }
    return episodes


def _rate_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    successes = sum(bool(row["success"]) for row in rows)
    lower, upper = wilson_interval(successes, len(rows))
    eligible = [row for row in rows if row["target_eligible"]]
    grasped = sum(bool(row["target_grasped"]) for row in eligible)
    lifted = sum(bool(row["target_lifted"]) for row in eligible)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "success_rate_percent": 100.0 * successes / len(rows),
        "wilson_95_percent": [100.0 * lower, 100.0 * upper],
        "grasp_eligible_episodes": len(eligible),
        "target_grasp_rate": (grasped / len(eligible) if eligible else None),
        "target_lift_rate": (lifted / len(eligible) if eligible else None),
    }


def _paired_report(
    base: Mapping[tuple[int, int, int], Mapping[str, Any]],
    v9: Mapping[tuple[int, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    if set(base) != set(v9):
        missing_base = sorted(set(v9) - set(base))[:10]
        missing_v9 = sorted(set(base) - set(v9))[:10]
        raise ValueError(
            "Base/V9 episode keys are not paired exactly; "
            f"missing_base={missing_base}, missing_v9={missing_v9}."
        )
    both = base_only = v9_only = neither = 0
    for key in sorted(base):
        base_success = bool(base[key]["success"])
        v9_success = bool(v9[key]["success"])
        if base_success and v9_success:
            both += 1
        elif base_success:
            base_only += 1
        elif v9_success:
            v9_only += 1
        else:
            neither += 1
    return {
        "both": both,
        "base_only": base_only,
        "v9_only": v9_only,
        "neither": neither,
        "mcnemar_exact_p": mcnemar_exact_p(base_only, v9_only),
    }


def _tasks_meet_rate(
    per_task: Mapping[str, Any],
    task_ids: Iterable[int],
    minimum: float,
) -> bool:
    """Fail closed when a required LIBERO-10 task is absent from a report."""
    return all(
        str(task_id) in per_task
        and per_task[str(task_id)]["V9"]["success_rate"] >= float(minimum)
        for task_id in task_ids
    )


def build_report(
    runs: list[RunSpec],
    *,
    expected_seeds: int = 3,
    expected_tasks: int = 10,
    trials_per_task: int = 5,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[tuple[int, int, int], dict[str, Any]]] = {}
    seed_sets: dict[tuple[str, str], set[int]] = defaultdict(set)
    for spec in runs:
        group = (spec.model, spec.condition)
        seed_sets[group].add(spec.seed)
        loaded = _load_run(spec)
        if group not in grouped:
            grouped[group] = {}
        overlap = set(grouped[group]) & set(loaded)
        if overlap:
            raise ValueError(f"Duplicate paired episode keys for {group}: {sorted(overlap)[:10]}")
        grouped[group].update(loaded)

    required = {(model, condition) for model in MODELS for condition in CONDITIONS}
    missing = sorted(required - set(grouped))
    if missing:
        raise ValueError(f"Missing required Base/V9 condition runs: {missing}")
    expected_episodes = expected_seeds * expected_tasks * trials_per_task
    for group in sorted(required):
        if len(seed_sets[group]) != expected_seeds:
            raise ValueError(
                f"{group} has {len(seed_sets[group])} seeds, expected {expected_seeds}."
            )
        if len(grouped[group]) != expected_episodes:
            raise ValueError(
                f"{group} has {len(grouped[group])} episodes, expected {expected_episodes}."
            )

    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        base = grouped[("Base", condition)]
        v9 = grouped[("V9", condition)]
        base_summary = _rate_summary(base.values())
        v9_summary = _rate_summary(v9.values())
        paired = _paired_report(base, v9)
        per_task = {}
        for task_id in range(expected_tasks):
            base_rows = [row for (_, task, _), row in base.items() if task == task_id]
            v9_rows = [row for (_, task, _), row in v9.items() if task == task_id]
            per_task[str(task_id)] = {
                "Base": _rate_summary(base_rows),
                "V9": _rate_summary(v9_rows),
            }
        per_seed = {}
        for seed in sorted(seed_sets[("V9", condition)]):
            base_seed = {
                key: row for key, row in base.items() if key[0] == seed
            }
            v9_seed = {
                key: row for key, row in v9.items() if key[0] == seed
            }
            seed_base_summary = _rate_summary(base_seed.values())
            seed_v9_summary = _rate_summary(v9_seed.values())
            seed_per_task = {}
            for task_id in range(expected_tasks):
                seed_per_task[str(task_id)] = {
                    "Base": _rate_summary(
                        row
                        for (_, task, _), row in base_seed.items()
                        if task == task_id
                    ),
                    "V9": _rate_summary(
                        row
                        for (_, task, _), row in v9_seed.items()
                        if task == task_id
                    ),
                }
            per_seed[str(seed)] = {
                "Base": seed_base_summary,
                "V9": seed_v9_summary,
                "delta_v9_minus_base_pp": (
                    seed_v9_summary["success_rate_percent"]
                    - seed_base_summary["success_rate_percent"]
                ),
                "paired": _paired_report(base_seed, v9_seed),
                "per_task": seed_per_task,
            }
        conditions[condition] = {
            "Base": base_summary,
            "V9": v9_summary,
            "delta_v9_minus_base_pp": (
                v9_summary["success_rate_percent"]
                - base_summary["success_rate_percent"]
            ),
            "paired": paired,
            "per_task": per_task,
            "per_seed": per_seed,
        }

    correct = conditions["correct"]
    raw = conditions["raw_cis"]
    strict = conditions["strict_cis"]
    correct_seeds = list(correct["per_seed"].values())
    raw_seeds = list(raw["per_seed"].values())
    strict_seeds = list(strict["per_seed"].values())
    checks = {
        "correct_at_least_48_of_50_equivalent": (
            all(seed["V9"]["success_rate"] >= 0.96 for seed in correct_seeds)
        ),
        "correct_regression_at_most_2pp": (
            all(seed["delta_v9_minus_base_pp"] >= -2.0 for seed in correct_seeds)
        ),
        "raw_cis_at_least_66pct": all(
            seed["V9"]["success_rate"] >= 0.66 for seed in raw_seeds
        ),
        "strict_cis_at_least_46pct": all(
            seed["V9"]["success_rate"] >= 0.46 for seed in strict_seeds
        ),
        "strict_cis_significantly_above_base": (
            strict["delta_v9_minus_base_pp"] > 0.0
            and strict["paired"]["mcnemar_exact_p"] < 0.05
        ),
        "graspable_target_grasp_at_least_70pct": (
            all(
                seed["V9"]["target_grasp_rate"] is not None
                and seed["V9"]["target_grasp_rate"] >= 0.70
                for seed in strict_seeds
            )
        ),
        "graspable_target_lift_at_least_70pct": (
            all(
                seed["V9"]["target_lift_rate"] is not None
                and seed["V9"]["target_lift_rate"] >= 0.70
                for seed in strict_seeds
            )
        ),
        "tasks_2_4_5_6_each_at_least_40pct": all(
            _tasks_meet_rate(seed["per_task"], (2, 4, 5, 6), 0.40)
            for seed in strict_seeds
        ),
        "tasks_0_1_each_at_least_60pct": all(
            _tasks_meet_rate(seed["per_task"], (0, 1), 0.60)
            for seed in strict_seeds
        ),
    }
    return {
        "format": "pgc_v9_three_seed_evaluation_v1",
        "design": {
            "expected_seeds": expected_seeds,
            "expected_tasks": expected_tasks,
            "trials_per_task_per_seed": trials_per_task,
            "episodes_per_condition_per_model": expected_episodes,
        },
        "conditions": conditions,
        "admission_checks": checks,
        "passed": all(checks.values()),
        "notes": [
            "Base-mode numerical equivalence is audited separately because "
            "it is a tensor-level invariant.",
            "Strict-manifest coverage >=8/10 is enforced by the "
            "strict-manifest builder before this report.",
            "All rate and per-task admission thresholds must pass "
            "independently for every seed; McNemar is computed on the fully "
            "paired aggregate.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run_spec,
        required=True,
        help="MODEL:CONDITION:SEED:PATH; repeat for Base and V9 runs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--expected-tasks", type=int, default=10)
    parser.add_argument("--trials-per-task", type=int, default=5)
    args = parser.parse_args()
    if min(args.expected_seeds, args.expected_tasks, args.trials_per_task) <= 0:
        parser.error("expected seed/task/trial counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(
        args.run,
        expected_seeds=args.expected_seeds,
        expected_tasks=args.expected_tasks,
        trials_per_task=args.trials_per_task,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
