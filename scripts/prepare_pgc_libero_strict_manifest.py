#!/usr/bin/env python3
"""Build a replay-audited strict-conflict PGC manifest.

Every accepted pair is checked on five source demonstrations and five donor
demonstrations.  The requested goal must start false, the demonstrated goal
must succeed, and the opposite goal must remain false.  Failed candidates are
rejected rather than weakened; uncovered tasks are reported explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from fastwam.datasets.pgc_libero import (  # noqa: E402
    classify_strict_conflict,
    filter_libero_noops,
    iter_libero_hdf5_demos,
    libero_problem_entity_catalog,
    parse_libero_goal_clauses,
    resolve_demo_file,
    validate_strict_conflict_audit,
)
from scripts.build_pgc_libero_data import (  # noqa: E402
    _make_source_env,
    _make_target_env,
    _prepare_source_initial_state,
    _replay,
)
from scripts.prepare_pgc_libero_manifests import (  # noqa: E402
    LIBERO_CANDIDATE_SUITES,
    LIBERO_SUITES,
    _candidate_records,
    _load_tasks,
)


STRICT_TYPES = (
    "entity_swap",
    "relation_swap",
    "direction_swap",
    "articulated_state",
    "conjunction",
    "compound_conflict",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=LIBERO_SUITES, default="libero_10")
    parser.add_argument(
        "--candidate-suite",
        action="append",
        choices=LIBERO_CANDIDATE_SUITES,
        dest="candidate_suites",
    )
    parser.add_argument("--demo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--demos-per-direction", type=int, default=5)
    parser.add_argument("--max-demos-per-candidate", type=int, default=50)
    parser.add_argument("--max-candidates-per-task", type=int, default=30)
    parser.add_argument("--camera-resolution", type=int, default=64)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--state-atol", type=float, default=1.0e-7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-coverage", type=int, default=8)
    parser.add_argument("--keep-noops", action="store_true")
    args = parser.parse_args()
    if min(
        args.demos_per_direction,
        args.max_demos_per_candidate,
        args.max_candidates_per_task,
        args.camera_resolution,
    ) <= 0:
        parser.error("strict-manifest counts and resolution must be positive")
    if not 0 <= int(args.min_coverage) <= 10:
        parser.error("--min-coverage must be in [0,10]")
    return args


def _manifest_record(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    goal: list[list[Any]],
    transfer_mode: str,
    candidate_rank: int,
    candidate_count: int,
    conflict_type: str,
) -> dict[str, Any]:
    source_task = source["task"]
    target_task = target["task"]
    return {
        "pair_id": (
            f"{source['suite']}_{int(source['task_id']):02d}_to_"
            f"{target['suite']}_{int(target['task_id']):02d}"
        ),
        "task_suite_name": str(source["suite"]),
        "task_id": int(source["task_id"]),
        "task_name": str(source_task.language),
        "scene_group": (
            f"{source['suite']}_source_{int(source['task_id']):02d}"
        ),
        "correct_instruction": str(source_task.language),
        "shuffled_instruction": str(target_task.language),
        "counterfactual_instruction": str(target_task.language),
        "counterfactual_task_suite_name": str(target["suite"]),
        "counterfactual_task_id": int(target["task_id"]),
        "counterfactual_task_name": str(target_task.language),
        "candidate_rank": int(candidate_rank),
        "candidate_count": int(candidate_count),
        "counterfactual_is_executable": True,
        "counterfactual_state_replay_compatible": transfer_mode == "flat_exact",
        "state_transfer_mode": str(transfer_mode),
        "counterfactual_goal_changed": True,
        "counterfactual_state_changed": True,
        "source_bddl_file": str(source["bddl_path"]),
        "counterfactual_bddl_file": str(target["bddl_path"]),
        "counterfactual_goal_state": goal,
        "strict_conflict": True,
        "strict_conflict_type": str(conflict_type),
    }


def _source_demo_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result.update(
        {
            "counterfactual_task_suite_name": str(record["task_suite_name"]),
            "counterfactual_task_id": int(record["task_id"]),
            "counterfactual_task_name": str(record["task_name"]),
            "counterfactual_instruction": str(record["correct_instruction"]),
            "counterfactual_bddl_file": str(record["source_bddl_file"]),
        }
    )
    return result


def _make_original_env(
    record: Mapping[str, Any], *, resolution: int, seed: int
) -> Any:
    from libero.libero import benchmark
    from experiments.libero.libero_utils import get_libero_env

    suite = benchmark.get_benchmark_dict()[str(record["task_suite_name"])]()
    task = suite.get_task(int(record["task_id"]))
    env, _ = get_libero_env(task, resolution, seed, env_num=1)
    return env


def _actions(demo: Any, *, keep_noops: bool) -> Any:
    return demo.actions if keep_noops else filter_libero_noops(demo.actions)


def _audit_candidate(
    record: Mapping[str, Any],
    *,
    demo_root: Path,
    required: int,
    max_demos: int,
    keep_noops: bool,
    resolution: int,
    settle_steps: int,
    state_atol: float,
    seed: int,
) -> dict[str, Any]:
    source_demo_path = resolve_demo_file(
        demo_root, _source_demo_record(record)
    )
    target_demo_path = resolve_demo_file(demo_root, record)
    source_env = _make_original_env(record, resolution=resolution, seed=seed)
    target_goal_env, _ = _make_source_env(
        record, resolution=resolution, seed=seed + 1
    )
    target_donor_env = None
    if str(record["state_transfer_mode"]) == "named_joint_remap":
        target_donor_env, _ = _make_target_env(
            record, resolution=64, seed=seed + 2
        )
    audit = Counter()
    source_demo_keys: list[str] = []
    target_demo_keys: list[str] = []
    rejection: str | None = None
    try:
        for demo_index, demo in enumerate(iter_libero_hdf5_demos(source_demo_path)):
            if demo_index >= max_demos or audit["source_demo_source_success"] >= required:
                break
            actions = _actions(demo, keep_noops=keep_noops)
            source_result = _replay(
                source_env,
                demo.initial_state,
                actions,
                state_atol=state_atol,
                settle_steps=settle_steps,
            )
            target_result = _replay(
                target_goal_env,
                demo.initial_state,
                actions,
                state_atol=state_atol,
                settle_steps=settle_steps,
            )
            if source_result["initial_goal_satisfied"] or target_result["initial_goal_satisfied"]:
                rejection = f"{demo.group_name}: source-state goal initially true"
                break
            if not source_result["success"]:
                continue
            if target_result["success"]:
                rejection = f"{demo.group_name}: source demo reaches target goal"
                break
            audit["source_demo_source_success"] += 1
            audit["target_initially_false"] += 1
            source_demo_keys.append(str(demo.group_name))

        if rejection is None:
            for demo_index, demo in enumerate(iter_libero_hdf5_demos(target_demo_path)):
                if demo_index >= max_demos or audit["target_demo_target_success"] >= required:
                    break
                actions = _actions(demo, keep_noops=keep_noops)
                transferred_state, _ = _prepare_source_initial_state(
                    target_goal_env,
                    target_donor_env,
                    demo.initial_state,
                    record,
                    state_atol=state_atol,
                )
                target_result = _replay(
                    target_goal_env,
                    transferred_state,
                    actions,
                    state_atol=state_atol,
                    settle_steps=settle_steps,
                )
                source_result = _replay(
                    source_env,
                    transferred_state,
                    actions,
                    state_atol=state_atol,
                    settle_steps=settle_steps,
                )
                if (
                    target_result["initial_goal_satisfied"]
                    or source_result["initial_goal_satisfied"]
                ):
                    rejection = f"{demo.group_name}: donor-state goal initially true"
                    break
                if not target_result["success"]:
                    continue
                if source_result["success"]:
                    rejection = f"{demo.group_name}: target demo reaches source goal"
                    break
                audit["target_demo_target_success"] += 1
                audit["source_initially_false"] += 1
                target_demo_keys.append(str(demo.group_name))
    finally:
        for env in (source_env, target_goal_env, target_donor_env):
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    audit.setdefault("source_demo_target_success", 0)
    audit.setdefault("target_demo_source_success", 0)
    payload = {
        **dict(audit),
        "source_demo_groups": source_demo_keys,
        "target_demo_groups": target_demo_keys,
        "source_demo_file": str(source_demo_path),
        "target_demo_file": str(target_demo_path),
    }
    if rejection is not None:
        raise ValueError(rejection)
    payload.update(validate_strict_conflict_audit(payload, required_demos=required))
    return payload


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    candidate_suites = args.candidate_suites or list(LIBERO_CANDIDATE_SUITES)
    sources = _load_tasks([args.suite])
    candidates = _load_tasks(candidate_suites)
    accepted: list[dict[str, Any]] = []
    uncovered: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for source in sources:
        available = _candidate_records(
            source, candidates, relaxed_scene_match=False
        )
        prepared: list[tuple[int, str, dict[str, Any]]] = []
        for candidate_rank, (_, target, goal, transfer_mode, goal_changed) in enumerate(available):
            if not goal_changed:
                continue
            source_clauses = parse_libero_goal_clauses(
                source["problem"]["goal_state"],
                regions=source["problem"].get("regions", {}),
                instruction=str(source["task"].language),
                entity_catalog=libero_problem_entity_catalog(source["problem"]),
            )
            target_clauses = parse_libero_goal_clauses(
                target["problem"]["goal_state"],
                regions=target["problem"].get("regions", {}),
                instruction=str(target["task"].language),
                entity_catalog=libero_problem_entity_catalog(target["problem"]),
            )
            conflict_type = classify_strict_conflict(source_clauses, target_clauses)
            if conflict_type is None:
                continue
            record = _manifest_record(
                source,
                target,
                goal=goal,
                transfer_mode=transfer_mode,
                candidate_rank=candidate_rank,
                candidate_count=len(available),
                conflict_type=conflict_type,
            )
            prepared.append((candidate_rank, conflict_type, record))
        prepared.sort(
            key=lambda item: (
                category_counts[item[1]],
                STRICT_TYPES.index(item[1]),
                item[0],
            )
        )
        failures: list[str] = []
        selected = None
        for _, conflict_type, record in prepared[: args.max_candidates_per_task]:
            try:
                replay_audit = _audit_candidate(
                    record,
                    demo_root=args.demo_root.expanduser().resolve(),
                    required=args.demos_per_direction,
                    max_demos=args.max_demos_per_candidate,
                    keep_noops=args.keep_noops,
                    resolution=args.camera_resolution,
                    settle_steps=args.settle_steps,
                    state_atol=args.state_atol,
                    seed=args.seed + int(source["task_id"]) * 100,
                )
            except Exception as exc:
                failures.append(f"{record['pair_id']}: {exc}")
                continue
            selected = dict(record)
            selected["strict_replay_audit"] = replay_audit
            selected["notes"] = (
                "Strict bidirectional replay passed without initial-goal, "
                "compatible-goal, or subgoal shortcut."
            )
            accepted.append(selected)
            category_counts[conflict_type] += 1
            print(
                f"ACCEPT {record['pair_id']} type={conflict_type} "
                f"coverage={len(accepted)}/10",
                flush=True,
            )
            break
        if selected is None:
            uncovered.append(
                {
                    "task_suite_name": args.suite,
                    "task_id": int(source["task_id"]),
                    "instruction": str(source["task"].language),
                    "reason": "no candidate passed strict bidirectional replay",
                    "candidate_failures": failures,
                }
            )
    accepted.sort(key=lambda item: int(item["task_id"]))
    _write_jsonl(args.output.expanduser().resolve(), accepted)
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else args.output.expanduser().resolve().with_suffix(".coverage.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "format": "pgc_libero_strict_conflict_manifest_v1",
                "suite": args.suite,
                "coverage": len(accepted),
                "coverage_required": int(args.min_coverage),
                "strict_pair_count": len(accepted),
                "conflict_type_counts": dict(sorted(category_counts.items())),
                "uncovered": uncovered,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"STRICT_MANIFEST={args.output} COVERAGE={len(accepted)}/10")
    print(f"STRICT_REPORT={report_path}")
    if len(accepted) < int(args.min_coverage):
        raise RuntimeError(
            f"Strict coverage {len(accepted)}/10 is below the required "
            f"{args.min_coverage}/10; no main LIBERO conclusion is allowed."
        )


if __name__ == "__main__":
    main()
