#!/usr/bin/env python3
"""Create executable cross-task PGC manifests for the four LIBERO suites.

Unlike a plain shuffle, every selected target contributes an executable
language-conditioned trajectory whose entities are present in the source
scene. Exact simulator layouts use flat-state replay; compatible layouts with
different distractors use named-joint transfer. LIBERO-Spatial is handled as a
state-grounded special case because its ten tasks intentionally share one final
predicate and differ through the initial bowl placement named by the language.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.language_interventions import (  # noqa: E402
    canonical_goal_state,
    goal_entity_names,
    problem_entity_names,
    validate_counterfactual_problem,
)
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import bddl_utils as BDDLUtils  # noqa: E402


LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)
LIBERO_CANDIDATE_SUITES = (*LIBERO_SUITES, "libero_90")


def _parse_candidate_rank_override(value: str) -> tuple[tuple[str, int], int]:
    """Parse ``suite:task_id=rank`` used to retry an unproductive donor."""
    text = str(value).strip()
    try:
        source_text, rank_text = text.rsplit("=", 1)
        suite_name, task_text = source_text.rsplit(":", 1)
        task_id = int(task_text)
        candidate_rank = int(rank_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "candidate-rank override must use suite:task_id=rank, for "
            "example libero_object:5=1"
        ) from exc
    if suite_name not in LIBERO_SUITES:
        raise argparse.ArgumentTypeError(
            f"unsupported source suite in candidate-rank override: {suite_name!r}"
        )
    if not 0 <= task_id < 10:
        raise argparse.ArgumentTypeError("candidate-rank task_id must be in [0, 9]")
    if candidate_rank < 0:
        raise argparse.ArgumentTypeError("candidate rank must be non-negative")
    return (suite_name, task_id), candidate_rank


def _task_bddl_path(task: Any) -> Path:
    return Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file


def _canonical_mapping(mapping: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(mapping, Mapping):
        return ()
    items = []
    for key, values in mapping.items():
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        items.append(
            (
                str(key).casefold(),
                tuple(sorted(str(value).casefold() for value in values)),
            )
        )
    return tuple(sorted(items))


def _ordered_mapping(mapping: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Preserve BDDL declaration order for exact flattened-state replay."""
    if not isinstance(mapping, Mapping):
        return ()
    items = []
    for key, values in mapping.items():
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        items.append(
            (
                str(key).casefold(),
                tuple(str(value).casefold() for value in values),
            )
        )
    return tuple(items)


def _flat_state_signature(problem: Mapping[str, Any]) -> tuple[Any, ...]:
    """Signature required to copy one simulator's flat state byte-for-byte."""
    return (
        str(problem.get("problem_name", "")).casefold(),
        _ordered_mapping(problem.get("objects", {})),
        _ordered_mapping(problem.get("fixtures", {})),
    )


def _semantic_transfer_compatible(
    source_problem: Mapping[str, Any], target_problem: Mapping[str, Any]
) -> bool:
    """Whether named-joint transfer keeps the same workspace/environment class."""
    return (
        str(source_problem.get("problem_name", "")).casefold()
        == str(target_problem.get("problem_name", "")).casefold()
        and _canonical_mapping(source_problem.get("fixtures", {}))
        == _canonical_mapping(target_problem.get("fixtures", {}))
    )


def _object_names(problem: Mapping[str, Any]) -> set[str]:
    mapping = problem.get("objects", {})
    if not isinstance(mapping, Mapping):
        return set()
    return {
        str(value).casefold()
        for values in mapping.values()
        for value in (values if isinstance(values, (list, tuple, set)) else [values])
    }


def _spatial_state_variant_goal(
    source: Mapping[str, Any], target: Mapping[str, Any], *, exact_scene: bool
) -> list[list[Any]] | None:
    """Allow Spatial positives whose language difference is in initial placement.

    LIBERO-Spatial deliberately shares one final predicate across its ten tasks.
    The supervision-changing signal is which initial bowl placement the instruction
    names, so a different official task state/action remains a direct positive even
    though the terminal BDDL predicate is unchanged.
    """
    if not exact_scene:
        return None
    if source["suite"] != "libero_spatial" or target["suite"] != "libero_spatial":
        return None
    source_problem = source["problem"]
    target_problem = target["problem"]
    source_goal = canonical_goal_state(source_problem.get("goal_state", []))
    target_goal = canonical_goal_state(target_problem.get("goal_state", []))
    source_initial = canonical_goal_state(source_problem.get("initial_state", []))
    target_initial = canonical_goal_state(target_problem.get("initial_state", []))
    if not target_goal or source_goal != target_goal or source_initial == target_initial:
        return None
    return [list(predicate) for predicate in target_problem["goal_state"]]


def _load_tasks(suite_names: list[str]) -> list[dict[str, Any]]:
    benchmark_dict = benchmark.get_benchmark_dict()
    entries: list[dict[str, Any]] = []
    for suite_name in suite_names:
        if suite_name not in benchmark_dict:
            raise ValueError(f"Unknown LIBERO suite: {suite_name!r}.")
        suite = benchmark_dict[suite_name]()
        for task_id in range(int(suite.n_tasks)):
            task = suite.get_task(task_id)
            bddl_path = _task_bddl_path(task)
            problem = BDDLUtils.robosuite_parse_problem(str(bddl_path))
            entries.append(
                {
                    "suite": suite_name,
                    "task_id": task_id,
                    "task": task,
                    "bddl_path": bddl_path,
                    "problem": problem,
                    "flat_state_signature": _flat_state_signature(problem),
                }
            )
    return entries


def _candidate_records(
    source: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    relaxed_scene_match: bool,
) -> list[
    tuple[
        tuple[int, int, int, str, int],
        dict[str, Any],
        list[list[Any]],
        str,
        bool,
    ]
]:
    ranked = []
    source_problem = source["problem"]
    source_initial = {
        canonical_goal_state([state])
        for state in source_problem.get("initial_state", [])
    }
    for target in candidates:
        if (target["suite"], target["task_id"]) == (
            source["suite"],
            source["task_id"],
        ):
            continue
        exact_scene = (
            target["flat_state_signature"] == source["flat_state_signature"]
        )
        semantic_transfer = _semantic_transfer_compatible(
            source_problem, target["problem"]
        )
        if not exact_scene and not semantic_transfer and not relaxed_scene_match:
            continue
        goal_changed = True
        try:
            goal = validate_counterfactual_problem(source_problem, target["problem"])
        except ValueError:
            goal = _spatial_state_variant_goal(
                source,
                target,
                exact_scene=exact_scene,
            )
            if goal is None:
                continue
            goal_changed = False
        if canonical_goal_state(goal) in source_initial:
            continue
        required_entities = goal_entity_names(goal)
        if not required_entities.issubset(problem_entity_names(source_problem)):
            continue
        same_suite = target["suite"] == source["suite"]
        transfer_mode = "flat_exact" if exact_scene else "named_joint_remap"
        object_overlap = len(
            _object_names(source_problem) & _object_names(target["problem"])
        )
        rank = (
            0
            if exact_scene and same_suite
            else 1
            if exact_scene
            else 2
            if semantic_transfer and same_suite
            else 3
            if semantic_transfer
            else 4,
            0 if goal_changed else 1,
            -object_overlap,
            str(target["suite"]),
            int(target["task_id"]),
        )
        ranked.append((rank, target, goal, transfer_mode, goal_changed))
    return sorted(ranked, key=lambda item: item[0])


def build_manifests(
    source_suites: list[str],
    candidate_suites: list[str],
    *,
    relaxed_scene_match: bool = False,
    candidate_rank_overrides: Mapping[tuple[str, int], int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    candidate_rank_overrides = dict(candidate_rank_overrides or {})
    sources = _load_tasks(source_suites)
    candidates = _load_tasks(candidate_suites)
    manifests: dict[str, list[dict[str, Any]]] = {
        suite: [] for suite in source_suites
    }
    uncovered: list[dict[str, Any]] = []
    for source in sources:
        available = _candidate_records(
            source,
            candidates,
            relaxed_scene_match=relaxed_scene_match,
        )
        if not available:
            uncovered.append(
                {
                    "task_suite_name": source["suite"],
                    "task_id": source["task_id"],
                    "instruction": source["task"].language,
                    "reason": "no executable state-compatible donor task",
                }
            )
            continue
        source_key = (str(source["suite"]), int(source["task_id"]))
        candidate_rank = int(candidate_rank_overrides.get(source_key, 0))
        if candidate_rank >= len(available):
            raise ValueError(
                "Candidate-rank override is out of range for "
                f"{source_key[0]}/{source_key[1]}: requested={candidate_rank}, "
                f"available={len(available)}."
            )
        _, target, goal, transfer_mode, goal_changed = available[candidate_rank]
        source_task = source["task"]
        target_task = target["task"]
        pair_id = (
            f"{source['suite']}_{source['task_id']:02d}_to_"
            f"{target['suite']}_{target['task_id']:02d}"
        )
        manifests[source["suite"]].append(
            {
                "pair_id": pair_id,
                "task_suite_name": source["suite"],
                "task_id": source["task_id"],
                "task_name": source_task.language,
                "scene_group": f"{source['suite']}_source_{source['task_id']:02d}",
                "correct_instruction": source_task.language,
                "shuffled_instruction": target_task.language,
                "counterfactual_instruction": target_task.language,
                "counterfactual_task_suite_name": target["suite"],
                "counterfactual_task_id": target["task_id"],
                "counterfactual_task_name": target_task.language,
                "candidate_rank": candidate_rank,
                "candidate_count": len(available),
                "counterfactual_is_executable": True,
                "counterfactual_state_replay_compatible": transfer_mode
                == "flat_exact",
                "state_transfer_mode": transfer_mode,
                "counterfactual_goal_changed": goal_changed,
                "counterfactual_state_changed": True,
                "source_bddl_file": str(source["bddl_path"]),
                "counterfactual_bddl_file": str(target["bddl_path"]),
                "counterfactual_goal_state": goal,
                "notes": (
                    "Target demo is replayed in the source environment using "
                    f"{transfer_mode}; the requested language-conditioned "
                    "outcome must succeed twice before export."
                ),
            }
        )
    return manifests, uncovered


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-suite",
        action="append",
        choices=LIBERO_SUITES,
        dest="source_suites",
    )
    parser.add_argument(
        "--candidate-suite",
        action="append",
        choices=LIBERO_CANDIDATE_SUITES,
        dest="candidate_suites",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--relaxed-scene-match",
        action="store_true",
        help=(
            "Diagnostic only: also consider donors whose fixtures differ. "
            "Normal mode already supports safe named-joint transfer across "
            "different distractor-object inventories."
        ),
    )
    parser.add_argument(
        "--candidate-rank-override",
        action="append",
        default=[],
        metavar="SUITE:TASK_ID=RANK",
        help=(
            "Select a later compatible donor for one source task. Rank 0 is "
            "the default; use rank 1 after rank 0 exhausts all demos."
        ),
    )
    args = parser.parse_args()
    source_suites = args.source_suites or list(LIBERO_SUITES)
    # LIBERO-10 is the held-out part of LIBERO-100.  A state-compatible
    # alternate goal for a held-out scene may live in LIBERO-90, so use it as
    # a donor pool even though it is not a source/evaluation suite here.
    candidate_suites = args.candidate_suites or list(LIBERO_CANDIDATE_SUITES)
    candidate_rank_overrides: dict[tuple[str, int], int] = {}
    for raw_override in args.candidate_rank_override:
        source_key, candidate_rank = _parse_candidate_rank_override(raw_override)
        if source_key in candidate_rank_overrides:
            parser.error(
                "duplicate candidate-rank override for "
                f"{source_key[0]}:{source_key[1]}"
            )
        if source_key[0] not in source_suites:
            parser.error(
                "candidate-rank override references a suite outside the "
                f"requested sources: {source_key[0]}"
            )
        candidate_rank_overrides[source_key] = candidate_rank

    manifests, uncovered = build_manifests(
        source_suites,
        candidate_suites,
        relaxed_scene_match=args.relaxed_scene_match,
        candidate_rank_overrides=candidate_rank_overrides,
    )
    combined: list[dict[str, Any]] = []
    for suite_name in source_suites:
        records = sorted(manifests[suite_name], key=lambda item: int(item["task_id"]))
        path = args.output_dir / f"{suite_name}_pgc.jsonl"
        _write_jsonl(path, records)
        combined.extend(records)
        print(f"Wrote {len(records)} PGC pairs: {path}")
    combined_path = args.output_dir / "libero_pgc_all.jsonl"
    _write_jsonl(combined_path, combined)
    report_path = args.output_dir / "pgc_manifest_coverage.json"
    report_path.write_text(
        json.dumps(
            {
                "source_suites": source_suites,
                "candidate_suites": candidate_suites,
                "candidate_rank_overrides": {
                    f"{suite_name}:{task_id}": candidate_rank
                    for (suite_name, task_id), candidate_rank in sorted(
                        candidate_rank_overrides.items()
                    )
                },
                "pair_count": len(combined),
                "state_transfer_modes": {
                    mode: sum(
                        record.get("state_transfer_mode") == mode
                        for record in combined
                    )
                    for mode in ("flat_exact", "named_joint_remap")
                },
                "uncovered_count": len(uncovered),
                "uncovered": uncovered,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Coverage report: {report_path}")
    if uncovered and not args.allow_incomplete:
        details = "; ".join(
            f"{item['task_suite_name']}/{item['task_id']}" for item in uncovered
        )
        raise RuntimeError(
            "No state-compatible counterfactual donor was found for: " + details
        )


if __name__ == "__main__":
    main()
