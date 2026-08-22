#!/usr/bin/env python3
"""Build V9.12 phase-rebinding supervision from passive Base rollouts.

Each input record was captured at an actual immutable-Base replanning state by
the V9 ERAF shadow observer.  This builder restores that exact MuJoCo state,
renders the two deployment cameras, and writes a one-state LeRobot episode plus
the ordinary training-only ERAF sidecar.  Privileged masks, predicates, poses,
and phase labels never enter the deployed policy.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from fastwam.datasets.pgc_libero import (  # noqa: E402
    PGC_ACTION_CONVENTION_FASTWAM,
    PGC_ACTION_REPLAY_FASTWAM_TO_LIBERO_ENV,
    PGC_ENTITY_RELATION_FORMAT,
    PGC_ENTITY_RELATION_INDEX,
    PGC_ENTITY_RELATION_PREDICATES,
    array_sha256,
    atomic_write_json,
    libero_lerobot_features,
    libero_problem_entity_catalog,
    parse_libero_goal_clauses,
    state_sha256,
)
from scripts.build_pgc_libero_data import (  # noqa: E402
    _frame_from_obs,
    _get_inner_env,
    _make_source_env,
    _reset_exact_state,
)
from scripts.build_pgc_libero_entity_relations import (  # noqa: E402
    CAMERA_NAMES,
    _body_position,
    _clause_truth,
    _empty_role_arrays,
    _entity_id,
    _entity_masks,
    _instance_geom_ids,
    _normalize_position,
    _per_view_mask_geometry,
    _region_anchor,
    _save_episode,
    _sha256,
)


LOGGER = logging.getLogger("pgc_v912_closed_loop_grounding")
CAPTURE_FORMAT = "pgc_v9_eraf_closed_loop_capture_v1"
DATA_INDEX_FORMAT = "pgc_v912_closed_loop_grounding_v1"
DATA_INDEX_PATH = Path("meta/pgc_v912_closed_loop_grounding.json")
ALLOWED_STAGES = (
    "initial_search",
    "holding",
    "released_unfinished",
    "next_clause_search",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build audited ERAF labels on immutable-Base closed-loop states."
    )
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidecar-output", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--max-per-task-stage", type=int, default=25)
    parser.add_argument(
        "--stages",
        default=",".join(ALLOWED_STAGES),
        help="Comma-separated state phases to retain.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera-resolution", type=int, default=512)
    parser.add_argument("--mask-height", type=int, default=56)
    parser.add_argument("--mask-width", type=int, default=112)
    parser.add_argument("--max-clauses", type=int, default=4)
    parser.add_argument("--state-atol", type=float, default=1.0e-7)
    parser.add_argument(
        "--workspace-min", type=float, nargs=3, default=(-0.65, -0.60, 0.70)
    )
    parser.add_argument(
        "--workspace-max", type=float, nargs=3, default=(0.65, 0.60, 1.45)
    )
    parser.add_argument(
        "--video-codec",
        choices=("h264", "hevc", "libsvtav1", "h264_nvenc"),
        default="h264",
    )
    args = parser.parse_args()
    if args.max_per_task_stage <= 0:
        parser.error("--max-per-task-stage must be positive")
    if args.max_clauses != 4:
        parser.error("V9.12 requires --max-clauses=4")
    if args.mask_height <= 0 or args.mask_width <= 0 or args.mask_width % 2:
        parser.error("mask dimensions must be positive and width must be even")
    if args.state_atol < 0:
        parser.error("--state-atol must be non-negative")
    return args


def _load_captures(args: argparse.Namespace) -> list[dict[str, Any]]:
    requested_stages = {
        value.strip() for value in str(args.stages).split(",") if value.strip()
    }
    unknown = requested_stages - set(ALLOWED_STAGES)
    if unknown or not requested_stages:
        raise ValueError(f"Invalid V9.12 capture stages: {sorted(unknown)}.")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for record_path in sorted(args.captures.expanduser().resolve().rglob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("format") != CAPTURE_FORMAT:
            continue
        capture_id = str(record.get("capture_id", "")).strip()
        if not capture_id or capture_id in seen:
            raise ValueError(f"Invalid or duplicate capture at {record_path}.")
        if str(record.get("task_suite_name")) != str(args.suite):
            continue
        if record.get("rollout_policy") != "immutable_base":
            raise ValueError(f"Capture {capture_id} was not produced by Base.")
        if record.get("action_integrity") != "selected_equals_immutable_base_exact":
            raise ValueError(
                f"Capture {capture_id} lacks the exact immutable-Base action proof."
            )
        if record.get("privileged_supervision") != "training_only":
            raise ValueError(f"Capture {capture_id} leaks privileged supervision.")
        instruction = str(record.get("correct_instruction", "")).strip()
        if not instruction or instruction.casefold() != str(
            record.get("policy_instruction", "")
        ).strip().casefold():
            raise ValueError(f"Capture {capture_id} is not a Correct rollout.")
        stage = str(record.get("online_stage_v2", ""))
        if stage not in requested_stages:
            continue
        state_relpath = Path(str(record.get("state_file", "")))
        if state_relpath.is_absolute() or ".." in state_relpath.parts:
            raise ValueError(f"Capture {capture_id} has an unsafe state path.")
        state_path = record_path.parent / state_relpath
        if not state_path.is_file():
            raise FileNotFoundError(f"Missing capture state: {state_path}.")
        with np.load(state_path, allow_pickle=False) as payload:
            state = np.asarray(payload["simulator_state"]).copy()
        if state_sha256(state) != str(record.get("capture_state_sha256", "")):
            raise ValueError(f"Capture state hash changed for {capture_id}.")
        normalized = dict(record)
        normalized["state"] = state
        normalized["record_path"] = str(record_path)
        grouped[(int(record["task_id"]), stage)].append(normalized)
        seen.add(capture_id)

    selected: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed))
    for key in sorted(grouped):
        values = sorted(
            grouped[key],
            key=lambda item: (
                int(item.get("trial_index", 0)),
                int(item.get("replan_index", 0)),
                str(item["capture_id"]),
            ),
        )
        if len(values) > int(args.max_per_task_stage):
            indices = np.sort(
                rng.choice(
                    len(values), size=int(args.max_per_task_stage), replace=False
                )
            )
            values = [values[int(index)] for index in indices]
        selected.extend(values)
    selected.sort(
        key=lambda item: (
            int(item["task_id"]),
            ALLOWED_STAGES.index(str(item["online_stage_v2"])),
            int(item.get("trial_index", 0)),
            int(item.get("replan_index", 0)),
        )
    )
    if not selected:
        raise RuntimeError("No compatible V9.12 closed-loop capture was found.")
    postgrasp = {
        "holding",
        "released_unfinished",
        "next_clause_search",
    }
    if not any(str(item["online_stage_v2"]) in postgrasp for item in selected):
        raise RuntimeError("V9.12 data has no post-grasp state.")
    return selected


def _state_labels(
    *,
    env: Any,
    obs: Mapping[str, Any],
    clauses: Sequence[Mapping[str, Any]],
    problem: Mapping[str, Any],
    phase_targets: Sequence[int],
    predicate_truth_audit: Sequence[bool],
    height: int,
    width: int,
    max_clauses: int,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> dict[str, np.ndarray]:
    if len(clauses) != len(phase_targets) or len(clauses) != len(
        predicate_truth_audit
    ):
        raise ValueError("Capture clause/phase/truth cardinality changed.")
    entities = sorted(
        {
            str(clause[role])
            for clause in clauses
            for role in ("subject", "reference")
        }
    )
    geom_ids = _instance_geom_ids(env, entities)
    masks = _entity_masks(obs, geom_ids, height=height, width=width)
    arrays = _empty_role_arrays(1, max_clauses, height, width)
    for clause_index, clause in enumerate(clauses):
        arrays["predicate_ids"][0, clause_index] = int(clause["predicate_id"])
        arrays["clause_valid"][0, clause_index] = True
        for role in ("subject", "reference"):
            entity = str(clause[role])
            arrays[f"{role}_entity_ids"][0, clause_index] = _entity_id(entity)
            mask = masks[entity]
            arrays[f"{role}_masks"][0, clause_index] = mask
            arrays[f"{role}_mask_valid"][0, clause_index] = bool(mask.any())
            visible, centers = _per_view_mask_geometry(mask)
            arrays[f"{role}_view_visible"][0, clause_index] = visible
            arrays[f"{role}_view_centers"][0, clause_index] = centers
            position, valid = _body_position(env, entity)
            arrays[f"{role}_position_valid"][0, clause_index] = bool(valid)
            if valid:
                arrays[f"{role}_positions"][0, clause_index] = _normalize_position(
                    position, workspace_min, workspace_max
                )
                if role == "subject" and clause["predicate"] in {
                    "in",
                    "on",
                    "left",
                    "right",
                    "front",
                    "back",
                }:
                    arrays["grasp_anchors"][0, clause_index] = arrays[
                        "subject_positions"
                    ][0, clause_index]
                    arrays["grasp_anchor_valid"][0, clause_index] = True
        goal_anchor, goal_valid = _region_anchor(env, clause, problem)
        arrays["goal_anchor_valid"][0, clause_index] = bool(goal_valid)
        if goal_valid:
            arrays["goal_anchors"][0, clause_index] = _normalize_position(
                goal_anchor, workspace_min, workspace_max
            )
        truth = bool(_clause_truth(env, clause["raw"]))
        if truth != bool(predicate_truth_audit[clause_index]):
            raise ValueError(
                "Restored closed-loop predicate truth differs from capture for "
                f"clause {clause_index}."
            )
        arrays["predicate_truth"][0, clause_index] = float(truth)
        arrays["predicate_truth_valid"][0, clause_index] = True
        phase = int(phase_targets[clause_index])
        if phase not in {0, 1, 2}:
            raise ValueError(f"Invalid captured phase {phase}.")
        arrays["phase_ids"][0, clause_index] = phase
        arrays["phase_valid"][0, clause_index] = True
    return arrays


def _create_dataset(args: argparse.Namespace):
    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

    output = args.output.expanduser().resolve()
    sidecar = args.sidecar_output.expanduser().resolve()
    for path in (output, sidecar):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite V9.12 output: {path}.")
    dataset = LeRobotDataset.create(
        repo_id=output.name,
        root=output,
        fps=int(args.fps),
        robot_type="panda",
        features=libero_lerobot_features(int(args.camera_resolution)),
        use_videos=True,
        video_codec=str(args.video_codec),
        is_compute_episode_stats_image=False,
    )
    return dataset, output, sidecar


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    captures = _load_captures(args)
    dataset, output, sidecar = _create_dataset(args)
    workspace_min = np.asarray(args.workspace_min, dtype=np.float32)
    workspace_max = np.asarray(args.workspace_max, dtype=np.float32)
    if np.any(workspace_max <= workspace_min):
        raise ValueError("Workspace bounds are invalid.")
    entries: list[dict[str, Any]] = []
    data_entries: list[dict[str, Any]] = []
    entity_vocabulary: dict[str, int] = {}
    stage_counts: Counter[str] = Counter()
    env = None
    active_task_id: int | None = None
    active_instruction: str | None = None
    try:
        for episode_index, capture in enumerate(captures):
            task_id = int(capture["task_id"])
            instruction = str(capture["correct_instruction"]).strip()
            if active_task_id != task_id:
                if env is not None:
                    env.close()
                record = {
                    "task_suite_name": str(args.suite),
                    "task_id": task_id,
                    "correct_instruction": instruction,
                }
                env, _ = _make_source_env(
                    record,
                    resolution=int(args.camera_resolution),
                    seed=int(args.seed) + task_id,
                    camera_segmentations="element",
                    apply_counterfactual_goal=False,
                )
                active_task_id = task_id
                active_instruction = instruction
            elif instruction.casefold() != str(active_instruction).casefold():
                raise ValueError(f"Task {task_id} instruction changed across captures.")

            state = np.asarray(capture["state"]).copy()
            obs, restored = _reset_exact_state(
                env, state, state_atol=float(args.state_atol), settle_steps=0
            )
            problem = _get_inner_env(env).parsed_problem
            clauses = parse_libero_goal_clauses(
                problem["goal_state"],
                regions=problem.get("regions", {}),
                max_clauses=int(args.max_clauses),
                instruction=instruction,
                entity_catalog=libero_problem_entity_catalog(problem),
            )
            for clause in clauses:
                for role in ("subject", "reference"):
                    entity = str(clause[role])
                    entity_vocabulary[entity] = _entity_id(entity)
            arrays = _state_labels(
                env=env,
                obs=obs,
                clauses=clauses,
                problem=problem,
                phase_targets=capture["phase_targets"],
                predicate_truth_audit=capture["predicate_truth"],
                height=int(args.mask_height),
                width=int(args.mask_width),
                max_clauses=int(args.max_clauses),
                workspace_min=workspace_min,
                workspace_max=workspace_max,
            )
            flattened = {
                f"{role}_{name}": value.copy()
                for role in ("target", "source")
                for name, value in arrays.items()
            }
            action = np.zeros(7, dtype=np.float32)
            action[-1] = 1.0
            dataset.add_frame(
                _frame_from_obs(obs, action),
                task=[instruction, instruction, instruction, instruction],
            )
            dataset.save_episode(raw_file_name=str(capture["capture_id"]))

            relpath = Path("episodes") / f"episode_{episode_index:06d}.npz"
            episode_path = sidecar / relpath
            _save_episode(episode_path, flattened)
            state_digest = state_sha256(restored)
            action_digest = array_sha256(action.reshape(1, 7))
            entries.append(
                {
                    "episode_index": episode_index,
                    "pair_id": f"{args.suite}_{task_id:02d}_closed_loop",
                    "file": str(relpath),
                    "sha256": _sha256(episode_path),
                    "state_sha256": state_digest,
                    "initial_state_sha256": state_digest,
                    "action_sha256": action_digest,
                    "frame_count": 1,
                    "source_demo": str(capture["record_path"]),
                    "source_demo_group": "immutable_base_closed_loop_replan",
                    "capture_id": str(capture["capture_id"]),
                    "online_stage_v2": str(capture["online_stage_v2"]),
                    "target_clauses": clauses,
                    "source_clauses": clauses,
                }
            )
            data_entries.append(
                {
                    "episode_index": episode_index,
                    "capture_id": str(capture["capture_id"]),
                    "task_id": task_id,
                    "instruction": instruction,
                    "online_stage_v2": str(capture["online_stage_v2"]),
                    "capture_state_sha256": str(
                        capture["capture_state_sha256"]
                    ),
                    "restored_state_sha256": state_digest,
                    "source_record": str(capture["record_path"]),
                }
            )
            stage_counts[str(capture["online_stage_v2"])] += 1
            LOGGER.info(
                "SAVED closed-loop episode=%d task=%d stage=%s progress=%d/%d",
                episode_index,
                task_id,
                capture["online_stage_v2"],
                episode_index + 1,
                len(captures),
            )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    atomic_write_json(
        output / DATA_INDEX_PATH,
        {
            "format": DATA_INDEX_FORMAT,
            "suite": str(args.suite),
            "rollout_policy": "immutable_base",
            "state_distribution": "closed_loop_replan",
            "privileged_supervision": "training_only",
            "deployment_inputs": "rgb_language_proprio",
            "episode_count": len(data_entries),
            "stage_counts": dict(stage_counts),
            "episodes": data_entries,
        },
    )
    atomic_write_json(
        sidecar / PGC_ENTITY_RELATION_INDEX,
        {
            "format": PGC_ENTITY_RELATION_FORMAT,
            "benchmark": "libero",
            "suite": str(args.suite),
            "dataset": str(output),
            "dataset_kind": "native",
            "state_distribution": "immutable_base_closed_loop_replan",
            "dataset_action_convention": PGC_ACTION_CONVENTION_FASTWAM,
            "simulator_replay_action_transform": (
                PGC_ACTION_REPLAY_FASTWAM_TO_LIBERO_ENV
            ),
            "privileged_supervision": "training_only",
            "deployment_inputs": "rgb_language_proprio",
            "max_clauses": int(args.max_clauses),
            "predicate_vocabulary": list(PGC_ENTITY_RELATION_PREDICATES),
            "entity_id_scheme": "sha256_63bit",
            "entity_vocabulary": dict(sorted(entity_vocabulary.items())),
            "camera_names": list(CAMERA_NAMES),
            "view_center_coordinate_system": "per_camera_normalized_xy",
            "mask_size": [int(args.mask_height), int(args.mask_width)],
            "workspace_min": workspace_min.tolist(),
            "workspace_max": workspace_max.tolist(),
            "episode_count": len(entries),
            "stage_counts": dict(stage_counts),
            "episodes": entries,
        },
    )
    print(
        json.dumps(
            {
                "dataset": str(output),
                "sidecar": str(sidecar),
                "episodes": len(entries),
                "stage_counts": dict(stage_counts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
