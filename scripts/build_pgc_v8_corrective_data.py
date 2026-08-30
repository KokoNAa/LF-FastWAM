#!/usr/bin/env python3
"""Build replay-verified PGC closed-loop corrective trajectories.

The input states are captured at actual PGC closed-loop replanning points. For
each state, this builder searches pre-correction action suffixes from the audited PGC
counterfactual demonstrations, restores the captured simulator state exactly,
and accepts a suffix only when two independent replays either lift a graspable
target or complete an articulated/fixture counterfactual goal. The resulting
LeRobot dataset is exact-state action supervision; it is never instruction
relabeling.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.libero.counterfactual_diagnostics import (  # noqa: E402
    CounterfactualEpisodeTracker,
)
from experiments.libero.libero_utils import quat2axisangle  # noqa: E402
from fastwam.datasets.pgc_libero import (  # noqa: E402
    PGC_CLOSED_LOOP_CORRECTIVE_FORMAT_V2,
    PGC_CLOSED_LOOP_CORRECTIVE_INDEX,
    append_jsonl,
    atomic_write_json,
    canonical_state_array,
    read_jsonl,
    state_sha256,
    validate_manifest_record,
)
from scripts.build_pgc_libero_data import (  # noqa: E402
    PGC_EPISODES,
    PGC_PENDING,
    PGC_PROVENANCE,
    PGC_STATE_DIR,
    _dataset_create_or_resume,
    _discard_episode,
    _existing_audits,
    _frame_from_obs,
    _get_inner_env,
    _goal_satisfied,
    _make_source_env,
    _recover_pending,
    _reset_exact_state,
)


LOGGER = logging.getLogger("pgc_v8_corrective_data")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact-state, replay-verified PGC V8 corrections."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--reference-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--episodes-per-pair", type=int, default=5)
    parser.add_argument("--max-captures-per-pair", type=int, default=80)
    parser.add_argument("--max-candidates-per-capture", type=int, default=20)
    parser.add_argument("--reference-index-stride", type=int, default=1)
    parser.add_argument("--post-lift-steps", type=int, default=8)
    parser.add_argument("--min-actions", type=int, default=12)
    parser.add_argument("--lift-threshold-m", type=float, default=0.04)
    parser.add_argument("--state-atol", type=float, default=1.0e-7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera-resolution", type=int, default=512)
    parser.add_argument(
        "--video-codec",
        choices=("h264", "hevc", "libsvtav1", "h264_nvenc"),
        default="h264",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    for name in (
        "episodes_per_pair",
        "max_captures_per_pair",
        "max_candidates_per_capture",
        "reference_index_stride",
        "min_actions",
        "fps",
        "camera_resolution",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.post_lift_steps < 0:
        parser.error("--post-lift-steps must be non-negative")
    if args.lift_threshold_m <= 0 or args.state_atol < 0:
        parser.error("lift threshold must be positive and state atol non-negative")
    # Compatibility fields consumed by the shared transactional dataset writer.
    args.settle_steps = 0
    args.keep_noops = True
    return args


def _load_manifest(path: Path, suite_name: str) -> list[dict[str, Any]]:
    records = [
        record
        for record in read_jsonl(path.expanduser().resolve())
        if str(record.get("task_suite_name", "")) == suite_name
    ]
    if not records:
        raise ValueError(f"No {suite_name!r} records in {path}.")
    for record in records:
        validate_manifest_record(record)
    pair_ids = [str(record["pair_id"]) for record in records]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("V8 manifest contains duplicate pair IDs.")
    return sorted(records, key=lambda record: int(record["task_id"]))


def _capture_digest(state: np.ndarray) -> str:
    return state_sha256(state)


def _load_captures(
    root: Path,
    records_by_pair: Mapping[str, Mapping[str, Any]],
    suite_name: str,
) -> dict[str, list[dict[str, Any]]]:
    root = root.expanduser().resolve()
    captures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for record_path in sorted(root.rglob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("format") != "pgc_libero_closed_loop_capture_v1":
            continue
        capture_id = str(record.get("capture_id", "")).strip()
        pair_id = str(record.get("pair_id", "")).strip()
        if not capture_id or capture_id in seen:
            raise ValueError(f"Invalid/duplicate capture ID at {record_path}.")
        if pair_id not in records_by_pair:
            raise ValueError(
                f"Capture {capture_id} references unknown pair {pair_id!r}."
            )
        manifest_record = records_by_pair[pair_id]
        if str(record.get("task_suite_name")) != suite_name or int(
            record.get("task_id", -1)
        ) != int(manifest_record["task_id"]):
            raise ValueError(f"Capture {capture_id} source task changed.")
        for capture_key, manifest_key in (
            ("correct_instruction", "correct_instruction"),
            ("counterfactual_instruction", "counterfactual_instruction"),
        ):
            if (
                str(record.get(capture_key, "")).strip().casefold()
                != str(manifest_record[manifest_key]).strip().casefold()
            ):
                raise ValueError(
                    f"Capture {capture_id} instruction does not match manifest."
                )
        state_path = Path(str(record.get("state_file", "")))
        if state_path.is_absolute() or ".." in state_path.parts:
            raise ValueError(f"Capture {capture_id} has unsafe state path.")
        state_path = record_path.parent / state_path
        if not state_path.is_file():
            raise FileNotFoundError(f"Missing capture state: {state_path}")
        with np.load(state_path, allow_pickle=False) as payload:
            state = np.asarray(payload["simulator_state"]).copy()
        actual_digest = _capture_digest(state)
        if actual_digest != str(record.get("capture_state_sha256", "")):
            raise ValueError(
                f"Capture state digest changed for {capture_id}: {actual_digest}."
            )
        normalized = dict(record)
        normalized["state"] = state
        normalized["record_path"] = str(record_path)
        captures[pair_id].append(normalized)
        seen.add(capture_id)
    for pair_id, values in captures.items():
        values.sort(
            key=lambda item: (
                int(item.get("policy_step", 0)),
                str(item["capture_id"]),
            )
        )
    return dict(captures)


def _episode_actions(dataset: Any, episode_index: int) -> np.ndarray:
    episode = dataset.get_episode_data(int(episode_index))
    actions = episode.get("action")
    if actions is None:
        raise KeyError(f"Reference episode {episode_index} has no actions.")
    if hasattr(actions, "detach"):
        actions = actions.detach().cpu().numpy()
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(
            f"Reference episode {episode_index} actions must be [T,7], got {actions.shape}."
        )
    return np.ascontiguousarray(actions)


def _target_object_names(record: Mapping[str, Any]) -> set[str]:
    names = {
        str(predicate[1])
        for predicate in record["counterfactual_goal_state"]
        if len(predicate) >= 2
    }
    if not names:
        raise ValueError(f"Pair {record['pair_id']} has no target object.")
    return names


def _named_site_position(inner: Any, name: str) -> np.ndarray | None:
    model = inner.sim.model
    candidates = [str(name)]
    object_sites = getattr(inner, "object_sites_dict", {})
    site = object_sites.get(str(name)) if isinstance(object_sites, Mapping) else None
    if site is not None:
        if isinstance(site, str):
            candidates.append(site)
        for attribute in ("name", "site_name"):
            value = getattr(site, attribute, None)
            if value:
                candidates.append(str(value))
    raw_names = getattr(model, "site_names", ())
    site_names = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_names
        if value is not None
    ]
    suffix_matches = [
        site_name
        for site_name in site_names
        if site_name == str(name) or site_name.endswith(str(name))
    ]
    if len(suffix_matches) == 1:
        candidates.append(suffix_matches[0])
    for candidate in dict.fromkeys(candidates):
        try:
            site_id = int(model.site_name2id(candidate))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        return np.asarray(inner.sim.data.site_xpos[site_id], dtype=np.float64)
    return None


def _entity_position(env: Any, name: str) -> np.ndarray | None:
    """Resolve movable objects, fixtures, and BDDL region sites structurally."""
    inner = _get_inner_env(env)
    body_ids = getattr(inner, "obj_body_id", {})
    body_id = body_ids.get(name) if isinstance(body_ids, Mapping) else None
    if body_id is None:
        try:
            body_id = int(inner.sim.model.body_name2id(name))
        except (AttributeError, KeyError, TypeError, ValueError):
            body_id = None
    if body_id is not None:
        return np.asarray(inner.sim.data.body_xpos[int(body_id)], dtype=np.float64)
    return _named_site_position(inner, name)


def _state_feature(
    env: Any,
    obs: Mapping[str, Any],
    target_objects: set[str],
) -> np.ndarray:
    positions = []
    missing = []
    for name in sorted(target_objects):
        position = _entity_position(env, name)
        if position is None:
            missing.append(name)
        else:
            positions.append(position)
    if missing:
        raise KeyError(
            "Corrective target entities have no LIBERO body/site position: "
            f"{missing}."
        )
    target_position = np.mean(np.stack(positions), axis=0)
    eef_position = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    eef_rotation = np.asarray(
        quat2axisangle(np.asarray(obs["robot0_eef_quat"])), dtype=np.float64
    )
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
    feature = np.concatenate(
        [
            (eef_position - target_position) / 0.20,
            eef_rotation / np.pi,
            gripper[:2] / 0.04,
        ]
    )
    if feature.shape != (8,) or not np.isfinite(feature).all():
        raise ValueError(f"Invalid corrective state feature: {feature}.")
    return feature.astype(np.float32)


def _new_tracker(
    env: Any, record: Mapping[str, Any], lift_threshold_m: float
) -> CounterfactualEpisodeTracker:
    return CounterfactualEpisodeTracker(
        env,
        source_goal_state=[list(x) for x in record["source_goal_state"]],
        counterfactual_goal_state=[
            list(x) for x in record["counterfactual_goal_state"]
        ],
        lift_threshold_m=lift_threshold_m,
    )


def _target_lifted(
    tracker: CounterfactualEpisodeTracker, target_objects: set[str]
) -> bool:
    return bool(target_objects & tracker.lifted_objects)


def _corrective_contract(
    tracker: CounterfactualEpisodeTracker,
    record: Mapping[str, Any],
) -> tuple[str, set[str]]:
    graspable = set(tracker.counterfactual_graspable_target_objects)
    if graspable:
        return "target_lift", graspable
    # Open/close/turn-on goals commonly use an articulated fixture or BDDL
    # region as predicate subject. Such entities are valid to _eval_predicate
    # even though they are deliberately absent from objects_dict/obj_body_id.
    return "counterfactual_goal", _target_object_names(record)


def _corrective_succeeded(
    tracker: CounterfactualEpisodeTracker,
    *,
    env: Any,
    done: bool,
    verification_kind: str,
    target_objects: set[str],
) -> bool:
    if verification_kind == "target_lift":
        return _target_lifted(tracker, target_objects)
    if verification_kind == "counterfactual_goal":
        # The source environment has already had its parsed goal replaced by
        # the manifest's counterfactual BDDL goal. Match the exact success
        # oracle used when the reference dataset was admitted; direct
        # _eval_predicate calls can disagree for articulated fixture states.
        return bool(_goal_satisfied(env, done=done))
    raise ValueError(
        f"Unsupported corrective verification kind: {verification_kind!r}."
    )


def _build_reference_bank(
    *,
    env: Any,
    record: Mapping[str, Any],
    reference_dataset: Any,
    reference_root: Path,
    reference_audits: list[dict[str, Any]],
    reference_settle_steps: int,
    state_atol: float,
    lift_threshold_m: float,
    index_stride: int,
    post_lift_steps: int,
) -> list[dict[str, Any]]:
    bank: list[dict[str, Any]] = []
    rejected_references: list[str] = []
    accepted_reference_episodes: set[int] = set()
    for audit in reference_audits:
        episode_index = int(audit["episode_index"])
        actions = _episode_actions(reference_dataset, episode_index)
        initial_state = np.load(
            reference_root / str(audit["source_initial_state_catalog"]),
            allow_pickle=False,
        )
        obs, _ = _reset_exact_state(
            env,
            initial_state,
            state_atol=state_atol,
            settle_steps=reference_settle_steps,
        )
        tracker = _new_tracker(env, record, lift_threshold_m)
        tracker.observe(policy_step=0)
        verification_kind, target_objects = _corrective_contract(tracker, record)
        if verification_kind == "counterfactual_goal" and _goal_satisfied(
            env, done=False
        ):
            reason = "counterfactual goal already satisfied at initial state"
            rejected_references.append(f"episode={episode_index}: {reason}")
            LOGGER.warning(
                "Rejected corrective reference episode=%d pair=%s: %s.",
                episode_index,
                record["pair_id"],
                reason,
            )
            continue
        trajectory_features: list[np.ndarray] = []
        first_target_acquisition_step: int | None = None
        first_verification_step: int | None = None
        reference_boundary_event: str | None = None
        for action_index, action in enumerate(actions):
            trajectory_features.append(_state_feature(env, obs, target_objects))
            obs, _, done, _ = env.step(np.asarray(action).tolist())
            tracker.observe(policy_step=action_index + 1)
            if first_target_acquisition_step is None and bool(
                target_objects & tracker.grasped_objects
            ):
                first_target_acquisition_step = action_index + 1
                reference_boundary_event = "grasp_contact"
            corrective_succeeded = _corrective_succeeded(
                tracker,
                env=env,
                done=bool(done),
                verification_kind=verification_kind,
                target_objects=target_objects,
            )
            if first_verification_step is None and corrective_succeeded:
                first_verification_step = action_index + 1
            if (
                verification_kind == "target_lift"
                and first_target_acquisition_step is None
                and corrective_succeeded
            ):
                # Robosuite contact-name heuristics occasionally miss a valid
                # grasp even though the object is observably lifted. The lift
                # event is the V8 task predicate and is therefore a safe,
                # conservative fallback for delimiting candidate starts.
                first_target_acquisition_step = action_index + 1
                reference_boundary_event = "target_lift_fallback"
            if verification_kind == "counterfactual_goal" and corrective_succeeded:
                first_target_acquisition_step = action_index + 1
                reference_boundary_event = "counterfactual_goal"
            if corrective_succeeded or bool(done):
                break
        if first_target_acquisition_step is None or first_verification_step is None:
            requested = (
                "lift the target"
                if verification_kind == "target_lift"
                else "achieve the counterfactual goal"
            )
            reason = f"never {requested}"
            rejected_references.append(f"episode={episode_index}: {reason}")
            LOGGER.warning(
                "Rejected corrective reference episode=%d pair=%s: %s.",
                episode_index,
                record["pair_id"],
                reason,
            )
            continue
        if reference_boundary_event == "target_lift_fallback":
            LOGGER.warning(
                "Reference episode %d uses target-lift fallback because the "
                "contact heuristic did not report a grasp (pair=%s).",
                episode_index,
                record["pair_id"],
            )
        last_start = min(first_target_acquisition_step, len(trajectory_features) - 1)
        candidate_indices = list(range(0, last_start + 1, index_stride))
        if last_start not in candidate_indices:
            candidate_indices.append(last_start)
        reference_stop_step = min(
            len(actions),
            first_verification_step + int(post_lift_steps),
        )
        accepted_reference_episodes.add(episode_index)
        for action_index in candidate_indices:
            bank.append(
                {
                    "episode_index": episode_index,
                    "action_index": int(action_index),
                    "verification_kind": verification_kind,
                    "feature_targets": tuple(sorted(target_objects)),
                    "reference_boundary_event": reference_boundary_event,
                    "feature": trajectory_features[action_index],
                    # Stop shortly after the audited corrective event. Keeping
                    # a full placement tail made each failed candidate replay
                    # hundreds of unnecessary MuJoCo steps.
                    "actions": actions[action_index:reference_stop_step],
                }
            )
    if not bank:
        raise RuntimeError(
            f"No replay-valid pre-correction reference actions for "
            f"{record['pair_id']}; rejected={rejected_references}."
        )
    suffix_lengths = [len(item["actions"]) for item in bank]
    LOGGER.info(
        "Built corrective reference bank pair=%s mode=%s episodes=%d candidates=%d "
        "suffix_steps=%d..%d.",
        record["pair_id"],
        bank[0]["verification_kind"],
        len(accepted_reference_episodes),
        len(bank),
        min(suffix_lengths),
        max(suffix_lengths),
    )
    return bank


def _rank_candidates(
    feature: np.ndarray,
    bank: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        bank,
        key=lambda item: float(np.mean((feature - np.asarray(item["feature"])) ** 2)),
    )
    selected: list[dict[str, Any]] = []
    used: dict[int, list[int]] = defaultdict(list)
    for item in ranked:
        episode_index = int(item["episode_index"])
        action_index = int(item["action_index"])
        if any(abs(action_index - previous) < 3 for previous in used[episode_index]):
            continue
        normalized = dict(item)
        normalized["feature_mse"] = float(
            np.mean((feature - np.asarray(item["feature"])) ** 2)
        )
        selected.append(normalized)
        used[episode_index].append(action_index)
        if len(selected) >= limit:
            break
    return selected


def _replay_for_corrective_success(
    *,
    env: Any,
    record: Mapping[str, Any],
    initial_state: np.ndarray,
    actions: np.ndarray,
    state_atol: float,
    lift_threshold_m: float,
    verification_kind: str,
    dataset: Any | None = None,
    stop_on_success: bool = False,
) -> dict[str, Any]:
    obs, actual_state = _reset_exact_state(
        env,
        initial_state,
        state_atol=state_atol,
        settle_steps=0,
    )
    tracker = _new_tracker(env, record, lift_threshold_m)
    tracker.observe(policy_step=0)
    contract_kind, target_objects = _corrective_contract(tracker, record)
    if contract_kind != verification_kind:
        raise ValueError(
            "Corrective replay contract changed between reference and capture: "
            f"{verification_kind!r} != {contract_kind!r}."
        )
    if verification_kind == "counterfactual_goal" and _goal_satisfied(env, done=False):
        raise ValueError(
            f"Corrective replay starts with the counterfactual goal already "
            f"satisfied for {record['pair_id']}."
        )
    verification_step = None
    used_actions = 0
    for action in np.asarray(actions, dtype=np.float32):
        if dataset is not None:
            instruction = str(record["counterfactual_instruction"]).strip()
            dataset.add_frame(
                _frame_from_obs(obs, action),
                task=[instruction, instruction, instruction, instruction],
            )
        obs, _, done, _ = env.step(action.tolist())
        used_actions += 1
        tracker.observe(policy_step=used_actions)
        if verification_step is None and _corrective_succeeded(
            tracker,
            env=env,
            done=bool(done),
            verification_kind=verification_kind,
            target_objects=target_objects,
        ):
            verification_step = used_actions
            if stop_on_success:
                break
        if bool(done):
            break
    return {
        "corrective_verified": verification_step is not None,
        "verification_kind": verification_kind,
        "verification_step": verification_step,
        "target_lifted": (
            verification_kind == "target_lift" and verification_step is not None
        ),
        "counterfactual_goal_verified": (
            verification_kind == "counterfactual_goal" and verification_step is not None
        ),
        "used_actions": used_actions,
        "actual_initial_state": actual_state,
        "tracker": tracker,
    }


def _write_v8_index(output: Path, audits: list[dict[str, Any]]) -> None:
    episodes = []
    for audit in sorted(audits, key=lambda item: int(item["episode_index"])):
        verification_kind = str(audit.get("verification_kind", "target_lift"))
        verification_step = int(
            audit.get("verification_step", audit.get("target_lift_step", 0))
        )
        episodes.append(
            {
                "episode_index": int(audit["episode_index"]),
                "pair_id": str(audit["pair_id"]),
                "capture_id": str(audit["capture_id"]),
                "capture_state_sha256": str(audit["capture_state_sha256"]),
                "recorded_action_count": int(audit["recorded_action_count"]),
                "corrective_verified": True,
                "verification_kind": verification_kind,
                "verification_step": verification_step,
                "target_lift_verified": bool(
                    audit.get(
                        "target_lift_verified", verification_kind == "target_lift"
                    )
                ),
                "counterfactual_goal_verified": bool(
                    audit.get(
                        "counterfactual_goal_verified",
                        verification_kind == "counterfactual_goal",
                    )
                ),
                "target_lift_step": (
                    verification_step if verification_kind == "target_lift" else None
                ),
                "reference_episode_index": int(audit["reference_episode_index"]),
                "reference_action_index": int(audit["reference_action_index"]),
                "reference_boundary_event": str(audit["reference_boundary_event"]),
                "reference_feature_mse": float(audit["reference_feature_mse"]),
            }
        )
    atomic_write_json(
        output / PGC_CLOSED_LOOP_CORRECTIVE_INDEX,
        {
            "format": PGC_CLOSED_LOOP_CORRECTIVE_FORMAT_V2,
            "acquisition_only": False,
            "verification_kinds": sorted(
                {str(item["verification_kind"]) for item in episodes}
            ),
            "episode_count": len(episodes),
            "episodes": episodes,
        },
    )


def _save_episode(
    *,
    dataset: Any,
    provenance: dict[str, Any],
    args: argparse.Namespace,
    record: Mapping[str, Any],
    capture: Mapping[str, Any],
    candidate: Mapping[str, Any],
    initial_state: np.ndarray,
    recorded_action_count: int,
    verification_kind: str,
    verification_step: int,
) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    episode_index = int(dataset.meta.total_episodes)
    state_relpath = PGC_STATE_DIR / f"v8_episode_{episode_index:06d}.npy"
    state_path = output / state_relpath
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(state_path, canonical_state_array(initial_state), allow_pickle=False)
    audit = {
        "episode_index": episode_index,
        "pair_id": str(record["pair_id"]),
        "source_suite": str(record["task_suite_name"]),
        "source_task_id": int(record["task_id"]),
        "source_initial_state_index": episode_index,
        "source_initial_state_catalog": str(state_relpath),
        "initial_state_sha256": state_sha256(initial_state),
        "initial_state_match": True,
        "capture_id": str(capture["capture_id"]),
        "capture_state_sha256": str(capture["capture_state_sha256"]),
        "capture_policy_step": int(capture["policy_step"]),
        "capture_replan_index": int(capture["replan_index"]),
        "capture_record": str(capture["record_path"]),
        "counterfactual_instruction": str(record["counterfactual_instruction"]).strip(),
        "counterfactual_goal_state": record["counterfactual_goal_state"],
        "counterfactual_goal_satisfied": (verification_kind == "counterfactual_goal"),
        "corrective_verified": True,
        "verification_kind": str(verification_kind),
        "verification_step": int(verification_step),
        "target_lift_verified": verification_kind == "target_lift",
        "counterfactual_goal_verified": (verification_kind == "counterfactual_goal"),
        "target_lift_step": (
            int(verification_step) if verification_kind == "target_lift" else None
        ),
        "recorded_action_count": int(recorded_action_count),
        "reference_episode_index": int(candidate["episode_index"]),
        "reference_action_index": int(candidate["action_index"]),
        "reference_boundary_event": str(candidate["reference_boundary_event"]),
        "reference_feature_mse": float(candidate["feature_mse"]),
        "collection_method": (
            "exact_closed_loop_state_plus_replay_verified_expert_suffix"
        ),
    }
    atomic_write_json(output / PGC_PENDING, audit)
    dataset.save_episode(
        raw_file_name=(
            f"{capture['capture_id']}::reference_episode="
            f"{candidate['episode_index']}::action={candidate['action_index']}"
        )
    )
    append_jsonl(output / PGC_EPISODES, audit)
    provenance["successful_episode_count"] = int(dataset.meta.total_episodes)
    atomic_write_json(output / PGC_PROVENANCE, provenance)
    (output / PGC_PENDING).unlink()
    return audit


def _quarantine_empty_incomplete_bootstrap(output: Path) -> Path | None:
    """Preserve and replace a zero-episode LeRobot bootstrap directory.

    LeRobot does not materialize tasks.jsonl / episodes.jsonl until the first
    episode is saved. If validation fails after ``create`` but before that
    first save, a later ``--resume`` cannot load the metadata. Such a directory
    is safe to replace only when both the PGC audit and provenance prove that
    zero episodes were committed. Renaming keeps the original fully
    recoverable for diagnosis.
    """
    required = (
        output / "meta/info.json",
        output / "meta/tasks.jsonl",
        output / "meta/episodes.jsonl",
    )
    missing = [path for path in required if not path.is_file()]
    if not missing:
        return None
    audits = _existing_audits(output)
    provenance_path = output / PGC_PROVENANCE
    successful_episode_count = None
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        successful_episode_count = int(provenance.get("successful_episode_count", -1))
    if audits or successful_episode_count not in (None, 0):
        raise RuntimeError(
            "Refusing to replace an incomplete V8 dataset that may contain "
            f"committed episodes: output={output} audits={len(audits)} "
            f"provenance_count={successful_episode_count} "
            f"missing={[str(path) for path in missing]}."
        )
    suffix = 0
    while True:
        label = ".bootstrap-incomplete" + ("" if suffix == 0 else f"-{suffix}")
        quarantine = output.with_name(output.name + label)
        if not quarantine.exists():
            break
        suffix += 1
    output.rename(quarantine)
    LOGGER.warning(
        "Quarantined zero-episode incomplete V8 bootstrap %s -> %s; "
        "starting a clean dataset.",
        output,
        quarantine,
    )
    return quarantine


def _main(args: argparse.Namespace) -> None:
    records = _load_manifest(args.manifest, args.suite)
    records_by_pair = {str(record["pair_id"]): record for record in records}
    captures = _load_captures(args.captures, records_by_pair, args.suite)
    if not captures:
        raise RuntimeError(
            "No failed closed-loop capture was found. Run the V5 CIS capture "
            "evaluation before building V8 data."
        )
    active_records = [
        record for record in records if str(record["pair_id"]) in captures
    ]
    active_by_pair = {str(record["pair_id"]): record for record in active_records}
    pairs_without_failed_capture = sorted(set(records_by_pair) - set(active_by_pair))
    for pair_id, record in active_by_pair.items():
        source_goals = {
            json.dumps(
                capture.get("source_goal_state"),
                sort_keys=True,
                separators=(",", ":"),
            )
            for capture in captures[pair_id]
        }
        if len(source_goals) != 1:
            raise ValueError(
                f"Closed-loop captures disagree on source goal for {pair_id}."
            )
        source_goal_state = json.loads(next(iter(source_goals)))
        if not isinstance(source_goal_state, list) or not source_goal_state:
            raise ValueError(f"Capture source goal is empty for {pair_id}.")
        record["source_goal_state"] = source_goal_state

    reference_root = args.reference_dataset.expanduser().resolve()
    reference_provenance = json.loads(
        (reference_root / PGC_PROVENANCE).read_text(encoding="utf-8")
    )
    reference_audits = read_jsonl(reference_root / PGC_EPISODES)
    references_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audit in reference_audits:
        pair_id = str(audit.get("pair_id", ""))
        if pair_id in active_by_pair:
            references_by_pair[pair_id].append(audit)
    missing_reference_pairs = sorted(set(active_by_pair) - set(references_by_pair))
    if missing_reference_pairs:
        raise RuntimeError(
            "Reference PGC dataset lacks pairs: " + ", ".join(missing_reference_pairs)
        )

    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

    reference_dataset = LeRobotDataset(repo_id=reference_root.name, root=reference_root)
    output = args.output.expanduser().resolve()
    if output.exists() and args.resume:
        _recover_pending(output)
        _quarantine_empty_incomplete_bootstrap(output)
    audits = _existing_audits(output)
    dataset, provenance = _dataset_create_or_resume(
        args, active_records, episode_audits=audits
    )
    if int(dataset.meta.total_episodes) != len(audits):
        raise RuntimeError(
            "V8 LeRobot/audit count mismatch: "
            f"{dataset.meta.total_episodes} != {len(audits)}."
        )
    provenance.update(
        {
            "collection_method": (
                "exact_closed_loop_state_plus_replay_verified_expert_suffix"
            ),
            "acquisition_only": False,
            "corrective_verification": "target_lift_or_counterfactual_goal",
            "reference_dataset": str(reference_root),
            "lift_threshold_m": float(args.lift_threshold_m),
            "post_lift_steps": int(args.post_lift_steps),
        }
    )
    atomic_write_json(output / PGC_PROVENANCE, provenance)
    _write_v8_index(output, audits)

    counts = Counter(str(audit["pair_id"]) for audit in audits)
    used_captures = {str(audit["capture_id"]) for audit in audits}
    failures: dict[str, list[str]] = defaultdict(list)
    reference_settle_steps = int(reference_provenance.get("settle_steps", 10))
    for pair_offset, record in enumerate(active_records):
        pair_id = str(record["pair_id"])
        required = int(args.episodes_per_pair)
        if counts[pair_id] >= required:
            LOGGER.info(
                "%s already has %d/%d V8 episodes.", pair_id, counts[pair_id], required
            )
            continue
        env = None
        try:
            env, _ = _make_source_env(
                record,
                resolution=args.camera_resolution,
                seed=args.seed + pair_offset,
            )
            bank = _build_reference_bank(
                env=env,
                record=record,
                reference_dataset=reference_dataset,
                reference_root=reference_root,
                reference_audits=references_by_pair[pair_id],
                reference_settle_steps=reference_settle_steps,
                state_atol=args.state_atol,
                lift_threshold_m=args.lift_threshold_m,
                index_stride=args.reference_index_stride,
                post_lift_steps=args.post_lift_steps,
            )
            ranked_captures: list[
                tuple[float, dict[str, Any], list[dict[str, Any]]]
            ] = []
            for capture in captures[pair_id][: args.max_captures_per_pair]:
                if str(capture["capture_id"]) in used_captures:
                    continue
                obs, _ = _reset_exact_state(
                    env,
                    np.asarray(capture["state"]),
                    state_atol=args.state_atol,
                    settle_steps=0,
                )
                feature = _state_feature(env, obs, set(bank[0]["feature_targets"]))
                candidates = _rank_candidates(
                    feature, bank, args.max_candidates_per_capture
                )
                if candidates:
                    ranked_captures.append(
                        (float(candidates[0]["feature_mse"]), capture, candidates)
                    )
            ranked_captures.sort(key=lambda item: (item[0], str(item[1]["capture_id"])))
            LOGGER.info(
                "Searching V8 corrections pair=%s captures=%d "
                "candidates_per_capture<=%d "
                "progress=%d/%d.",
                pair_id,
                len(ranked_captures),
                int(args.max_candidates_per_capture),
                counts[pair_id],
                required,
            )

            for capture_rank, (nearest_mse, capture, candidates) in enumerate(
                ranked_captures, start=1
            ):
                if counts[pair_id] >= required:
                    break
                LOGGER.info(
                    "Trying V8 capture pair=%s capture=%d/%d id=%s "
                    "nearest_mse=%.6g candidates=%d accepted=%d/%d.",
                    pair_id,
                    capture_rank,
                    len(ranked_captures),
                    capture["capture_id"],
                    nearest_mse,
                    len(candidates),
                    counts[pair_id],
                    required,
                )
                accepted = False
                for candidate_rank, candidate in enumerate(candidates, start=1):
                    if candidate_rank == 1 or candidate_rank % 10 == 0:
                        LOGGER.info(
                            "Validating V8 candidate pair=%s capture=%s "
                            "candidate=%d/%d reference_episode=%d action=%d.",
                            pair_id,
                            capture["capture_id"],
                            candidate_rank,
                            len(candidates),
                            int(candidate["episode_index"]),
                            int(candidate["action_index"]),
                        )
                    suffix = np.asarray(candidate["actions"], dtype=np.float32)
                    verification_kind = str(candidate["verification_kind"])
                    validation = _replay_for_corrective_success(
                        env=env,
                        record=record,
                        initial_state=np.asarray(capture["state"]),
                        actions=suffix,
                        state_atol=args.state_atol,
                        lift_threshold_m=args.lift_threshold_m,
                        verification_kind=verification_kind,
                        stop_on_success=True,
                    )
                    if not validation["corrective_verified"]:
                        continue
                    verification_step = int(validation["verification_step"])
                    recorded_count = min(
                        len(suffix),
                        max(
                            int(args.min_actions),
                            verification_step + int(args.post_lift_steps),
                        ),
                    )
                    if recorded_count < int(args.min_actions):
                        continue
                    recording = _replay_for_corrective_success(
                        env=env,
                        record=record,
                        initial_state=np.asarray(capture["state"]),
                        actions=suffix[:recorded_count],
                        state_atol=args.state_atol,
                        lift_threshold_m=args.lift_threshold_m,
                        verification_kind=verification_kind,
                        dataset=dataset,
                    )
                    if not recording["corrective_verified"]:
                        _discard_episode(dataset)
                        failures[pair_id].append(
                            f"{capture['capture_id']}: nondeterministic "
                            f"{verification_kind}"
                        )
                        continue
                    actual_recorded_count = int(recording["used_actions"])
                    if actual_recorded_count < int(args.min_actions):
                        _discard_episode(dataset)
                        failures[pair_id].append(
                            f"{capture['capture_id']}: recording ended after "
                            f"{actual_recorded_count} actions"
                        )
                        continue
                    audit = _save_episode(
                        dataset=dataset,
                        provenance=provenance,
                        args=args,
                        record=record,
                        capture=capture,
                        candidate=candidate,
                        initial_state=np.asarray(capture["state"]),
                        recorded_action_count=actual_recorded_count,
                        verification_kind=verification_kind,
                        verification_step=int(recording["verification_step"]),
                    )
                    audits.append(audit)
                    _write_v8_index(output, audits)
                    counts[pair_id] += 1
                    used_captures.add(str(capture["capture_id"]))
                    accepted = True
                    print(
                        f"SAVED V8 episode={audit['episode_index']} pair={pair_id} "
                        f"capture={capture['capture_id']} mode={verification_kind} "
                        f"actions={actual_recorded_count} "
                        f"progress={counts[pair_id]}/{required}",
                        flush=True,
                    )
                    break
                if not accepted:
                    failures[pair_id].append(
                        f"{capture['capture_id']}: no expert suffix passed "
                        "corrective verification"
                    )
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    missing = {
        pair_id: int(args.episodes_per_pair) - counts[pair_id]
        for pair_id in active_by_pair
        if counts[pair_id] < int(args.episodes_per_pair)
    }
    summary = {
        "format": PGC_CLOSED_LOOP_CORRECTIVE_FORMAT_V2,
        "output": str(output),
        "successful_episodes": len(audits),
        "corrective_pair_count": len(active_by_pair),
        "pairs_without_failed_capture": pairs_without_failed_capture,
        "episodes_per_pair": dict(sorted(counts.items())),
        "missing_episodes_per_pair": dict(sorted(missing.items())),
        "recent_rejections": {
            pair_id: messages[-10:]
            for pair_id, messages in sorted(failures.items())
            if messages
        },
    }
    atomic_write_json(output / "meta/pgc_v8_collection_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if missing and not args.allow_partial:
        raise RuntimeError(
            "PGC V8 corrective collection is incomplete. Capture more failed "
            "rollout states, then rerun with --resume."
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _main(_parse_args())
