"""Pure data-contract helpers for collecting PGC LIBERO demonstrations.

The simulator-facing entry point lives in ``scripts/build_pgc_libero_data.py``.
This module intentionally has no torch, LIBERO, or robosuite imports so that
manifest/demo validation can run on a CPU workstation before a server job is
started.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


PGC_DATA_FORMAT = "pgc_counterfactual_actions_v1"
PGC_ACTION_SUPERVISION = "executed_counterfactual_success_trajectory"
PGC_TARGET_MASK_FORMAT = "pgc_libero_element_target_masks_v1"
PGC_TARGET_MASK_INDEX = Path("meta/pgc_v7_target_masks/index.json")
PGC_COMPLETION_PHASE_FORMAT = "pgc_libero_completion_phases_v1"
PGC_COMPLETION_PHASE_INDEX = Path("meta/pgc_v5_completion_phases.json")
PGC_CLOSED_LOOP_CORRECTIVE_FORMAT = "pgc_libero_closed_loop_corrective_v1"
PGC_CLOSED_LOOP_CORRECTIVE_INDEX = Path("meta/pgc_v8_closed_loop/index.json")
PGC_ENTITY_RELATION_FORMAT = "pgc_libero_entity_relation_v1"
PGC_ROBOTWIN_ENTITY_RELATION_FORMAT = "pgc_robotwin_entity_relation_v1"
PGC_ENTITY_RELATION_INDEX = Path("index.json")
PGC_ENTITY_RELATION_WORKSPACE_MIN = (-0.8, -0.8, 0.0)
PGC_ENTITY_RELATION_WORKSPACE_MAX = (0.8, 0.8, 1.2)
PGC_ACTION_CONVENTION_FASTWAM = "fastwam_gripper_open_1_close_0"
PGC_ACTION_CONVENTION_LIBERO_ENV = (
    "libero_env_gripper_open_minus1_close_plus1"
)
PGC_ACTION_CONVENTION_ROBOTWIN_QPOS = "robotwin_dual_arm_qpos14"
PGC_ACTION_REPLAY_IDENTITY = "identity"
PGC_ACTION_REPLAY_FASTWAM_TO_LIBERO_ENV = "fastwam_to_libero_env"
PGC_ENTITY_RELATION_PREDICATES = (
    "pad",
    "in",
    "on",
    "left",
    "right",
    "front",
    "back",
    "open",
    "close",
    "turnon",
    "turnoff",
)
PGC_ENTITY_RELATION_ARRAY_NAMES = (
    "predicate_ids",
    "clause_valid",
    "subject_entity_ids",
    "reference_entity_ids",
    "subject_masks",
    "reference_masks",
    "subject_mask_valid",
    "reference_mask_valid",
    "subject_view_visible",
    "reference_view_visible",
    "subject_view_centers",
    "reference_view_centers",
    "subject_positions",
    "reference_positions",
    "subject_position_valid",
    "reference_position_valid",
    "grasp_anchors",
    "grasp_anchor_valid",
    "goal_anchors",
    "goal_anchor_valid",
    "interaction_anchors",
    "interaction_anchor_valid",
    "predicate_truth",
    "predicate_truth_valid",
    "phase_ids",
    "phase_valid",
)
LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)
PGC_STATE_TRANSFER_MODES = ("flat_exact", "named_joint_remap")


def build_pgc_pair_balanced_sample_indices(
    dataset_index_groups: list[list[int]],
) -> list[int]:
    """Interleave matched native/CF direction datasets with equal weight."""
    groups = [[int(index) for index in group] for group in dataset_index_groups]
    if not groups or any(not group for group in groups):
        raise ValueError("PGC pair-balanced sampling requires non-empty datasets.")
    flattened = [index for group in groups for index in group]
    if len(set(flattened)) != len(flattened):
        raise ValueError("PGC pair-balanced dataset frame ranges must be disjoint.")
    target_count = max(len(group) for group in groups)

    def repeat_to(group: list[int]) -> list[int]:
        repeats = (target_count + len(group) - 1) // len(group)
        return (group * repeats)[:target_count]

    balanced = [repeat_to(group) for group in groups]
    return [
        balanced[group_index][position]
        for position in range(target_count)
        for group_index in range(len(balanced))
    ]


def pgc_entity_relation_workspace_bounds(
    indices: Mapping[int, Mapping[str, Any]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the shared ERAF workspace or reject mixed coordinate frames."""
    bounds = {
        (
            tuple(
                float(value)
                for value in np.asarray(
                    index["workspace_min"], dtype=np.float32
                ).tolist()
            ),
            tuple(
                float(value)
                for value in np.asarray(
                    index["workspace_max"], dtype=np.float32
                ).tolist()
            ),
        )
        for index in indices.values()
    }
    if len(bounds) != 1:
        raise ValueError(
            "PGC v9 sidecars disagree on workspace bounds: "
            f"{sorted(bounds)!r}."
        )
    return next(iter(bounds))


def _validated_libero_actions(actions: np.ndarray) -> np.ndarray:
    """Return a finite float32 copy of an action array ending in 7 dims."""
    result = np.asarray(actions, dtype=np.float32)
    if result.ndim < 1 or result.shape[-1] != 7:
        raise ValueError(
            f"LIBERO actions must have final dimension 7, got {result.shape}."
        )
    if not np.isfinite(result).all():
        raise ValueError("LIBERO actions must be finite.")
    return np.array(result, dtype=np.float32, copy=True, order="C")


def libero_env_actions_to_fastwam(actions: np.ndarray) -> np.ndarray:
    """Convert MuJoCo gripper ``open=-1/close=+1`` to FastWAM ``1/0``.

    The first six action dimensions are already identical. This conversion
    deliberately remains affine instead of thresholding so action audits can
    detect unexpected intermediate gripper values rather than hiding them.
    """
    result = _validated_libero_actions(actions)
    result[..., -1] = (1.0 - result[..., -1]) * 0.5
    return result


def fastwam_actions_to_libero_env(actions: np.ndarray) -> np.ndarray:
    """Convert FastWAM gripper ``open=1/close=0`` to MuJoCo ``-1/+1``."""
    result = _validated_libero_actions(actions)
    result[..., -1] = 1.0 - 2.0 * result[..., -1]
    return result


def detect_pgc_completion_phase(
    actions: np.ndarray,
    *,
    close_threshold: float = 0.0,
) -> dict[str, int | None]:
    """Locate grasp-close and optional release transitions in a LIBERO demo.

    LIBERO actions use the seventh dimension for the binary gripper command.
    The collected PGC trajectory is aligned one-to-one with these commands, so
    the first positive command is a conservative boundary between target
    acquisition and post-grasp completion. A later non-positive command, when
    present, marks release. Successful collection may stop before a release is
    recorded; in that case ``release_open_step`` is intentionally ``None``.
    """
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7 or actions.shape[0] <= 0:
        raise ValueError(
            "PGC completion phases require non-empty LIBERO actions [T,7], "
            f"got {actions.shape}."
        )
    if not np.isfinite(actions).all():
        raise ValueError("PGC completion actions contain NaN or infinity.")
    closed = np.asarray(actions[:, -1] > float(close_threshold), dtype=np.bool_)
    close_indices = np.flatnonzero(closed)
    if close_indices.size == 0:
        raise ValueError("PGC successful trajectory has no gripper-close command.")
    grasp_close_step = int(close_indices[0])
    release_indices = np.flatnonzero(~closed[grasp_close_step + 1 :])
    release_open_step = (
        None
        if release_indices.size == 0
        else int(grasp_close_step + 1 + release_indices[0])
    )
    return {
        "grasp_close_step": grasp_close_step,
        "release_open_step": release_open_step,
    }


def load_pgc_completion_phase_index(
    dataset_root: str | Path,
) -> dict[int, dict[str, Any]]:
    """Load audited grasp-to-completion boundaries for a PGC action dataset."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    index_path = dataset_root / PGC_COMPLETION_PHASE_INDEX
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing PGC V5 completion-phase sidecar: {index_path}. Run "
            "scripts/build_pgc_completion_phases.py first."
        )
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("format") != PGC_COMPLETION_PHASE_FORMAT:
        raise ValueError(
            f"Unsupported PGC completion-phase format at {index_path}: "
            f"{payload.get('format')!r}."
        )
    audited_pairs = load_pgc_episode_language_pairs(dataset_root)
    records = payload.get("episodes")
    if not isinstance(records, list) or len(records) != len(audited_pairs):
        raise ValueError(
            "PGC completion-phase episode count does not match action audit: "
            f"phases={len(records or [])} audits={len(audited_pairs)}."
        )
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("PGC completion-phase records must be objects.")
        episode_index = int(record.get("episode_index", -1))
        if episode_index in indexed or episode_index not in audited_pairs:
            raise ValueError(
                f"Invalid or duplicate PGC completion episode {episode_index}."
            )
        action_count = int(record.get("action_count", 0))
        grasp_close_step = int(record.get("grasp_close_step", -1))
        release_raw = record.get("release_open_step")
        release_open_step = None if release_raw is None else int(release_raw)
        if action_count <= 0 or not 0 <= grasp_close_step < action_count:
            raise ValueError(
                f"Invalid completion boundary for episode {episode_index}: "
                f"count={action_count} close={grasp_close_step}."
            )
        if release_open_step is not None and not (
            grasp_close_step < release_open_step < action_count
        ):
            raise ValueError(
                f"Invalid release boundary for episode {episode_index}: "
                f"close={grasp_close_step} release={release_open_step} "
                f"count={action_count}."
            )
        pair = audited_pairs[episode_index]
        if str(record.get("pair_id", "")) != str(pair["pair_id"]):
            raise ValueError(
                f"PGC completion pair mismatch for episode {episode_index}."
            )
        normalized = dict(record)
        normalized.update(
            {
                "episode_index": episode_index,
                "action_count": action_count,
                "grasp_close_step": grasp_close_step,
                "release_open_step": release_open_step,
            }
        )
        indexed[episode_index] = normalized
    return indexed


def load_pgc_closed_loop_corrective_index(
    dataset_root: str | Path,
) -> dict[int, dict[str, Any]]:
    """Load the audited V8 closed-loop target-acquisition contract.

    V8 data are not relabeled offline demonstrations.  Every episode must
    begin at an exact simulator state captured from a deployed PGC rollout,
    and its recorded action suffix must have been replay-verified to lift the
    requested counterfactual target from that exact state.  Keeping this
    contract in a sidecar lets the ordinary LeRobot action/video schema remain
    unchanged while preventing unverified rollout states from entering the
    optimizer.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    index_path = dataset_root / PGC_CLOSED_LOOP_CORRECTIVE_INDEX
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing PGC V8 closed-loop corrective index: {index_path}. "
            "Run scripts/build_pgc_v8_corrective_data.py first."
        )
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("format") != PGC_CLOSED_LOOP_CORRECTIVE_FORMAT:
        raise ValueError(
            f"Unsupported PGC V8 corrective format at {index_path}: "
            f"{payload.get('format')!r}."
        )
    if payload.get("acquisition_only") is not True:
        raise ValueError(
            "PGC V8 corrective data must declare acquisition_only=true."
        )
    audited_pairs = load_pgc_episode_language_pairs(dataset_root)
    if int(payload.get("episode_count", -1)) != len(audited_pairs):
        raise ValueError(
            "PGC V8 corrective episode_count does not match action audits."
        )
    episode_audits = {
        int(audit["episode_index"]): audit
        for audit in read_jsonl(dataset_root / "meta/pgc_episodes.jsonl")
    }
    records = payload.get("episodes")
    if not isinstance(records, list) or len(records) != len(audited_pairs):
        raise ValueError(
            "PGC V8 corrective episode count does not match action audit: "
            f"index={len(records or [])} audits={len(audited_pairs)}."
        )
    indexed: dict[int, dict[str, Any]] = {}
    capture_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("PGC V8 corrective records must be objects.")
        episode_index = int(record.get("episode_index", -1))
        if episode_index in indexed or episode_index not in audited_pairs:
            raise ValueError(
                f"Invalid or duplicate PGC V8 episode {episode_index}."
            )
        capture_id = str(record.get("capture_id", "")).strip()
        pair_id = str(record.get("pair_id", "")).strip()
        state_digest = str(record.get("capture_state_sha256", "")).strip().lower()
        action_count = int(record.get("recorded_action_count", 0))
        reference_boundary_event = str(
            record.get("reference_boundary_event", "")
        )
        if not capture_id or capture_id in capture_ids:
            raise ValueError(
                f"PGC V8 episode {episode_index} has an invalid/duplicate capture_id."
            )
        if not re.fullmatch(r"[0-9a-f]{64}", state_digest):
            raise ValueError(
                f"PGC V8 episode {episode_index} has no valid state SHA256."
            )
        if action_count <= 0:
            raise ValueError(
                f"PGC V8 episode {episode_index} has no corrective actions."
            )
        if record.get("target_lift_verified") is not True:
            raise ValueError(
                f"PGC V8 episode {episode_index} was not target-lift verified."
            )
        if reference_boundary_event not in {
            "grasp_contact",
            "target_lift_fallback",
        }:
            raise ValueError(
                f"PGC V8 episode {episode_index} has no valid reference "
                "acquisition boundary."
            )
        pair = audited_pairs[episode_index]
        if pair_id != str(pair["pair_id"]):
            raise ValueError(
                f"PGC V8 pair mismatch for episode {episode_index}: "
                f"{pair_id!r} != {pair['pair_id']!r}."
            )
        audit = episode_audits.get(episode_index)
        if audit is None or (
            str(audit.get("capture_id", "")) != capture_id
            or str(audit.get("capture_state_sha256", "")).lower()
            != state_digest
            or int(audit.get("recorded_action_count", 0)) != action_count
            or audit.get("target_lift_verified") is not True
            or str(audit.get("reference_boundary_event", ""))
            != reference_boundary_event
        ):
            raise ValueError(
                f"PGC V8 index/audit mismatch for episode {episode_index}."
            )
        state_relpath = Path(
            str(audit.get("source_initial_state_catalog", ""))
        )
        if state_relpath.is_absolute() or ".." in state_relpath.parts:
            raise ValueError(
                f"PGC V8 episode {episode_index} has an unsafe state path."
            )
        state_path = dataset_root / state_relpath
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Missing PGC V8 captured simulator state: {state_path}."
            )
        expected_initial_digest = str(
            audit.get("initial_state_sha256", "")
        ).lower()
        if state_digest != expected_initial_digest:
            raise ValueError(
                "PGC V8 captured and recorded initial-state hashes differ for "
                f"episode {episode_index}."
            )
        actual_initial_digest = state_sha256(
            np.load(state_path, allow_pickle=False)
        )
        if expected_initial_digest != actual_initial_digest:
            raise ValueError(
                "PGC V8 captured simulator state hash changed for episode "
                f"{episode_index}."
            )
        capture_ids.add(capture_id)
        indexed[episode_index] = dict(record)
    return indexed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def goal_subject(goal_state: Sequence[Sequence[Any]]) -> str:
    """Return the manipulated entity from a single-object LIBERO goal.

    PGC currently collects one-object manipulation goals. Keeping this parser
    strict prevents a silently wrong segmentation label when a future suite
    introduces a compound predicate or omits the manipulated entity.
    """
    if not isinstance(goal_state, Sequence) or isinstance(goal_state, (str, bytes)):
        raise ValueError("LIBERO goal_state must be a sequence of predicates.")
    subjects = {
        str(predicate[1]).strip()
        for predicate in goal_state
        if isinstance(predicate, Sequence)
        and not isinstance(predicate, (str, bytes))
        and len(predicate) >= 2
        and str(predicate[1]).strip()
    }
    if len(subjects) != 1:
        raise ValueError(
            "PGC target-mask supervision requires exactly one manipulated "
            f"goal entity, got {sorted(subjects)} from {goal_state!r}."
        )
    return next(iter(subjects))


def _normalize_libero_predicate(value: Any) -> str:
    predicate = re.sub(r"[^a-z]", "", str(value).strip().casefold())
    aliases = {
        "inside": "in",
        "contains": "in",
        "leftof": "left",
        "rightof": "right",
        "infrontof": "front",
        "frontof": "front",
        "behind": "back",
        "backof": "back",
        "turnon": "turnon",
        "turnoff": "turnoff",
    }
    return aliases.get(predicate, predicate)


def resolve_libero_region_target(
    value: Any,
    regions: Mapping[str, Any],
) -> tuple[str, str | None]:
    """Resolve a BDDL region through ``regions[name]['target']``.

    The original region name is returned separately because it defines the
    goal anchor, while the resolved target is the reference fixture/entity.
    String suffix guessing is intentionally forbidden.
    """
    name = str(value).strip()
    region = regions.get(name)
    if not isinstance(region, Mapping):
        return name, None
    target = region.get("target")
    if isinstance(target, Sequence) and not isinstance(target, (str, bytes)):
        target = next((item for item in target if str(item).strip()), None)
    target_name = "" if target is None else str(target).strip()
    if not target_name:
        raise ValueError(f"LIBERO region {name!r} has no structural target.")
    return target_name, name


def libero_problem_entity_catalog(
    problem: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Merge structurally declared LIBERO objects and fixtures by type."""
    result: dict[str, list[str]] = {}
    for section_name in ("objects", "fixtures"):
        section = problem.get(section_name, {})
        if not isinstance(section, Mapping):
            continue
        for declared_type, raw_instances in section.items():
            instances = (
                raw_instances
                if isinstance(raw_instances, Sequence)
                and not isinstance(raw_instances, (str, bytes))
                else [raw_instances]
            )
            bucket = result.setdefault(str(declared_type), [])
            for raw_instance in instances:
                instance = str(raw_instance).strip()
                if instance and instance not in bucket:
                    bucket.append(instance)
    return result


def _instruction_spatial_relation(instruction: str | None) -> str | None:
    """Extract an explicit directional relation from task language.

    LIBERO represents directional placement goals as ``on(subject, region)``.
    The reference entity must still come from the structural BDDL
    ``regions[name]['target']`` mapping; language is used only to label which
    directional relation that already-resolved region denotes.
    """
    normalized = re.sub(
        r"\s+", " ", str(instruction or "").strip().casefold()
    )
    patterns = (
        (r"\b(?:to (?:the )?)?left of\b", "left"),
        (r"\b(?:to (?:the )?)?right of\b", "right"),
        (r"\b(?:in |to (?:the )?)?front of\b", "front"),
        (r"\b(?:behind|(?:to (?:the )?)?back of)\b", "back"),
    )
    matches = [relation for pattern, relation in patterns if re.search(pattern, normalized)]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous directional relations in instruction {instruction!r}: "
            f"{matches}."
        )
    return matches[0] if matches else None


def _instruction_reference_entity(
    instruction: str | None,
    *,
    entity_catalog: Mapping[str, Any] | None,
    subject: str,
) -> str | None:
    """Resolve a directional reference from declared BDDL entities.

    The catalog constrains matching to actual scene instances, so this never
    invents an entity from a region-name substring.  Suffix aliases handle
    LIBERO names such as ``flat_stove_1`` -> ``stove``.
    """
    if not entity_catalog:
        return None
    normalized_instruction = re.sub(
        r"[^a-z0-9]+", " ", str(instruction or "").casefold()
    ).strip()
    candidates: list[tuple[int, str]] = []
    for declared_type, raw_instances in entity_catalog.items():
        instances = (
            raw_instances
            if isinstance(raw_instances, Sequence)
            and not isinstance(raw_instances, (str, bytes))
            else [raw_instances]
        )
        for raw_instance in instances:
            instance = str(raw_instance).strip()
            if not instance or instance == str(subject):
                continue
            stems = {
                re.sub(r"_\d+$", "", instance.casefold()),
                re.sub(r"_\d+$", "", str(declared_type).strip().casefold()),
            }
            aliases: set[str] = set()
            for stem in stems:
                words = [word for word in re.split(r"[^a-z0-9]+", stem) if word]
                aliases.update(" ".join(words[index:]) for index in range(len(words)))
            matched = [
                alias
                for alias in aliases
                if alias
                and re.search(
                    rf"\b{re.escape(alias)}\b", normalized_instruction
                )
            ]
            if matched:
                candidates.append(
                    (max(len(alias.split()) for alias in matched), instance)
                )
    if not candidates:
        return None
    best_score = max(score for score, _ in candidates)
    best = sorted({instance for score, instance in candidates if score == best_score})
    return best[0] if len(best) == 1 else None


def parse_libero_goal_clauses(
    goal_state: Sequence[Sequence[Any]],
    *,
    regions: Mapping[str, Any] | None = None,
    max_clauses: int = 4,
    instruction: str | None = None,
    entity_catalog: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Parse LIBERO predicates into explicit subject/reference clauses."""
    if not isinstance(goal_state, Sequence) or isinstance(
        goal_state, (str, bytes)
    ):
        raise ValueError("LIBERO goal_state must be a predicate sequence.")
    regions = dict(regions or {})
    clauses: list[dict[str, Any]] = []
    for raw_clause in goal_state:
        if not isinstance(raw_clause, Sequence) or isinstance(
            raw_clause, (str, bytes)
        ):
            raise ValueError(f"Invalid LIBERO goal predicate: {raw_clause!r}.")
        values = [str(value).strip() for value in raw_clause]
        if len(values) < 2:
            raise ValueError(f"Incomplete LIBERO goal predicate: {raw_clause!r}.")
        predicate = _normalize_libero_predicate(values[0])
        if predicate not in PGC_ENTITY_RELATION_PREDICATES[1:]:
            raise ValueError(
                f"Unsupported ERAF predicate {predicate!r} in {raw_clause!r}."
            )
        subject, subject_region = resolve_libero_region_target(
            values[1], regions
        )
        reference = subject
        reference_region = subject_region
        reference_region_target = reference if reference_region is not None else None
        if predicate in {"in", "on", "left", "right", "front", "back"}:
            if len(values) < 3:
                raise ValueError(
                    f"Binary ERAF predicate lacks a reference: {raw_clause!r}."
                )
            reference, reference_region = resolve_libero_region_target(
                values[2], regions
            )
            reference_region_target = (
                reference if reference_region is not None else None
            )
        if predicate == "on" and reference_region is not None:
            # Directional LIBERO goals are encoded as ON over a BDDL region.
            # Structural parsing resolves the true reference entity; explicit
            # task language supplies the otherwise absent relation class.
            directional = _instruction_spatial_relation(instruction)
            if directional is not None:
                predicate = directional
                reference = _instruction_reference_entity(
                    instruction,
                    entity_catalog=entity_catalog,
                    subject=subject,
                ) or reference
        clauses.append(
            {
                "clause_index": len(clauses),
                "predicate": predicate,
                "predicate_id": PGC_ENTITY_RELATION_PREDICATES.index(
                    predicate
                ),
                "subject": subject,
                "subject_region": subject_region,
                "reference": reference,
                "reference_region": reference_region,
                "reference_region_target": reference_region_target,
                "raw": values,
            }
        )
    if not clauses:
        raise ValueError("ERAF requires at least one LIBERO goal clause.")
    if len(clauses) > int(max_clauses):
        raise ValueError(
            f"ERAF supports at most {max_clauses} clauses, got {len(clauses)}."
        )
    return clauses


def classify_strict_conflict(
    source_clauses: Sequence[Mapping[str, Any]],
    target_clauses: Sequence[Mapping[str, Any]],
) -> str | None:
    """Classify a candidate pair without weakening the strict-conflict rules."""
    source = [dict(clause) for clause in source_clauses]
    target = [dict(clause) for clause in target_clauses]
    if not source or not target or source == target:
        return None

    def semantic_signature(clause: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(
            str(clause.get(field) or "")
            for field in (
                "predicate",
                "subject",
                "reference",
                "subject_region",
                "reference_region",
                "reference_region_target",
            )
        )

    # A shared clause makes either side a compatible subgoal.  Such pairs are
    # useful as raw CIS interventions but are forbidden in strict-conflict CIS.
    source_signatures = {semantic_signature(clause) for clause in source}
    target_signatures = {semantic_signature(clause) for clause in target}
    if source_signatures & target_signatures:
        return None
    if len(source) > 1 or len(target) > 1:
        return "conjunction"
    source_clause, target_clause = source[0], target[0]
    source_predicate = str(source_clause.get("predicate", ""))
    target_predicate = str(target_clause.get("predicate", ""))
    source_subject = str(source_clause.get("subject", ""))
    target_subject = str(target_clause.get("subject", ""))
    source_reference = str(source_clause.get("reference", ""))
    target_reference = str(target_clause.get("reference", ""))
    source_region = str(source_clause.get("reference_region") or "")
    target_region = str(target_clause.get("reference_region") or "")
    articulated = {"open", "close", "turnon", "turnoff"}
    if source_predicate in articulated or target_predicate in articulated:
        return "articulated_state"
    inverse_directions = {
        ("left", "right"),
        ("right", "left"),
        ("front", "back"),
        ("back", "front"),
    }
    if (
        (source_predicate, target_predicate) in inverse_directions
        and source_subject == target_subject
        and source_reference == target_reference
    ):
        return "direction_swap"
    if (
        source_predicate != target_predicate
        and source_subject == target_subject
        and source_reference == target_reference
    ):
        return "relation_swap"
    if (
        source_predicate == target_predicate
        and (
            source_subject != target_subject
            or source_reference != target_reference
        )
    ):
        return "entity_swap"
    if (
        source_predicate == target_predicate
        and source_subject == target_subject
        and source_reference == target_reference
        and source_region != target_region
    ):
        return "relation_swap"
    # A pair can alter both entity and relation.  It is still a strict
    # semantic conflict, but is kept separate from the cleaner ablations.
    return "compound_conflict"


def validate_strict_conflict_audit(
    audit: Mapping[str, Any], *, required_demos: int = 5
) -> dict[str, Any]:
    """Fail closed on initial-goal and bidirectional demonstration shortcuts."""
    required_demos = int(required_demos)
    if required_demos <= 0:
        raise ValueError("Strict-conflict required_demos must be positive.")
    required_counts = {
        "source_demo_source_success": required_demos,
        "target_demo_target_success": required_demos,
        "target_initially_false": required_demos,
        "source_initially_false": required_demos,
    }
    forbidden_counts = {
        "source_demo_target_success": 0,
        "target_demo_source_success": 0,
    }
    normalized = {name: int(audit.get(name, -1)) for name in (*required_counts, *forbidden_counts)}
    failures = [
        f"{name}={normalized[name]} expected>={minimum}"
        for name, minimum in required_counts.items()
        if normalized[name] < minimum
    ]
    failures.extend(
        f"{name}={normalized[name]} expected={expected}"
        for name, expected in forbidden_counts.items()
        if normalized[name] != expected
    )
    if failures:
        raise ValueError("Strict-conflict audit rejected pair: " + "; ".join(failures))
    normalized["required_demos"] = required_demos
    normalized["strict_conflict_passed"] = True
    return normalized


def load_pgc_entity_relation_index(
    sidecar_root: str | Path,
) -> dict[str, Any]:
    """Load the standalone, hash-audited V9 entity--relation sidecar."""
    sidecar_root = Path(sidecar_root).expanduser().resolve()
    index_path = sidecar_root / PGC_ENTITY_RELATION_INDEX
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing PGC v9 entity-relation index: {index_path}. Run "
            "scripts/build_pgc_libero_entity_relations.py first."
        )
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    sidecar_format = payload.get("format")
    if sidecar_format not in {
        PGC_ENTITY_RELATION_FORMAT,
        PGC_ROBOTWIN_ENTITY_RELATION_FORMAT,
    }:
        raise ValueError(
            f"Unsupported PGC v9 sidecar format at {index_path}: "
            f"{payload.get('format')!r}."
        )
    if payload.get("privileged_supervision") != "training_only":
        raise ValueError("PGC v9 sidecars must declare training-only privilege.")
    if payload.get("deployment_inputs") != "rgb_language_proprio":
        raise ValueError(
            "PGC v9 deployment must be limited to RGB, language, and proprio."
        )
    dataset_kind = str(payload.get("dataset_kind", ""))
    action_convention = str(payload.get("dataset_action_convention", ""))
    replay_transform = str(
        payload.get("simulator_replay_action_transform", "")
    )
    is_robotwin = sidecar_format == PGC_ROBOTWIN_ENTITY_RELATION_FORMAT
    expected_action_contract = (
        {
            "native": (
                PGC_ACTION_CONVENTION_ROBOTWIN_QPOS,
                PGC_ACTION_REPLAY_IDENTITY,
            ),
            "counterfactual": (
                PGC_ACTION_CONVENTION_ROBOTWIN_QPOS,
                PGC_ACTION_REPLAY_IDENTITY,
            ),
        }
        if is_robotwin
        else {
            "native": (
                PGC_ACTION_CONVENTION_FASTWAM,
                PGC_ACTION_REPLAY_FASTWAM_TO_LIBERO_ENV,
            ),
            "counterfactual": (
                PGC_ACTION_CONVENTION_LIBERO_ENV,
                PGC_ACTION_REPLAY_IDENTITY,
            ),
        }
    )
    if dataset_kind not in expected_action_contract:
        raise ValueError(
            "PGC v9 sidecars must declare dataset_kind as native or "
            "counterfactual."
        )
    expected_convention, expected_replay = expected_action_contract[
        dataset_kind
    ]
    if action_convention != expected_convention:
        raise ValueError(
            "PGC v9 sidecar action convention is incompatible with "
            f"dataset_kind={dataset_kind!r}: {action_convention!r}."
        )
    if replay_transform != expected_replay:
        raise ValueError(
            "PGC v9 sidecar simulator replay transform is incompatible with "
            f"dataset_kind={dataset_kind!r}: {replay_transform!r}."
        )
    if payload.get("entity_id_scheme") != "sha256_63bit":
        raise ValueError("PGC v9 sidecar entity ID scheme is incompatible.")
    if tuple(payload.get("predicate_vocabulary", ())) != (
        PGC_ENTITY_RELATION_PREDICATES
    ):
        raise ValueError("PGC v9 sidecar predicate vocabulary is incompatible.")
    entity_vocabulary = payload.get("entity_vocabulary")
    if not isinstance(entity_vocabulary, Mapping) or not entity_vocabulary:
        raise ValueError("PGC v9 sidecar entity vocabulary is empty.")
    if any(
        not str(name).strip() or not isinstance(entity_id, int) or entity_id < 0
        for name, entity_id in entity_vocabulary.items()
    ):
        raise ValueError("PGC v9 sidecar entity vocabulary is invalid.")
    if len(set(entity_vocabulary.values())) != len(entity_vocabulary):
        raise ValueError("PGC v9 sidecar entity IDs must be unique.")
    max_clauses = int(payload.get("max_clauses", 0))
    if max_clauses != 4:
        raise ValueError("PGC v9 sidecars must use max_clauses=4.")
    mask_size = payload.get("mask_size")
    if (
        not isinstance(mask_size, list)
        or len(mask_size) != 2
        or any(int(value) <= 0 for value in mask_size)
        or int(mask_size[1]) % 2
        or (is_robotwin and int(mask_size[0]) % 3)
    ):
        raise ValueError(
            "PGC v9 sidecar mask_size must be positive with an even width."
        )
    expected_camera_names = (
        ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        if is_robotwin
        else ["agentview", "robot0_eye_in_hand"]
    )
    if payload.get("camera_names") != expected_camera_names:
        raise ValueError("PGC v9 sidecar camera order is incompatible.")
    action_dim = int(payload.get("action_dim", 14 if is_robotwin else 7))
    if action_dim != (14 if is_robotwin else 7):
        raise ValueError(
            f"PGC v9 sidecar action_dim is incompatible: {action_dim}."
        )
    if (
        payload.get("view_center_coordinate_system")
        != "per_camera_normalized_xy"
    ):
        raise ValueError(
            "PGC v9 sidecar per-camera center coordinate system is "
            "incompatible."
        )
    try:
        workspace_min = np.asarray(payload["workspace_min"], dtype=np.float32)
        workspace_max = np.asarray(payload["workspace_max"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("PGC v9 sidecar has invalid workspace bounds.") from exc
    if (
        workspace_min.shape != (3,)
        or workspace_max.shape != (3,)
        or not np.isfinite(workspace_min).all()
        or not np.isfinite(workspace_max).all()
        or np.any(workspace_max <= workspace_min)
    ):
        raise ValueError("PGC v9 sidecar has invalid workspace bounds.")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("PGC v9 sidecar has no episode records.")
    indexed: dict[int, dict[str, Any]] = {}
    for record in episodes:
        if not isinstance(record, Mapping):
            raise ValueError("PGC v9 episode records must be objects.")
        episode_index = int(record.get("episode_index", -1))
        frame_count = int(record.get("frame_count", 0))
        if episode_index < 0 or episode_index in indexed or frame_count <= 0:
            raise ValueError(
                f"Invalid/duplicate PGC v9 episode record {episode_index}."
            )
        relpath = Path(str(record.get("file", "")))
        if relpath.is_absolute() or ".." in relpath.parts:
            raise ValueError("PGC v9 episode paths must stay inside the sidecar.")
        episode_path = sidecar_root / relpath
        if not episode_path.is_file():
            raise FileNotFoundError(f"Missing PGC v9 episode: {episode_path}.")
        expected_digest = str(record.get("sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise ValueError(
                f"PGC v9 episode {episode_index} has no valid sidecar SHA256."
            )
        if _file_sha256(episode_path) != expected_digest:
            raise ValueError(
                f"PGC v9 episode {episode_index} sidecar hash changed."
            )
        for audit_name in ("state_sha256", "action_sha256"):
            digest = str(record.get(audit_name, "")).lower()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(
                    f"PGC v9 episode {episode_index} lacks {audit_name}."
                )
        normalized = dict(record)
        normalized["path"] = str(episode_path)
        indexed[episode_index] = normalized
    if int(payload.get("episode_count", -1)) != len(indexed):
        raise ValueError("PGC v9 sidecar episode_count is inconsistent.")
    if set(indexed) != set(range(len(indexed))):
        raise ValueError("PGC v9 sidecar episode indices must be dense.")
    result = dict(payload)
    result["camera_count"] = len(expected_camera_names)
    result["action_dim"] = action_dim
    result["index_path"] = str(index_path)
    result["episodes_by_index"] = indexed
    return result


def load_pgc_target_mask_index(
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Load and validate the V7 per-episode object-mask sidecar.

    The sidecar is deliberately separate from the LeRobot feature table so an
    already audited RGB/action dataset can be augmented without rewriting its
    videos, actions, statistics, or episode hashes.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    index_path = dataset_root / PGC_TARGET_MASK_INDEX
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing PGC V7 target-mask index: {index_path}. Run "
            "scripts/build_pgc_libero_target_masks.py first."
        )
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("format") != PGC_TARGET_MASK_FORMAT:
        raise ValueError(
            f"Unsupported PGC target-mask format at {index_path}: "
            f"{payload.get('format')!r}."
        )
    mask_size = payload.get("mask_size")
    if (
        not isinstance(mask_size, list)
        or len(mask_size) != 2
        or any(not isinstance(value, int) or value <= 0 for value in mask_size)
    ):
        raise ValueError(f"Invalid PGC target-mask size in {index_path}: {mask_size!r}.")
    if int(mask_size[1]) % 2:
        raise ValueError("PGC two-camera target-mask width must be even.")
    camera_names = payload.get("camera_names")
    if camera_names != ["agentview", "robot0_eye_in_hand"]:
        raise ValueError(
            "PGC target masks must use the FastWAM camera order "
            "['agentview', 'robot0_eye_in_hand']."
        )
    catalog = payload.get("object_catalog")
    if not isinstance(catalog, list) or not catalog:
        raise ValueError(f"PGC target-mask catalog is empty: {index_path}.")
    object_names: set[str] = set()
    instructions: set[str] = set()
    for catalog_index, entry in enumerate(catalog):
        if not isinstance(entry, dict):
            raise ValueError("PGC target-mask catalog entries must be objects.")
        if int(entry.get("catalog_index", -1)) != catalog_index:
            raise ValueError("PGC target-mask catalog indices must be dense and ordered.")
        object_name = str(entry.get("object_name", "")).strip()
        instruction = str(entry.get("instruction", "")).strip()
        if not object_name or not instruction:
            raise ValueError("PGC target-mask catalog entries require object/instruction.")
        if object_name in object_names or instruction.casefold() in instructions:
            raise ValueError("PGC target-mask catalog contains duplicate labels.")
        object_names.add(object_name)
        instructions.add(instruction.casefold())

    audited_pairs = load_pgc_episode_language_pairs(dataset_root)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(audited_pairs):
        raise ValueError(
            "PGC target-mask episode count does not match the audited action "
            f"dataset: masks={len(episodes or [])} audits={len(audited_pairs)}."
        )
    indexed_episodes: dict[int, dict[str, Any]] = {}
    for entry in episodes:
        if not isinstance(entry, dict):
            raise ValueError("PGC target-mask episode entries must be objects.")
        episode_index = int(entry.get("episode_index", -1))
        if episode_index in indexed_episodes or episode_index not in audited_pairs:
            raise ValueError(
                f"Invalid or duplicate PGC target-mask episode {episode_index}."
            )
        frame_count = int(entry.get("frame_count", 0))
        if frame_count <= 0:
            raise ValueError("PGC target-mask episodes require positive frame_count.")
        relpath = Path(str(entry.get("file", "")))
        if relpath.is_absolute() or ".." in relpath.parts:
            raise ValueError("PGC target-mask episode paths must stay inside the dataset.")
        mask_path = dataset_root / relpath
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing PGC target-mask episode file: {mask_path}")
        expected_sha256 = str(entry.get("sha256", "")).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(
                f"PGC target-mask episode {episode_index} has no valid SHA256."
            )
        actual_sha256 = _file_sha256(mask_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "PGC target-mask episode file hash changed for episode "
                f"{episode_index}: expected={expected_sha256} "
                f"actual={actual_sha256}."
            )
        target_index = int(entry.get("target_catalog_index", -1))
        source_index = int(entry.get("source_catalog_index", -1))
        if not (0 <= target_index < len(catalog)) or not (
            0 <= source_index < len(catalog)
        ):
            raise ValueError("PGC target/source mask catalog index is out of range.")
        pair = audited_pairs[episode_index]
        if str(entry.get("pair_id", "")) != str(pair["pair_id"]):
            raise ValueError(
                f"PGC target-mask pair mismatch for episode {episode_index}."
            )
        if (
            catalog[target_index]["instruction"].strip().casefold()
            != str(pair["counterfactual_instruction"]).strip().casefold()
            or catalog[source_index]["instruction"].strip().casefold()
            != str(pair["source_instruction"]).strip().casefold()
        ):
            raise ValueError(
                f"PGC target-mask language labels mismatch episode {episode_index}."
            )
        normalized = dict(entry)
        normalized["mask_path"] = str(mask_path)
        indexed_episodes[episode_index] = normalized

    result = dict(payload)
    result["index_path"] = str(index_path)
    result["episodes_by_index"] = indexed_episodes
    return result


@dataclass(frozen=True)
class LiberoDemo:
    """One action demonstration lazily copied out of a LIBERO HDF5 file."""

    group_name: str
    initial_state: np.ndarray
    actions: np.ndarray


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            records.append(record)
    return records


def load_pgc_episode_language_pairs(
    dataset_root: str | Path,
) -> dict[int, dict[str, Any]]:
    """Map each audited PGC episode to its same-state language pair.

    PGC v5 reuses the already collected successful counterfactual trajectory
    twice: once with its recorded counterfactual instruction (positive action
    supervision), and once with the source instruction on the *identical*
    visual state (strict zero-residual supervision).  The collection contract
    deliberately stores the language pair at dataset level and the pair ID at
    episode level, so the training dataset can recover this mapping without
    duplicating videos or actions.
    """
    dataset_root = Path(dataset_root).expanduser()
    provenance_path = dataset_root / "meta/pgc_provenance.json"
    episodes_path = dataset_root / "meta/pgc_episodes.jsonl"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing PGC provenance: {provenance_path}")
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing PGC episode audit: {episodes_path}")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("format") != PGC_DATA_FORMAT:
        raise ValueError(
            f"Unsupported PGC data format at {provenance_path}: "
            f"{provenance.get('format')!r}."
        )
    if provenance.get("state_aligned") is not True:
        raise ValueError(
            f"PGC paired-language training requires state_aligned=true: "
            f"{provenance_path}."
        )

    pairs_by_id: dict[str, dict[str, Any]] = {}
    for pair in provenance.get("pairs") or []:
        pair_id = str(pair.get("pair_id", "")).strip()
        source_instruction = str(pair.get("source_instruction", "")).strip()
        counterfactual_instruction = str(
            pair.get("counterfactual_instruction", "")
        ).strip()
        if not pair_id or not source_instruction or not counterfactual_instruction:
            raise ValueError(
                f"PGC provenance contains an incomplete language pair: {pair!r}."
            )
        if source_instruction.casefold() == counterfactual_instruction.casefold():
            raise ValueError(
                f"PGC pair {pair_id!r} must change the instruction for v5."
            )
        if pair_id in pairs_by_id:
            raise ValueError(f"Duplicate PGC provenance pair ID: {pair_id!r}.")
        pairs_by_id[pair_id] = {
            "pair_id": pair_id,
            "source_instruction": source_instruction,
            "counterfactual_instruction": counterfactual_instruction,
            "source_suite": str(pair.get("source_suite", "")),
            "source_task_id": int(pair.get("source_task_id", -1)),
            "strict_conflict": bool(pair.get("strict_conflict", False)),
            "strict_conflict_type": pair.get("strict_conflict_type"),
            "strict_replay_audit": pair.get("strict_replay_audit"),
        }
    if not pairs_by_id:
        raise ValueError(f"PGC provenance has no language pairs: {provenance_path}.")

    result: dict[int, dict[str, Any]] = {}
    for audit in read_jsonl(episodes_path):
        try:
            episode_index = int(audit["episode_index"])
            pair_id = str(audit["pair_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid PGC episode audit in {episodes_path}: {audit!r}."
            ) from exc
        if episode_index < 0:
            raise ValueError("PGC episode indices must be non-negative.")
        if episode_index in result:
            raise ValueError(
                f"Duplicate PGC episode audit index {episode_index} in "
                f"{episodes_path}."
            )
        try:
            pair = pairs_by_id[pair_id]
        except KeyError as exc:
            raise ValueError(
                f"PGC episode {episode_index} references unknown pair "
                f"{pair_id!r}."
            ) from exc
        result[episode_index] = dict(pair)

    expected_count = int(provenance.get("successful_episode_count", len(result)))
    if expected_count != len(result):
        raise ValueError(
            "PGC successful episode count does not match its audit table: "
            f"provenance={expected_count}, audits={len(result)}."
        )
    if not result:
        raise ValueError(f"PGC episode audit is empty: {episodes_path}.")
    return result


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def canonical_state_array(state: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a stable little-endian float64 representation of simulator state."""
    array = np.asarray(state)
    if array.ndim == 0:
        raise ValueError("Simulator state must have at least one dimension.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Simulator state must be numeric, got {array.dtype}.")
    array = np.ascontiguousarray(array.astype("<f8", copy=False))
    if not np.isfinite(array).all():
        raise ValueError("Simulator state contains NaN or infinity.")
    return array


def state_sha256(state: np.ndarray | Sequence[float]) -> str:
    """Hash state values and shape independently of host byte order/dtype."""
    array = canonical_state_array(state)
    header = json.dumps(
        {"dtype": "float64-le", "shape": list(array.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def array_sha256(array: np.ndarray | Sequence[Any]) -> str:
    """Hash a typed tensor with its exact dtype and shape for data audits."""
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def states_match(
    requested: np.ndarray | Sequence[float],
    actual: np.ndarray | Sequence[float],
    *,
    atol: float = 1e-7,
) -> bool:
    requested_array = canonical_state_array(requested)
    actual_array = canonical_state_array(actual)
    return requested_array.shape == actual_array.shape and bool(
        np.allclose(requested_array, actual_array, rtol=0.0, atol=float(atol))
    )


def filter_libero_noops(
    actions: np.ndarray | Sequence[Sequence[float]],
    *,
    threshold: float = 1e-4,
) -> np.ndarray:
    """Match the no-op removal used by LIBERO dataset regeneration.

    A stationary action is kept when it changes the gripper command.  The
    previous *kept* action is used, matching the released no-noops converter.
    """
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"LIBERO actions must be [T,7], got {array.shape}.")
    kept: list[np.ndarray] = []
    for action in array:
        previous = kept[-1] if kept else None
        stationary = float(np.linalg.norm(action[:-1])) < float(threshold)
        same_gripper = previous is None or bool(action[-1] == previous[-1])
        if stationary and same_gripper:
            continue
        kept.append(action)
    if not kept:
        return np.empty((0, 7), dtype=np.float32)
    return np.ascontiguousarray(np.stack(kept).astype(np.float32, copy=False))


def _normalise_stem(value: str) -> str:
    value = Path(value).stem.casefold()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    for suffix in ("_demo", "_demonstration"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value


def demo_file_candidates(
    demo_root: str | Path,
    record: Mapping[str, Any],
) -> list[Path]:
    """Find HDF5 files whose name matches the target task/BDDL stem."""
    root = Path(demo_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LIBERO demonstration root not found: {root}")

    desired_stems = {
        _normalise_stem(str(record.get("counterfactual_bddl_file", ""))),
        _normalise_stem(str(record.get("counterfactual_task_name", ""))),
        _normalise_stem(str(record.get("counterfactual_instruction", ""))),
    }
    desired_stems.discard("")
    if not desired_stems:
        raise ValueError(
            f"Manifest pair {record.get('pair_id')!r} has no target BDDL/task name."
        )

    # Prefer the canonical ``<root>/<suite>/<task>_demo.hdf5`` layout used by
    # the official LIBERO HDF5 release. Apart from being substantially cheaper
    # than recursively scanning a large, partially downloaded LIBERO-90 tree,
    # this deterministic lookup avoids depending on recursive-discovery
    # semantics across storage backends. Fall back to the recursive search for
    # legacy or flattened layouts.
    target_suite = str(
        record.get("counterfactual_task_suite_name", "")
    ).casefold()
    suite_root = root / target_suite if target_suite else None
    if suite_root is not None and suite_root.is_dir():
        suite_matches = sorted(
            {
                path.resolve()
                for path in suite_root.iterdir()
                if path.is_file()
                and path.suffix.casefold() in {".hdf5", ".h5"}
                and _normalise_stem(path.name) in desired_stems
            }
        )
        if suite_matches:
            return suite_matches

    matches = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in {".hdf5", ".h5"}:
            continue
        candidate_stem = _normalise_stem(path.name)
        if candidate_stem in desired_stems:
            matches.append(path.resolve())
    matches = sorted(set(matches))
    if len(matches) > 1 and target_suite:
        suite_matches = [
            path
            for path in matches
            if target_suite in {part.casefold() for part in path.parts}
        ]
        if suite_matches:
            matches = suite_matches
    return matches


def resolve_demo_file(
    demo_root: str | Path,
    record: Mapping[str, Any],
) -> Path:
    matches = demo_file_candidates(demo_root, record)
    if len(matches) != 1:
        root = Path(demo_root).expanduser().resolve()
        desired_stems = sorted(
            {
                stem
                for stem in (
                    _normalise_stem(
                        str(record.get("counterfactual_bddl_file", ""))
                    ),
                    _normalise_stem(
                        str(record.get("counterfactual_task_name", ""))
                    ),
                    _normalise_stem(
                        str(record.get("counterfactual_instruction", ""))
                    ),
                )
                if stem
            }
        )
        raise ValueError(
            f"Expected exactly one target demo file for pair "
            f"{record.get('pair_id')!r}, found {len(matches)}: "
            f"{[str(path) for path in matches]}; demo_root={str(root)!r}; "
            f"target_suite="
            f"{str(record.get('counterfactual_task_suite_name', ''))!r}; "
            f"desired_stems={desired_stems!r}"
        )
    return matches[0]


def _demo_sort_key(group_name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(group_name))
    return (int(match.group(1)) if match else 2**31 - 1, str(group_name))


def iter_libero_hdf5_demos(path: str | Path) -> Iterator[LiberoDemo]:
    """Yield standard LIBERO ``states``/``actions`` demonstration groups."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - server dependency check
        raise ImportError("PGC data building requires h5py.") from exc

    path = Path(path)
    with h5py.File(path, "r") as handle:
        root = handle["data"] if "data" in handle else handle
        group_names = sorted(root.keys(), key=_demo_sort_key)
        found = 0
        for group_name in group_names:
            group = root[group_name]
            if not hasattr(group, "keys") or not {"states", "actions"}.issubset(
                set(group.keys())
            ):
                continue
            states = np.asarray(group["states"])
            actions = np.asarray(group["actions"], dtype=np.float32)
            if states.ndim != 2 or states.shape[0] == 0:
                raise ValueError(
                    f"{path}:{group_name}/states must be non-empty [T,D], "
                    f"got {states.shape}."
                )
            if actions.ndim != 2 or actions.shape[0] == 0:
                raise ValueError(
                    f"{path}:{group_name}/actions must be non-empty [T,A], "
                    f"got {actions.shape}."
                )
            if not np.isfinite(actions).all():
                raise ValueError(f"{path}:{group_name}/actions contains NaN or infinity.")
            found += 1
            yield LiberoDemo(
                group_name=str(group_name),
                initial_state=np.asarray(states[0]).copy(),
                actions=np.ascontiguousarray(actions),
            )
        if found == 0:
            raise ValueError(f"No LIBERO states/actions demos found in {path}.")


def validate_manifest_record(record: Mapping[str, Any]) -> None:
    required = {
        "pair_id",
        "task_suite_name",
        "task_id",
        "correct_instruction",
        "counterfactual_instruction",
        "counterfactual_goal_state",
        "counterfactual_bddl_file",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(
            f"Manifest pair {record.get('pair_id')!r} is missing {sorted(missing)}."
        )
    if str(record["task_suite_name"]) not in LIBERO_SUITES:
        raise ValueError(
            f"Unsupported source suite {record['task_suite_name']!r}."
        )
    if not 0 <= int(record["task_id"]) < 10:
        raise ValueError(
            f"Pair {record['pair_id']!r} source task ID must be in [0,9]."
        )
    if not str(record["counterfactual_instruction"]).strip():
        raise ValueError(f"Pair {record['pair_id']!r} has an empty target instruction.")
    goal = record["counterfactual_goal_state"]
    if not isinstance(goal, list) or not goal:
        raise ValueError(f"Pair {record['pair_id']!r} has no alternate goal state.")
    transfer_mode = str(record.get("state_transfer_mode", "flat_exact"))
    if transfer_mode not in PGC_STATE_TRANSFER_MODES:
        raise ValueError(
            f"Pair {record['pair_id']!r} has unsupported state_transfer_mode "
            f"{transfer_mode!r}; expected one of {PGC_STATE_TRANSFER_MODES}."
        )
    if (
        record.get("counterfactual_goal_changed") is False
        and str(record["task_suite_name"]) != "libero_spatial"
    ):
        raise ValueError(
            f"Pair {record['pair_id']!r} may keep the terminal goal only for "
            "LIBERO-Spatial state-grounded supervision."
        )


def provenance_pair(record: Mapping[str, Any]) -> dict[str, Any]:
    validate_manifest_record(record)
    pair = {
        "pair_id": str(record["pair_id"]),
        "source_suite": str(record["task_suite_name"]),
        "source_task_id": int(record["task_id"]),
        "source_instruction": str(record["correct_instruction"]).strip(),
        "counterfactual_instruction": str(
            record["counterfactual_instruction"]
        ).strip(),
        "counterfactual_goal_state": record["counterfactual_goal_state"],
        "counterfactual_task_suite_name": str(
            record.get(
                "counterfactual_task_suite_name", record["task_suite_name"]
            )
        ),
        "counterfactual_task_id": int(
            record.get("counterfactual_task_id", record["task_id"])
        ),
        "source_bddl_file": str(record.get("source_bddl_file", "")),
        "counterfactual_bddl_file": str(record["counterfactual_bddl_file"]),
        "state_transfer_mode": str(
            record.get("state_transfer_mode", "flat_exact")
        ),
        "counterfactual_goal_changed": bool(
            record.get("counterfactual_goal_changed", True)
        ),
        "counterfactual_state_changed": bool(
            record.get("counterfactual_state_changed", True)
        ),
    }
    if bool(record.get("strict_conflict", False)):
        conflict_type = str(record.get("strict_conflict_type", "")).strip()
        replay_audit = record.get("strict_replay_audit")
        if not conflict_type or not isinstance(replay_audit, Mapping):
            raise ValueError(
                f"Strict pair {record['pair_id']!r} lacks its conflict type or "
                "bidirectional replay audit."
            )
        required_demos = int(replay_audit.get("required_demos", 5))
        normalized_audit = validate_strict_conflict_audit(
            replay_audit, required_demos=required_demos
        )
        pair.update(
            {
                "strict_conflict": True,
                "strict_conflict_type": conflict_type,
                "strict_replay_audit": normalized_audit,
            }
        )
    return pair


def build_provenance(
    records: Sequence[Mapping[str, Any]],
    *,
    successful_episode_count: int = 0,
) -> dict[str, Any]:
    if not records:
        raise ValueError("A PGC dataset requires at least one intervention pair.")
    pairs = [provenance_pair(record) for record in records]
    pair_ids = [pair["pair_id"] for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("PGC manifest contains duplicate pair IDs.")
    suites = sorted({pair["source_suite"] for pair in pairs})
    return {
        "format": PGC_DATA_FORMAT,
        "benchmark": "libero",
        "action_supervision": PGC_ACTION_SUPERVISION,
        "collection_method": (
            "audited_target_demo_replay_with_exact_or_named_joint_state_transfer"
        ),
        "state_aligned": True,
        "state_match_tolerance": 1e-7,
        "state_catalog": "meta/pgc_initial_states/episode_{episode_index:06d}.npy",
        "successful_only": True,
        "successful_episode_count": int(successful_episode_count),
        "source_suites": suites,
        "pairs": pairs,
    }


def libero_lerobot_features(resolution: int = 512) -> dict[str, dict[str, Any]]:
    resolution = int(resolution)
    if resolution <= 0:
        raise ValueError("Camera resolution must be positive.")
    return {
        "observation.images.image": {
            "dtype": "video",
            "shape": (3, resolution, resolution),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": (3, resolution, resolution),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "eef_x",
                "eef_y",
                "eef_z",
                "eef_axis_x",
                "eef_axis_y",
                "eef_axis_z",
                "gripper_left",
                "gripper_right",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "delta_x",
                "delta_y",
                "delta_z",
                "delta_axis_x",
                "delta_axis_y",
                "delta_axis_z",
                "gripper",
            ],
        },
    }
