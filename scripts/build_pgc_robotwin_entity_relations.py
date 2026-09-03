#!/usr/bin/env python3
"""Build hash-audited ERAF sidecars from raw RoboTwin PGC captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.robotwin.pgc_data import direction_from_task
from fastwam.datasets.pgc_libero import (
    PGC_ACTION_CONVENTION_ROBOTWIN_QPOS,
    PGC_ACTION_REPLAY_IDENTITY,
    PGC_ENTITY_RELATION_ARRAY_NAMES,
    PGC_ENTITY_RELATION_PREDICATES,
    PGC_ROBOTWIN_ENTITY_RELATION_FORMAT,
    read_jsonl,
)


MASK_SIZE = (24, 20)
WORKSPACE_MIN = np.asarray((-0.8, -0.8, 0.0), dtype=np.float32)
WORKSPACE_MAX = np.asarray((0.8, 0.8, 1.2), dtype=np.float32)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entity_id(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _resize_mask(mask: np.ndarray, *, height: int, width: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    return np.asarray(
        image.resize((width, height), resample=Image.Resampling.NEAREST),
        dtype=np.uint8,
    ).astype(np.bool_)


def _mosaic_mask(
    head: np.ndarray, left: np.ndarray, right: np.ndarray, actor_id: int
) -> np.ndarray:
    head_mask = _resize_mask(head == actor_id, height=16, width=20)
    left_mask = _resize_mask(left == actor_id, height=8, width=10)
    right_mask = _resize_mask(right == actor_id, height=8, width=10)
    return np.concatenate(
        (head_mask, np.concatenate((left_mask, right_mask), axis=1)), axis=0
    )


def _view_geometry(
    head: np.ndarray, left: np.ndarray, right: np.ndarray, actor_id: int
) -> tuple[np.ndarray, np.ndarray]:
    visible = np.zeros(3, dtype=np.bool_)
    centers = np.zeros((3, 2), dtype=np.float32)
    for index, labels in enumerate((head, left, right)):
        ys, xs = np.nonzero(labels == actor_id)
        if xs.size == 0:
            continue
        visible[index] = True
        width = max(int(labels.shape[1]) - 1, 1)
        height = max(int(labels.shape[0]) - 1, 1)
        centers[index] = (
            2.0 * float(xs.mean()) / width - 1.0,
            2.0 * float(ys.mean()) / height - 1.0,
        )
    return visible, centers


def _normalize_position(position: np.ndarray) -> np.ndarray:
    value = (
        2.0
        * (np.asarray(position, dtype=np.float32) - WORKSPACE_MIN)
        / (WORKSPACE_MAX - WORKSPACE_MIN)
        - 1.0
    )
    return np.clip(value, -1.0, 1.0).astype(np.float32)


def _truth(subject: np.ndarray, reference: np.ndarray, direction: str) -> np.ndarray:
    distance = np.linalg.norm(subject[:, :2] - reference[:, :2], axis=-1)
    side = (
        subject[:, 0] < reference[:, 0]
        if direction == "left"
        else subject[:, 0] > reference[:, 0]
    )
    return (
        (distance < 0.2)
        & (distance > 0.08)
        & side
        & (np.abs(subject[:, 1] - reference[:, 1]) < 0.05)
    ).astype(np.float32)


def _phase_ids(subject: np.ndarray, target_truth: np.ndarray) -> np.ndarray:
    displacement = np.linalg.norm(subject - subject[:1], axis=-1)
    lifted = subject[:, 2] > float(subject[0, 2]) + 0.03
    moving = lifted | (displacement > 0.03)
    phase = np.zeros(len(subject), dtype=np.int64)
    phase[moving] = 1
    satisfied = np.flatnonzero(target_truth >= 0.5)
    if satisfied.size:
        phase[int(satisfied[0]) :] = 2
    return phase


def _empty_arrays(frame_count: int) -> dict[str, np.ndarray]:
    scalar_bool = np.zeros((frame_count, 4), dtype=np.bool_)
    scalar_float = np.zeros((frame_count, 4), dtype=np.float32)
    scalar_int = np.zeros((frame_count, 4), dtype=np.int64)
    vector = np.zeros((frame_count, 4, 3), dtype=np.float32)
    masks = np.zeros((frame_count, 4, *MASK_SIZE), dtype=np.bool_)
    view_visible = np.zeros((frame_count, 4, 3), dtype=np.bool_)
    view_centers = np.zeros((frame_count, 4, 3, 2), dtype=np.float32)
    arrays: dict[str, np.ndarray] = {}
    for name in PGC_ENTITY_RELATION_ARRAY_NAMES:
        if name.endswith("_masks"):
            value = masks.copy()
        elif name.endswith("_view_visible"):
            value = view_visible.copy()
        elif name.endswith("_view_centers"):
            value = view_centers.copy()
        elif name.endswith("_positions") or name.endswith("_anchors"):
            value = vector.copy()
        elif name.endswith("_valid") or name == "clause_valid":
            value = scalar_bool.copy()
        elif name in {
            "predicate_truth",
        }:
            value = scalar_float.copy()
        else:
            value = scalar_int.copy()
        arrays[name] = value
    return arrays


def _h5_array(handle: h5py.File, key: str) -> np.ndarray:
    if key not in handle:
        raise KeyError(f"Raw RoboTwin capture is missing HDF5 dataset {key!r}.")
    return np.asarray(handle[key])


def _role_arrays(
    *,
    handle: h5py.File,
    direction: str,
    phase_ids: np.ndarray,
    subject_entity_id: int,
    reference_entity_id: int,
) -> dict[str, np.ndarray]:
    subject = _h5_array(handle, "pgc_entity_state/subject_position").astype(np.float32)
    reference = _h5_array(handle, "pgc_entity_state/reference_position").astype(
        np.float32
    )
    subject_ids = _h5_array(handle, "pgc_entity_state/subject_actor_id").reshape(-1)
    reference_ids = _h5_array(handle, "pgc_entity_state/reference_actor_id").reshape(-1)
    head = _h5_array(handle, "observation/head_camera/actor_segmentation_ids")
    left = _h5_array(handle, "observation/left_camera/actor_segmentation_ids")
    right = _h5_array(handle, "observation/right_camera/actor_segmentation_ids")
    frame_count = len(subject)
    arrays = _empty_arrays(frame_count)
    arrays["clause_valid"][:, 0] = True
    arrays["predicate_ids"][:, 0] = PGC_ENTITY_RELATION_PREDICATES.index(direction)
    arrays["subject_entity_ids"][:, 0] = subject_entity_id
    arrays["reference_entity_ids"][:, 0] = reference_entity_id
    arrays["phase_ids"][:, 0] = phase_ids
    arrays["phase_valid"][:, 0] = True
    truth = _truth(subject, reference, direction)
    arrays["predicate_truth"][:, 0] = truth
    arrays["predicate_truth_valid"][:, 0] = True
    goal = reference.copy()
    goal[:, 0] += -0.13 if direction == "left" else 0.13
    for frame in range(frame_count):
        subject_mask = _mosaic_mask(
            head[frame], left[frame], right[frame], int(subject_ids[frame])
        )
        reference_mask = _mosaic_mask(
            head[frame], left[frame], right[frame], int(reference_ids[frame])
        )
        subject_visible, subject_centers = _view_geometry(
            head[frame], left[frame], right[frame], int(subject_ids[frame])
        )
        reference_visible, reference_centers = _view_geometry(
            head[frame], left[frame], right[frame], int(reference_ids[frame])
        )
        arrays["subject_masks"][frame, 0] = subject_mask
        arrays["reference_masks"][frame, 0] = reference_mask
        arrays["subject_mask_valid"][frame, 0] = bool(subject_mask.any())
        arrays["reference_mask_valid"][frame, 0] = bool(reference_mask.any())
        arrays["subject_view_visible"][frame, 0] = subject_visible
        arrays["reference_view_visible"][frame, 0] = reference_visible
        arrays["subject_view_centers"][frame, 0] = subject_centers
        arrays["reference_view_centers"][frame, 0] = reference_centers
    arrays["subject_positions"][:, 0] = np.stack(
        [_normalize_position(value) for value in subject]
    )
    arrays["reference_positions"][:, 0] = np.stack(
        [_normalize_position(value) for value in reference]
    )
    arrays["subject_position_valid"][:, 0] = True
    arrays["reference_position_valid"][:, 0] = True
    arrays["grasp_anchors"][:, 0] = arrays["subject_positions"][:, 0]
    arrays["goal_anchors"][:, 0] = np.stack(
        [_normalize_position(value) for value in goal]
    )
    arrays["interaction_anchors"][:, 0] = arrays["goal_anchors"][:, 0]
    arrays["grasp_anchor_valid"][:, 0] = True
    arrays["goal_anchor_valid"][:, 0] = True
    arrays["interaction_anchor_valid"][:, 0] = True
    return arrays


def _generic_role_arrays(
    *,
    handle: h5py.File,
    prefix: str,
    entity_ids: list[int],
) -> dict[str, np.ndarray]:
    positions = _h5_array(handle, "pgc_entity_state/entity_positions").astype(
        np.float32
    )
    actor_ids = _h5_array(handle, "pgc_entity_state/entity_actor_ids").astype(np.int64)
    entity_valid = _h5_array(handle, "pgc_entity_state/entity_valid").astype(np.bool_)
    subject_indices = _h5_array(
        handle, f"pgc_entity_state/{prefix}_subject_indices"
    ).astype(np.int64)
    reference_indices = _h5_array(
        handle, f"pgc_entity_state/{prefix}_reference_indices"
    ).astype(np.int64)
    predicate_ids = _h5_array(
        handle, f"pgc_entity_state/{prefix}_predicate_ids"
    ).astype(np.int64)
    goal_positions = _h5_array(
        handle, f"pgc_entity_state/{prefix}_goal_positions"
    ).astype(np.float32)
    predicate_truth = _h5_array(
        handle, f"pgc_entity_state/{prefix}_predicate_truth"
    ).astype(np.float32)
    clause_valid = _h5_array(handle, f"pgc_entity_state/{prefix}_clause_valid").astype(
        np.bool_
    )
    head = _h5_array(handle, "observation/head_camera/actor_segmentation_ids")
    left = _h5_array(handle, "observation/left_camera/actor_segmentation_ids")
    right = _h5_array(handle, "observation/right_camera/actor_segmentation_ids")
    frame_count = int(positions.shape[0])
    expected_shapes = {
        "positions": (frame_count, 4, 3),
        "actor_ids": (frame_count, 4),
        "entity_valid": (frame_count, 4),
        "subject_indices": (frame_count, 4),
        "reference_indices": (frame_count, 4),
        "predicate_ids": (frame_count, 4),
        "goal_positions": (frame_count, 4, 3),
        "predicate_truth": (frame_count, 4),
        "clause_valid": (frame_count, 4),
    }
    values = {
        "positions": positions,
        "actor_ids": actor_ids,
        "entity_valid": entity_valid,
        "subject_indices": subject_indices,
        "reference_indices": reference_indices,
        "predicate_ids": predicate_ids,
        "goal_positions": goal_positions,
        "predicate_truth": predicate_truth,
        "clause_valid": clause_valid,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(
                f"Raw RoboTwin {prefix} {name} has shape {values[name].shape}; "
                f"expected {expected}."
            )
    arrays = _empty_arrays(frame_count)
    for clause_index in range(4):
        valid_frames = np.flatnonzero(clause_valid[:, clause_index])
        if valid_frames.size == 0:
            continue
        if valid_frames.size != frame_count:
            raise ValueError(
                f"RoboTwin {prefix} clause {clause_index} validity changes by frame."
            )
        subject_index = int(subject_indices[0, clause_index])
        reference_index = int(reference_indices[0, clause_index])
        predicate_id = int(predicate_ids[0, clause_index])
        if not (0 <= subject_index < len(entity_ids)) or not (
            0 <= reference_index < len(entity_ids)
        ):
            raise ValueError(
                f"RoboTwin {prefix} clause {clause_index} references an invalid entity."
            )
        if not (
            np.all(subject_indices[:, clause_index] == subject_index)
            and np.all(reference_indices[:, clause_index] == reference_index)
            and np.all(predicate_ids[:, clause_index] == predicate_id)
        ):
            raise ValueError(
                f"RoboTwin {prefix} clause {clause_index} semantics drift by frame."
            )
        if not (0 < predicate_id < len(PGC_ENTITY_RELATION_PREDICATES)):
            raise ValueError(
                f"RoboTwin {prefix} clause {clause_index} has predicate {predicate_id}."
            )
        if not (
            np.all(entity_valid[:, subject_index])
            and np.all(entity_valid[:, reference_index])
        ):
            raise ValueError(
                f"RoboTwin {prefix} clause {clause_index} uses an invalid entity."
            )
        subject = positions[:, subject_index]
        reference = positions[:, reference_index]
        truth = predicate_truth[:, clause_index]
        arrays["clause_valid"][:, clause_index] = True
        arrays["predicate_ids"][:, clause_index] = predicate_id
        arrays["subject_entity_ids"][:, clause_index] = entity_ids[subject_index]
        arrays["reference_entity_ids"][:, clause_index] = entity_ids[reference_index]
        arrays["predicate_truth"][:, clause_index] = truth
        arrays["predicate_truth_valid"][:, clause_index] = True
        arrays["phase_ids"][:, clause_index] = _phase_ids(subject, truth)
        arrays["phase_valid"][:, clause_index] = True
        arrays["subject_positions"][:, clause_index] = np.stack(
            [_normalize_position(value) for value in subject]
        )
        arrays["reference_positions"][:, clause_index] = np.stack(
            [_normalize_position(value) for value in reference]
        )
        normalized_goal = np.stack(
            [_normalize_position(value) for value in goal_positions[:, clause_index]]
        )
        arrays["subject_position_valid"][:, clause_index] = True
        arrays["reference_position_valid"][:, clause_index] = True
        arrays["grasp_anchors"][:, clause_index] = arrays["subject_positions"][
            :, clause_index
        ]
        arrays["goal_anchors"][:, clause_index] = normalized_goal
        arrays["interaction_anchors"][:, clause_index] = normalized_goal
        arrays["grasp_anchor_valid"][:, clause_index] = True
        arrays["goal_anchor_valid"][:, clause_index] = True
        arrays["interaction_anchor_valid"][:, clause_index] = True
        for frame in range(frame_count):
            subject_actor_id = int(actor_ids[frame, subject_index])
            reference_actor_id = int(actor_ids[frame, reference_index])
            subject_mask = _mosaic_mask(
                head[frame], left[frame], right[frame], subject_actor_id
            )
            reference_mask = _mosaic_mask(
                head[frame], left[frame], right[frame], reference_actor_id
            )
            subject_visible, subject_centers = _view_geometry(
                head[frame], left[frame], right[frame], subject_actor_id
            )
            reference_visible, reference_centers = _view_geometry(
                head[frame], left[frame], right[frame], reference_actor_id
            )
            arrays["subject_masks"][frame, clause_index] = subject_mask
            arrays["reference_masks"][frame, clause_index] = reference_mask
            arrays["subject_mask_valid"][frame, clause_index] = bool(subject_mask.any())
            arrays["reference_mask_valid"][frame, clause_index] = bool(
                reference_mask.any()
            )
            arrays["subject_view_visible"][frame, clause_index] = subject_visible
            arrays["reference_view_visible"][frame, clause_index] = reference_visible
            arrays["subject_view_centers"][frame, clause_index] = subject_centers
            arrays["reference_view_centers"][frame, clause_index] = reference_centers
    return arrays


def build_sidecar(*, raw_root: Path, dataset_root: Path, output_root: Path) -> Path:
    records = read_jsonl(raw_root / "meta" / "pgc_episodes.jsonl")
    if not records:
        raise ValueError(f"Raw RoboTwin PGC capture has no episodes: {raw_root}.")
    provenance = json.loads(
        (raw_root / "meta" / "pgc_provenance.json").read_text(encoding="utf-8")
    )
    dataset_kind = str(provenance.get("dataset_kind", ""))
    if dataset_kind not in {"native", "counterfactual"}:
        raise ValueError(f"Invalid raw RoboTwin dataset_kind: {dataset_kind!r}.")
    pairs = provenance.get("pairs") or []
    if len(pairs) > 1:
        raise ValueError("Raw RoboTwin ERAF provenance must contain at most one pair.")
    pair = pairs[0] if pairs else {}
    source_task = str(pair.get("source_task", ""))
    entity_names = tuple(str(name) for name in pair.get("entity_names") or ())
    generic_capture = bool(entity_names)
    if generic_capture and not (1 <= len(entity_names) <= 4):
        raise ValueError("Raw RoboTwin ERAF entity_names must contain 1..4 names.")
    output_root.mkdir(parents=True, exist_ok=True)
    episode_dir = output_root / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    subject_entity_id = _entity_id("robotwin:place_a2b:subject")
    reference_entity_id = _entity_id("robotwin:place_a2b:reference")
    entity_keys = [f"robotwin:{source_task}:{name}" for name in entity_names]
    entity_ids = [_entity_id(key) for key in entity_keys]
    episode_records = []
    for episode_index, record in enumerate(records):
        hdf5_path = raw_root / str(record["raw_hdf5"])
        with h5py.File(hdf5_path, "r") as handle:
            if generic_capture:
                target = _generic_role_arrays(
                    handle=handle,
                    prefix="target",
                    entity_ids=entity_ids,
                )
                source = _generic_role_arrays(
                    handle=handle,
                    prefix="source",
                    entity_ids=entity_ids,
                )
                phase = target["phase_ids"]
            else:
                source_direction = direction_from_task(record["source_task"])
                target_direction = direction_from_task(record["counterfactual_task"])
                subject = _h5_array(handle, "pgc_entity_state/subject_position").astype(
                    np.float32
                )
                reference = _h5_array(
                    handle, "pgc_entity_state/reference_position"
                ).astype(np.float32)
                target_truth = _truth(subject, reference, target_direction)
                phase = _phase_ids(subject, target_truth)
                target = _role_arrays(
                    handle=handle,
                    direction=target_direction,
                    phase_ids=phase,
                    subject_entity_id=subject_entity_id,
                    reference_entity_id=reference_entity_id,
                )
                source = _role_arrays(
                    handle=handle,
                    direction=source_direction,
                    phase_ids=phase,
                    subject_entity_id=subject_entity_id,
                    reference_entity_id=reference_entity_id,
                )
        payload = {
            **{f"target_{key}": value for key, value in target.items()},
            **{f"source_{key}": value for key, value in source.items()},
        }
        for role in ("source", "target"):
            valid_clauses = np.flatnonzero(payload[f"{role}_clause_valid"][0])
            for clause_index in valid_clauses:
                if not bool(
                    payload[f"{role}_subject_masks"][:, clause_index].any()
                ) or not bool(
                    payload[f"{role}_reference_masks"][:, clause_index].any()
                ):
                    raise ValueError(
                        "Actor-ID supervision is empty for "
                        f"episode {episode_index}, {role} clause {clause_index}."
                    )
        output_path = episode_dir / f"episode{episode_index}.npz"
        np.savez_compressed(output_path, **payload)
        episode_records.append(
            {
                "episode_index": episode_index,
                "pair_id": record["pair_id"],
                "file": output_path.relative_to(output_root).as_posix(),
                "sha256": _file_sha256(output_path),
                "frame_count": int(phase.shape[0]),
                "state_sha256": record["initial_state_sha256"],
                "initial_state_sha256": record["initial_state_sha256"],
                "action_sha256": record["action_sha256"],
            }
        )
    index = {
        "format": PGC_ROBOTWIN_ENTITY_RELATION_FORMAT,
        "artifact_role": provenance.get("artifact_role", "eraf_grounding_supervision"),
        "allowed_training_stages": list(
            provenance.get("allowed_training_stages") or ["grounding"]
        ),
        "full_goal_verified": False,
        "privileged_supervision": "training_only",
        "deployment_inputs": "rgb_language_proprio",
        "dataset": str(dataset_root.resolve()),
        "dataset_kind": dataset_kind,
        "dataset_action_convention": PGC_ACTION_CONVENTION_ROBOTWIN_QPOS,
        "simulator_replay_action_transform": PGC_ACTION_REPLAY_IDENTITY,
        "action_dim": 14,
        "entity_id_scheme": "sha256_63bit",
        "predicate_vocabulary": list(PGC_ENTITY_RELATION_PREDICATES),
        "entity_vocabulary": (
            dict(zip(entity_keys, entity_ids))
            if generic_capture
            else {
                "robotwin:place_a2b:subject": subject_entity_id,
                "robotwin:place_a2b:reference": reference_entity_id,
            }
        ),
        "max_clauses": 4,
        "mask_size": list(MASK_SIZE),
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "camera_layout": "robotwin_mosaic",
        "view_center_coordinate_system": "per_camera_normalized_xy",
        "workspace_min": WORKSPACE_MIN.tolist(),
        "workspace_max": WORKSPACE_MAX.tolist(),
        "episode_count": len(episode_records),
        "episodes": episode_records,
    }
    index_path = output_root / "index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    index_path = build_sidecar(
        raw_root=args.raw_root.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
    )
    print(index_path)


if __name__ == "__main__":
    main()
