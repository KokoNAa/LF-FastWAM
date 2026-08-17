#!/usr/bin/env python3
"""Replay audited PGC episodes and add current-state object-mask sidecars.

V7 never consumes these masks at deployment. They are training-only labels for
the language-to-visible-object binding module. The RGB videos, actions,
normalization statistics, and PGC episode provenance remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from fastwam.datasets.counterfactual import stable_instruction_id  # noqa: E402
from fastwam.datasets.pgc_libero import (  # noqa: E402
    PGC_TARGET_MASK_FORMAT,
    PGC_TARGET_MASK_INDEX,
    atomic_write_json,
    goal_subject,
    read_jsonl,
)
from scripts.build_pgc_libero_data import (  # noqa: E402
    PGC_EPISODES,
    PGC_PROVENANCE,
    _get_inner_env,
    _goal_satisfied,
    _load_suite_manifest,
    _make_source_env,
    _reset_exact_state,
)


CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an audited PGC LeRobot dataset with robosuite element "
            "segmentation and save compact V7 object-mask labels."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-height", type=int, default=56)
    parser.add_argument("--mask-width", type=int, default=112)
    parser.add_argument("--state-atol", type=float)
    parser.add_argument("--settle-steps", type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing episode mask files in place.",
    )
    args = parser.parse_args()
    if args.mask_height <= 0 or args.mask_width <= 0 or args.mask_width % 2:
        parser.error("mask dimensions must be positive and --mask-width must be even")
    if args.state_atol is not None and args.state_atol < 0:
        parser.error("--state-atol must be non-negative")
    if args.settle_steps is not None and args.settle_steps < 0:
        parser.error("--settle-steps must be non-negative")
    return args


def _source_goal_state(record: Mapping[str, Any]) -> list[list[Any]]:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import bddl_utils as BDDLUtils

    suite = benchmark.get_benchmark_dict()[str(record["task_suite_name"])]()
    task = suite.get_task(int(record["task_id"]))
    bddl_path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    problem = BDDLUtils.robosuite_parse_problem(str(bddl_path))
    goal = problem.get("goal_state") or []
    if not isinstance(goal, list) or not goal:
        raise ValueError(f"Source task has no goal state: {bddl_path}")
    return [list(predicate) for predicate in goal]


def _build_object_catalog(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_instruction: dict[str, dict[str, Any]] = {}
    for record in records:
        candidates = (
            (
                str(record["correct_instruction"]).strip(),
                goal_subject(_source_goal_state(record)),
                str(record["task_suite_name"]),
                int(record["task_id"]),
            ),
            (
                str(record["counterfactual_instruction"]).strip(),
                goal_subject(record["counterfactual_goal_state"]),
                str(record["counterfactual_task_suite_name"]),
                int(record["counterfactual_task_id"]),
            ),
        )
        for instruction, object_name, suite_name, task_id in candidates:
            key = instruction.casefold()
            entry = {
                "instruction": instruction,
                "object_name": object_name,
                "suite": suite_name,
                "task_id": task_id,
                "goal_id": stable_instruction_id(instruction),
            }
            previous = by_instruction.get(key)
            if previous is not None and previous["object_name"] != object_name:
                raise ValueError(
                    f"Instruction {instruction!r} maps to conflicting objects: "
                    f"{previous['object_name']!r} and {object_name!r}."
                )
            by_instruction[key] = entry
    catalog = sorted(
        by_instruction.values(),
        key=lambda item: (str(item["suite"]), int(item["task_id"]), item["instruction"]),
    )
    for catalog_index, entry in enumerate(catalog):
        entry["catalog_index"] = catalog_index
    lookup = {
        str(entry["instruction"]).strip().casefold(): int(entry["catalog_index"])
        for entry in catalog
    }
    return catalog, lookup


def _episode_actions(dataset: Any, episode_index: int) -> np.ndarray:
    episode = dataset.get_episode_data(int(episode_index))
    action = episode.get("action")
    if action is None:
        raise KeyError(f"PGC episode {episode_index} has no action column.")
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError(
            f"PGC episode {episode_index} actions must be [T,7], got {action.shape}."
        )
    return np.ascontiguousarray(action)


def _instance_geom_ids(env: Any, object_names: list[str]) -> list[np.ndarray]:
    model = _get_inner_env(env).model
    mapping = getattr(model, "instances_to_ids", None)
    if not isinstance(mapping, Mapping):
        raise RuntimeError("robosuite model has no instances_to_ids mapping.")
    resolved: list[np.ndarray] = []
    for object_name in object_names:
        instance = mapping.get(object_name)
        if not isinstance(instance, Mapping):
            # Some object XMLs expose a model name without the trailing instance
            # suffix. Accept it only when the fallback is unique.
            stem = object_name.rsplit("_", 1)[0]
            candidates = [name for name in mapping if name == stem or name.startswith(stem + "_")]
            if len(candidates) == 1:
                instance = mapping[candidates[0]]
        geom_ids = None if not isinstance(instance, Mapping) else instance.get("geom")
        geom_ids = np.asarray(geom_ids if geom_ids is not None else [], dtype=np.int32)
        if geom_ids.ndim != 1:
            raise ValueError(
                f"Object instance {object_name!r} has invalid robosuite geom IDs."
            )
        # The suite-level catalog is intentionally stable across episodes, but
        # an individual BDDL scene need not instantiate every auxiliary object.
        # Missing catalog entries therefore become empty masks. The episode's
        # target and source objects are checked for visibility after replay.
        resolved.append(geom_ids)
    return resolved


def _resize_binary(mask: np.ndarray, *, height: int, width: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    resized = image.resize((int(width), int(height)), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def _object_masks_from_obs(
    obs: Mapping[str, Any],
    geom_ids: list[np.ndarray],
    *,
    mask_height: int,
    mask_width: int,
) -> np.ndarray:
    half_width = mask_width // 2
    camera_masks: list[np.ndarray] = []
    for camera_name in CAMERA_NAMES:
        key = f"{camera_name}_segmentation_element"
        if key not in obs:
            raise KeyError(
                f"Missing robosuite element-segmentation observation {key!r}; "
                f"available segmentation keys={sorted(k for k in obs if 'segmentation' in k)}."
            )
        segmentation = np.asarray(obs[key])
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[..., 0]
        if segmentation.ndim != 2:
            raise ValueError(f"Segmentation {key} must be HxW or HxWx1.")
        # FastWAM applies this 180-degree conversion to both RGB cameras.
        segmentation = np.ascontiguousarray(segmentation[::-1, ::-1])
        per_object = [
            _resize_binary(
                np.isin(segmentation, object_geom_ids),
                height=mask_height,
                width=half_width,
            )
            for object_geom_ids in geom_ids
        ]
        camera_masks.append(np.stack(per_object, axis=0))
    return np.concatenate(camera_masks, axis=-1)


def _replay_masks(
    env: Any,
    *,
    initial_state: np.ndarray,
    actions: np.ndarray,
    object_names: list[str],
    state_atol: float,
    settle_steps: int,
    mask_height: int,
    mask_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    obs, _ = _reset_exact_state(
        env,
        initial_state,
        state_atol=state_atol,
        settle_steps=settle_steps,
    )
    if _goal_satisfied(env, done=False):
        raise RuntimeError("Counterfactual goal is already satisfied at episode start.")
    geom_ids = _instance_geom_ids(env, object_names)
    frames: list[np.ndarray] = []
    done = False
    for action in actions:
        frames.append(
            _object_masks_from_obs(
                obs,
                geom_ids,
                mask_height=mask_height,
                mask_width=mask_width,
            )
        )
        obs, _, done, _ = env.step(np.asarray(action, dtype=np.float32).tolist())
    if not _goal_satisfied(env, done=bool(done)):
        raise RuntimeError(
            "Segmentation replay did not reproduce the audited counterfactual success."
        )
    masks = np.stack(frames, axis=0).astype(np.bool_, copy=False)
    visible = masks.reshape(masks.shape[0], masks.shape[1], -1).any(axis=-1)
    return masks, visible


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_masks(
    path: Path,
    *,
    masks: np.ndarray,
    visible: np.ndarray,
    target_catalog_index: int,
    source_catalog_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        packed_masks=np.packbits(masks, axis=-1, bitorder="little"),
        visible=visible.astype(np.bool_, copy=False),
        frame_count=np.asarray(masks.shape[0], dtype=np.int64),
        object_count=np.asarray(masks.shape[1], dtype=np.int64),
        mask_height=np.asarray(masks.shape[2], dtype=np.int64),
        mask_width=np.asarray(masks.shape[3], dtype=np.int64),
        target_catalog_index=np.asarray(target_catalog_index, dtype=np.int64),
        source_catalog_index=np.asarray(source_catalog_index, dtype=np.int64),
    )
    temporary.replace(path)


def _validate_existing_mask_file(
    path: Path,
    *,
    frame_count: int,
    object_count: int,
    mask_height: int,
    mask_width: int,
    target_catalog_index: int,
    source_catalog_index: int,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        expected_scalars = {
            "frame_count": frame_count,
            "object_count": object_count,
            "mask_height": mask_height,
            "mask_width": mask_width,
            "target_catalog_index": target_catalog_index,
            "source_catalog_index": source_catalog_index,
        }
        for name, expected in expected_scalars.items():
            if int(payload[name]) != int(expected):
                raise ValueError(
                    f"Existing PGC mask file {path} has {name}={int(payload[name])}, "
                    f"expected {expected}. Use --overwrite after checking the dataset."
                )
        visible = np.asarray(payload["visible"], dtype=np.bool_)
        if visible.shape != (frame_count, object_count):
            raise ValueError(f"Existing PGC mask visibility shape is invalid: {path}.")
        expected_packed = (frame_count, object_count, mask_height, (mask_width + 7) // 8)
        if tuple(payload["packed_masks"].shape) != expected_packed:
            raise ValueError(f"Existing PGC packed-mask shape is invalid: {path}.")
        return visible


def _write_index(
    dataset_root: Path,
    *,
    suite: str,
    seed: int,
    camera_resolution: int,
    mask_height: int,
    mask_width: int,
    catalog: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> None:
    atomic_write_json(
        dataset_root / PGC_TARGET_MASK_INDEX,
        {
            "format": PGC_TARGET_MASK_FORMAT,
            "benchmark": "libero",
            "suite": suite,
            "training_only": True,
            "render_segmentation": "robosuite_element",
            "camera_names": list(CAMERA_NAMES),
            "camera_resolution": camera_resolution,
            "mask_size": [mask_height, mask_width],
            "packing": {"axis": -1, "bitorder": "little"},
            "collection_seed": seed,
            "object_catalog": catalog,
            "episodes": sorted(entries, key=lambda item: int(item["episode_index"])),
        },
    )


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    provenance_path = dataset_root / PGC_PROVENANCE
    episodes_path = dataset_root / PGC_EPISODES
    if not provenance_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(
            f"{dataset_root} is not an audited PGC action dataset."
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_suites = {str(item) for item in provenance.get("source_suites") or []}
    if source_suites != {args.suite}:
        raise ValueError(
            f"PGC mask suite mismatch: dataset={sorted(source_suites)} requested={args.suite!r}."
        )
    records = _load_suite_manifest(manifest_path, args.suite)
    records_by_pair = {str(record["pair_id"]): record for record in records}
    audits = sorted(
        read_jsonl(episodes_path), key=lambda item: int(item["episode_index"])
    )
    if len(audits) != int(provenance.get("successful_episode_count", -1)):
        raise ValueError("PGC provenance/audit episode counts disagree.")
    catalog, catalog_lookup = _build_object_catalog(records)
    object_names = [str(entry["object_name"]) for entry in catalog]
    state_atol = (
        float(args.state_atol)
        if args.state_atol is not None
        else float(provenance.get("state_match_tolerance", 1.0e-7))
    )
    settle_steps = (
        int(args.settle_steps)
        if args.settle_steps is not None
        else int(provenance.get("settle_steps", 10))
    )
    camera_resolution = int(provenance.get("camera_resolution", 512))

    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)
    entries: list[dict[str, Any]] = []
    for audit in audits:
        episode_index = int(audit["episode_index"])
        pair_id = str(audit["pair_id"])
        if pair_id not in records_by_pair:
            raise ValueError(f"Episode {episode_index} references unknown pair {pair_id!r}.")
        record = records_by_pair[pair_id]
        pair_offset = records.index(record)
        target_instruction = str(record["counterfactual_instruction"]).strip()
        source_instruction = str(record["correct_instruction"]).strip()
        target_index = catalog_lookup[target_instruction.casefold()]
        source_index = catalog_lookup[source_instruction.casefold()]
        actions = _episode_actions(dataset, episode_index)
        if actions.shape[0] != int(audit["recorded_action_count"]):
            raise ValueError(
                f"Episode {episode_index} action count changed: "
                f"dataset={actions.shape[0]} audit={audit['recorded_action_count']}."
            )
        mask_relpath = Path("meta/pgc_v7_target_masks") / f"episode_{episode_index:06d}.npz"
        mask_path = dataset_root / mask_relpath
        if mask_path.is_file() and not args.overwrite:
            visible = _validate_existing_mask_file(
                mask_path,
                frame_count=actions.shape[0],
                object_count=len(catalog),
                mask_height=args.mask_height,
                mask_width=args.mask_width,
                target_catalog_index=target_index,
                source_catalog_index=source_index,
            )
            status = "VERIFIED"
        else:
            initial_state = np.load(
                dataset_root / str(audit["source_initial_state_catalog"]),
                allow_pickle=False,
            )
            env = None
            try:
                env, _ = _make_source_env(
                    record,
                    resolution=camera_resolution,
                    seed=args.seed + pair_offset,
                    camera_segmentations="element",
                )
                masks, visible = _replay_masks(
                    env,
                    initial_state=initial_state,
                    actions=actions,
                    object_names=object_names,
                    state_atol=state_atol,
                    settle_steps=settle_steps,
                    mask_height=args.mask_height,
                    mask_width=args.mask_width,
                )
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
            _atomic_save_masks(
                mask_path,
                masks=masks,
                visible=visible,
                target_catalog_index=target_index,
                source_catalog_index=source_index,
            )
            status = "SAVED"
        if not bool(visible[:, target_index].any()):
            raise RuntimeError(
                f"Counterfactual target {target_instruction!r} is never visible "
                f"in episode {episode_index}."
            )
        if not bool(visible[:, source_index].any()):
            raise RuntimeError(
                f"Source target {source_instruction!r} is never visible in "
                f"episode {episode_index}."
            )
        entries.append(
            {
                "episode_index": episode_index,
                "pair_id": pair_id,
                "file": str(mask_relpath),
                "sha256": _sha256(mask_path),
                "frame_count": int(actions.shape[0]),
                "target_catalog_index": target_index,
                "source_catalog_index": source_index,
                "target_visible_frames": int(visible[:, target_index].sum()),
                "source_visible_frames": int(visible[:, source_index].sum()),
                "visible_catalog_entries": int(visible.any(axis=0).sum()),
            }
        )
        _write_index(
            dataset_root,
            suite=args.suite,
            seed=args.seed,
            camera_resolution=camera_resolution,
            mask_height=args.mask_height,
            mask_width=args.mask_width,
            catalog=catalog,
            entries=entries,
        )
        print(
            f"{status} masks episode={episode_index} pair={pair_id} "
            f"frames={actions.shape[0]} progress={len(entries)}/{len(audits)}",
            flush=True,
        )
    if len(entries) != len(audits):
        raise RuntimeError("PGC V7 target-mask construction is incomplete.")
    print(dataset_root / PGC_TARGET_MASK_INDEX)


if __name__ == "__main__":
    main()
