import hashlib
import os
from pathlib import Path
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from collections import OrderedDict
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from ..counterfactual import (
    load_counterfactual_instruction_map,
    stable_instruction_id,
)
from ..pgc_libero import (
    PGC_ACTION_CONVENTION_FASTWAM,
    PGC_ENTITY_RELATION_ARRAY_NAMES,
    array_sha256,
    classify_strict_conflict,
    load_pgc_closed_loop_corrective_index,
    load_pgc_completion_phase_index,
    load_pgc_entity_relation_index,
    load_pgc_episode_language_pairs,
    load_pgc_target_mask_index,
    read_jsonl,
    state_sha256,
)
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def build_pgc_sample_indices(
    *,
    native_frame_count: int,
    total_frame_count: int,
    counterfactual_oversample_factor: int = 1,
    balance_native_counterfactual: bool = False,
) -> list[int]:
    """Build deterministic native/CF indices with an optional exact 1:1 mix."""
    native_frame_count = int(native_frame_count)
    total_frame_count = int(total_frame_count)
    counterfactual_oversample_factor = int(
        counterfactual_oversample_factor
    )
    if not 0 <= native_frame_count <= total_frame_count:
        raise ValueError("PGC native/total frame counts are inconsistent.")
    if counterfactual_oversample_factor < 1:
        raise ValueError("`counterfactual_oversample_factor` must be >= 1.")

    native = list(range(native_frame_count))
    counterfactual = list(range(native_frame_count, total_frame_count))
    if not counterfactual:
        return native
    if not native:
        raise ValueError("PGC counterfactual data requires native policy data.")
    if balance_native_counterfactual:
        if counterfactual_oversample_factor != 1:
            raise ValueError(
                "Exact PGC 1:1 balancing is mutually exclusive with a manual "
                "counterfactual oversample factor."
            )

        target_count = max(len(native), len(counterfactual))

        def _repeat_to(values: list[int], count: int) -> list[int]:
            repeats = (count + len(values) - 1) // len(values)
            return (values * repeats)[:count]

        return _repeat_to(native, target_count) + _repeat_to(
            counterfactual, target_count
        )

    return native + counterfactual * counterfactual_oversample_factor


def build_pgc_v8_sample_indices(
    *,
    native_frame_count: int,
    offline_counterfactual_frame_count: int,
    total_frame_count: int,
    closed_loop_oversample_factor: int = 4,
    balance_native_counterfactual: bool = True,
) -> list[int]:
    """Build a deterministic V8 native/offline/closed-loop sample mixture.

    The corrective set is usually much smaller than the original successful
    demonstration set.  It is repeated *inside* the counterfactual half, then
    the native half is repeated to the same total size.  This retains an exact
    1:1 policy-protection mix without allowing broad native zero-residual data
    to drown out the deployment-state corrections.
    """
    native_frame_count = int(native_frame_count)
    offline_counterfactual_frame_count = int(
        offline_counterfactual_frame_count
    )
    total_frame_count = int(total_frame_count)
    closed_loop_oversample_factor = int(closed_loop_oversample_factor)
    if not (
        0 <= native_frame_count <= offline_counterfactual_frame_count
        <= total_frame_count
    ):
        raise ValueError("PGC V8 frame boundaries are inconsistent.")
    if closed_loop_oversample_factor < 1:
        raise ValueError("PGC V8 closed-loop oversample factor must be >= 1.")
    native = list(range(native_frame_count))
    offline = list(
        range(native_frame_count, offline_counterfactual_frame_count)
    )
    corrective = list(
        range(offline_counterfactual_frame_count, total_frame_count)
    )
    if not native or not offline or not corrective:
        raise ValueError(
            "PGC V8 requires non-empty native, offline CF, and closed-loop CF data."
        )
    counterfactual = offline + corrective * closed_loop_oversample_factor
    if not balance_native_counterfactual:
        return native + counterfactual

    def _repeat_to(values: list[int], count: int) -> list[int]:
        repeats = (count + len(values) - 1) // len(values)
        return (values * repeats)[:count]

    target_count = max(len(native), len(counterfactual))
    return _repeat_to(native, target_count) + _repeat_to(
        counterfactual, target_count
    )


def _repeat_indices(values: list[int], count: int) -> list[int]:
    if not values:
        raise ValueError("Cannot repeat an empty PGC sample pool.")
    repeats = (int(count) + len(values) - 1) // len(values)
    return (list(values) * repeats)[: int(count)]


def build_pgc_v9_sample_indices(
    *,
    native_indices: list[int],
    original_counterfactual_indices: list[int],
    strict_counterfactual_indices: list[int],
    strict_relation_categories: list[str],
) -> list[int]:
    """Build the exact V9 native/original/strict training mixture.

    The returned deterministic pool has native:counterfactual=1:1 and, inside
    the counterfactual half, original:strict=1:1.  Strict samples are first
    balanced over their audited conflict categories.  DistributedSampler may
    shuffle this pool later without changing any of those multiplicities.
    """
    native = [int(index) for index in native_indices]
    original = [int(index) for index in original_counterfactual_indices]
    strict = [int(index) for index in strict_counterfactual_indices]
    categories = [str(value).strip() for value in strict_relation_categories]
    if not native or not original or not strict:
        raise ValueError(
            "PGC v9 requires non-empty native, original-CF, and strict-CF pools."
        )
    if len(strict) != len(categories) or any(not value for value in categories):
        raise ValueError(
            "PGC v9 strict relation labels must cover every strict-CF sample."
        )
    if len(set(native + original + strict)) != len(native) + len(original) + len(strict):
        raise ValueError("PGC v9 sample pools must be disjoint.")

    grouped: dict[str, list[int]] = {}
    for index, category in zip(strict, categories, strict=True):
        grouped.setdefault(category, []).append(index)
    category_count = max(len(values) for values in grouped.values())
    balanced_groups = {
        category: _repeat_indices(values, category_count)
        for category, values in sorted(grouped.items())
    }
    strict_balanced = [
        balanced_groups[category][position]
        for position in range(category_count)
        for category in sorted(balanced_groups)
    ]

    # Choose a common CF-subset size that is also a multiple of the number of
    # strict categories.  This preserves all three ratios exactly even when
    # the native dataset is much larger than either CF dataset:
    # native:CF=1:1, original:strict=1:1, and equal strict-category counts.
    minimum_subset_count = max(
        len(original),
        len(strict_balanced),
        (len(native) + 1) // 2,
    )
    category_total = len(balanced_groups)
    counterfactual_subset_count = (
        (minimum_subset_count + category_total - 1) // category_total
    ) * category_total
    original_balanced = _repeat_indices(original, counterfactual_subset_count)
    strict_balanced = _repeat_indices(strict_balanced, counterfactual_subset_count)
    counterfactual = original_balanced + strict_balanced
    return _repeat_indices(native, len(counterfactual)) + counterfactual


class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        counterfactual_manifest_path: Optional[str] = None,
        counterfactual_negative_probability: float = 0.0,
        pgc_counterfactual_dataset_dirs: Optional[list[str]] = None,
        pgc_closed_loop_corrective_dataset_dirs: Optional[list[str]] = None,
        pgc_counterfactual_oversample_factor: int = 1,
        pgc_closed_loop_corrective_oversample_factor: int = 4,
        pgc_balance_native_counterfactual: bool = False,
        pgc_target_mask_supervision_required: bool = False,
        pgc_completion_phase_supervision_required: bool = False,
        pgc_entity_relation_supervision_required: bool = False,
        pgc_entity_relation_sidecar_dirs: Optional[list[str]] = None,
        pgc_v9_balanced_sampling: bool = False,
    ):
        native_dataset_dirs = [str(path) for path in dataset_dirs]
        pgc_counterfactual_dataset_dirs = [
            str(path) for path in (pgc_counterfactual_dataset_dirs or [])
        ]
        pgc_closed_loop_corrective_dataset_dirs = [
            str(path)
            for path in (pgc_closed_loop_corrective_dataset_dirs or [])
        ]
        duplicate_dirs = set(native_dataset_dirs) & set(
            pgc_counterfactual_dataset_dirs
            + pgc_closed_loop_corrective_dataset_dirs
        )
        duplicate_counterfactual_dirs = set(
            pgc_counterfactual_dataset_dirs
        ) & set(pgc_closed_loop_corrective_dataset_dirs)
        if duplicate_dirs:
            raise ValueError(
                "PGC direct-counterfactual datasets must be distinct from the "
                f"native datasets; duplicates={sorted(duplicate_dirs)}."
            )
        if duplicate_counterfactual_dirs:
            raise ValueError(
                "PGC V8 closed-loop datasets must be distinct from offline "
                "counterfactual datasets; duplicates="
                f"{sorted(duplicate_counterfactual_dirs)}."
            )
        self.pgc_counterfactual_oversample_factor = int(
            pgc_counterfactual_oversample_factor
        )
        if self.pgc_counterfactual_oversample_factor < 1:
            raise ValueError("`pgc_counterfactual_oversample_factor` must be >= 1.")
        self.pgc_native_dataset_count = len(native_dataset_dirs)
        self.pgc_offline_counterfactual_dataset_count = len(
            pgc_counterfactual_dataset_dirs
        )
        self.pgc_closed_loop_corrective_dataset_dirs = (
            pgc_closed_loop_corrective_dataset_dirs
        )
        self.pgc_counterfactual_dataset_dirs = (
            pgc_counterfactual_dataset_dirs
            + self.pgc_closed_loop_corrective_dataset_dirs
        )
        self.pgc_has_counterfactual_data = bool(
            self.pgc_counterfactual_dataset_dirs
        )
        self.pgc_has_closed_loop_corrective_data = bool(
            self.pgc_closed_loop_corrective_dataset_dirs
        )
        self.pgc_closed_loop_corrective_oversample_factor = int(
            pgc_closed_loop_corrective_oversample_factor
        )
        if self.pgc_closed_loop_corrective_oversample_factor < 1:
            raise ValueError(
                "`pgc_closed_loop_corrective_oversample_factor` must be >= 1."
            )
        self.pgc_episode_language_pairs: dict[
            int, dict[int, dict[str, object]]
        ] = {
            self.pgc_native_dataset_count + offset: (
                load_pgc_episode_language_pairs(dataset_dir)
            )
            for offset, dataset_dir in enumerate(
                self.pgc_counterfactual_dataset_dirs
            )
        }
        self.pgc_closed_loop_corrective_indices: dict[
            int, dict[int, dict[str, object]]
        ] = {
            (
                self.pgc_native_dataset_count
                + self.pgc_offline_counterfactual_dataset_count
                + offset
            ): load_pgc_closed_loop_corrective_index(dataset_dir)
            for offset, dataset_dir in enumerate(
                self.pgc_closed_loop_corrective_dataset_dirs
            )
        }
        self.pgc_target_mask_supervision_required = bool(
            pgc_target_mask_supervision_required
        )
        self.pgc_completion_phase_supervision_required = bool(
            pgc_completion_phase_supervision_required
        )
        self.pgc_entity_relation_supervision_required = bool(
            pgc_entity_relation_supervision_required
        )
        self.pgc_v9_balanced_sampling = bool(pgc_v9_balanced_sampling)
        if self.pgc_completion_phase_supervision_required and not (
            self.pgc_counterfactual_dataset_dirs
        ):
            raise ValueError(
                "PGC V5 completion supervision requires at least one direct "
                "counterfactual dataset."
            )
        self.pgc_completion_phase_indices: dict[
            int, dict[int, dict[str, object]]
        ] = {}
        if self.pgc_completion_phase_supervision_required:
            self.pgc_completion_phase_indices = {
                self.pgc_native_dataset_count + offset: (
                    load_pgc_completion_phase_index(dataset_dir)
                )
                for offset, dataset_dir in enumerate(
                    self.pgc_counterfactual_dataset_dirs
                )
            }
        if self.pgc_target_mask_supervision_required and not (
            self.pgc_counterfactual_dataset_dirs
        ):
            raise ValueError(
                "PGC V7 target-mask supervision requires at least one direct "
                "counterfactual dataset."
            )
        self.pgc_target_mask_indices: dict[int, dict[str, object]] = {}
        if self.pgc_target_mask_supervision_required:
            self.pgc_target_mask_indices = {
                self.pgc_native_dataset_count + offset: (
                    load_pgc_target_mask_index(dataset_dir)
                )
                for offset, dataset_dir in enumerate(
                    self.pgc_counterfactual_dataset_dirs
                )
            }
            mask_shapes = {
                tuple(index["mask_size"])
                for index in self.pgc_target_mask_indices.values()
            }
            if len(mask_shapes) != 1:
                raise ValueError(
                    f"PGC V7 target-mask datasets disagree on shape: {mask_shapes}."
                )
            self.pgc_target_mask_shape = next(iter(mask_shapes))
        else:
            self.pgc_target_mask_shape = (56, 112)
        # Each DataLoader worker owns a small decompressed packed-mask cache.
        self._pgc_target_mask_cache: OrderedDict[
            tuple[int, int], tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()
        self._pgc_target_mask_cache_size = 4
        combined_dataset_dirs = (
            native_dataset_dirs + self.pgc_counterfactual_dataset_dirs
        )
        pgc_entity_relation_sidecar_dirs = [
            str(path)
            for path in (pgc_entity_relation_sidecar_dirs or [])
        ]
        if self.pgc_entity_relation_supervision_required and len(
            pgc_entity_relation_sidecar_dirs
        ) != len(combined_dataset_dirs):
            raise ValueError(
                "PGC v9 requires one entity-relation sidecar per native/CF "
                "dataset in the same order: "
                f"datasets={len(combined_dataset_dirs)} "
                f"sidecars={len(pgc_entity_relation_sidecar_dirs)}."
            )
        self.pgc_entity_relation_indices: dict[int, dict[str, object]] = {}
        if self.pgc_entity_relation_supervision_required:
            self.pgc_entity_relation_indices = {
                dataset_index: load_pgc_entity_relation_index(path)
                for dataset_index, path in enumerate(
                    pgc_entity_relation_sidecar_dirs
                )
            }
            for dataset_index, dataset_dir in enumerate(combined_dataset_dirs):
                audited_dataset = os.path.realpath(
                    str(self.pgc_entity_relation_indices[dataset_index]["dataset"])
                )
                if audited_dataset != os.path.realpath(dataset_dir):
                    raise ValueError(
                        "PGC v9 sidecar/dataset order mismatch at index "
                        f"{dataset_index}: sidecar={audited_dataset} "
                        f"dataset={os.path.realpath(dataset_dir)}."
                    )
            eraf_mask_shapes = {
                tuple(index["mask_size"])
                for index in self.pgc_entity_relation_indices.values()
            }
            if len(eraf_mask_shapes) != 1:
                raise ValueError(
                    "PGC v9 sidecars disagree on mask size: "
                    f"{sorted(eraf_mask_shapes)}."
                )
            self.pgc_entity_relation_mask_shape = next(
                iter(eraf_mask_shapes)
            )
        else:
            self.pgc_entity_relation_mask_shape = (56, 112)
        self._pgc_entity_relation_cache: OrderedDict[
            tuple[int, int], dict[str, np.ndarray]
        ] = OrderedDict()
        self._pgc_entity_relation_cache_size = 2
        self.pgc_balance_native_counterfactual = bool(
            pgc_balance_native_counterfactual
        )
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=combined_dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )
        if self.pgc_entity_relation_supervision_required:
            action_conventions = {
                dataset_index: str(index["dataset_action_convention"])
                for dataset_index, index in self.pgc_entity_relation_indices.items()
            }
            self.lerobot_dataset.set_action_conventions_by_dataset_index(
                action_conventions
            )
            aligned_count = sum(
                convention != PGC_ACTION_CONVENTION_FASTWAM
                for convention in action_conventions.values()
            )
            logger.info(
                "Configured PGC v9 action conventions for %d datasets "
                "(%d aligned to FastWAM before preprocessing).",
                len(action_conventions),
                aligned_count,
            )
        underlying = self.lerobot_dataset.multi_dataset._datasets
        if self.pgc_entity_relation_supervision_required:
            self._validate_pgc_entity_relation_dataset_audits(
                underlying=underlying,
                combined_dataset_dirs=combined_dataset_dirs,
            )
        self.pgc_native_frame_count = sum(
            int(dataset.num_frames)
            for dataset in underlying[: self.pgc_native_dataset_count]
        )
        offline_dataset_end = (
            self.pgc_native_dataset_count
            + self.pgc_offline_counterfactual_dataset_count
        )
        self.pgc_offline_counterfactual_frame_end = sum(
            int(dataset.num_frames)
            for dataset in underlying[:offline_dataset_end]
        )
        total_frame_count = sum(int(dataset.num_frames) for dataset in underlying)
        if self.pgc_v9_balanced_sampling:
            if not self.pgc_entity_relation_supervision_required:
                raise ValueError(
                    "PGC v9 balanced sampling requires audited ERAF sidecars."
                )
            if self.pgc_has_closed_loop_corrective_data:
                raise ValueError(
                    "PGC v9 strict sampling cannot mix V8 corrective datasets."
                )
            if self.pgc_offline_counterfactual_dataset_count != 2:
                raise ValueError(
                    "PGC v9 requires exactly two CF datasets ordered as "
                    "[original, strict]."
                )
            dataset_offsets: list[int] = []
            offset = 0
            for dataset in underlying:
                dataset_offsets.append(offset)
                offset += int(dataset.num_frames)
            original_dataset_index = self.pgc_native_dataset_count
            strict_dataset_index = original_dataset_index + 1
            original_start = dataset_offsets[original_dataset_index]
            strict_start = dataset_offsets[strict_dataset_index]
            original = list(range(original_start, strict_start))
            strict_dataset = underlying[strict_dataset_index]
            strict = list(range(strict_start, total_frame_count))
            episode_column = strict_dataset.hf_dataset["episode_index"]
            strict_categories: list[str] = []
            strict_index = self.pgc_entity_relation_indices[strict_dataset_index]
            strict_pairs = self.pgc_episode_language_pairs[strict_dataset_index]
            strict_covered_tasks = {
                int(pair["source_task_id"]) for pair in strict_pairs.values()
            }
            if len(strict_covered_tasks) < 8:
                raise ValueError(
                    "PGC v9 strict-conflict training requires audited coverage "
                    f"of at least 8/10 source tasks, got "
                    f"{len(strict_covered_tasks)}/10."
                )
            for raw_episode_index in episode_column:
                episode_index = int(
                    torch.as_tensor(raw_episode_index).reshape(-1)[0].item()
                )
                try:
                    episode_record = strict_index["episodes_by_index"][episode_index]
                    pair_audit = strict_pairs[episode_index]
                except KeyError as exc:
                    raise KeyError(
                        "PGC v9 strict sidecar does not cover episode "
                        f"{episode_index}."
                    ) from exc
                category = classify_strict_conflict(
                    episode_record["source_clauses"],
                    episode_record["target_clauses"],
                )
                if category is None:
                    raise ValueError(
                        "PGC v9 strict dataset contains a non-conflicting pair: "
                        f"{episode_record.get('pair_id')}."
                    )
                if (
                    pair_audit.get("strict_conflict") is not True
                    or pair_audit.get("strict_conflict_type") != category
                    or not isinstance(
                        pair_audit.get("strict_replay_audit"), dict
                    )
                    or pair_audit["strict_replay_audit"].get(
                        "strict_conflict_passed"
                    )
                    is not True
                ):
                    raise ValueError(
                        "PGC v9 strict dataset lacks a matching successful "
                        f"bidirectional audit for episode {episode_index}."
                    )
                strict_categories.append(category)
            self._sample_indices = build_pgc_v9_sample_indices(
                native_indices=list(range(self.pgc_native_frame_count)),
                original_counterfactual_indices=original,
                strict_counterfactual_indices=strict,
                strict_relation_categories=strict_categories,
            )
        elif self.pgc_has_closed_loop_corrective_data:
            if self.pgc_counterfactual_oversample_factor != 1:
                raise ValueError(
                    "PGC V8 uses its dedicated closed-loop oversample factor; "
                    "set pgc_counterfactual_oversample_factor=1."
                )
            self._sample_indices = build_pgc_v8_sample_indices(
                native_frame_count=self.pgc_native_frame_count,
                offline_counterfactual_frame_count=(
                    self.pgc_offline_counterfactual_frame_end
                ),
                total_frame_count=total_frame_count,
                closed_loop_oversample_factor=(
                    self.pgc_closed_loop_corrective_oversample_factor
                ),
                balance_native_counterfactual=(
                    self.pgc_balance_native_counterfactual
                ),
            )
        else:
            self._sample_indices = build_pgc_sample_indices(
                native_frame_count=self.pgc_native_frame_count,
                total_frame_count=total_frame_count,
                counterfactual_oversample_factor=(
                    self.pgc_counterfactual_oversample_factor
                ),
                balance_native_counterfactual=(
                    self.pgc_balance_native_counterfactual
                ),
            )
        self.pgc_effective_native_sample_count = sum(
            int(index < self.pgc_native_frame_count)
            for index in self._sample_indices
        )
        self.pgc_effective_counterfactual_sample_count = (
            len(self._sample_indices)
            - self.pgc_effective_native_sample_count
        )
        self.pgc_effective_closed_loop_corrective_sample_count = sum(
            int(index >= self.pgc_offline_counterfactual_frame_end)
            for index in self._sample_indices
        )
        if self.pgc_has_counterfactual_data:
            logger.info(
                "PGC sampling: native=%d counterfactual=%d closed_loop=%d balanced=%s",
                self.pgc_effective_native_sample_count,
                self.pgc_effective_counterfactual_sample_count,
                self.pgc_effective_closed_loop_corrective_sample_count,
                self.pgc_balance_native_counterfactual,
            )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        if self.pgc_has_counterfactual_data and self.override_instruction is not None:
            raise ValueError(
                "PGC direct-action datasets cannot be combined with "
                "`override_instruction`; their recorded task text is part of "
                "the positive supervision contract."
            )
        if self.pgc_has_counterfactual_data and counterfactual_manifest_path:
            raise ValueError(
                "PGC direct-action datasets cannot be combined with the TC "
                "negative-only counterfactual manifest."
            )
        self.counterfactual_negative_probability = float(
            counterfactual_negative_probability
        )
        if not 0.0 <= self.counterfactual_negative_probability <= 1.0:
            raise ValueError(
                "`counterfactual_negative_probability` must be in [0,1]."
            )
        if self.override_instruction is not None and counterfactual_manifest_path:
            raise ValueError(
                "Counterfactual ranking cannot be combined with override_instruction."
            )
        if self.counterfactual_negative_probability > 0 and (
            not counterfactual_manifest_path
        ):
            raise ValueError(
                "A counterfactual manifest is required when negative sampling is enabled."
            )
        self.counterfactual_instruction_map = (
            load_counterfactual_instruction_map(counterfactual_manifest_path)
            if counterfactual_manifest_path
            else None
        )

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self._sample_indices)

    def _validate_pgc_entity_relation_dataset_audits(
        self,
        *,
        underlying,
        combined_dataset_dirs: list[str],
    ) -> None:
        """Cross-check every ERAF sidecar against the loaded policy dataset.

        The standalone sidecar loader protects each NPZ with a file digest.
        This second audit binds those labels to the actual LeRobot action rows
        and, for collected PGC datasets, to the exact saved simulator state and
        pair provenance.  It runs once before any optimizer step and never on
        the deployment path.
        """
        for dataset_index, (dataset, dataset_dir) in enumerate(
            zip(underlying, combined_dataset_dirs, strict=True)
        ):
            index = self.pgc_entity_relation_indices[dataset_index]
            records = index["episodes_by_index"]
            expected_episode_count = int(dataset.meta.total_episodes)
            if int(index.get("episode_count", -1)) != expected_episode_count:
                raise ValueError(
                    "PGC v9 ERAF sidecar episode_count does not match the "
                    f"LeRobot dataset at index {dataset_index}: "
                    f"sidecar={index.get('episode_count')} "
                    f"dataset={expected_episode_count}."
                )
            if set(records) != set(range(expected_episode_count)):
                raise ValueError(
                    "PGC v9 ERAF sidecar episode indices must be dense and "
                    f"complete at dataset index {dataset_index}."
                )

            selected_episode_ids = (
                list(dataset.episodes)
                if dataset.episodes is not None
                else list(range(expected_episode_count))
            )
            pgc_audits: dict[int, dict[str, object]] | None = None
            dataset_root = Path(dataset_dir).expanduser().resolve()
            audit_path = dataset_root / "meta/pgc_episodes.jsonl"
            if dataset_index >= self.pgc_native_dataset_count:
                if not audit_path.is_file():
                    raise FileNotFoundError(
                        f"PGC v9 counterfactual audit is missing: {audit_path}."
                    )
                pgc_audits = {
                    int(record["episode_index"]): dict(record)
                    for record in read_jsonl(audit_path)
                }
                if set(pgc_audits) != set(range(expected_episode_count)):
                    raise ValueError(
                        "PGC v9 counterfactual episode audit must be dense and "
                        f"complete at dataset index {dataset_index}."
                    )

            for local_episode_index, episode_index in enumerate(
                selected_episode_ids
            ):
                episode_index = int(episode_index)
                record = records[episode_index]
                episode = dataset.get_episode_data(local_episode_index)
                if "action" not in episode:
                    raise KeyError(
                        "PGC v9 dataset episode has no raw `action` column at "
                        f"dataset/episode {dataset_index}/{episode_index}."
                    )
                action = episode["action"]
                if hasattr(action, "detach"):
                    action = action.detach().cpu().numpy()
                action = np.ascontiguousarray(
                    np.asarray(action, dtype=np.float32)
                )
                if action.ndim != 2 or action.shape[1] != 7:
                    raise ValueError(
                        "PGC v9 audited actions must be [T,7], got "
                        f"{action.shape} at dataset/episode "
                        f"{dataset_index}/{episode_index}."
                    )
                if int(record["frame_count"]) != int(action.shape[0]):
                    raise ValueError(
                        "PGC v9 ERAF frame/action count mismatch at "
                        f"dataset/episode {dataset_index}/{episode_index}."
                    )
                if array_sha256(action) != str(record["action_sha256"]):
                    raise ValueError(
                        "PGC v9 ERAF action hash does not match the loaded "
                        f"LeRobot episode {dataset_index}/{episode_index}."
                    )

                if pgc_audits is None:
                    continue
                audit = pgc_audits[episode_index]
                if str(audit.get("pair_id", "")) != str(
                    record.get("pair_id", "")
                ):
                    raise ValueError(
                        "PGC v9 ERAF pair audit mismatch at dataset/episode "
                        f"{dataset_index}/{episode_index}."
                    )
                expected_state_digest = str(
                    record.get("initial_state_sha256", "")
                )
                if expected_state_digest != str(
                    audit.get("initial_state_sha256", "")
                ):
                    raise ValueError(
                        "PGC v9 ERAF initial-state provenance mismatch at "
                        f"dataset/episode {dataset_index}/{episode_index}."
                    )
                state_relpath = Path(
                    str(audit.get("source_initial_state_catalog", ""))
                )
                if (
                    not str(state_relpath)
                    or state_relpath.is_absolute()
                    or ".." in state_relpath.parts
                ):
                    raise ValueError(
                        "PGC v9 initial-state audit path is unsafe at "
                        f"dataset/episode {dataset_index}/{episode_index}."
                    )
                state_path = dataset_root / state_relpath
                if not state_path.is_file():
                    raise FileNotFoundError(
                        f"PGC v9 initial-state audit is missing: {state_path}."
                    )
                initial_state = np.load(state_path, allow_pickle=False)
                if state_sha256(initial_state) != expected_state_digest:
                    raise ValueError(
                        "PGC v9 initial-state file hash changed at "
                        f"dataset/episode {dataset_index}/{episode_index}."
                    )

        logger.info(
            "Validated PGC v9 ERAF action/state/hash audits for %d datasets.",
            len(combined_dataset_dirs),
        )

    def _load_pgc_target_mask_episode(
        self,
        *,
        dataset_index: int,
        episode_index: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
        """Return packed-mask data and audited metadata for one CF episode."""
        try:
            index = self.pgc_target_mask_indices[int(dataset_index)]
            episode = index["episodes_by_index"][int(episode_index)]
        except KeyError as exc:
            raise KeyError(
                "No PGC V7 target-mask audit for dataset/episode "
                f"{dataset_index}/{episode_index}."
            ) from exc
        cache_key = (int(dataset_index), int(episode_index))
        cached = self._pgc_target_mask_cache.get(cache_key)
        if cached is None:
            with np.load(str(episode["mask_path"]), allow_pickle=False) as payload:
                packed = np.asarray(payload["packed_masks"], dtype=np.uint8).copy()
                visible = np.asarray(payload["visible"], dtype=np.bool_).copy()
            expected_height, expected_width = map(int, index["mask_size"])
            expected_shape = (
                int(episode["frame_count"]),
                len(index["object_catalog"]),
                expected_height,
                (expected_width + 7) // 8,
            )
            if packed.shape != expected_shape or visible.shape != expected_shape[:2]:
                raise ValueError(
                    "PGC V7 packed target-mask tensor shape changed for "
                    f"dataset/episode {dataset_index}/{episode_index}: "
                    f"packed={packed.shape} visible={visible.shape}."
                )
            cached = (packed, visible)
            self._pgc_target_mask_cache[cache_key] = cached
            self._pgc_target_mask_cache.move_to_end(cache_key)
            while len(self._pgc_target_mask_cache) > self._pgc_target_mask_cache_size:
                self._pgc_target_mask_cache.popitem(last=False)
        else:
            self._pgc_target_mask_cache.move_to_end(cache_key)
        return cached[0], cached[1], episode, index

    def _get_pgc_target_mask_sample(
        self,
        *,
        dataset_index: int,
        episode_index: int,
        frame_index: int,
    ) -> dict[str, object]:
        packed, visible, episode, index = self._load_pgc_target_mask_episode(
            dataset_index=dataset_index,
            episode_index=episode_index,
        )
        frame_index = int(frame_index)
        if not 0 <= frame_index < packed.shape[0]:
            raise IndexError(
                f"PGC V7 mask frame {frame_index} is outside episode "
                f"{episode_index} with {packed.shape[0]} frames."
            )
        width = int(index["mask_size"][1])
        frame_masks = np.unpackbits(
            packed[frame_index], axis=-1, count=width, bitorder="little"
        ).astype(np.bool_, copy=False)
        target_index = int(episode["target_catalog_index"])
        source_index = int(episode["source_catalog_index"])
        visible_indices = [
            idx
            for idx, is_visible in enumerate(visible[frame_index].tolist())
            if is_visible and idx not in {target_index, source_index}
        ]
        if not visible_indices:
            visible_indices = [
                idx
                for idx, is_visible in enumerate(visible[frame_index].tolist())
                if is_visible and idx != target_index
            ]
        aux_valid = bool(visible_indices)
        if aux_valid:
            aux_index = visible_indices[
                (int(episode_index) + frame_index) % len(visible_indices)
            ]
        else:
            aux_index = source_index
        catalog = index["object_catalog"]
        return {
            "target_mask": torch.from_numpy(frame_masks[target_index].copy()),
            "source_mask": torch.from_numpy(frame_masks[source_index].copy()),
            "aux_mask": torch.from_numpy(frame_masks[aux_index].copy()),
            "target_valid": bool(visible[frame_index, target_index]),
            "source_valid": bool(visible[frame_index, source_index]),
            "aux_valid": aux_valid and bool(visible[frame_index, aux_index]),
            "aux_instruction": str(catalog[aux_index]["instruction"]),
            "aux_goal_id": int(catalog[aux_index]["goal_id"]),
        }

    def _load_pgc_entity_relation_episode(
        self,
        *,
        dataset_index: int,
        episode_index: int,
    ) -> dict[str, np.ndarray]:
        try:
            index = self.pgc_entity_relation_indices[int(dataset_index)]
            episode = index["episodes_by_index"][int(episode_index)]
        except KeyError as exc:
            raise KeyError(
                "No PGC v9 entity-relation audit for dataset/episode "
                f"{dataset_index}/{episode_index}."
            ) from exc
        cache_key = (int(dataset_index), int(episode_index))
        cached = self._pgc_entity_relation_cache.get(cache_key)
        if cached is None:
            with np.load(str(episode["path"]), allow_pickle=False) as payload:
                cached = {
                    name: np.asarray(payload[name]).copy()
                    for name in payload.files
                }
            required = {
                f"{role}_{name}"
                for role in ("target", "source")
                for name in PGC_ENTITY_RELATION_ARRAY_NAMES
            }
            missing = sorted(required - set(cached))
            if missing:
                raise ValueError(
                    "PGC v9 episode is missing ERAF arrays: "
                    f"{missing}."
                )
            frame_count = int(episode["frame_count"])
            max_clauses = int(index["max_clauses"])
            mask_height, mask_width = map(int, index["mask_size"])
            for name, array in cached.items():
                if name in required and array.shape[:2] != (
                    frame_count,
                    max_clauses,
                ):
                    raise ValueError(
                        f"PGC v9 {name} must start with "
                        f"[{frame_count},{max_clauses}], got {array.shape}."
                    )
            for role in ("target", "source"):
                expected_scalar_shape = (frame_count, max_clauses)
                expected_vector_shape = (*expected_scalar_shape, 3)
                expected_view_visible_shape = (*expected_scalar_shape, 2)
                expected_view_center_shape = (*expected_scalar_shape, 2, 2)
                expected_mask_shape = (
                    *expected_scalar_shape,
                    mask_height,
                    mask_width,
                )
                for entity_role in ("subject", "reference"):
                    mask = cached[f"{role}_{entity_role}_masks"]
                    if mask.shape != expected_mask_shape or mask.dtype != np.bool_:
                        raise ValueError(
                            f"PGC v9 {role}_{entity_role}_masks has "
                            f"incompatible shape/dtype {mask.shape}/{mask.dtype}."
                        )
                    view_visible = cached[
                        f"{role}_{entity_role}_view_visible"
                    ]
                    if (
                        view_visible.shape != expected_view_visible_shape
                        or view_visible.dtype != np.bool_
                    ):
                        raise ValueError(
                            f"PGC v9 {role}_{entity_role}_view_visible must be "
                            f"bool {expected_view_visible_shape}, got "
                            f"{view_visible.shape}/{view_visible.dtype}."
                        )
                    view_centers = cached[
                        f"{role}_{entity_role}_view_centers"
                    ]
                    if (
                        view_centers.shape != expected_view_center_shape
                        or not np.issubdtype(
                            view_centers.dtype, np.floating
                        )
                        or not np.isfinite(view_centers).all()
                        or bool((np.abs(view_centers) > 1.00001).any())
                    ):
                        raise ValueError(
                            f"PGC v9 {role}_{entity_role}_view_centers must "
                            f"be finite normalized {expected_view_center_shape}, "
                            f"got {view_centers.shape}/{view_centers.dtype}."
                        )
                    if bool(
                        (
                            np.abs(view_centers[~view_visible]) > 1.0e-7
                        ).any()
                    ):
                        raise ValueError(
                            f"PGC v9 {role}_{entity_role}_view_centers must be "
                            "zero for invisible camera views."
                        )
                for suffix in (
                    "clause_valid",
                    "subject_mask_valid",
                    "reference_mask_valid",
                    "subject_position_valid",
                    "reference_position_valid",
                    "grasp_anchor_valid",
                    "goal_anchor_valid",
                    "interaction_anchor_valid",
                    "predicate_truth_valid",
                    "phase_valid",
                ):
                    value = cached[f"{role}_{suffix}"]
                    if value.shape != expected_scalar_shape or value.dtype != np.bool_:
                        raise ValueError(
                            f"PGC v9 {role}_{suffix} must be bool "
                            f"{expected_scalar_shape}, got {value.shape}/{value.dtype}."
                        )
                for suffix in (
                    "subject_positions",
                    "reference_positions",
                    "grasp_anchors",
                    "goal_anchors",
                    "interaction_anchors",
                ):
                    value = cached[f"{role}_{suffix}"]
                    if (
                        value.shape != expected_vector_shape
                        or not np.issubdtype(value.dtype, np.floating)
                        or not np.isfinite(value).all()
                        or bool((np.abs(value) > 1.00001).any())
                    ):
                        raise ValueError(
                            f"PGC v9 {role}_{suffix} must be finite normalized "
                            f"{expected_vector_shape}, got {value.shape}/{value.dtype}."
                        )
                clause_valid = cached[f"{role}_clause_valid"]
                predicate_ids = cached[f"{role}_predicate_ids"]
                phase_ids = cached[f"{role}_phase_ids"]
                subject_ids = cached[f"{role}_subject_entity_ids"]
                reference_ids = cached[f"{role}_reference_entity_ids"]
                if (
                    predicate_ids.shape != expected_scalar_shape
                    or not np.issubdtype(predicate_ids.dtype, np.integer)
                    or bool((predicate_ids < 0).any())
                    or bool(
                        (
                            predicate_ids
                            >= len(index["predicate_vocabulary"])
                        ).any()
                    )
                    or bool((predicate_ids[clause_valid] == 0).any())
                    or bool((predicate_ids[~clause_valid] != 0).any())
                ):
                    raise ValueError(
                        f"PGC v9 {role}_predicate_ids violate the clause schema."
                    )
                if (
                    phase_ids.shape != expected_scalar_shape
                    or not np.issubdtype(phase_ids.dtype, np.integer)
                    or bool(((phase_ids < 0) | (phase_ids > 2)).any())
                ):
                    raise ValueError(
                        f"PGC v9 {role}_phase_ids must contain only 0, 1, or 2."
                    )
                for suffix, value in (
                    ("subject_entity_ids", subject_ids),
                    ("reference_entity_ids", reference_ids),
                ):
                    if (
                        value.shape != expected_scalar_shape
                        or not np.issubdtype(value.dtype, np.integer)
                        or bool((value[clause_valid] < 0).any())
                    ):
                        raise ValueError(
                            f"PGC v9 {role}_{suffix} violates the entity schema."
                        )
                truth = cached[f"{role}_predicate_truth"]
                if (
                    truth.shape != expected_scalar_shape
                    or not np.issubdtype(truth.dtype, np.floating)
                    or not np.isfinite(truth).all()
                    or bool(((truth < 0.0) | (truth > 1.0)).any())
                ):
                    raise ValueError(
                        f"PGC v9 {role}_predicate_truth must be finite in [0,1]."
                    )
            self._pgc_entity_relation_cache[cache_key] = cached
            self._pgc_entity_relation_cache.move_to_end(cache_key)
            while (
                len(self._pgc_entity_relation_cache)
                > self._pgc_entity_relation_cache_size
            ):
                self._pgc_entity_relation_cache.popitem(last=False)
        else:
            self._pgc_entity_relation_cache.move_to_end(cache_key)
        return cached

    def _get_pgc_entity_relation_sample(
        self,
        *,
        dataset_index: int,
        episode_index: int,
        frame_index: int,
    ) -> dict[str, torch.Tensor]:
        payload = self._load_pgc_entity_relation_episode(
            dataset_index=dataset_index,
            episode_index=episode_index,
        )
        frame_count = next(iter(payload.values())).shape[0]
        if not 0 <= int(frame_index) < int(frame_count):
            raise IndexError(
                f"PGC v9 frame {frame_index} is outside episode "
                f"{episode_index} with {frame_count} frames."
            )
        result: dict[str, torch.Tensor] = {}
        for role, output_prefix in (("target", ""), ("source", "source_")):
            for name in PGC_ENTITY_RELATION_ARRAY_NAMES:
                result[f"pgc_eraf_{output_prefix}{name}"] = torch.from_numpy(
                    np.asarray(payload[f"{role}_{name}"][int(frame_index)]).copy()
                )
        return result

    def _get(self, idx):
        if not 0 <= int(idx) < len(self._sample_indices):
            raise IndexError(f"Sample index {idx} is out of bounds.")
        sample_idx = self._sample_indices[int(idx)]
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = self._sample_indices[np.random.randint(len(self))]
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        dataset_index = int(
            torch.as_tensor(sample.get("dataset_index", 0)).item()
        )
        pgc_is_counterfactual = (
            dataset_index >= self.pgc_native_dataset_count
        )
        pgc_is_closed_loop_corrective = dataset_index >= (
            self.pgc_native_dataset_count
            + self.pgc_offline_counterfactual_dataset_count
        )
        pgc_source_task = str(task)
        pgc_pair_valid = False
        pgc_completion_phase = 0
        pgc_completion_phase_valid = False
        episode_index = (
            int(torch.as_tensor(sample["episode_index"]).reshape(-1)[0].item())
            if "episode_index" in sample
            else -1
        )
        frame_index = (
            int(torch.as_tensor(sample["frame_index"]).reshape(-1)[0].item())
            if "frame_index" in sample
            else -1
        )
        if pgc_is_counterfactual:
            if "episode_index" not in sample:
                raise KeyError(
                    "PGC counterfactual samples require `episode_index` to "
                    "recover their paired source instruction."
                )
            if (
                (
                    self.pgc_target_mask_supervision_required
                    or pgc_is_closed_loop_corrective
                )
                and "frame_index" not in sample
            ):
                raise KeyError(
                    "PGC counterfactual samples require `frame_index` for "
                    "V7 masks or V8 corrective audit boundaries."
                )
            try:
                pair = self.pgc_episode_language_pairs[dataset_index][
                    episode_index
                ]
            except KeyError as exc:
                raise KeyError(
                    "No audited PGC language pair for dataset/episode "
                    f"{dataset_index}/{episode_index}."
                ) from exc
            recorded_counterfactual = str(
                pair["counterfactual_instruction"]
            ).strip()
            if str(task).strip().casefold() != recorded_counterfactual.casefold():
                raise ValueError(
                    "PGC recorded task text does not match its audited "
                    "counterfactual instruction: "
                    f"{task!r} != {recorded_counterfactual!r}."
                )
            pgc_source_task = str(pair["source_instruction"]).strip()
            pgc_pair_valid = True
            if pgc_is_closed_loop_corrective:
                try:
                    corrective_record = self.pgc_closed_loop_corrective_indices[
                        dataset_index
                    ][episode_index]
                except KeyError as exc:
                    raise KeyError(
                        "No audited PGC V8 corrective record for "
                        f"dataset/episode {dataset_index}/{episode_index}."
                    ) from exc
                if frame_index >= int(
                    corrective_record["recorded_action_count"]
                ):
                    raise ValueError(
                        "PGC V8 frame exceeds its verified correction range: "
                        f"frame={frame_index} actions="
                        f"{corrective_record['recorded_action_count']}."
                    )
            if self.pgc_completion_phase_supervision_required:
                if frame_index < 0:
                    raise KeyError("PGC completion supervision requires frame_index.")
                try:
                    phase_record = self.pgc_completion_phase_indices[dataset_index][
                        episode_index
                    ]
                except KeyError as exc:
                    raise KeyError(
                        "No audited PGC completion phase for dataset/episode "
                        f"{dataset_index}/{episode_index}."
                    ) from exc
                action_count = int(phase_record["action_count"])
                if not 0 <= frame_index < action_count:
                    raise ValueError(
                        "PGC completion frame is outside its audited action "
                        f"range: frame={frame_index} count={action_count}."
                    )
                grasp_close_step = int(phase_record["grasp_close_step"])
                release_open_step = phase_record["release_open_step"]
                pgc_completion_phase = int(frame_index >= grasp_close_step)
                if release_open_step is not None and frame_index >= int(
                    release_open_step
                ):
                    pgc_completion_phase = 2
                pgc_completion_phase_valid = True

        pgc_eraf_sample: dict[str, torch.Tensor] = {}
        if self.pgc_entity_relation_supervision_required:
            if episode_index < 0 or frame_index < 0:
                raise KeyError(
                    "PGC v9 entity-relation supervision requires episode_index "
                    "and frame_index for every native and counterfactual row."
                )
            pgc_eraf_sample = self._get_pgc_entity_relation_sample(
                dataset_index=dataset_index,
                episode_index=episode_index,
                frame_index=frame_index,
            )
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        if pgc_pair_valid:
            pgc_source_prompt = DEFAULT_PROMPT.format(task=pgc_source_task)
            pgc_source_context, pgc_source_context_mask = (
                self._get_cached_text_context(pgc_source_prompt)
            )
            pgc_source_context[~pgc_source_context_mask] = 0.0
            pgc_source_context_mask = torch.ones_like(
                pgc_source_context_mask
            )
        else:
            # Keep the batch schema uniform. Native rows do not use this copy;
            # their ordinary zero-residual objective remains authoritative.
            pgc_source_prompt = instruction
            pgc_source_context = context.clone()
            pgc_source_context_mask = context_mask.clone()

        mask_height, mask_width = map(int, self.pgc_target_mask_shape)
        empty_mask = torch.zeros((mask_height, mask_width), dtype=torch.bool)
        pgc_target_object_mask = empty_mask
        pgc_source_object_mask = empty_mask.clone()
        pgc_aux_object_mask = empty_mask.clone()
        pgc_target_mask_valid = False
        pgc_source_mask_valid = False
        pgc_aux_mask_valid = False
        pgc_aux_prompt = instruction
        pgc_aux_context = context.clone()
        pgc_aux_context_mask = context_mask.clone()
        pgc_aux_goal_id = stable_instruction_id(str(task))
        if self.pgc_target_mask_supervision_required and pgc_pair_valid:
            mask_sample = self._get_pgc_target_mask_sample(
                dataset_index=dataset_index,
                episode_index=episode_index,
                frame_index=frame_index,
            )
            pgc_target_object_mask = mask_sample["target_mask"]
            pgc_source_object_mask = mask_sample["source_mask"]
            pgc_aux_object_mask = mask_sample["aux_mask"]
            pgc_target_mask_valid = bool(mask_sample["target_valid"])
            pgc_source_mask_valid = bool(mask_sample["source_valid"])
            pgc_aux_mask_valid = bool(mask_sample["aux_valid"])
            pgc_aux_prompt = DEFAULT_PROMPT.format(
                task=mask_sample["aux_instruction"]
            )
            pgc_aux_context, pgc_aux_context_mask = (
                self._get_cached_text_context(pgc_aux_prompt)
            )
            pgc_aux_context[~pgc_aux_context_mask] = 0.0
            pgc_aux_context_mask = torch.ones_like(pgc_aux_context_mask)
            pgc_aux_goal_id = int(mask_sample["aux_goal_id"])

        negative_prompt = None
        negative_context = None
        negative_context_mask = None
        negative_valid = False
        if self.counterfactual_instruction_map is not None:
            task_key = str(task).strip().casefold()
            if task_key not in self.counterfactual_instruction_map:
                raise KeyError(
                    f"No audited counterfactual instruction for dataset task {task!r}."
                )
            negative_task = self.counterfactual_instruction_map[task_key]
            negative_prompt = DEFAULT_PROMPT.format(task=negative_task)
            negative_context, negative_context_mask = self._get_cached_text_context(
                negative_prompt
            )
            negative_context[~negative_context_mask] = 0.0
            negative_context_mask = torch.ones_like(negative_context_mask)
            has_realized_transition = not bool(image_is_pad[1:].all().item())
            has_action = not bool(sample["action_is_pad"].all().item())
            negative_valid = bool(
                has_realized_transition
                and has_action
                and np.random.random()
                < self.counterfactual_negative_probability
            )
        
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
            "pgc_is_counterfactual": torch.tensor(
                pgc_is_counterfactual, dtype=torch.bool
            ),
            "pgc_is_closed_loop_corrective": torch.tensor(
                pgc_is_closed_loop_corrective, dtype=torch.bool
            ),
            "pgc_direct_action_valid": torch.tensor(
                (
                    not bool(sample["action_is_pad"].all().item())
                    and not bool(image_is_pad[1:].all().item())
                ),
                dtype=torch.bool,
            ),
            "pgc_goal_id": torch.tensor(
                stable_instruction_id(task), dtype=torch.long
            ),
            "pgc_source_prompt": pgc_source_prompt,
            "pgc_source_context": pgc_source_context,
            "pgc_source_context_mask": pgc_source_context_mask,
            "pgc_source_goal_id": torch.tensor(
                stable_instruction_id(pgc_source_task), dtype=torch.long
            ),
            "pgc_paired_language_valid": torch.tensor(
                pgc_pair_valid, dtype=torch.bool
            ),
            "pgc_completion_phase": torch.tensor(
                pgc_completion_phase, dtype=torch.long
            ),
            "pgc_completion_phase_valid": torch.tensor(
                pgc_completion_phase_valid, dtype=torch.bool
            ),
            "pgc_target_object_mask": pgc_target_object_mask,
            "pgc_source_object_mask": pgc_source_object_mask,
            "pgc_aux_object_mask": pgc_aux_object_mask,
            "pgc_target_mask_valid": torch.tensor(
                pgc_target_mask_valid, dtype=torch.bool
            ),
            "pgc_source_mask_valid": torch.tensor(
                pgc_source_mask_valid, dtype=torch.bool
            ),
            "pgc_aux_mask_valid": torch.tensor(
                pgc_aux_mask_valid, dtype=torch.bool
            ),
            "pgc_aux_prompt": pgc_aux_prompt,
            "pgc_aux_context": pgc_aux_context,
            "pgc_aux_context_mask": pgc_aux_context_mask,
            "pgc_aux_goal_id": torch.tensor(pgc_aux_goal_id, dtype=torch.long),
            "pgc_dataset_index": torch.tensor(dataset_index, dtype=torch.long),
        }
        data.update(pgc_eraf_sample)
        if negative_context is not None:
            data.update(
                {
                    "negative_prompt": negative_prompt,
                    "negative_context": negative_context,
                    "negative_context_mask": negative_context_mask,
                    "negative_valid": torch.tensor(
                        negative_valid, dtype=torch.bool
                    ),
                    "negative_type": "cross_task",
                    "transition_task_id": torch.tensor(
                        stable_instruction_id(task), dtype=torch.long
                    ),
                    "counterfactual_task_id": torch.tensor(
                        stable_instruction_id(negative_task), dtype=torch.long
                    ),
                }
            )
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
