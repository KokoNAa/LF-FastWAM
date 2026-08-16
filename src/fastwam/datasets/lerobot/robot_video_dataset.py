import hashlib
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from ..counterfactual import (
    load_counterfactual_instruction_map,
    stable_instruction_id,
)
from ..pgc_libero import load_pgc_episode_language_pairs
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
        pgc_counterfactual_oversample_factor: int = 1,
        pgc_balance_native_counterfactual: bool = False,
    ):
        native_dataset_dirs = [str(path) for path in dataset_dirs]
        pgc_counterfactual_dataset_dirs = [
            str(path) for path in (pgc_counterfactual_dataset_dirs or [])
        ]
        duplicate_dirs = set(native_dataset_dirs) & set(
            pgc_counterfactual_dataset_dirs
        )
        if duplicate_dirs:
            raise ValueError(
                "PGC direct-counterfactual datasets must be distinct from the "
                f"native datasets; duplicates={sorted(duplicate_dirs)}."
            )
        self.pgc_counterfactual_oversample_factor = int(
            pgc_counterfactual_oversample_factor
        )
        if self.pgc_counterfactual_oversample_factor < 1:
            raise ValueError("`pgc_counterfactual_oversample_factor` must be >= 1.")
        self.pgc_native_dataset_count = len(native_dataset_dirs)
        self.pgc_counterfactual_dataset_dirs = pgc_counterfactual_dataset_dirs
        self.pgc_has_counterfactual_data = bool(
            self.pgc_counterfactual_dataset_dirs
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
        self.pgc_balance_native_counterfactual = bool(
            pgc_balance_native_counterfactual
        )
        combined_dataset_dirs = (
            native_dataset_dirs + self.pgc_counterfactual_dataset_dirs
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
        underlying = self.lerobot_dataset.multi_dataset._datasets
        self.pgc_native_frame_count = sum(
            int(dataset.num_frames)
            for dataset in underlying[: self.pgc_native_dataset_count]
        )
        total_frame_count = sum(int(dataset.num_frames) for dataset in underlying)
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
        if self.pgc_has_counterfactual_data:
            logger.info(
                "PGC sampling: native=%d counterfactual=%d balanced=%s",
                self.pgc_effective_native_sample_count,
                self.pgc_effective_counterfactual_sample_count,
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
        pgc_source_task = str(task)
        pgc_pair_valid = False
        if pgc_is_counterfactual:
            if "episode_index" not in sample:
                raise KeyError(
                    "PGC counterfactual samples require `episode_index` to "
                    "recover their paired source instruction."
                )
            episode_index = int(
                torch.as_tensor(sample["episode_index"]).reshape(-1)[0].item()
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
            "pgc_dataset_index": torch.tensor(dataset_index, dtype=torch.long),
        }
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
