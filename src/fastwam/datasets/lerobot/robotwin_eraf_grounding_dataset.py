"""Grounding-only RoboTwin ERAF dataset with domain-aware pair balancing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..robotwin_eraf_sampling import (
    build_robotwin_eraf_grounding_sample_indices,
)
from .robot_video_dataset import RobotVideoDataset


logger = logging.getLogger(__name__)

FULL_GOAL_INDEX = Path("meta/pgc_robotwin_full_goal/index.json")


class RoboTwinERAFGroundingDataset(RobotVideoDataset):
    """RoboTwin grounding dataset that rejects full-goal and balances 10 groups."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        dataset_dirs = list(kwargs.get("dataset_dirs") or (args[0] if args else ()))
        counterfactual_dirs = list(
            kwargs.get("pgc_counterfactual_dataset_dirs") or ()
        )
        corrective_dirs = list(
            kwargs.get("pgc_closed_loop_corrective_dataset_dirs") or ()
        )
        if corrective_dirs:
            raise ValueError(
                "Full-goal/corrective datasets are forbidden in ERAF grounding."
            )
        leaked = [
            str(path)
            for path in dataset_dirs + counterfactual_dirs
            if (Path(path).expanduser().resolve() / FULL_GOAL_INDEX).is_file()
        ]
        if leaked:
            raise ValueError(
                "Full-goal data is forbidden in ERAF grounding; "
                f"leaked={leaked}."
            )
        if kwargs.get("pgc_pair_balanced_sampling") is True:
            raise ValueError(
                "Use the RoboTwin domain-aware sampler, not the one-shard "
                "generic pair sampler."
            )
        if kwargs.get("pgc_v9_balanced_sampling") is True:
            raise ValueError(
                "RoboTwin ERAF grounding cannot use the LIBERO strict-CF sampler."
            )
        kwargs["pgc_pair_balanced_sampling"] = False
        kwargs["pgc_v9_balanced_sampling"] = False
        kwargs["pgc_balance_native_counterfactual"] = False
        kwargs["pgc_entity_relation_supervision_required"] = True
        super().__init__(*args, **kwargs)

        offset = 0
        frame_groups: list[list[int]] = []
        for dataset in self.lerobot_dataset.multi_dataset._datasets:
            count = int(dataset.num_frames)
            frame_groups.append(list(range(offset, offset + count)))
            offset += count
        self._sample_indices, self.pgc_robotwin_eraf_group_labels = (
            build_robotwin_eraf_grounding_sample_indices(
                frame_groups=frame_groups,
                sidecar_indices=self.pgc_entity_relation_indices,
                native_dataset_count=self.pgc_native_dataset_count,
            )
        )
        self.pgc_balance_native_counterfactual = True
        self.pgc_effective_native_sample_count = sum(
            int(index < self.pgc_native_frame_count) for index in self._sample_indices
        )
        self.pgc_effective_counterfactual_sample_count = (
            len(self._sample_indices) - self.pgc_effective_native_sample_count
        )
        self.pgc_effective_closed_loop_corrective_sample_count = 0
        self.pgc_effective_eraf_closed_loop_sample_count = 0
        logger.info(
            "RoboTwin ERAF grounding sampling: groups=%s native=%d "
            "counterfactual=%d total=%d",
            self.pgc_robotwin_eraf_group_labels,
            self.pgc_effective_native_sample_count,
            self.pgc_effective_counterfactual_sample_count,
            len(self._sample_indices),
        )
