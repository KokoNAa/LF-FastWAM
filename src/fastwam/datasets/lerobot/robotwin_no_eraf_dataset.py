"""RoboTwin four-pool dataset used by the no-ERAF control experiment.

The no-ERAF control deliberately reuses the LIBERO V9.12 sampling contract:
offline native, immutable-Base closed-loop native, historical
counterfactual, and strict counterfactual frames are interleaved 1:1:1:1.
Full-goal corrective data is forbidden at this stage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from fastwam.datasets.lerobot.robot_video_dataset import (
    RobotVideoDataset,
    build_pgc_v912_sample_plan,
)


logger = logging.getLogger(__name__)

FULL_GOAL_INDEX = Path("meta/pgc_robotwin_full_goal/index.json")
POOL_ORDER = (
    "offline_native",
    "closed_loop_native",
    "historical_cf",
    "strict_cf",
)
ALLOWED_CLOSED_LOOP_STAGES = {
    "initial_search",
    "holding",
    "released_unfinished",
    "next_clause_search",
}


def _frame_ranges(frame_counts: Sequence[int]) -> list[list[int]]:
    ranges: list[list[int]] = []
    offset = 0
    for raw_count in frame_counts:
        count = int(raw_count)
        if count <= 0:
            raise ValueError("RoboTwin no-ERAF datasets must contain frames.")
        ranges.append(list(range(offset, offset + count)))
        offset += count
    return ranges


def _frame_categories(
    *,
    dataset: Any,
    episode_records: Mapping[int, Mapping[str, Any]],
    field: str,
    allowed: set[str] | None = None,
) -> list[str]:
    categories: list[str] = []
    for raw_episode_index in dataset.hf_dataset["episode_index"]:
        episode_index = int(torch.as_tensor(raw_episode_index).reshape(-1)[0].item())
        try:
            value = str(episode_records[episode_index][field]).strip()
        except KeyError as exc:
            raise KeyError(
                "RoboTwin no-ERAF sidecar does not cover "
                f"episode {episode_index} field {field!r}."
            ) from exc
        if not value or (allowed is not None and value not in allowed):
            raise ValueError(f"Unsupported RoboTwin no-ERAF {field} value {value!r}.")
        categories.append(value)
    return categories


def build_robotwin_no_eraf_sample_plan(
    *,
    dataset_frame_counts: Sequence[int],
    offline_native_dataset_count: int,
    closed_loop_native_dataset_count: int,
    historical_cf_dataset_count: int,
    strict_cf_dataset_count: int,
    closed_loop_stage_categories: Sequence[str],
    strict_relation_categories: Sequence[str],
) -> tuple[list[int], list[int]]:
    """Build the deterministic four-pool sample plan from dataset boundaries."""

    counts = (
        int(offline_native_dataset_count),
        int(closed_loop_native_dataset_count),
        int(historical_cf_dataset_count),
        int(strict_cf_dataset_count),
    )
    if any(count <= 0 for count in counts):
        raise ValueError("All four RoboTwin no-ERAF pools must be non-empty.")
    if sum(counts) != len(dataset_frame_counts):
        raise ValueError(
            "RoboTwin no-ERAF dataset-count contract does not match loaded "
            f"datasets: pools={counts} datasets={len(dataset_frame_counts)}."
        )
    ranges = _frame_ranges(dataset_frame_counts)
    cursor = 0
    pool_indices: list[list[int]] = []
    for count in counts:
        pool_indices.append(
            [index for group in ranges[cursor : cursor + count] for index in group]
        )
        cursor += count
    if len(closed_loop_stage_categories) != len(pool_indices[1]):
        raise ValueError("Closed-loop stage labels do not cover every frame.")
    if len(strict_relation_categories) != len(pool_indices[3]):
        raise ValueError("Strict relation labels do not cover every frame.")
    return build_pgc_v912_sample_plan(
        offline_native_indices=pool_indices[0],
        closed_loop_native_indices=pool_indices[1],
        original_counterfactual_indices=pool_indices[2],
        strict_counterfactual_indices=pool_indices[3],
        closed_loop_stage_categories=[
            str(value) for value in closed_loop_stage_categories
        ],
        strict_relation_categories=[str(value) for value in strict_relation_categories],
    )


class RoboTwinNoERAFFourPoolDataset(RobotVideoDataset):
    """RoboTwin action dataset with a full-goal-free four-pool sampler."""

    def __init__(
        self,
        dataset_dirs: Sequence[str],
        *,
        pgc_counterfactual_dataset_dirs: Sequence[str],
        pgc_entity_relation_sidecar_dirs: Sequence[str],
        pgc_robotwin_offline_native_dataset_count: int,
        pgc_robotwin_closed_loop_native_dataset_count: int,
        pgc_robotwin_historical_cf_dataset_count: int,
        pgc_robotwin_strict_cf_dataset_count: int,
        **kwargs: Any,
    ) -> None:
        native_dirs = [str(path) for path in dataset_dirs]
        counterfactual_dirs = [str(path) for path in pgc_counterfactual_dataset_dirs]
        sidecar_dirs = [str(path) for path in pgc_entity_relation_sidecar_dirs]
        counts = (
            int(pgc_robotwin_offline_native_dataset_count),
            int(pgc_robotwin_closed_loop_native_dataset_count),
            int(pgc_robotwin_historical_cf_dataset_count),
            int(pgc_robotwin_strict_cf_dataset_count),
        )
        if counts[0] + counts[1] != len(native_dirs):
            raise ValueError("RoboTwin no-ERAF native pool order/count is invalid.")
        if counts[2] + counts[3] != len(counterfactual_dirs):
            raise ValueError("RoboTwin no-ERAF CF pool order/count is invalid.")
        combined_dirs = native_dirs + counterfactual_dirs
        if len(sidecar_dirs) != len(combined_dirs):
            raise ValueError(
                "RoboTwin no-ERAF requires one ordered sidecar per dataset."
            )
        leaked = [
            path
            for path in combined_dirs
            if (Path(path).expanduser().resolve() / FULL_GOAL_INDEX).exists()
        ]
        if leaked:
            raise ValueError(
                "full-goal data is forbidden in no-ERAF training: " f"{sorted(leaked)}."
            )

        incompatible_defaults = {
            "pgc_balance_native_counterfactual": False,
            "pgc_v9_balanced_sampling": False,
            "pgc_pair_balanced_sampling": False,
            "pgc_v9_closed_loop_rebinding": False,
            "pgc_v9_phase_safe_memory": False,
            "pgc_v9_closed_loop_native_dataset_count": 0,
        }
        for name, expected in incompatible_defaults.items():
            actual = kwargs.pop(name, expected)
            if actual != expected:
                raise ValueError(
                    f"RoboTwin no-ERAF owns {name}; expected {expected!r}, got {actual!r}."
                )
        required_defaults = {
            "pgc_entity_relation_supervision_required": True,
            "pgc_bidirectional_language_supervision_required": True,
        }
        for name, expected in required_defaults.items():
            actual = kwargs.pop(name, expected)
            if actual != expected:
                raise ValueError(
                    f"RoboTwin no-ERAF requires {name}={expected!r}, got {actual!r}."
                )
        if kwargs.pop("pgc_closed_loop_corrective_dataset_dirs", []):
            raise ValueError("no-ERAF cannot load a corrective/full-goal dataset.")

        super().__init__(
            dataset_dirs=native_dirs,
            pgc_counterfactual_dataset_dirs=counterfactual_dirs,
            pgc_closed_loop_corrective_dataset_dirs=[],
            pgc_balance_native_counterfactual=False,
            pgc_entity_relation_supervision_required=True,
            pgc_entity_relation_sidecar_dirs=sidecar_dirs,
            pgc_bidirectional_language_supervision_required=True,
            pgc_v9_balanced_sampling=False,
            pgc_pair_balanced_sampling=False,
            pgc_v9_closed_loop_rebinding=False,
            pgc_v9_phase_safe_memory=False,
            pgc_v9_closed_loop_native_dataset_count=0,
            **kwargs,
        )

        expected_roles = (
            ["offline_native"] * counts[0]
            + ["closed_loop_native"] * counts[1]
            + ["historical_cf"] * counts[2]
            + ["strict_cf"] * counts[3]
        )
        for dataset_index, expected_role in enumerate(expected_roles):
            index = self.pgc_entity_relation_indices[dataset_index]
            if index.get("artifact_role") != expected_role:
                raise ValueError(
                    "RoboTwin no-ERAF sidecar role/order mismatch at dataset "
                    f"{dataset_index}: expected={expected_role!r} "
                    f"actual={index.get('artifact_role')!r}."
                )
            if index.get("full_goal_verified") is True:
                raise ValueError("full-goal verification leaked into no-ERAF sidecar.")
            if "no_eraf" not in tuple(index.get("allowed_training_stages", ())):
                raise ValueError(f"Sidecar role {expected_role!r} forbids no-ERAF.")
            expected_kind = (
                "native" if expected_role.endswith("native") else "counterfactual"
            )
            if index.get("dataset_kind") != expected_kind:
                raise ValueError(
                    f"Sidecar role {expected_role!r} has wrong dataset_kind."
                )
            if (
                expected_role == "closed_loop_native"
                and index.get("state_distribution")
                != "immutable_base_closed_loop_replan"
            ):
                raise ValueError("Closed-loop native state distribution is invalid.")

        underlying = self.lerobot_dataset.multi_dataset._datasets
        frame_counts = [int(dataset.num_frames) for dataset in underlying]
        closed_start = counts[0]
        closed_end = closed_start + counts[1]
        strict_start = sum(counts[:3])
        closed_stages: list[str] = []
        for dataset_index in range(closed_start, closed_end):
            closed_stages.extend(
                _frame_categories(
                    dataset=underlying[dataset_index],
                    episode_records=self.pgc_entity_relation_indices[dataset_index][
                        "episodes_by_index"
                    ],
                    field="online_stage_v2",
                    allowed=ALLOWED_CLOSED_LOOP_STAGES,
                )
            )
        strict_categories: list[str] = []
        for dataset_index in range(strict_start, len(underlying)):
            strict_categories.extend(
                _frame_categories(
                    dataset=underlying[dataset_index],
                    episode_records=self.pgc_entity_relation_indices[dataset_index][
                        "episodes_by_index"
                    ],
                    field="pair_id",
                )
            )
        self._sample_indices, self.pgc_robotwin_no_eraf_group_ids = (
            build_robotwin_no_eraf_sample_plan(
                dataset_frame_counts=frame_counts,
                offline_native_dataset_count=counts[0],
                closed_loop_native_dataset_count=counts[1],
                historical_cf_dataset_count=counts[2],
                strict_cf_dataset_count=counts[3],
                closed_loop_stage_categories=closed_stages,
                strict_relation_categories=strict_categories,
            )
        )
        # ResumableEpochSampler recognizes this established four-group label
        # name and constructs rank-aware optimizer windows. The ERAF data flag
        # remains false because pgc_v9_closed_loop_grounding is disabled.
        self.pgc_v9_closed_loop_group_ids = self.pgc_robotwin_no_eraf_group_ids
        group_counts = tuple(
            self.pgc_robotwin_no_eraf_group_ids.count(group) for group in range(4)
        )
        if len(set(group_counts)) != 1:
            raise AssertionError(
                f"RoboTwin no-ERAF mixture is not 1:1:1:1: {group_counts}."
            )
        # The trainer's paired-language guard names the original LIBERO
        # four-pool contract fields. Advertise the equivalent post-build
        # invariants here; pgc_v9_closed_loop_grounding stays false, so no
        # ERAF module is constructed or marked active for these rows.
        self.pgc_balance_native_counterfactual = True
        self.pgc_v9_balanced_sampling = True
        self.pgc_v9_phase_safe_memory = True
        self.pgc_v9_closed_loop_native_dataset_count = counts[1]
        self.pgc_effective_native_sample_count = group_counts[0] + group_counts[1]
        self.pgc_effective_counterfactual_sample_count = (
            group_counts[2] + group_counts[3]
        )
        self.pgc_effective_closed_loop_corrective_sample_count = 0
        self.pgc_effective_eraf_closed_loop_sample_count = 0
        logger.info(
            "RoboTwin no-ERAF four-pool curriculum: offline_native=%d "
            "closed_loop_native=%d historical_cf=%d strict_cf=%d",
            *group_counts,
        )
