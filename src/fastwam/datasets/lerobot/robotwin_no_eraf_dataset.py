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

import numpy as np
import torch

from fastwam.datasets.pgc_libero import (
    build_pgc_bidirectional_language_pair_index,
    load_pgc_episode_language_pairs,
    read_jsonl,
)
from fastwam.datasets.robotwin_eraf_sampling import robotwin_array_sha256
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
CLOSED_LOOP_CAPTURE_FORMAT = "pgc_robotwin_closed_loop_native_capture_v2"
CLOSED_LOOP_CAPTURE_FRAME_COUNT = 9
CLOSED_LOOP_ACTION_VIDEO_FREQ_RATIO = 4
CLOSED_LOOP_PRODUCTIVE_START_COUNT = 5
CLOSED_LOOP_TEMPORAL_CONTRACT = (
    "contiguous_pre_action_qpos_stride1_with_realized_video_at_t_plus_4"
)


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


def _scalar_int(value: Any) -> int:
    return int(torch.as_tensor(value).reshape(-1)[0].item())


def _closed_loop_productive_rows(
    *,
    dataset: Any,
    index: Mapping[str, Any],
    dataset_offset: int,
    action_video_freq_ratio: int,
) -> tuple[list[int], list[str]]:
    """Return only starts with a realized observation at video offset ``t+4``."""

    expected_index = {
        "capture_format": CLOSED_LOOP_CAPTURE_FORMAT,
        "capture_frame_count": CLOSED_LOOP_CAPTURE_FRAME_COUNT,
        "action_video_freq_ratio": CLOSED_LOOP_ACTION_VIDEO_FREQ_RATIO,
        "productive_start_count_per_episode": CLOSED_LOOP_PRODUCTIVE_START_COUNT,
        "temporal_contract": CLOSED_LOOP_TEMPORAL_CONTRACT,
    }
    mismatches = {
        key: (index.get(key), expected)
        for key, expected in expected_index.items()
        if index.get(key) != expected
    }
    if int(action_video_freq_ratio) != CLOSED_LOOP_ACTION_VIDEO_FREQ_RATIO:
        mismatches["training_action_video_freq_ratio"] = (
            int(action_video_freq_ratio),
            CLOSED_LOOP_ACTION_VIDEO_FREQ_RATIO,
        )
    if mismatches:
        raise ValueError(
            "RoboTwin closed-loop native temporal contract mismatch: "
            f"{mismatches}."
        )

    episode_records = index["episodes_by_index"]
    raw_episode_indices = dataset.hf_dataset["episode_index"]
    raw_frame_indices = dataset.hf_dataset["frame_index"]
    if len(raw_episode_indices) != len(raw_frame_indices):
        raise ValueError("Closed-loop episode/frame columns have different lengths.")

    frames_by_episode: dict[int, list[int]] = {}
    productive_indices: list[int] = []
    productive_stages: list[str] = []
    for local_index, (raw_episode_index, raw_frame_index) in enumerate(
        zip(raw_episode_indices, raw_frame_indices, strict=True)
    ):
        episode_index = _scalar_int(raw_episode_index)
        frame_index = _scalar_int(raw_frame_index)
        try:
            record = episode_records[episode_index]
        except KeyError as exc:
            raise KeyError(
                "Closed-loop sidecar does not cover dataset episode "
                f"{episode_index}."
            ) from exc
        record_expected = {
            "frame_count": CLOSED_LOOP_CAPTURE_FRAME_COUNT,
            "action_video_freq_ratio": CLOSED_LOOP_ACTION_VIDEO_FREQ_RATIO,
            "productive_start_count": CLOSED_LOOP_PRODUCTIVE_START_COUNT,
            "temporal_contract": CLOSED_LOOP_TEMPORAL_CONTRACT,
        }
        if any(record.get(key) != value for key, value in record_expected.items()):
            raise ValueError(
                "Closed-loop episode violates the productive temporal contract: "
                f"episode={episode_index}."
            )
        stage = str(record.get("online_stage_v2", "")).strip()
        if stage not in ALLOWED_CLOSED_LOOP_STAGES:
            raise ValueError(f"Unsupported closed-loop online stage {stage!r}.")
        frames_by_episode.setdefault(episode_index, []).append(frame_index)
        if frame_index < CLOSED_LOOP_PRODUCTIVE_START_COUNT:
            productive_indices.append(int(dataset_offset) + local_index)
            productive_stages.append(stage)

    if set(frames_by_episode) != set(episode_records):
        raise ValueError("Closed-loop sidecar/dataset episode coverage differs.")
    for episode_index, frame_indices in frames_by_episode.items():
        if frame_indices != list(range(CLOSED_LOOP_CAPTURE_FRAME_COUNT)):
            raise ValueError(
                "Closed-loop episode must contain dense ordered frame indices "
                f"0..{CLOSED_LOOP_CAPTURE_FRAME_COUNT - 1}: "
                f"episode={episode_index} frames={frame_indices}."
            )
    expected_productive = len(episode_records) * CLOSED_LOOP_PRODUCTIVE_START_COUNT
    if (
        len(productive_indices) != expected_productive
        or int(index.get("productive_frame_count", -1)) != expected_productive
    ):
        raise ValueError(
            "Closed-loop productive-row count changed: "
            f"selected={len(productive_indices)} expected={expected_productive}."
        )
    return productive_indices, productive_stages


def build_robotwin_no_eraf_sample_plan(
    *,
    dataset_frame_counts: Sequence[int],
    offline_native_dataset_count: int,
    closed_loop_native_dataset_count: int,
    historical_cf_dataset_count: int,
    strict_cf_dataset_count: int,
    closed_loop_productive_indices: Sequence[int],
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
    closed_loop_productive = [int(index) for index in closed_loop_productive_indices]
    if not closed_loop_productive or len(set(closed_loop_productive)) != len(
        closed_loop_productive
    ):
        raise ValueError("Closed-loop productive starts must be non-empty and unique.")
    if not set(closed_loop_productive).issubset(set(pool_indices[1])):
        raise ValueError(
            "Closed-loop productive starts must stay inside the declared pool."
        )
    pool_indices[1] = closed_loop_productive
    if len(closed_loop_stage_categories) != len(closed_loop_productive):
        raise ValueError("Closed-loop stage labels do not cover productive starts.")
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

    def _validate_pgc_entity_relation_dataset_audits(
        self, *, underlying: list[Any], combined_dataset_dirs: list[str]
    ) -> None:
        """Bind all four pools to RoboTwin's typed action/state hashes."""

        for dataset_index, (dataset, dataset_dir) in enumerate(
            zip(underlying, combined_dataset_dirs, strict=True)
        ):
            index = self.pgc_entity_relation_indices[dataset_index]
            records = index["episodes_by_index"]
            expected_episode_count = int(dataset.meta.total_episodes)
            if int(index.get("episode_count", -1)) != expected_episode_count or set(
                records
            ) != set(range(expected_episode_count)):
                raise ValueError(
                    "RoboTwin no-ERAF sidecar episodes do not exactly cover "
                    f"dataset index {dataset_index}."
                )
            selected_episode_ids = (
                list(dataset.episodes)
                if dataset.episodes is not None
                else list(range(expected_episode_count))
            )
            dataset_root = Path(dataset_dir).expanduser().resolve()
            audit_path = dataset_root / "meta/pgc_episodes.jsonl"
            if not audit_path.is_file():
                raise FileNotFoundError(
                    f"RoboTwin no-ERAF episode audit is missing: {audit_path}."
                )
            audits = {
                int(record["episode_index"]): dict(record)
                for record in read_jsonl(audit_path)
            }
            if set(audits) != set(range(expected_episode_count)):
                raise ValueError(
                    "RoboTwin no-ERAF episode audit must be dense and complete "
                    f"at dataset index {dataset_index}."
                )

            for local_episode_index, episode_index in enumerate(selected_episode_ids):
                episode_index = int(episode_index)
                record = records[episode_index]
                audit = audits[episode_index]
                episode = dataset.get_episode_data(local_episode_index)
                if "action" not in episode:
                    raise KeyError(
                        "RoboTwin no-ERAF dataset has no raw action column at "
                        f"{dataset_index}/{episode_index}."
                    )
                action = episode["action"]
                if hasattr(action, "detach"):
                    action = action.detach().cpu().numpy()
                action = np.ascontiguousarray(np.asarray(action, dtype=np.float32))
                expected_action_dim = int(index["action_dim"])
                if action.ndim != 2 or action.shape[1] != expected_action_dim:
                    raise ValueError(
                        "RoboTwin no-ERAF actions have the wrong shape at "
                        f"{dataset_index}/{episode_index}: {action.shape}."
                    )
                if int(record["frame_count"]) != int(action.shape[0]):
                    raise ValueError(
                        "RoboTwin no-ERAF frame/action count mismatch at "
                        f"{dataset_index}/{episode_index}."
                    )
                expected_action_digest = str(record["action_sha256"])
                if (
                    robotwin_array_sha256(action) != expected_action_digest
                    or str(audit.get("action_sha256", "")) != expected_action_digest
                ):
                    raise ValueError(
                        "RoboTwin no-ERAF typed qpos hash does not match the "
                        f"loaded LeRobot episode {dataset_index}/{episode_index}."
                    )
                if str(audit.get("pair_id", "")) != str(record.get("pair_id", "")):
                    raise ValueError(
                        "RoboTwin no-ERAF pair audit mismatch at "
                        f"{dataset_index}/{episode_index}."
                    )
                expected_state_digest = str(record.get("initial_state_sha256", ""))
                if expected_state_digest != str(audit.get("initial_state_sha256", "")):
                    raise ValueError(
                        "RoboTwin no-ERAF initial-state provenance mismatch at "
                        f"{dataset_index}/{episode_index}."
                    )
                if index.get("artifact_role") == "closed_loop_native":
                    if (
                        index.get("state_distribution")
                        != "immutable_base_closed_loop_replan"
                        or not str(audit.get("capture_id", "")).strip()
                        or int(record.get("frame_count", -1))
                        != CLOSED_LOOP_CAPTURE_FRAME_COUNT
                        or int(record.get("action_video_freq_ratio", -1))
                        != CLOSED_LOOP_ACTION_VIDEO_FREQ_RATIO
                        or int(record.get("productive_start_count", -1))
                        != CLOSED_LOOP_PRODUCTIVE_START_COUNT
                        or record.get("temporal_contract")
                        != CLOSED_LOOP_TEMPORAL_CONTRACT
                        or int(audit.get("action_count", -1))
                        != CLOSED_LOOP_CAPTURE_FRAME_COUNT
                        or int(audit.get("action_video_freq_ratio", -1))
                        != CLOSED_LOOP_ACTION_VIDEO_FREQ_RATIO
                        or int(audit.get("productive_start_count", -1))
                        != CLOSED_LOOP_PRODUCTIVE_START_COUNT
                        or audit.get("temporal_contract")
                        != CLOSED_LOOP_TEMPORAL_CONTRACT
                    ):
                        raise ValueError(
                            "RoboTwin closed-loop native audit contract changed at "
                            f"{dataset_index}/{episode_index}."
                        )
                    continue
                state_relpath = Path(str(audit.get("source_initial_state_catalog", "")))
                if (
                    not str(state_relpath)
                    or state_relpath.is_absolute()
                    or ".." in state_relpath.parts
                ):
                    raise ValueError(
                        "RoboTwin no-ERAF initial-state audit path is unsafe at "
                        f"{dataset_index}/{episode_index}."
                    )
                state_path = dataset_root / state_relpath
                if not state_path.is_file():
                    raise FileNotFoundError(
                        f"RoboTwin initial-state file is missing: {state_path}."
                    )
                if (
                    robotwin_array_sha256(np.load(state_path, allow_pickle=False))
                    != expected_state_digest
                ):
                    raise ValueError(
                        "RoboTwin no-ERAF typed initial-state hash changed at "
                        f"{dataset_index}/{episode_index}."
                    )

        logger.info(
            "Validated RoboTwin no-ERAF typed action/state hashes for %d datasets.",
            len(combined_dataset_dirs),
        )

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

        # The immutable-Base closed-loop pool records the exact task wording
        # seen during rollout. RoboTwin may paraphrase the same task with
        # scene-specific entity names, so those audited per-episode language
        # pairs must participate in native reverse-ranking too. The base class
        # only loads language pairs from counterfactual datasets; merge the
        # closed-loop provenance explicitly, while binding every pair ID back
        # to its already validated sidecar episode.
        closed_start = counts[0]
        closed_end = closed_start + counts[1]
        closed_loop_language_pair_count = 0
        for dataset_index in range(closed_start, closed_end):
            pairs = load_pgc_episode_language_pairs(native_dirs[dataset_index])
            records = self.pgc_entity_relation_indices[dataset_index][
                "episodes_by_index"
            ]
            if set(pairs) != set(records):
                raise ValueError(
                    "RoboTwin closed-loop language provenance does not exactly "
                    f"cover sidecar episodes at dataset index {dataset_index}."
                )
            for episode_index, pair in pairs.items():
                if str(pair.get("pair_id", "")) != str(
                    records[episode_index].get("pair_id", "")
                ):
                    raise ValueError(
                        "RoboTwin closed-loop language/sidecar pair mismatch at "
                        f"{dataset_index}/{episode_index}."
                    )
            self.pgc_episode_language_pairs[dataset_index] = pairs
            closed_loop_language_pair_count += len(pairs)
        self.pgc_bidirectional_language_pairs_by_source = (
            build_pgc_bidirectional_language_pair_index(self.pgc_episode_language_pairs)
        )
        logger.info(
            "Merged %d audited closed-loop language pairs into RoboTwin "
            "no-ERAF bidirectional supervision.",
            closed_loop_language_pair_count,
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
                and (
                    index.get("state_distribution")
                    != "immutable_base_closed_loop_replan"
                    or index.get("capture_format")
                    != CLOSED_LOOP_CAPTURE_FORMAT
                    or int(index.get("capture_frame_count", -1))
                    != CLOSED_LOOP_CAPTURE_FRAME_COUNT
                    or int(index.get("action_video_freq_ratio", -1))
                    != CLOSED_LOOP_ACTION_VIDEO_FREQ_RATIO
                    or int(index.get("productive_start_count_per_episode", -1))
                    != CLOSED_LOOP_PRODUCTIVE_START_COUNT
                    or index.get("temporal_contract")
                    != CLOSED_LOOP_TEMPORAL_CONTRACT
                )
            ):
                raise ValueError(
                    "Closed-loop native productive temporal contract is invalid."
                )

        underlying = self.lerobot_dataset.multi_dataset._datasets
        frame_counts = [int(dataset.num_frames) for dataset in underlying]
        dataset_offsets: list[int] = []
        offset = 0
        for frame_count in frame_counts:
            dataset_offsets.append(offset)
            offset += frame_count
        strict_start = sum(counts[:3])
        closed_productive_indices: list[int] = []
        closed_stages: list[str] = []
        for dataset_index in range(closed_start, closed_end):
            productive_indices, productive_stages = _closed_loop_productive_rows(
                dataset=underlying[dataset_index],
                index=self.pgc_entity_relation_indices[dataset_index],
                dataset_offset=dataset_offsets[dataset_index],
                action_video_freq_ratio=int(self.action_video_freq_ratio),
            )
            closed_productive_indices.extend(productive_indices)
            closed_stages.extend(productive_stages)
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
                closed_loop_productive_indices=closed_productive_indices,
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
