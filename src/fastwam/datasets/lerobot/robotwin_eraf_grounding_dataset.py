"""Grounding-only RoboTwin ERAF dataset with domain-aware pair balancing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..pgc_libero import read_jsonl
from ..robotwin_eraf_sampling import (
    build_robotwin_eraf_grounding_sample_indices,
    robotwin_array_sha256,
)
from .robot_video_dataset import RobotVideoDataset


logger = logging.getLogger(__name__)

FULL_GOAL_INDEX = Path("meta/pgc_robotwin_full_goal/index.json")


class RoboTwinERAFGroundingDataset(RobotVideoDataset):
    """RoboTwin grounding dataset that rejects full-goal and balances 10 groups."""

    def _validate_pgc_entity_relation_dataset_audits(
        self, *, underlying: list[Any], combined_dataset_dirs: list[str]
    ) -> None:
        """Bind sidecars to RoboTwin's typed qpos/state hash contract.

        The generic LIBERO audit uses a JSON-header tensor digest. RoboTwin raw
        collection instead committed ``dtype.str|shape`` before conversion;
        using the LIBERO digest rejects intact data even when every float32
        qpos value is unchanged.
        """

        for dataset_index, (dataset, dataset_dir) in enumerate(
            zip(underlying, combined_dataset_dirs, strict=True)
        ):
            index = self.pgc_entity_relation_indices[dataset_index]
            records = index["episodes_by_index"]
            expected_episode_count = int(dataset.meta.total_episodes)
            if (
                int(index.get("episode_count", -1)) != expected_episode_count
                or set(records) != set(range(expected_episode_count))
            ):
                raise ValueError(
                    "RoboTwin ERAF sidecar episodes do not exactly cover "
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
                    f"RoboTwin ERAF episode audit is missing: {audit_path}."
                )
            audits = {
                int(record["episode_index"]): dict(record)
                for record in read_jsonl(audit_path)
            }
            if set(audits) != set(range(expected_episode_count)):
                raise ValueError(
                    "RoboTwin ERAF episode audit must be dense and complete at "
                    f"dataset index {dataset_index}."
                )

            for local_episode_index, episode_index in enumerate(selected_episode_ids):
                episode_index = int(episode_index)
                record = records[episode_index]
                audit = audits[episode_index]
                episode = dataset.get_episode_data(local_episode_index)
                if "action" not in episode:
                    raise KeyError(
                        "RoboTwin ERAF dataset has no raw action column at "
                        f"{dataset_index}/{episode_index}."
                    )
                action = episode["action"]
                if hasattr(action, "detach"):
                    action = action.detach().cpu().numpy()
                action = np.ascontiguousarray(np.asarray(action, dtype=np.float32))
                expected_action_dim = int(index["action_dim"])
                if action.ndim != 2 or action.shape[1] != expected_action_dim:
                    raise ValueError(
                        "RoboTwin ERAF actions have the wrong shape at "
                        f"{dataset_index}/{episode_index}: {action.shape}."
                    )
                if int(record["frame_count"]) != int(action.shape[0]):
                    raise ValueError(
                        "RoboTwin ERAF frame/action count mismatch at "
                        f"{dataset_index}/{episode_index}."
                    )
                expected_action_digest = str(record["action_sha256"])
                if (
                    robotwin_array_sha256(action) != expected_action_digest
                    or str(audit.get("action_sha256", ""))
                    != expected_action_digest
                ):
                    raise ValueError(
                        "RoboTwin ERAF typed qpos hash does not match the loaded "
                        f"LeRobot episode {dataset_index}/{episode_index}."
                    )
                if str(audit.get("pair_id", "")) != str(record.get("pair_id", "")):
                    raise ValueError(
                        "RoboTwin ERAF pair audit mismatch at "
                        f"{dataset_index}/{episode_index}."
                    )
                expected_state_digest = str(record.get("initial_state_sha256", ""))
                if expected_state_digest != str(
                    audit.get("initial_state_sha256", "")
                ):
                    raise ValueError(
                        "RoboTwin ERAF initial-state provenance mismatch at "
                        f"{dataset_index}/{episode_index}."
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
                        "RoboTwin ERAF initial-state audit path is unsafe at "
                        f"{dataset_index}/{episode_index}."
                    )
                state_path = dataset_root / state_relpath
                if not state_path.is_file():
                    raise FileNotFoundError(
                        f"RoboTwin ERAF initial-state file is missing: {state_path}."
                    )
                initial_state = np.load(state_path, allow_pickle=False)
                if robotwin_array_sha256(initial_state) != expected_state_digest:
                    raise ValueError(
                        "RoboTwin ERAF typed initial-state hash changed at "
                        f"{dataset_index}/{episode_index}."
                    )

        logger.info(
            "Validated RoboTwin ERAF typed action/state hashes for %d datasets.",
            len(combined_dataset_dirs),
        )

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
