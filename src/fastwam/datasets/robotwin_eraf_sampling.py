"""Pure sampling-contract helpers for formal RoboTwin ERAF grounding."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .pgc_libero import build_pgc_pair_balanced_sample_indices


EXPECTED_DOMAIN_EPISODE_COUNTS = (2, 3)


def build_robotwin_eraf_grounding_sample_indices(
    *,
    frame_groups: Sequence[Sequence[int]],
    sidecar_indices: Mapping[int, Mapping[str, Any]],
    native_dataset_count: int,
) -> tuple[list[int], tuple[str, ...]]:
    """Equalize five semantic pairs and native/CF while retaining both domains.

    Each semantic group combines the clean (3-episode) and randomized
    (2-episode) shards before frame-level balancing. This preserves the
    collected 3:2 domain composition inside a group instead of giving both
    shards an artificial 1:1 weight.
    """

    groups = [list(map(int, group)) for group in frame_groups]
    if len(groups) != len(sidecar_indices):
        raise ValueError(
            "RoboTwin ERAF frame/sidecar dataset counts differ: "
            f"frames={len(groups)} sidecars={len(sidecar_indices)}."
        )
    if native_dataset_count <= 0 or native_dataset_count >= len(groups):
        raise ValueError("RoboTwin ERAF requires native and counterfactual shards.")
    if len(groups) != 20 or native_dataset_count != 10:
        raise ValueError(
            "Formal RoboTwin ERAF grounding requires 10 native plus 10 "
            f"counterfactual dataset shards; got {native_dataset_count}+"
            f"{len(groups) - native_dataset_count}."
        )
    if any(not group for group in groups):
        raise ValueError("RoboTwin ERAF dataset shards must contain frames.")

    grouped_frames: dict[tuple[str, str], list[int]] = defaultdict(list)
    grouped_episode_counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    pair_order: list[str] = []
    for dataset_index, frames in enumerate(groups):
        try:
            index = sidecar_indices[dataset_index]
        except KeyError as exc:
            raise ValueError(
                f"Missing RoboTwin ERAF sidecar at dataset index {dataset_index}."
            ) from exc
        kind = "native" if dataset_index < native_dataset_count else "counterfactual"
        if str(index.get("dataset_kind")) != kind:
            raise ValueError(
                "RoboTwin ERAF dataset/sidecar kind mismatch at index "
                f"{dataset_index}."
            )
        if (
            index.get("artifact_role") != "eraf_grounding_supervision"
            or index.get("allowed_training_stages") != ["grounding"]
            or index.get("full_goal_verified") is not False
        ):
            raise ValueError(
                "RoboTwin ERAF sidecars must be grounding-only and explicitly "
                f"exclude full-goal data; dataset_index={dataset_index}."
            )
        records = index.get("episodes_by_index")
        if not isinstance(records, Mapping):
            raise ValueError(
                f"RoboTwin ERAF sidecar has no episode map: {dataset_index}."
            )
        pair_ids = {
            str(record.get("pair_id", "")).strip()
            for record in records.values()
            if isinstance(record, Mapping)
        }
        if len(pair_ids) != 1 or not next(iter(pair_ids)):
            raise ValueError(
                "Every RoboTwin ERAF shard must contain one non-empty pair_id; "
                f"dataset_index={dataset_index}."
            )
        pair_id = next(iter(pair_ids))
        if kind == "native" and pair_id not in pair_order:
            pair_order.append(pair_id)
        key = (kind, pair_id)
        grouped_frames[key].extend(frames)
        grouped_episode_counts[key].append(int(index.get("episode_count", -1)))

    if len(pair_order) != 5:
        raise ValueError(
            f"Formal RoboTwin ERAF grounding requires five pairs; got {pair_order}."
        )
    expected_keys = {
        (kind, pair_id)
        for kind in ("native", "counterfactual")
        for pair_id in pair_order
    }
    if set(grouped_frames) != expected_keys:
        raise ValueError(
            "RoboTwin ERAF native/counterfactual pair coverage differs: "
            f"got={sorted(grouped_frames)}."
        )
    for key in sorted(expected_keys):
        if tuple(sorted(grouped_episode_counts[key])) != EXPECTED_DOMAIN_EPISODE_COUNTS:
            raise ValueError(
                "Every RoboTwin ERAF pair/kind must contain clean=3 and "
                f"randomized=2 episodes; {key} has "
                f"{grouped_episode_counts[key]}."
            )

    group_labels = tuple(
        f"{kind}:{pair_id}"
        for kind in ("native", "counterfactual")
        for pair_id in pair_order
    )
    balanced_groups = [
        grouped_frames[tuple(label.split(":", 1))] for label in group_labels
    ]
    return build_pgc_pair_balanced_sample_indices(balanced_groups), group_labels
