#!/usr/bin/env python3
"""Build audited post-grasp completion labels for an existing PGC dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastwam.datasets.pgc_libero import (  # noqa: E402
    PGC_COMPLETION_PHASE_FORMAT,
    PGC_COMPLETION_PHASE_INDEX,
    atomic_write_json,
    detect_pgc_completion_phase,
    filter_libero_noops,
    iter_libero_hdf5_demos,
    read_jsonl,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive grasp-close / release boundaries from the exact donor "
            "actions used to build a PGC counterfactual LeRobot dataset."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_demo_actions(path: Path, group_name: str):
    for demo in iter_libero_hdf5_demos(path):
        if demo.group_name == group_name:
            return demo.actions
    raise KeyError(f"Demo group {group_name!r} is absent from {path}.")


def build_completion_phase_index(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    provenance_path = dataset_root / "meta/pgc_provenance.json"
    episodes_path = dataset_root / "meta/pgc_episodes.jsonl"
    if not provenance_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(
            "Expected meta/pgc_provenance.json and meta/pgc_episodes.jsonl "
            f"under {dataset_root}."
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    audits = read_jsonl(episodes_path)
    keep_noops = not bool(provenance.get("noop_filter_enabled", True))
    records: list[dict[str, Any]] = []
    close_steps: list[int] = []
    release_count = 0
    for audit in sorted(audits, key=lambda item: int(item["episode_index"])):
        episode_index = int(audit["episode_index"])
        demo_path = Path(str(audit["donor_demo_file"])).expanduser().resolve()
        group_name = str(audit["donor_demo_group"])
        actions = _load_demo_actions(demo_path, group_name)
        if not keep_noops:
            actions = filter_libero_noops(actions)
        action_count = int(audit["recorded_action_count"])
        if action_count <= 0 or action_count > len(actions):
            raise ValueError(
                f"Episode {episode_index} recorded_action_count={action_count} "
                f"is incompatible with donor action count {len(actions)}."
            )
        # The LeRobot episode was truncated at the first satisfied success
        # predicate, so phase labels must use that identical action prefix.
        phase = detect_pgc_completion_phase(actions[:action_count])
        close_steps.append(int(phase["grasp_close_step"]))
        release_count += int(phase["release_open_step"] is not None)
        records.append(
            {
                "episode_index": episode_index,
                "pair_id": str(audit["pair_id"]),
                "action_count": action_count,
                "grasp_close_step": int(phase["grasp_close_step"]),
                "release_open_step": phase["release_open_step"],
                "donor_demo_key": str(audit["donor_demo_key"]),
                "boundary_source": "executed_gripper_command",
            }
        )
    if not records:
        raise ValueError(f"PGC dataset has no audited episodes: {dataset_root}.")
    return {
        "format": PGC_COMPLETION_PHASE_FORMAT,
        "dataset_root": str(dataset_root),
        "episode_count": len(records),
        "gripper_contract": "libero_action_dim_6_negative_open_positive_close",
        "completion_definition": "frame_at_or_after_first_close_command",
        "release_definition": "first_negative_command_after_close_when_recorded",
        "release_label_count": release_count,
        "grasp_close_step_min": min(close_steps),
        "grasp_close_step_max": max(close_steps),
        "episodes": records,
    }


def main() -> None:
    args = _parse_args()
    output = args.dataset_root.expanduser().resolve() / PGC_COMPLETION_PHASE_INDEX
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Completion sidecar already exists: {output}")
    payload = build_completion_phase_index(args.dataset_root)
    atomic_write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "episode_count": payload["episode_count"],
                "release_label_count": payload["release_label_count"],
                "grasp_close_step_range": [
                    payload["grasp_close_step_min"],
                    payload["grasp_close_step_max"],
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
