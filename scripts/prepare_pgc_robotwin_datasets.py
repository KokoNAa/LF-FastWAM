#!/usr/bin/env python3
"""Prepare all matched RoboTwin PGC LeRobot datasets and ERAF sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from fastwam.datasets.pgc_libero import load_pgc_entity_relation_index
from scripts.build_pgc_robotwin_entity_relations import build_sidecar
from scripts.convert_pgc_robotwin_to_lerobot import convert_dataset
from scripts.validate_pgc_robotwin_raw import validate_raw_dataset


PAIR_IDS = (
    "place_a2b_left_to_place_a2b_right",
    "place_a2b_right_to_place_a2b_left",
)
DATASET_KINDS = ("native", "counterfactual")


@dataclass(frozen=True)
class DatasetSpec:
    pair_id: str
    dataset_kind: str
    raw_root: Path
    dataset_root: Path
    sidecar_root: Path


def dataset_specs(
    *, raw_root: Path, dataset_root: Path, sidecar_root: Path
) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            pair_id=pair_id,
            dataset_kind=kind,
            raw_root=raw_root / pair_id / kind,
            dataset_root=dataset_root / pair_id / kind,
            sidecar_root=sidecar_root / pair_id / kind,
        )
        for pair_id in PAIR_IDS
        for kind in DATASET_KINDS
    ]


def _sidecar_signatures(index: dict) -> list[tuple[str, str]]:
    return [
        (
            str(index["episodes_by_index"][episode_index]["initial_state_sha256"]),
            str(index["episodes_by_index"][episode_index]["pair_id"]),
        )
        for episode_index in range(int(index["episode_count"]))
    ]


def prepare(
    *,
    raw_root: Path,
    dataset_root: Path,
    sidecar_root: Path,
    fps: int,
    video_codec: str,
) -> dict:
    reports = []
    indices = {}
    for spec in dataset_specs(
        raw_root=raw_root,
        dataset_root=dataset_root,
        sidecar_root=sidecar_root,
    ):
        raw_report = validate_raw_dataset(spec.raw_root)
        if raw_report["pair_id"] != spec.pair_id:
            raise ValueError(
                f"Raw pair mismatch at {spec.raw_root}: {raw_report['pair_id']}."
            )
        if raw_report["dataset_kind"] != spec.dataset_kind:
            raise ValueError(
                f"Raw dataset_kind mismatch at {spec.raw_root}: "
                f"{raw_report['dataset_kind']}."
            )
        dataset_exists = spec.dataset_root.exists()
        sidecar_exists = (spec.sidecar_root / "index.json").is_file()
        if dataset_exists != sidecar_exists:
            raise FileExistsError(
                "Refusing ambiguous partial prepared output; dataset and "
                f"sidecar must both exist or both be absent: {spec}."
            )
        if not dataset_exists:
            convert_dataset(
                raw_root=spec.raw_root,
                output=spec.dataset_root,
                fps=fps,
                video_codec=video_codec,
            )
            build_sidecar(
                raw_root=spec.raw_root,
                dataset_root=spec.dataset_root,
                output_root=spec.sidecar_root,
            )
        index = load_pgc_entity_relation_index(spec.sidecar_root)
        if Path(str(index["dataset"])).resolve() != spec.dataset_root.resolve():
            raise ValueError(
                f"Sidecar does not bind its exact dataset: {spec.sidecar_root}."
            )
        if str(index["dataset_kind"]) != spec.dataset_kind:
            raise ValueError(
                f"Prepared sidecar kind mismatch: {spec.sidecar_root}."
            )
        if int(index["episode_count"]) != int(raw_report["episodes"]):
            raise ValueError(
                f"Prepared episode count mismatch: {spec.sidecar_root}."
            )
        indices[(spec.pair_id, spec.dataset_kind)] = index
        reports.append(
            {
                "pair_id": spec.pair_id,
                "dataset_kind": spec.dataset_kind,
                "episodes": int(index["episode_count"]),
                "dataset": str(spec.dataset_root.resolve()),
                "sidecar": str(spec.sidecar_root.resolve()),
                "camera_count": int(index["camera_count"]),
                "action_dim": int(index["action_dim"]),
                "valid": True,
            }
        )
    for pair_id in PAIR_IDS:
        native = indices[(pair_id, "native")]
        counterfactual = indices[(pair_id, "counterfactual")]
        if _sidecar_signatures(native) != _sidecar_signatures(counterfactual):
            raise ValueError(
                f"Prepared native/counterfactual sidecars are unmatched for {pair_id}."
            )
    payload = {
        "format": "pgc_robotwin_prepared_matrix_v1",
        "complete": True,
        "pairs": list(PAIR_IDS),
        "datasets": reports,
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "pgc_robotwin_prepared.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--video-codec",
        choices=("h264", "hevc", "libsvtav1", "h264_nvenc"),
        default="h264",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    payload = prepare(
        raw_root=args.raw_root.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        sidecar_root=args.sidecar_root.expanduser().resolve(),
        fps=args.fps,
        video_codec=args.video_codec,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
