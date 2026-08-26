#!/usr/bin/env python3
"""Convert audited raw RoboTwin PGC HDF5 episodes to LeRobot v2.1."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from fastwam.datasets.pgc_libero import read_jsonl
from scripts.validate_pgc_robotwin_raw import validate_raw_dataset


def robotwin_lerobot_features(height: int, width: int) -> dict[str, dict]:
    image = {
        "dtype": "video",
        "shape": (3, int(height), int(width)),
        "names": ["channels", "height", "width"],
    }
    qpos_names = [
        *(f"left_joint_{index}" for index in range(6)),
        "left_gripper",
        *(f"right_joint_{index}" for index in range(6)),
        "right_gripper",
    ]
    return {
        "observation.images.cam_high": dict(image),
        "observation.images.cam_left_wrist": dict(image),
        "observation.images.cam_right_wrist": dict(image),
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": qpos_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": qpos_names,
        },
    }


def _decode_rgb(value: np.bytes_) -> np.ndarray:
    encoded = bytes(value).rstrip(b"\0")
    if not encoded:
        raise ValueError("Raw RoboTwin frame contains an empty JPEG payload.")
    with Image.open(io.BytesIO(encoded)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _array(handle: h5py.File, key: str) -> np.ndarray:
    if key not in handle:
        raise KeyError(f"Raw RoboTwin HDF5 is missing {key!r}.")
    return np.asarray(handle[key])


def convert_dataset(
    *, raw_root: Path, output: Path, fps: int, video_codec: str
) -> Path:
    validate_raw_dataset(raw_root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite LeRobot dataset: {output}.")
    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

    records = read_jsonl(raw_root / "meta" / "pgc_episodes.jsonl")
    provenance = json.loads(
        (raw_root / "meta" / "pgc_provenance.json").read_text(encoding="utf-8")
    )
    dataset_kind = str(provenance.get("dataset_kind", ""))
    if dataset_kind not in {"native", "counterfactual"}:
        raise ValueError(f"Invalid raw RoboTwin dataset_kind: {dataset_kind!r}.")
    first_path = raw_root / str(records[0]["raw_hdf5"])
    with h5py.File(first_path, "r") as handle:
        first_rgb = _decode_rgb(_array(handle, "observation/head_camera/rgb")[0])
    height, width = map(int, first_rgb.shape[:2])
    dataset = LeRobotDataset.create(
        repo_id=output.name,
        root=output,
        fps=int(fps),
        robot_type="aloha-agilex",
        features=robotwin_lerobot_features(height, width),
        use_videos=True,
        video_codec=str(video_codec),
        is_compute_episode_stats_image=False,
    )
    output_audits = []
    for episode_index, record in enumerate(records):
        raw_path = raw_root / str(record["raw_hdf5"])
        instruction_key = (
            "source_instruction"
            if dataset_kind == "native"
            else "counterfactual_instruction"
        )
        instruction = str(record[instruction_key]).strip()
        with h5py.File(raw_path, "r") as handle:
            actions = _array(handle, "joint_action/vector").astype(np.float32)
            rgb = {
                "cam_high": _array(handle, "observation/head_camera/rgb"),
                "cam_left_wrist": _array(handle, "observation/left_camera/rgb"),
                "cam_right_wrist": _array(handle, "observation/right_camera/rgb"),
            }
            frame_count = int(actions.shape[0])
            if any(len(value) != frame_count for value in rgb.values()):
                raise ValueError(
                    f"RGB/qpos frame mismatch in raw episode {episode_index}."
                )
            for frame_index in range(frame_count):
                frame = {
                    f"observation.images.{camera}": _decode_rgb(values[frame_index])
                    for camera, values in rgb.items()
                }
                frame["observation.state"] = actions[frame_index]
                frame["action"] = actions[frame_index]
                dataset.add_frame(
                    frame,
                    task=[instruction, instruction, instruction, instruction],
                )
        dataset.save_episode(raw_file_name=str(raw_path))
        normalized = dict(record)
        normalized["episode_index"] = episode_index
        output_audits.append(normalized)

    provenance["converted_format"] = "lerobot_v2.1"
    provenance["converted_dataset"] = str(output.resolve())
    provenance["fps"] = int(fps)
    (output / "meta" / "pgc_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "meta" / "pgc_episodes.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in output_audits:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    source_state_dir = raw_root / "meta" / "initial_states"
    target_state_dir = output / "meta" / "initial_states"
    shutil.copytree(source_state_dir, target_state_dir)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--video-codec",
        choices=("h264", "hevc", "libsvtav1", "h264_nvenc"),
        default="h264",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    print(
        convert_dataset(
            raw_root=args.raw_root.expanduser().resolve(),
            output=args.output.expanduser().resolve(),
            fps=args.fps,
            video_codec=args.video_codec,
        )
    )


if __name__ == "__main__":
    main()
