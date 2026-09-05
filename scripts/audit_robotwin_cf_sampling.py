#!/usr/bin/env python3
"""Reconstruct the saved no-ERAF run's frame draws without training a model.

Loads the real dataset and sampler, but never calls dataset.__getitem__ or
loads model weights. Exact common prefixes require equal qpos and decoded RGB.
The supplied process count must match the original launch, not current GPUs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]


def common_prefix(native, target):
    import h5py
    import numpy as np
    from PIL import Image

    with h5py.File(native, "r") as s, h5py.File(target, "r") as t:
        sa, ta = s["joint_action/vector"][:], t["joint_action/vector"][:]
        count = 0
        for frame in range(min(len(sa), len(ta))):
            if not np.array_equal(sa[frame], ta[frame]):
                break
            same = True
            for camera in ("head_camera", "left_camera", "right_camera"):
                key = f"observation/{camera}/rgb"
                images = [np.array(Image.open(io.BytesIO(bytes(h[key][frame]).rstrip(b"\0"))).convert("RGB"))
                          for h in (s, t)]
                if not np.array_equal(*images):
                    same = False
                    break
            if not same:
                break
            count += 1
        return count


def main():
    from hydra.utils import instantiate
    import numpy as np
    from omegaconf import OmegaConf
    from fastwam.utils.samplers import ResumableEpochSampler

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.world_size <= 0 or args.output.exists():
        ap.error("Require positive world-size and a fresh output file.")
    cfg = OmegaConf.load(args.config)
    if (cfg.data.train._target_ != "fastwam.datasets.lerobot.robotwin_no_eraf_dataset.RoboTwinNoERAFFourPoolDataset"
            or cfg.data.train.skip_padding_as_possible or cfg.data.train.global_sample_stride != 1):
        raise ValueError("Only the audited four-pool, stride1, non-resampled loader is supported.")
    print("[sampling-audit] constructing real dataset (no model)", flush=True)
    dataset = instantiate(cfg.data.train)
    underlying = dataset.lerobot_dataset.multi_dataset._datasets
    dirs = list(cfg.data.train.dataset_dirs) + list(cfg.data.train.pgc_counterfactual_dataset_dirs)
    indices = np.asarray(dataset._sample_indices, dtype=np.int64)
    window = int(cfg.batch_size) * args.world_size * int(cfg.gradient_accumulation_steps)
    count = int(cfg.max_steps) * window
    if count > len(dataset):
        raise ValueError("This audit only supports a run that fits inside its first epoch.")
    sampler = ResumableEpochSampler(dataset, cfg.seed, cfg.batch_size, args.world_size,
                                   cfg.gradient_accumulation_steps)
    positions = np.fromiter(itertools.islice(iter(sampler), count), dtype=np.int64, count=count)
    draws = indices[positions]
    groups = np.asarray(dataset.pgc_v9_closed_loop_group_ids)[positions]
    rows = []
    prefixes = {}
    cooccurrence = defaultdict(set)
    offset = 0
    for di, (data, folder) in enumerate(zip(underlying, dirs, strict=True)):
        folder = Path(folder)
        role = dataset.pgc_entity_relation_indices[di]["artifact_role"]
        metadata = dataset.pgc_entity_relation_indices[di]["episodes_by_index"]
        episodes = np.asarray(data.hf_dataset["episode_index"], dtype=np.int64)
        frames = np.asarray(data.hf_dataset["frame_index"], dtype=np.int64)
        chosen = np.flatnonzero((draws >= offset) & (draws < offset + len(episodes)))
        local = draws[chosen] - offset
        unique_episodes = sorted(set(episodes.tolist()))
        prefix_lengths = {}
        if role != "closed_loop_native":
            for ep in unique_episodes:
                # .../domain/lerobot/pair/kind -> .../domain/raw/pair
                raw_pair = folder.parents[2] / "raw" / folder.parent.name
                key = (str(raw_pair), int(ep))
                if key not in prefixes:
                    prefixes[key] = common_prefix(raw_pair / "native/data" / f"episode{ep}.hdf5",
                                                  raw_pair / "counterfactual/data" / f"episode{ep}.hdf5")
                prefix_lengths[ep] = prefixes[key]
        common_draws = []
        for draw_position, li in zip(chosen, local):
            ep, frame = int(episodes[li]), int(frames[li])
            if role != "closed_loop_native" and frame < prefix_lengths[ep]:
                common_draws.append(int(li))
                record = metadata[ep]
                key = (int(draw_position) // window, str(record["pair_id"]),
                       str(record["initial_state_sha256"]), frame)
                cooccurrence[key].add(role)
        rows.append({"dataset_index": di, "path": str(folder), "role": role,
                     "loaded_episodes": len(unique_episodes), "loaded_frames": len(episodes),
                     "draws": len(local), "unique_drawn_frames": len(set(local.tolist())),
                     "initial_frame_draws": int(np.sum(frames[local] == 0)),
                     "first_four_frame_draws": int(np.sum(frames[local] < 4)),
                     "exact_common_prefix_draws": len(common_draws) if role != "closed_loop_native" else None,
                     "exact_common_prefix_available_frames": sum(prefix_lengths.values()) if prefix_lengths else None,
                     "common_prefix_length_histogram": dict(Counter(prefix_lengths.values())),
                     "drawn_initial_episodes": sorted(set(episodes[local][frames[local] == 0].tolist()))})
        offset += len(episodes)
        print(f"[sampling-audit] {di+1}/{len(dirs)} {role} draws={len(local)} common={len(common_draws)}", flush=True)
    if offset <= int(draws.max()) or sum(r["draws"] for r in rows) != count:
        raise ValueError("Sampler-to-underlying-index coverage failed.")
    result = {"format": "robotwin_cf_sampling_audit_v1", "complete": True,
              "config": str(args.config.resolve()), "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "world_size_supplied": args.world_size, "optimizer_steps": int(cfg.max_steps),
              "global_batch": window, "draws": count, "draws_per_pool": dict(Counter(map(int, groups))),
              "same_optimizer_same_common_state_both_historical_positive_branches": sum(
                  {"offline_native", "historical_cf"}.issubset(v) for v in cooccurrence.values()),
              "datasets": rows,
              "runtime_source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (REPO / "src/fastwam/utils/samplers.py",
                            REPO / "src/fastwam/datasets/lerobot/robotwin_no_eraf_dataset.py",
                            REPO / "src/fastwam/datasets/lerobot/robot_video_dataset.py")},
              "notes": ["Reconstructed draws, not captured training events; assumes saved seed, epoch0, supplied world size and unchanged runtime sampler.",
                        "Common prefix stops at first unequal decoded RGB or qpos; later reconvergence and approximate equivalence are not measured.",
                        "Matching states in the same optimizer window still do not share training diffusion noise.",
                        "Closed-loop native frame0 is a captured replan state, not necessarily an episode initial state."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("[complete]", args.output, flush=True)


if __name__ == "__main__":
    main()
