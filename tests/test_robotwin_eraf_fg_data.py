from pathlib import Path

import h5py
import numpy as np

from experiments.robotwin.eraf_fg_data import RawReplay


def test_single_frame_ground_labels_keep_real_trajectory_phase_and_cache(tmp_path):
    for domain in ("demo_clean", "demo_randomized"):
        folder = tmp_path / domain
        folder.mkdir()
        (folder / "plan.json").write_text('{"pairs": []}')
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as h:
        positions = np.zeros((3, 4, 3), dtype=np.float32)
        positions[:, :, 2] = .74
        positions[2, 0, 2] = .84
        h["pgc_entity_state/entity_positions"] = positions
        h["pgc_entity_state/entity_actor_ids"] = np.tile([11, 22, 33, 44], (3, 1))
        h["pgc_entity_state/entity_valid"] = np.ones((3, 4), dtype=bool)
        for prefix in ("source", "target"):
            for name, value in {
                "subject_indices": np.zeros((3, 4), dtype=np.int64),
                "reference_indices": np.ones((3, 4), dtype=np.int64),
                "predicate_ids": np.ones((3, 4), dtype=np.int64),
                "goal_positions": np.zeros((3, 4, 3), dtype=np.float32),
                "predicate_truth": np.zeros((3, 4), dtype=np.float32),
                "clause_valid": np.tile([True, False, False, False], (3, 1)),
            }.items():
                h[f"pgc_entity_state/{prefix}_{name}"] = value
        image = np.zeros((3, 24, 20), dtype=np.uint32)
        image[:, :12] = 11
        image[:, 12:] = 22
        for camera in ("head_camera", "left_camera", "right_camera"):
            h[f"observation/{camera}/actor_segmentation_ids"] = image
    row = {"raw_paths": {"native": str(path), "counterfactual": str(path)},
           "source_frame_index": 2, "target_frame_index": 2, "pair_id": "test"}
    reader = RawReplay(tmp_path, label_cache=tmp_path / "cache")
    labels = reader.grounding(row, "source")
    assert labels["phase_ids"][0, 0].item() == 1
    assert not labels["phase_safe_memory_state_valid"].any()
    assert labels["subject_mask_valid"][0, 0]
    assert labels["reference_mask_valid"][0, 0]
    # Both role labels use the same cached raw frame without reading a full
    # trajectory of camera masks on every optimization sample.
    del reader
    again = RawReplay(tmp_path, label_cache=tmp_path / "cache").grounding(row, "target")
    assert again["phase_ids"][0, 0].item() == 1
