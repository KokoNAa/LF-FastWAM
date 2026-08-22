from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fastwam.datasets.pgc_libero import (
    PGC_ACTION_CONVENTION_FASTWAM,
    PGC_ACTION_REPLAY_FASTWAM_TO_LIBERO_ENV,
    PGC_ENTITY_RELATION_PREDICATES,
    load_pgc_entity_relation_index,
)
from scripts.migrate_pgc_eraf_workspace import migrate


class PGCERAFWorkspaceMigrationTest(unittest.TestCase):
    def test_out_of_place_migration_preserves_world_positions_and_audits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "canonical"
            episode = source / "episodes" / "episode_000000.npz"
            episode.parent.mkdir(parents=True)
            old_min = np.asarray([-0.65, -0.60, 0.70], dtype=np.float32)
            old_max = np.asarray([0.65, 0.60, 1.45], dtype=np.float32)
            new_min = np.asarray([-0.80, -0.80, 0.00], dtype=np.float32)
            new_max = np.asarray([0.80, 0.80, 1.20], dtype=np.float32)
            world = np.asarray([0.20, -0.10, 0.90], dtype=np.float32)
            old_normalized = 2.0 * (world - old_min) / (old_max - old_min) - 1.0
            arrays: dict[str, np.ndarray] = {}
            for role in ("target", "source"):
                for name, valid_name in (
                    ("subject_positions", "subject_position_valid"),
                    ("reference_positions", "reference_position_valid"),
                    ("grasp_anchors", "grasp_anchor_valid"),
                    ("goal_anchors", "goal_anchor_valid"),
                    ("interaction_anchors", "interaction_anchor_valid"),
                ):
                    values = np.zeros((1, 4, 3), dtype=np.float32)
                    values[0, 0] = old_normalized
                    valid = np.zeros((1, 4), dtype=np.bool_)
                    valid[0, 0] = True
                    arrays[f"{role}_{name}"] = values
                    arrays[f"{role}_{valid_name}"] = valid
            np.savez_compressed(episode, **arrays)
            original_digest = hashlib.sha256(episode.read_bytes()).hexdigest()
            state_digest = "0" * 64
            action_digest = "1" * 64
            index = {
                "format": "pgc_libero_entity_relation_v1",
                "dataset": str(root / "dataset"),
                "dataset_kind": "native",
                "dataset_action_convention": PGC_ACTION_CONVENTION_FASTWAM,
                "simulator_replay_action_transform": (
                    PGC_ACTION_REPLAY_FASTWAM_TO_LIBERO_ENV
                ),
                "privileged_supervision": "training_only",
                "deployment_inputs": "rgb_language_proprio",
                "max_clauses": 4,
                "predicate_vocabulary": list(PGC_ENTITY_RELATION_PREDICATES),
                "entity_id_scheme": "sha256_63bit",
                "entity_vocabulary": {"subject_1": 11, "reference_1": 21},
                "camera_names": ["agentview", "robot0_eye_in_hand"],
                "view_center_coordinate_system": "per_camera_normalized_xy",
                "mask_size": [2, 4],
                "workspace_min": old_min.tolist(),
                "workspace_max": old_max.tolist(),
                "episode_count": 1,
                "episodes": [
                    {
                        "episode_index": 0,
                        "pair_id": "pair",
                        "file": "episodes/episode_000000.npz",
                        "sha256": original_digest,
                        "state_sha256": state_digest,
                        "action_sha256": action_digest,
                        "frame_count": 1,
                    }
                ],
            }
            (source / "index.json").write_text(json.dumps(index), encoding="utf-8")
            report = migrate(
                argparse.Namespace(
                    input=source,
                    output=output,
                    workspace_min=tuple(new_min.tolist()),
                    workspace_max=tuple(new_max.tolist()),
                )
            )
            self.assertTrue(report["validated"])
            loaded = load_pgc_entity_relation_index(output)
            self.assertEqual(loaded["episodes"][0]["state_sha256"], state_digest)
            self.assertEqual(loaded["episodes"][0]["action_sha256"], action_digest)
            self.assertNotEqual(loaded["episodes"][0]["sha256"], original_digest)
            with np.load(output / "episodes" / "episode_000000.npz") as archive:
                migrated = archive["target_subject_positions"]
                expected = np.clip(
                    2.0 * (world - new_min) / (new_max - new_min) - 1.0,
                    -1.0,
                    1.0,
                )
                np.testing.assert_allclose(migrated[0, 0], expected, atol=2e-7)
                np.testing.assert_array_equal(migrated[0, 1:], 0.0)


if __name__ == "__main__":
    unittest.main()
