import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from fastwam.datasets.pgc_libero import (
    build_provenance,
    demo_file_candidates,
    filter_libero_noops,
    iter_libero_hdf5_demos,
    libero_lerobot_features,
    resolve_demo_file,
    state_sha256,
    states_match,
)


class PGCLiberoDataBuilderTest(unittest.TestCase):
    def setUp(self):
        self.record = {
            "pair_id": "libero_object_00_to_libero_object_01",
            "task_suite_name": "libero_object",
            "task_id": 0,
            "correct_instruction": "pick up the alphabet soup",
            "counterfactual_instruction": "pick up the cream cheese",
            "counterfactual_task_suite_name": "libero_object",
            "counterfactual_task_id": 1,
            "counterfactual_bddl_file": "/bddl/pick_up_the_cream_cheese.bddl",
            "counterfactual_goal_state": [
                ["in", "cream_cheese_1", "basket_1_contain_region"]
            ],
        }

    def test_state_hash_is_dtype_and_byte_order_stable(self):
        state32 = np.array([1.0, -2.5, 3.25], dtype=np.float32)
        state64_be = state32.astype(">f8")
        self.assertEqual(state_sha256(state32), state_sha256(state64_be))
        self.assertTrue(states_match(state32, state64_be))
        self.assertFalse(states_match(state32, state64_be + 1e-3))
        self.assertRegex(state_sha256(state32), r"^[0-9a-f]{64}$")

    def test_resolves_and_reads_standard_libero_hdf5(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "pick_up_the_cream_cheese_demo.hdf5"
            with h5py.File(path, "w") as handle:
                data = handle.create_group("data")
                for name, offset in (("demo_10", 10), ("demo_2", 2)):
                    demo = data.create_group(name)
                    demo.create_dataset(
                        "states",
                        data=np.arange(24, dtype=np.float64).reshape(3, 8) + offset,
                    )
                    demo.create_dataset(
                        "actions",
                        data=np.arange(21, dtype=np.float32).reshape(3, 7),
                    )

            self.assertEqual(demo_file_candidates(root, self.record), [path.resolve()])
            self.assertEqual(resolve_demo_file(root, self.record), path.resolve())
            demos = list(iter_libero_hdf5_demos(path))
            self.assertEqual([demo.group_name for demo in demos], ["demo_2", "demo_10"])
            self.assertEqual(demos[0].initial_state.shape, (8,))
            self.assertEqual(demos[0].actions.shape, (3, 7))

    def test_ambiguous_demo_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for directory in (root / "a", root / "b"):
                directory.mkdir()
                (directory / "pick_up_the_cream_cheese_demo.hdf5").touch()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                resolve_demo_file(root, self.record)

    def test_provenance_and_lerobot_feature_contract(self):
        provenance = build_provenance([self.record])
        self.assertEqual(provenance["benchmark"], "libero")
        self.assertTrue(provenance["state_aligned"])
        self.assertTrue(provenance["successful_only"])
        self.assertEqual(provenance["source_suites"], ["libero_object"])
        self.assertEqual(provenance["pairs"][0]["source_task_id"], 0)

        features = libero_lerobot_features(512)
        self.assertEqual(features["observation.state"]["shape"], (8,))
        self.assertEqual(features["action"]["shape"], (7,))
        self.assertEqual(
            features["observation.images.image"]["shape"], (3, 512, 512)
        )
        json.dumps(provenance)

    def test_noop_filter_preserves_gripper_transitions(self):
        actions = np.array(
            [
                [0.1, 0, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 1],
                [0.2, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        filtered = filter_libero_noops(actions)
        self.assertEqual(filtered.shape, (3, 7))
        np.testing.assert_array_equal(filtered[0], actions[0])
        np.testing.assert_array_equal(filtered[1], actions[2])
        np.testing.assert_array_equal(filtered[2], actions[3])


if __name__ == "__main__":
    unittest.main()
