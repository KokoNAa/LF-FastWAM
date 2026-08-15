import unittest

from fastwam.datasets.lerobot.robot_video_dataset import (
    build_pgc_sample_indices,
)


class PGCSamplingTest(unittest.TestCase):
    def test_balanced_indices_are_exactly_one_to_one(self):
        indices = build_pgc_sample_indices(
            native_frame_count=10,
            total_frame_count=14,
            balance_native_counterfactual=True,
        )
        native = [index for index in indices if index < 10]
        counterfactual = [index for index in indices if index >= 10]
        self.assertEqual(len(native), 10)
        self.assertEqual(len(counterfactual), 10)
        self.assertEqual(set(native), set(range(10)))
        self.assertEqual(set(counterfactual), set(range(10, 14)))

    def test_balancing_rejects_manual_oversampling(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_pgc_sample_indices(
                native_frame_count=10,
                total_frame_count=14,
                counterfactual_oversample_factor=2,
                balance_native_counterfactual=True,
            )

    def test_legacy_oversampling_is_preserved(self):
        indices = build_pgc_sample_indices(
            native_frame_count=3,
            total_frame_count=5,
            counterfactual_oversample_factor=2,
        )
        self.assertEqual(indices, [0, 1, 2, 3, 4, 3, 4])


if __name__ == "__main__":
    unittest.main()
