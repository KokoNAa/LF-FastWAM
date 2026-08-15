import unittest

from fastwam.training_progress import optimizer_step_to_sampler_position


class WeightOnlyResumePositionTest(unittest.TestCase):
    def test_maps_pgc_step_500_to_accumulated_micro_batches(self):
        position = optimizer_step_to_sampler_position(
            dataset_size=134_618,
            batch_size=1,
            num_processes=4,
            gradient_accumulation_steps=4,
            optimizer_step=500,
        )
        self.assertEqual(position["micro_steps_per_epoch"], 33_655)
        self.assertEqual(position["optimizer_steps_per_epoch"], 8_414)
        self.assertEqual(position["epoch"], 0)
        self.assertEqual(position["optimizer_step_in_epoch"], 500)
        self.assertEqual(position["batch_in_epoch"], 2_000)

    def test_exact_epoch_boundary_starts_next_epoch_without_offset(self):
        position = optimizer_step_to_sampler_position(
            dataset_size=134_618,
            batch_size=1,
            num_processes=4,
            gradient_accumulation_steps=4,
            optimizer_step=8_414,
        )
        self.assertEqual(position["epoch"], 1)
        self.assertEqual(position["optimizer_step_in_epoch"], 0)
        self.assertEqual(position["batch_in_epoch"], 0)

    def test_rejects_negative_optimizer_step(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            optimizer_step_to_sampler_position(
                dataset_size=10,
                batch_size=1,
                num_processes=1,
                gradient_accumulation_steps=1,
                optimizer_step=-1,
            )


if __name__ == "__main__":
    unittest.main()
