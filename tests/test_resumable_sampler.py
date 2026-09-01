import unittest
from collections import Counter

from fastwam.training_progress import optimizer_step_to_sampler_position
from fastwam.utils.samplers import ResumableEpochSampler


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


class ClosedLoopCurriculumSamplerTest(unittest.TestCase):
    def test_four_ranks_three_accumulations_balance_global_window(self):
        class Dataset:
            pgc_v9_closed_loop_group_ids = [0, 1, 2, 3] * 12

            def __len__(self):
                return len(self.pgc_v9_closed_loop_group_ids)

        dataset = Dataset()
        sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=42,
            batch_size=1,
            num_processes=4,
            gradient_accumulation_steps=3,
        )
        labels = [dataset.pgc_v9_closed_loop_group_ids[index] for index in sampler]
        for start in range(0, len(labels), 12):
            self.assertEqual(
                Counter(labels[start : start + 12]),
                Counter({0: 3, 1: 3, 2: 3, 3: 3}),
            )


if __name__ == "__main__":
    unittest.main()
