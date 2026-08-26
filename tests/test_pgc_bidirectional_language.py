import unittest

from fastwam.datasets.pgc_libero import (
    build_pgc_bidirectional_language_pair_index,
)


class PGCBidirectionalLanguageTest(unittest.TestCase):
    def test_pair_index_deduplicates_and_sorts_targets(self):
        shared = {
            "pair_id": "pair-a",
            "source_instruction": "pick up the blue cup",
            "counterfactual_instruction": "pick up the red cup",
            "source_suite": "libero_10",
            "source_task_id": 3,
        }
        pairs = {
            2: {1: shared, 2: dict(shared)},
            3: {
                0: {
                    **shared,
                    "pair_id": "pair-b",
                    "counterfactual_instruction": "pick up the green cup",
                }
            },
        }

        indexed = build_pgc_bidirectional_language_pair_index(pairs)
        candidates = indexed["pick up the blue cup"]

        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            [item["counterfactual_instruction"] for item in candidates],
            ["pick up the green cup", "pick up the red cup"],
        )

    def test_pair_index_rejects_identical_languages(self):
        pairs = {
            0: {
                0: {
                    "source_instruction": "open the drawer",
                    "counterfactual_instruction": " OPEN THE DRAWER ",
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "distinct instructions"):
            build_pgc_bidirectional_language_pair_index(pairs)


if __name__ == "__main__":
    unittest.main()
