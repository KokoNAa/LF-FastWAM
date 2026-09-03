import unittest
from pathlib import Path

from fastwam.datasets.robotwin_eraf_sampling import (
    build_robotwin_eraf_grounding_sample_indices,
)
from scripts.train_pgc_robotwin_eraf_grounding import build_overrides


PAIR_IDS = tuple(f"pair_{index}" for index in range(5))


def _formal_sampling_fixture():
    frame_groups = []
    sidecars = {}
    offset = 0
    dataset_index = 0
    for kind in ("native", "counterfactual"):
        for pair_id in PAIR_IDS:
            for episodes, frames in ((3, 6), (2, 4)):
                frame_groups.append(list(range(offset, offset + frames)))
                offset += frames
                sidecars[dataset_index] = {
                    "dataset_kind": kind,
                    "episode_count": episodes,
                    "artifact_role": "eraf_grounding_supervision",
                    "allowed_training_stages": ["grounding"],
                    "full_goal_verified": False,
                    "episodes_by_index": {
                        episode: {"pair_id": pair_id}
                        for episode in range(episodes)
                    },
                }
                dataset_index += 1
    return frame_groups, sidecars


class RoboTwinERAFGroundingTrainingTest(unittest.TestCase):
    def test_sampler_balances_pair_kind_after_combining_domains(self):
        frame_groups, sidecars = _formal_sampling_fixture()
        sample_indices, labels = build_robotwin_eraf_grounding_sample_indices(
            frame_groups=frame_groups,
            sidecar_indices=sidecars,
            native_dataset_count=10,
        )
        self.assertEqual(len(labels), 10)
        self.assertEqual(len(sample_indices), 100)
        self.assertEqual(labels[:5], tuple(f"native:{pair}" for pair in PAIR_IDS))
        self.assertEqual(
            labels[5:], tuple(f"counterfactual:{pair}" for pair in PAIR_IDS)
        )
        native_frame_end = sum(len(group) for group in frame_groups[:10])
        self.assertEqual(
            sum(index < native_frame_end for index in sample_indices), 50
        )

    def test_sampler_rejects_wrong_domain_episode_composition(self):
        frame_groups, sidecars = _formal_sampling_fixture()
        sidecars[0]["episode_count"] = 4
        with self.assertRaisesRegex(ValueError, "clean=3 and randomized=2"):
            build_robotwin_eraf_grounding_sample_indices(
                frame_groups=frame_groups,
                sidecar_indices=sidecars,
                native_dataset_count=10,
            )

    def test_launcher_matches_libero_grounding_contract(self):
        matrix = {
            "dataset_dirs": [f"/native/{index}" for index in range(10)],
            "counterfactual_dirs": [f"/cf/{index}" for index in range(10)],
            "sidecar_dirs": [f"/sidecar/{index}" for index in range(20)],
        }
        overrides = build_overrides(
            matrix=matrix,
            base_checkpoint=Path("/base.pt"),
            stats_path=Path("/stats.json"),
            cache_dir=Path("/cache"),
            seed=42,
            steps=1500,
            save_every=250,
            gradient_accumulation_steps=4,
        )
        self.assertIn("max_steps=1500", overrides)
        self.assertIn("gradient_accumulation_steps=4", overrides)
        self.assertIn("learning_rate=1.0e-4", overrides)
        self.assertIn("model.lora.enabled=false", overrides)
        self.assertIn(
            "model.policy_guard.entity_relation_grounding."
            "grounding_objective_version=2",
            overrides,
        )
        self.assertTrue(
            any("RoboTwinERAFGroundingDataset" in value for value in overrides)
        )
        self.assertFalse(any("full_goal" in value for value in overrides))


if __name__ == "__main__":
    unittest.main()
