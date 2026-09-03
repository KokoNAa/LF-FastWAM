import unittest
from pathlib import Path

from fastwam.datasets.robotwin_eraf_sampling import (
    build_robotwin_eraf_grounding_sample_indices,
    robotwin_array_sha256,
)
from experiments.robotwin.pgc_data import array_sha256 as collection_array_sha256
from scripts.train_pgc_robotwin_eraf_grounding import build_overrides
from scripts.train_pgc_robotwin_eraf_grounding_role import (
    build_role_overrides,
    validate_grounding_payload,
)


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
    def test_runtime_hash_matches_robotwin_collection_contract(self):
        import numpy as np

        actions = np.arange(42, dtype=np.float32).reshape(3, 14)
        self.assertEqual(
            robotwin_array_sha256(actions), collection_array_sha256(actions)
        )

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

    def test_role_continuation_matches_libero_objective_three_contract(self):
        matrix = {
            "dataset_dirs": [f"/native/{index}" for index in range(10)],
            "counterfactual_dirs": [f"/cf/{index}" for index in range(10)],
            "sidecar_dirs": [f"/sidecar/{index}" for index in range(20)],
        }
        overrides = build_role_overrides(
            matrix=matrix,
            grounding_checkpoint=Path("/step_001500.pt"),
            stats_path=Path("/stats.json"),
            cache_dir=Path("/cache"),
            seed=42,
            stage_steps=1000,
            save_every=250,
            gradient_accumulation_steps=4,
        )
        self.assertIn("resume=/step_001500.pt", overrides)
        self.assertIn("weight_only_start_step=1500", overrides)
        self.assertIn("max_steps=2500", overrides)
        self.assertIn("learning_rate=2.0e-5", overrides)
        self.assertIn(
            "model.policy_guard.entity_relation_grounding."
            "grounding_objective_version=3",
            overrides,
        )
        self.assertIn(
            "model.policy_guard.entity_relation_grounding."
            "role_assignment_weight=4.0",
            overrides,
        )
        self.assertIn(
            "model.policy_guard.entity_relation_grounding."
            "role_assignment_hard_weight=2.0",
            overrides,
        )
        self.assertFalse(any("full_goal" in value for value in overrides))

    def test_role_continuation_admits_exact_objective_two_checkpoint(self):
        payload = {
            "format": "fastwam_policy_guard_v9",
            "step": 1500,
            "architecture_metadata": {
                "policy_guard_version": 9,
                "eraf_grounding_objective_version": 2,
                "eraf_training_stage": "grounding",
                "warm_start_contract": "released_base_fresh_eraf",
                "action_output_dim": 14,
                "proprio_dim": 14,
                "eraf_camera_count": 3,
                "eraf_camera_layout": "robotwin_mosaic",
                "eraf_visual_aspect_ratio": 5.0 / 6.0,
            },
        }
        metadata = validate_grounding_payload(payload)
        self.assertEqual(metadata["eraf_grounding_objective_version"], 2)
        payload["architecture_metadata"]["eraf_grounding_objective_version"] = 3
        with self.assertRaisesRegex(ValueError, "mismatches"):
            validate_grounding_payload(payload)


if __name__ == "__main__":
    unittest.main()
