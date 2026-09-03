import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_pgc_robotwin_no_eraf_manifest import (
    MANIFEST_FORMAT,
    POOL_ORDER,
    load_no_eraf_manifest,
)
from scripts.train_pgc_robotwin_no_eraf import MODES, build_overrides


class RoboTwinNoERAFTrainingTest(unittest.TestCase):
    def test_sample_plan_is_four_way_and_uses_only_declared_rows(self):
        try:
            from fastwam.datasets.lerobot.robotwin_no_eraf_dataset import (
                build_robotwin_no_eraf_sample_plan,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional training dependency is unavailable: {exc.name}")
        indices, groups = build_robotwin_no_eraf_sample_plan(
            dataset_frame_counts=[3, 2, 4, 1, 2, 3],
            offline_native_dataset_count=2,
            closed_loop_native_dataset_count=1,
            historical_cf_dataset_count=1,
            strict_cf_dataset_count=2,
            closed_loop_stage_categories=[
                "initial_search",
                "holding",
                "released_unfinished",
                "holding",
            ],
            strict_relation_categories=["left_right", "stack", "left_right"],
        )
        self.assertEqual(len(indices), 20)
        self.assertEqual([groups.count(group) for group in range(4)], [5, 5, 5, 5])
        pools = [set(range(0, 5)), set(range(5, 9)), {9}, set(range(10, 15))]
        for index, group in zip(indices, groups, strict=True):
            self.assertIn(index, pools[group])

    def test_manifest_loader_rejects_full_goal_and_preserves_order(self):
        pools = {}
        for pool in POOL_ORDER:
            pools[pool] = [
                {
                    "dataset": f"/{pool}/dataset",
                    "sidecar": f"/{pool}/sidecar",
                    "episodes": 1,
                    "dataset_kind": (
                        "native" if pool.endswith("native") else "counterfactual"
                    ),
                    "artifact_role": pool,
                    "full_goal_verified": False,
                    "valid": True,
                }
            ]
        payload = {
            "format": MANIFEST_FORMAT,
            "complete": True,
            "training_stage": "no_eraf",
            "sampling_contract": "deterministic_1_1_1_1",
            "pool_order": list(POOL_ORDER),
            "full_goal_usage": "forbidden_not_present",
            "allowed_training_stage": "no_eraf",
            "pools": pools,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            matrix = load_no_eraf_manifest(path, verify_files=False)
            self.assertEqual(matrix["pool_order"], list(POOL_ORDER))
            self.assertEqual(matrix["dataset_counts"], {pool: 1 for pool in POOL_ORDER})
            payload["pools"]["strict_cf"][0]["full_goal_verified"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "full-goal"):
                load_no_eraf_manifest(path, verify_files=False)

    def test_launcher_matches_libero_no_eraf_hyperparameters(self):
        matrix = {
            "offline_native_dirs": ["/offline/a", "/offline/b"],
            "closed_loop_native_dirs": ["/closed"],
            "historical_cf_dirs": ["/historical/a"],
            "strict_cf_dirs": ["/strict/a"],
            "sidecar_dirs": ["/s0", "/s1", "/s2", "/s3", "/s4"],
            "dataset_counts": {
                "offline_native": 2,
                "closed_loop_native": 1,
                "historical_cf": 1,
                "strict_cf": 1,
            },
        }
        overrides = build_overrides(
            matrix=matrix,
            base_checkpoint=Path("/base.pt"),
            stats_path=Path("/stats.json"),
            cache_dir=Path("/cache"),
            seed=42,
            mode="formal",
        )
        joined = "\n".join(overrides)
        self.assertIn("max_steps=10000", overrides)
        self.assertIn("gradient_accumulation_steps=4", overrides)
        self.assertIn("learning_rate=5.0e-6", overrides)
        self.assertIn("model.policy_guard.enabled=false", overrides)
        self.assertIn("model.lora.rank=16", overrides)
        self.assertIn("pgc_closed_loop_corrective_dataset_dirs=[]", joined)
        self.assertNotIn("full_goal", joined)
        self.assertEqual(MODES["formal"]["save_every"], 250)


if __name__ == "__main__":
    unittest.main()
