import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_pgc_robotwin_no_eraf_manifest import (
    MANIFEST_FORMAT,
    POOL_ORDER,
    _closed_loop_entry,
    load_no_eraf_manifest,
)
from experiments.robotwin.closed_loop_capture import (
    CAPTURE_ACTION_VIDEO_FREQ_RATIO,
    CAPTURE_FORMAT,
    CAPTURE_FRAME_COUNT,
    CAPTURE_PRODUCTIVE_START_COUNT,
    CAPTURE_TEMPORAL_CONTRACT,
)
from scripts.train_pgc_robotwin_no_eraf import MODES, build_overrides


class RoboTwinNoERAFTrainingTest(unittest.TestCase):
    def test_closed_loop_entry_rejects_legacy_two_frame_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "dataset"
            sidecar = Path(temp_dir) / "sidecar"
            index_path = dataset / "meta/pgc_robotwin_closed_loop_native.json"
            index_path.parent.mkdir(parents=True)
            payload = {
                "format": "pgc_robotwin_closed_loop_native_v2",
                "complete": True,
                "episode_count": 2,
                "frame_count": 2 * CAPTURE_FRAME_COUNT,
                "productive_frame_count": 2 * CAPTURE_PRODUCTIVE_START_COUNT,
                "capture_format": CAPTURE_FORMAT,
                "capture_frame_count": CAPTURE_FRAME_COUNT,
                "action_video_freq_ratio": CAPTURE_ACTION_VIDEO_FREQ_RATIO,
                "productive_start_count_per_episode": (
                    CAPTURE_PRODUCTIVE_START_COUNT
                ),
                "temporal_contract": CAPTURE_TEMPORAL_CONTRACT,
                "state_distribution": "immutable_base_closed_loop_replan",
                "full_goal_usage": "forbidden_not_present",
            }
            index_path.write_text(json.dumps(payload), encoding="utf-8")
            entry = _closed_loop_entry(dataset, sidecar)
            self.assertEqual(entry["productive_frames"], 10)
            payload.update(
                {
                    "format": "pgc_robotwin_closed_loop_native_v1",
                    "frame_count": 4,
                    "productive_frame_count": 0,
                }
            )
            index_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "productively unusable"):
                _closed_loop_entry(dataset, sidecar)

    def test_sample_plan_is_four_way_and_uses_only_declared_rows(self):
        try:
            from fastwam.datasets.lerobot.robotwin_no_eraf_dataset import (
                _closed_loop_productive_rows,
                build_robotwin_no_eraf_sample_plan,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional training dependency is unavailable: {exc.name}")
        fake_dataset = type(
            "FakeDataset",
            (),
            {
                "hf_dataset": {
                    "episode_index": [0] * CAPTURE_FRAME_COUNT,
                    "frame_index": list(range(CAPTURE_FRAME_COUNT)),
                }
            },
        )()
        productive, stages = _closed_loop_productive_rows(
            dataset=fake_dataset,
            index={
                "capture_format": CAPTURE_FORMAT,
                "capture_frame_count": CAPTURE_FRAME_COUNT,
                "action_video_freq_ratio": CAPTURE_ACTION_VIDEO_FREQ_RATIO,
                "productive_start_count_per_episode": (
                    CAPTURE_PRODUCTIVE_START_COUNT
                ),
                "productive_frame_count": CAPTURE_PRODUCTIVE_START_COUNT,
                "temporal_contract": CAPTURE_TEMPORAL_CONTRACT,
                "episodes_by_index": {
                    0: {
                        "frame_count": CAPTURE_FRAME_COUNT,
                        "action_video_freq_ratio": (
                            CAPTURE_ACTION_VIDEO_FREQ_RATIO
                        ),
                        "productive_start_count": (
                            CAPTURE_PRODUCTIVE_START_COUNT
                        ),
                        "temporal_contract": CAPTURE_TEMPORAL_CONTRACT,
                        "online_stage_v2": "initial_search",
                    }
                },
            },
            dataset_offset=10,
            action_video_freq_ratio=CAPTURE_ACTION_VIDEO_FREQ_RATIO,
        )
        self.assertEqual(productive, [10, 11, 12, 13, 14])
        self.assertEqual(stages, ["initial_search"] * 5)
        indices, groups = build_robotwin_no_eraf_sample_plan(
            dataset_frame_counts=[3, 2, 4, 1, 2, 3],
            offline_native_dataset_count=2,
            closed_loop_native_dataset_count=1,
            historical_cf_dataset_count=1,
            strict_cf_dataset_count=2,
            closed_loop_productive_indices=[5, 6],
            closed_loop_stage_categories=[
                "initial_search",
                "holding",
            ],
            strict_relation_categories=[
                "left_right",
                "stack",
                "left_right",
                "stack",
                "left_right",
            ],
        )
        self.assertEqual(len(indices), 24)
        self.assertEqual([groups.count(group) for group in range(4)], [6, 6, 6, 6])
        pools = [set(range(0, 5)), {5, 6}, {9}, set(range(10, 15))]
        for index, group in zip(indices, groups, strict=True):
            self.assertIn(index, pools[group])

    def test_manifest_loader_rejects_full_goal_and_preserves_order(self):
        pools = {}
        for pool in POOL_ORDER:
            entry = {
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
            if pool == "closed_loop_native":
                entry.update(
                    {
                        "frames": CAPTURE_FRAME_COUNT,
                        "productive_frames": CAPTURE_PRODUCTIVE_START_COUNT,
                        "capture_format": CAPTURE_FORMAT,
                        "capture_frame_count": CAPTURE_FRAME_COUNT,
                        "action_video_freq_ratio": (
                            CAPTURE_ACTION_VIDEO_FREQ_RATIO
                        ),
                        "productive_start_count_per_episode": (
                            CAPTURE_PRODUCTIVE_START_COUNT
                        ),
                        "temporal_contract": CAPTURE_TEMPORAL_CONTRACT,
                    }
                )
            pools[pool] = [entry]
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
            payload["pools"]["closed_loop_native"][0]["productive_frames"] = 0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "productive temporal"):
                load_no_eraf_manifest(path, verify_files=False)
            payload["pools"]["closed_loop_native"][0][
                "productive_frames"
            ] = CAPTURE_PRODUCTIVE_START_COUNT
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
        self.assertIn(
            "model.lora.paired_language_control.deployment_matched_action_cache=true",
            overrides,
        )
        self.assertIn(
            "model.lora.paired_language_control.correct_branch_action_ranking=true",
            overrides,
        )
        self.assertIn("pgc_closed_loop_corrective_dataset_dirs=[]", joined)
        self.assertNotIn("full_goal", joined)
        self.assertEqual(MODES["diagnostic"]["steps"], 1000)
        self.assertEqual(MODES["diagnostic"]["save_every"], 250)
        self.assertEqual(MODES["formal"]["save_every"], 250)


if __name__ == "__main__":
    unittest.main()
