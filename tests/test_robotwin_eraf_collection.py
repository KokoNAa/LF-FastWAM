import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from experiments.robotwin.pgc_data import (
    ROBOTWIN_ERAF_PAIR_IDS,
    ROBOTWIN_ERAF_PAIR_SPECS,
    pair_spec_from_source_task,
    scene_state_vector,
    validate_pair_record,
)
from experiments.robotwin.pgc_task_variants import (
    PREDICATE_IDS,
    eraf_snapshot,
    install_pgc_task_contract,
)
from scripts.build_pgc_robotwin_entity_relations import build_sidecar
from scripts.prepare_pgc_robotwin_eraf_data import build_plan, raw_dataset_roots


class _Pose:
    def __init__(self, position):
        self.p = np.asarray(position, dtype=np.float32)
        self.q = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)


class _Entity:
    def __init__(self, actor_id):
        self.per_scene_id = actor_id


class _Actor:
    def __init__(self, position, actor_id):
        self._pose = _Pose(position)
        self.actor = _Entity(actor_id)

    def get_pose(self):
        return self._pose


class _Robot:
    def get_left_arm_real_jointState(self):
        return [0.0] * 7

    def get_right_arm_real_jointState(self):
        return [0.0] * 7


class _RankingTask:
    def __init__(self):
        self.block1 = _Actor((-0.08, -0.15, 0.765), 11)
        self.block2 = _Actor((0.00, -0.15, 0.765), 22)
        self.block3 = _Actor((0.08, -0.15, 0.765), 33)
        self.block1_target_pose = [-0.08, -0.15, 0.74, 0, 1, 0, 0]
        self.block2_target_pose = [0.00, -0.15, 0.74, 0, 1, 0, 0]
        self.block3_target_pose = [0.08, -0.15, 0.74, 0, 1, 0, 0]
        self.robot = _Robot()

    def check_success(self):
        return False

    def is_left_gripper_open(self):
        return True

    def is_right_gripper_open(self):
        return True


def _write_generic_hdf5(path: Path) -> None:
    frames = 2
    positions = np.zeros((frames, 4, 3), dtype=np.float32)
    positions[:, 0] = (-0.08, -0.15, 0.765)
    positions[:, 1] = (0.00, -0.15, 0.765)
    positions[:, 2] = (0.08, -0.15, 0.765)
    actor_ids = np.zeros((frames, 4), dtype=np.uint32)
    actor_ids[:, :3] = (11, 22, 33)
    entity_valid = np.zeros((frames, 4), dtype=np.bool_)
    entity_valid[:, :3] = True
    source_subject = np.zeros((frames, 4), dtype=np.int64)
    source_reference = np.zeros((frames, 4), dtype=np.int64)
    source_subject[:, :2] = (0, 1)
    source_reference[:, :2] = (1, 2)
    target_subject = np.zeros((frames, 4), dtype=np.int64)
    target_reference = np.zeros((frames, 4), dtype=np.int64)
    target_subject[:, :2] = (2, 1)
    target_reference[:, :2] = (1, 0)
    predicates = np.zeros((frames, 4), dtype=np.int64)
    predicates[:, :2] = PREDICATE_IDS["left"]
    clause_valid = np.zeros((frames, 4), dtype=np.bool_)
    clause_valid[:, :2] = True
    source_truth = np.zeros((frames, 4), dtype=np.float32)
    source_truth[:, :2] = 1.0
    target_truth = np.zeros((frames, 4), dtype=np.float32)
    goals = np.zeros((frames, 4, 3), dtype=np.float32)
    goals[:, 0] = (-0.08, -0.15, 0.74)
    goals[:, 1] = (0.00, -0.15, 0.74)
    head = np.zeros((frames, 16, 20), dtype=np.uint32)
    left = np.zeros((frames, 8, 10), dtype=np.uint32)
    right = np.zeros((frames, 8, 10), dtype=np.uint32)
    for frame in range(frames):
        head[frame, 1:4, 1:4] = 11
        head[frame, 5:8, 5:8] = 22
        head[frame, 9:12, 9:12] = 33
        left[frame, 1:3, 1:3] = 11
        left[frame, 3:5, 3:5] = 22
        left[frame, 5:7, 5:7] = 33
        right[frame] = left[frame]
    with h5py.File(path, "w") as handle:
        state = handle.create_group("pgc_entity_state")
        state.create_dataset("entity_positions", data=positions)
        state.create_dataset("entity_actor_ids", data=actor_ids)
        state.create_dataset("entity_valid", data=entity_valid)
        for prefix, subject, reference, truth in (
            ("source", source_subject, source_reference, source_truth),
            ("target", target_subject, target_reference, target_truth),
        ):
            state.create_dataset(f"{prefix}_subject_indices", data=subject)
            state.create_dataset(f"{prefix}_reference_indices", data=reference)
            state.create_dataset(f"{prefix}_predicate_ids", data=predicates)
            state.create_dataset(f"{prefix}_goal_positions", data=goals)
            state.create_dataset(f"{prefix}_predicate_truth", data=truth)
            state.create_dataset(f"{prefix}_clause_valid", data=clause_valid)
        observation = handle.create_group("observation")
        for camera, labels in (
            ("head_camera", head),
            ("left_camera", left),
            ("right_camera", right),
        ):
            observation.create_group(camera).create_dataset(
                "actor_segmentation_ids", data=labels
            )


class RoboTwinERAFCollectionTest(unittest.TestCase):
    def test_pair_catalog_covers_five_directed_pairs(self):
        self.assertEqual(
            ROBOTWIN_ERAF_PAIR_IDS,
            (
                "place_a2b_left_to_right",
                "place_a2b_right_to_left",
                "stack_blocks_two_green_on_red_to_red_on_green",
                "blocks_ranking_rgb_to_bgr",
                "place_burger_fries_native_to_swapped_slots",
            ),
        )
        self.assertEqual(len(ROBOTWIN_ERAF_PAIR_SPECS), 5)

    def test_ranking_snapshot_has_source_and_counterfactual_clauses(self):
        spec = pair_spec_from_source_task("blocks_ranking_rgb")
        task = install_pgc_task_contract(_RankingTask(), spec)
        snapshot = eraf_snapshot(task, spec)
        np.testing.assert_array_equal(snapshot["source_subject_indices"][:2], (0, 1))
        np.testing.assert_array_equal(snapshot["target_subject_indices"][:2], (2, 1))
        np.testing.assert_array_equal(
            snapshot["source_predicate_truth"][:2], (1.0, 1.0)
        )
        np.testing.assert_array_equal(
            snapshot["target_predicate_truth"][:2], (0.0, 0.0)
        )
        state = scene_state_vector(task)
        self.assertGreater(state.size, 35)
        task._pgc_active_variant = "rgb"
        self.assertTrue(task.check_success())
        task._pgc_active_variant = "bgr"
        self.assertFalse(task.check_success())

    def test_generic_pair_record_rejects_full_goal_aliasing(self):
        record = {
            "pair_id": "blocks_ranking_rgb_to_bgr",
            "dataset_kind": "counterfactual",
            "source_task": "blocks_ranking_rgb",
            "counterfactual_task": "blocks_ranking_bgr",
            "source_variant": "rgb",
            "counterfactual_variant": "bgr",
            "executed_variant": "bgr",
            "scene_seed": 44,
            "initial_state_sha256": "a" * 64,
            "action_sha256": "b" * 64,
            "action_count": 2,
        }
        self.assertEqual(validate_pair_record(record)["executed_variant"], "bgr")
        record["executed_variant"] = "rgb"
        with self.assertRaisesRegex(ValueError, "dataset_kind"):
            validate_pair_record(record)

    def test_five_pair_plan_has_ten_datasets(self):
        plan = build_plan(
            robotwin_root=Path("/repo/third_party/RoboTwin"),
            work_root=Path("/data/pgc_robotwin_eraf_v1"),
            stage="formal",
            episodes=5,
            task_config="demo_clean",
            start_seed=4_400_000,
            fps=10,
            video_codec="h264",
            skip_collection=False,
            python="python",
        )
        self.assertEqual(plan["pair_count"], 5)
        self.assertEqual(plan["dataset_count"], 10)
        self.assertEqual(plan["total_successful_trajectories"], 50)
        self.assertEqual(len(raw_dataset_roots(Path(plan["raw_root"]))), 10)

    def test_generic_sidecar_preserves_two_clause_semantics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            (raw / "meta").mkdir(parents=True)
            (raw / "data").mkdir()
            _write_generic_hdf5(raw / "data" / "episode0.hdf5")
            record = {
                "episode_index": 0,
                "pair_id": "blocks_ranking_rgb_to_bgr",
                "raw_hdf5": "data/episode0.hdf5",
                "initial_state_sha256": "a" * 64,
                "action_sha256": "b" * 64,
            }
            (raw / "meta" / "pgc_episodes.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            provenance = {
                "dataset_kind": "native",
                "artifact_role": "eraf_grounding_supervision",
                "allowed_training_stages": ["grounding"],
                "pairs": [
                    {
                        "source_task": "blocks_ranking_rgb",
                        "entity_names": [
                            "red_block",
                            "green_block",
                            "blue_block",
                        ],
                    }
                ],
            }
            (raw / "meta" / "pgc_provenance.json").write_text(
                json.dumps(provenance), encoding="utf-8"
            )
            output = root / "sidecar"
            build_sidecar(
                raw_root=raw,
                dataset_root=root / "lerobot",
                output_root=output,
            )
            index = json.loads((output / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["artifact_role"], "eraf_grounding_supervision")
            self.assertFalse(index["full_goal_verified"])
            with np.load(output / "episodes" / "episode0.npz") as arrays:
                np.testing.assert_array_equal(
                    arrays["source_clause_valid"][0], (True, True, False, False)
                )
                np.testing.assert_array_equal(
                    arrays["target_predicate_ids"][0, :2],
                    (PREDICATE_IDS["left"], PREDICATE_IDS["left"]),
                )
                self.assertTrue(arrays["target_subject_masks"][:, :2].any())


if __name__ == "__main__":
    unittest.main()
