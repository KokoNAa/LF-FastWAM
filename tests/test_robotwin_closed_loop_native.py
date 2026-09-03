import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.robotwin.closed_loop_capture import (
    CAPTURE_STATE_DISTRIBUTION,
    capture_frame,
    classify_online_stage,
    write_capture_segment,
)
from experiments.robotwin.pgc_data import pair_spec_from_source_task
from experiments.robotwin.pgc_task_variants import install_pgc_observation_contract
from scripts.build_pgc_robotwin_closed_loop_native import _training_pair_id


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


class _Cameras:
    def get_raw_segmentation(self, level="actor"):
        assert level == "actor"
        head = np.zeros((32, 40), dtype=np.uint32)
        left = np.zeros((16, 20), dtype=np.uint32)
        right = np.zeros((16, 20), dtype=np.uint32)
        head[2:10, 2:10] = 11
        head[15:25, 20:32] = 22
        left[1:6, 1:6] = 11
        left[8:14, 10:18] = 22
        right[:] = left
        return {
            "head_camera": head,
            "left_camera": left,
            "right_camera": right,
        }


class _PlaceTask:
    def __init__(self):
        self.object = _Actor((-0.2, 0.0, 0.76), 11)
        self.target_object = _Actor((0.1, 0.0, 0.76), 22)
        self.robot = _Robot()
        self.cameras = _Cameras()

    def check_success(self):
        return False

    def is_left_gripper_open(self):
        return True

    def is_right_gripper_open(self):
        return True


def _observation():
    return {
        "observation": {
            "head_camera": {"rgb": np.zeros((32, 40, 3), dtype=np.uint8)},
            "left_camera": {"rgb": np.zeros((16, 20, 3), dtype=np.uint8)},
            "right_camera": {"rgb": np.zeros((16, 20, 3), dtype=np.uint8)},
        },
        "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
    }


class RoboTwinClosedLoopNativeTest(unittest.TestCase):
    def test_observation_contract_does_not_replace_success(self):
        task = _PlaceTask()
        original = task.check_success
        spec = pair_spec_from_source_task("place_a2b_left")
        install_pgc_observation_contract(task, spec)
        self.assertEqual(task.check_success(), original())
        self.assertFalse(hasattr(task, "_pgc_native_check_success"))
        self.assertTrue(callable(task.pgc_eraf_snapshot))

    def test_capture_writes_two_frame_base_segment_without_full_goal(self):
        task = install_pgc_observation_contract(
            _PlaceTask(), pair_spec_from_source_task("place_a2b_left")
        )
        first = capture_frame(task, _observation(), np.zeros(14, dtype=np.float32))
        second = capture_frame(task, _observation(), np.ones(14, dtype=np.float32))
        metadata = {
            "pair_id": "place_a2b_left_to_right",
            "source_task": "place_a2b_left",
            "counterfactual_task": "place_a2b_right",
            "task_config": "demo_clean",
            "scene_seed": 5_500_000,
            "episode_index": 0,
            "source_instruction": "Place object A to the left of object B.",
            "counterfactual_instruction": "Place object A to the right of object B.",
            "policy_instruction": "Place object A to the left of object B.",
            "condition": "correct",
            "instruction_goal": "source",
            "checkpoint": "/weights/base.pt",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            record_path = write_capture_segment(
                capture_root=Path(temp_dir),
                metadata=metadata,
                replan_index=0,
                online_stage="initial_search",
                frames=(first, second),
            )
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["frame_count"], 2)
            self.assertEqual(record["artifact_role"], "closed_loop_native")
            self.assertEqual(record["state_distribution"], CAPTURE_STATE_DISTRIBUTION)
            self.assertFalse(record["full_goal_verified"])
            self.assertNotIn("full_goal", record["allowed_training_stages"])
            with np.load(record_path.parent / record["capture_file"]) as payload:
                self.assertEqual(payload["action"].shape, (2, 14))
                self.assertEqual(payload["head_actor_ids"].shape, (2, 16, 20))
            replay_path = write_capture_segment(
                capture_root=Path(temp_dir),
                metadata=metadata,
                replan_index=0,
                online_stage="initial_search",
                frames=(first, second),
            )
            self.assertEqual(replay_path, record_path)

    def test_stage_classifier_covers_holding_and_release(self):
        task = install_pgc_observation_contract(
            _PlaceTask(), pair_spec_from_source_task("place_a2b_left")
        )
        snapshot = task.pgc_eraf_snapshot()
        self.assertEqual(
            classify_online_stage(
                snapshot,
                replan_index=1,
                left_gripper_open=False,
                right_gripper_open=True,
                previous_stage="initial_search",
            ),
            "holding",
        )
        self.assertEqual(
            classify_online_stage(
                snapshot,
                replan_index=2,
                left_gripper_open=True,
                right_gripper_open=True,
                previous_stage="holding",
            ),
            "released_unfinished",
        )

    def test_language_specific_pair_ids_do_not_alias(self):
        base = {
            "pair_id": "place_a2b_left_to_right",
            "source_instruction": "Place A left of B.",
            "counterfactual_instruction": "Place A right of B.",
        }
        alternate = dict(base, source_instruction="Move A to B's left.")
        self.assertNotEqual(_training_pair_id(base), _training_pair_id(alternate))


if __name__ == "__main__":
    unittest.main()
