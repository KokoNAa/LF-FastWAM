import json
import io
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from experiments.robotwin.pgc_data import (
    array_sha256,
    direction_from_task,
    opposite_direction,
    validate_action_array,
    validate_pair_record,
)
from scripts.inspect_pgc_checkpoint import hydra_overrides, validate_payload
from scripts.build_pgc_robotwin_entity_relations import build_sidecar
from scripts.convert_pgc_robotwin_to_lerobot import _decode_rgb, robotwin_lerobot_features
from scripts.prepare_pgc_robotwin_datasets import dataset_specs
from fastwam.datasets.pgc_libero import load_pgc_entity_relation_index


def _checkpoint_payload():
    metadata = {
        "architecture": "pgc_fastwam",
        "policy_guard_version": 9,
        "eraf_grounding_objective_version": 26,
        "eraf_single_path": True,
        "gate_mode": "eraf_only",
        "policy_protection": "single_eraf_path_no_candidate_gate",
        "eraf_post_action_residual_active": False,
        "counterfactual_action_interface": "single_eraf_conditioned_action_denoising_path",
        "rollout_num_inference_steps": 10,
        "action_output_dim": 14,
        "proprio_dim": 14,
        "eraf_camera_count": 3,
        "eraf_visual_aspect_ratio": 5.0 / 6.0,
        "eraf_camera_layout": "robotwin_mosaic",
        "eraf_training_stage": "action",
        "eraf_entity_only": False,
        "eraf_use_anchors": True,
    }
    return {
        "format": "fastwam_policy_guard_v9",
        "architecture_metadata": metadata,
        "eraf_shared_expert_lora": {"video.q.lora_A": object()},
        "eraf_shared_expert_lora_config": {
            "experts": ["video", "action"]
        },
    }


class RoboTwinPGCCheckpointTest(unittest.TestCase):
    def test_robotwin_v926_checkpoint_contract_and_overrides(self):
        metadata = validate_payload(
            _checkpoint_payload(), target="robotwin", inference_steps=10
        )
        overrides = hydra_overrides(metadata)
        self.assertIn("model.policy_guard.version=9", overrides)
        self.assertIn(
            "model.policy_guard.entity_relation_grounding.camera_layout=robotwin_mosaic",
            overrides,
        )
        self.assertIn(
            "model.policy_guard.entity_relation_grounding.camera_count=3",
            overrides,
        )

    def test_rejects_libero_geometry_or_step_drift(self):
        payload = _checkpoint_payload()
        payload["architecture_metadata"]["eraf_camera_count"] = 2
        with self.assertRaisesRegex(ValueError, "camera_count"):
            validate_payload(payload, target="robotwin", inference_steps=10)
        payload = _checkpoint_payload()
        with self.assertRaisesRegex(ValueError, "denoising-step"):
            validate_payload(payload, target="robotwin", inference_steps=5)

    def test_rejects_action_only_lora(self):
        payload = _checkpoint_payload()
        payload["eraf_shared_expert_lora_config"]["experts"] = ["action"]
        with self.assertRaisesRegex(ValueError, "Video and Action"):
            validate_payload(payload, target="robotwin", inference_steps=10)


class RoboTwinPGCDataContractTest(unittest.TestCase):
    def test_prepared_matrix_contains_both_kinds_for_both_directions(self):
        specs = dataset_specs(
            raw_root=Path("/raw"),
            dataset_root=Path("/dataset"),
            sidecar_root=Path("/sidecar"),
        )
        self.assertEqual(len(specs), 4)
        self.assertEqual(
            {(spec.pair_id, spec.dataset_kind) for spec in specs},
            {
                ("place_a2b_left_to_place_a2b_right", "native"),
                ("place_a2b_left_to_place_a2b_right", "counterfactual"),
                ("place_a2b_right_to_place_a2b_left", "native"),
                ("place_a2b_right_to_place_a2b_left", "counterfactual"),
            },
        )

    def test_lerobot_conversion_schema_keeps_three_rgb_and_qpos14(self):
        features = robotwin_lerobot_features(480, 640)
        self.assertEqual(features["observation.images.cam_high"]["shape"], (3, 480, 640))
        self.assertEqual(features["observation.state"]["shape"], (14,))
        self.assertEqual(features["action"]["shape"], (14,))
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG")
        decoded = _decode_rgb(np.bytes_(buffer.getvalue() + b"\0\0"))
        self.assertEqual(decoded.shape, image.shape)

    def test_direction_pair_and_action_contract(self):
        self.assertEqual(direction_from_task("place_a2b_left"), "left")
        self.assertEqual(opposite_direction("left"), "right")
        actions = validate_action_array(np.zeros((32, 14), dtype=np.float32))
        state = np.arange(28, dtype=np.float32)
        record = validate_pair_record(
            {
                "pair_id": "left_to_right",
                "source_task": "place_a2b_left",
                "counterfactual_task": "place_a2b_right",
                "source_direction": "left",
                "counterfactual_direction": "right",
                "scene_seed": 42,
                "initial_state_sha256": array_sha256(state),
                "action_sha256": array_sha256(actions),
                "action_count": len(actions),
            }
        )
        self.assertEqual(record["counterfactual_direction"], "right")

    def test_rejects_independent_or_mislabeled_pair(self):
        digest = "0" * 64
        with self.assertRaisesRegex(ValueError, "reverse"):
            validate_pair_record(
                {
                    "pair_id": "bad",
                    "source_task": "place_a2b_left",
                    "counterfactual_task": "place_a2b_left",
                    "source_direction": "left",
                    "counterfactual_direction": "left",
                    "scene_seed": 42,
                    "initial_state_sha256": digest,
                    "action_sha256": digest,
                    "action_count": 1,
                }
            )

    def test_builds_three_camera_eraf_sidecar_from_raw_actor_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_root = root / "raw"
            (raw_root / "meta").mkdir(parents=True)
            (raw_root / "data").mkdir()
            dataset_root = root / "lerobot"
            dataset_root.mkdir()
            hdf5_path = raw_root / "data" / "episode0.hdf5"
            frames, height, width = 3, 12, 16
            subject_ids = np.full(frames, 11, dtype=np.uint32)
            reference_ids = np.full(frames, 22, dtype=np.uint32)
            labels = np.zeros((frames, height, width), dtype=np.uint32)
            labels[:, 2:5, 2:5] = 11
            labels[:, 6:9, 10:13] = 22
            subject = np.asarray(
                [[0.0, -0.1, 0.75], [0.0, -0.1, 0.85], [0.13, -0.1, 0.75]],
                dtype=np.float32,
            )
            reference = np.asarray(
                [[0.0, -0.1, 0.75]] * frames, dtype=np.float32
            )
            with h5py.File(hdf5_path, "w") as handle:
                for camera in ("head_camera", "left_camera", "right_camera"):
                    handle.create_dataset(
                        f"observation/{camera}/actor_segmentation_ids", data=labels
                    )
                handle.create_dataset("pgc_entity_state/subject_position", data=subject)
                handle.create_dataset("pgc_entity_state/reference_position", data=reference)
                handle.create_dataset("pgc_entity_state/subject_actor_id", data=subject_ids)
                handle.create_dataset("pgc_entity_state/reference_actor_id", data=reference_ids)
            digest = "0" * 64
            audit = {
                "episode_index": 0,
                "pair_id": "place_a2b_left_to_place_a2b_right",
                "source_task": "place_a2b_left",
                "counterfactual_task": "place_a2b_right",
                "raw_hdf5": "data/episode0.hdf5",
                "initial_state_sha256": digest,
                "action_sha256": digest,
            }
            (raw_root / "meta" / "pgc_episodes.jsonl").write_text(
                json.dumps(audit) + "\n", encoding="utf-8"
            )
            (raw_root / "meta" / "pgc_provenance.json").write_text(
                json.dumps({"dataset_kind": "counterfactual"}), encoding="utf-8"
            )
            sidecar_root = root / "sidecar"
            build_sidecar(
                raw_root=raw_root,
                dataset_root=dataset_root,
                output_root=sidecar_root,
            )
            index = load_pgc_entity_relation_index(sidecar_root)
            self.assertEqual(index["camera_count"], 3)
            self.assertEqual(index["action_dim"], 14)
            with np.load(index["episodes_by_index"][0]["path"]) as payload:
                self.assertEqual(payload["target_subject_masks"].shape, (3, 4, 24, 20))
                self.assertEqual(
                    payload["target_subject_view_visible"].shape, (3, 4, 3)
                )


if __name__ == "__main__":
    unittest.main()
