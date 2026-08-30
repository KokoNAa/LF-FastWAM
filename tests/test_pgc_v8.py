import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fastwam.datasets.lerobot.robot_video_dataset import (
    build_pgc_v8_sample_indices,
)
from fastwam.datasets.pgc_libero import (
    PGC_CLOSED_LOOP_CORRECTIVE_FORMAT,
    PGC_CLOSED_LOOP_CORRECTIVE_FORMAT_V2,
    load_pgc_closed_loop_corrective_index,
    state_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class PGCV8DataContractTest(unittest.TestCase):
    def _write_dataset(self, root: Path, *, verified: bool = True) -> Path:
        dataset = root / "v8"
        meta = dataset / "meta"
        (meta / "pgc_v8_closed_loop").mkdir(parents=True)
        state_dir = meta / "pgc_initial_states"
        state_dir.mkdir(parents=True)
        state = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        np.save(state_dir / "v8_episode_000000.npy", state, allow_pickle=False)
        initial_digest = state_sha256(state)
        (meta / "pgc_provenance.json").write_text(
            json.dumps(
                {
                    "format": "pgc_counterfactual_actions_v1",
                    "state_aligned": True,
                    "successful_episode_count": 1,
                    "pairs": [
                        {
                            "pair_id": "object_00_to_01",
                            "source_suite": "libero_object",
                            "source_task_id": 0,
                            "source_instruction": "pick up alphabet soup",
                            "counterfactual_instruction": "pick up cream cheese",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (meta / "pgc_episodes.jsonl").write_text(
            json.dumps(
                {
                    "episode_index": 0,
                    "pair_id": "object_00_to_01",
                    "capture_id": "capture_000",
                    "capture_state_sha256": initial_digest,
                    "recorded_action_count": 24,
                    "target_lift_verified": verified,
                    "reference_boundary_event": "grasp_contact",
                    "source_initial_state_catalog": (
                        "meta/pgc_initial_states/v8_episode_000000.npy"
                    ),
                    "initial_state_sha256": initial_digest,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (meta / "pgc_v8_closed_loop/index.json").write_text(
            json.dumps(
                {
                    "format": PGC_CLOSED_LOOP_CORRECTIVE_FORMAT,
                    "acquisition_only": True,
                    "episode_count": 1,
                    "episodes": [
                        {
                            "episode_index": 0,
                            "pair_id": "object_00_to_01",
                            "capture_id": "capture_000",
                            "capture_state_sha256": initial_digest,
                            "recorded_action_count": 24,
                            "target_lift_verified": verified,
                            "reference_boundary_event": "grasp_contact",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return dataset

    def test_v8_index_requires_replay_verified_target_lift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = load_pgc_closed_loop_corrective_index(
                self._write_dataset(Path(tmpdir))
            )
            self.assertEqual(set(index), {0})
            self.assertEqual(index[0]["recorded_action_count"], 24)
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "not target-lift verified"):
                load_pgc_closed_loop_corrective_index(
                    self._write_dataset(Path(tmpdir), verified=False)
                )

    def test_v8_sampler_balances_native_and_weighted_corrective_data(self):
        indices = build_pgc_v8_sample_indices(
            native_frame_count=3,
            offline_counterfactual_frame_count=5,
            total_frame_count=6,
            closed_loop_oversample_factor=4,
            balance_native_counterfactual=True,
        )
        self.assertEqual(len(indices), 12)
        self.assertEqual(sum(index < 3 for index in indices), 6)
        self.assertEqual(sum(index == 5 for index in indices), 4)
        self.assertEqual(sum(3 <= index < 5 for index in indices), 2)

    def test_v2_accepts_replay_verified_counterfactual_goal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self._write_dataset(Path(tmpdir), verified=False)
            audit_path = dataset / "meta/pgc_episodes.jsonl"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit.update(
                {
                    "corrective_verified": True,
                    "verification_kind": "counterfactual_goal",
                    "verification_step": 17,
                    "counterfactual_goal_verified": True,
                    "reference_boundary_event": "counterfactual_goal",
                }
            )
            audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")
            index_path = dataset / "meta/pgc_v8_closed_loop/index.json"
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            payload.update(
                {
                    "format": PGC_CLOSED_LOOP_CORRECTIVE_FORMAT_V2,
                    "acquisition_only": False,
                }
            )
            payload["episodes"][0].update(
                {
                    "corrective_verified": True,
                    "verification_kind": "counterfactual_goal",
                    "verification_step": 17,
                    "counterfactual_goal_verified": True,
                    "reference_boundary_event": "counterfactual_goal",
                }
            )
            index_path.write_text(json.dumps(payload), encoding="utf-8")
            index = load_pgc_closed_loop_corrective_index(dataset)
            self.assertEqual(index[0]["verification_kind"], "counterfactual_goal")

            payload["episodes"][0]["counterfactual_goal_verified"] = False
            index_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not replay-verified"):
                load_pgc_closed_loop_corrective_index(dataset)

    def test_v8_pipeline_scripts_preserve_v5_and_verify_rollouts(self):
        train = (REPO_ROOT / "scripts/train_pgc_v8_libero_suite.sh").read_text(
            encoding="utf-8"
        )
        builder = (REPO_ROOT / "scripts/build_pgc_v8_corrective_data.py").read_text(
            encoding="utf-8"
        )
        evaluation = (REPO_ROOT / "experiments/libero/eval_libero_single.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("PGC_WARM_START_V5=true", train)
        self.assertIn("PGC_CLOSED_LOOP_TRAIN_PROPOSAL_ONLY=true", train)
        self.assertIn("target_lift_verified", builder)
        self.assertIn("target_lift_fallback", builder)
        self.assertIn("bootstrap-incomplete", builder)
        self.assertIn("_replay_for_corrective_success", builder)
        self.assertIn("stop_on_success=True", builder)
        self.assertIn("counterfactual_goal_verified", builder)
        self.assertIn("_named_site_position", builder)
        self.assertIn("_goal_satisfied(env, done=done)", builder)
        self.assertIn("rejected_references", builder)
        self.assertIn("Trying V8 capture", builder)
        self.assertIn("closed_loop_capture_dir", evaluation)
        self.assertIn("_capture_libero_sim_state", evaluation)


if __name__ == "__main__":
    unittest.main()
