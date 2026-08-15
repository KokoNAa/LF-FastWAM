import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/validate_pgc_counterfactual_datasets.py"
)
SPEC = importlib.util.spec_from_file_location("pgc_data_validator", SCRIPT_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)


class PolicyGuardDataContractTest(unittest.TestCase):
    def _write_dataset(self, root: Path, *, state_aligned: bool = True) -> Path:
        dataset = root / "cf_object"
        meta = dataset / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text("{}\n", encoding="utf-8")
        (meta / "tasks.jsonl").write_text(
            json.dumps({"task": "pick up the milk and place it in the basket"})
            + "\n",
            encoding="utf-8",
        )
        (meta / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "tasks": [0]}) + "\n",
            encoding="utf-8",
        )
        state_dir = meta / "pgc_initial_states"
        state_dir.mkdir()
        state = np.arange(12, dtype=np.float64)
        np.save(state_dir / "episode_000000.npy", state, allow_pickle=False)
        (meta / "pgc_episodes.jsonl").write_text(
            json.dumps(
                {
                    "episode_index": 0,
                    "pair_id": "libero_object_00_to_07",
                    "source_initial_state_index": 3,
                    "source_initial_state_catalog": (
                        "meta/pgc_initial_states/episode_000000.npy"
                    ),
                    "initial_state_sha256": VALIDATOR._state_sha256(state),
                    "initial_state_match": True,
                    "counterfactual_goal_satisfied": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        provenance = {
            "format": "pgc_counterfactual_actions_v1",
            "benchmark": "libero",
            "action_supervision": "executed_counterfactual_success_trajectory",
            "state_aligned": state_aligned,
            "successful_only": True,
            "successful_episode_count": 1,
            "state_catalog": (
                "meta/pgc_initial_states/episode_{episode_index:06d}.npy"
            ),
            "source_suites": ["libero_object"],
            "pairs": [
                {
                    "pair_id": "libero_object_00_to_07",
                    "source_suite": "libero_object",
                    "source_task_id": 0,
                    "source_instruction": (
                        "pick up the alphabet soup and place it in the basket"
                    ),
                    "counterfactual_instruction": (
                        "pick up the milk and place it in the basket"
                    ),
                    "counterfactual_goal_state": [
                        ["in", "milk_1", "basket_1_contain_region"]
                    ],
                }
            ],
        }
        (meta / "pgc_provenance.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )
        return dataset

    def test_accepts_direct_success_trajectory_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = VALIDATOR.validate_dataset(
                self._write_dataset(Path(tmpdir))
            )
        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["pairs"], 1)

    def test_rejects_non_state_aligned_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self._write_dataset(
                Path(tmpdir), state_aligned=False
            )
            with self.assertRaisesRegex(ValueError, "state_aligned"):
                VALIDATOR.validate_dataset(dataset)

    def test_rejects_missing_episode_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self._write_dataset(Path(tmpdir))
            (dataset / "meta/pgc_episodes.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "pgc_episodes"):
                VALIDATOR.validate_dataset(dataset)

    def test_rejects_pair_without_successful_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self._write_dataset(Path(tmpdir))
            provenance_path = dataset / "meta/pgc_provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["pairs"].append(
                {
                    "pair_id": "libero_object_01_to_07",
                    "source_suite": "libero_object",
                    "source_task_id": 1,
                    "source_instruction": "pick up the cream cheese",
                    "counterfactual_instruction": "pick up the milk",
                    "counterfactual_goal_state": [
                        ["in", "milk_1", "basket_1_contain_region"]
                    ],
                }
            )
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no successful"):
                VALIDATOR.validate_dataset(dataset)

    def test_rejects_tampered_initial_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self._write_dataset(Path(tmpdir))
            state_path = (
                dataset / "meta/pgc_initial_states/episode_000000.npy"
            )
            np.save(state_path, np.ones(12), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                VALIDATOR.validate_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
