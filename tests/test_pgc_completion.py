import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fastwam.datasets.pgc_libero import (
    PGC_COMPLETION_PHASE_FORMAT,
    PGC_COMPLETION_PHASE_INDEX,
    PGC_DATA_FORMAT,
    detect_pgc_completion_phase,
    load_pgc_completion_phase_index,
)


class PGCCompletionPhaseTest(unittest.TestCase):
    def test_detects_grasp_and_release_commands(self):
        actions = np.zeros((8, 7), dtype=np.float32)
        actions[:3, -1] = -1.0
        actions[3:7, -1] = 1.0
        actions[7, -1] = -1.0
        phase = detect_pgc_completion_phase(actions)
        self.assertEqual(phase["grasp_close_step"], 3)
        self.assertEqual(phase["release_open_step"], 7)

    def test_allows_success_truncation_before_release(self):
        actions = np.zeros((5, 7), dtype=np.float32)
        actions[:2, -1] = -1.0
        actions[2:, -1] = 1.0
        phase = detect_pgc_completion_phase(actions)
        self.assertEqual(phase["grasp_close_step"], 2)
        self.assertIsNone(phase["release_open_step"])

    def test_loads_sidecar_against_episode_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            meta = root / "meta"
            meta.mkdir()
            provenance = {
                "format": PGC_DATA_FORMAT,
                "state_aligned": True,
                "successful_episode_count": 1,
                "pairs": [
                    {
                        "pair_id": "object_00_to_01",
                        "source_instruction": "pick source",
                        "counterfactual_instruction": "pick target",
                        "source_suite": "libero_object",
                        "source_task_id": 0,
                    }
                ],
            }
            (meta / "pgc_provenance.json").write_text(
                json.dumps(provenance), encoding="utf-8"
            )
            (meta / "pgc_episodes.jsonl").write_text(
                json.dumps(
                    {
                        "episode_index": 0,
                        "pair_id": "object_00_to_01",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sidecar = {
                "format": PGC_COMPLETION_PHASE_FORMAT,
                "episodes": [
                    {
                        "episode_index": 0,
                        "pair_id": "object_00_to_01",
                        "action_count": 12,
                        "grasp_close_step": 4,
                        "release_open_step": 10,
                    }
                ],
            }
            (root / PGC_COMPLETION_PHASE_INDEX).write_text(
                json.dumps(sidecar), encoding="utf-8"
            )
            loaded = load_pgc_completion_phase_index(root)
        self.assertEqual(loaded[0]["grasp_close_step"], 4)
        self.assertEqual(loaded[0]["release_open_step"], 10)


if __name__ == "__main__":
    unittest.main()
