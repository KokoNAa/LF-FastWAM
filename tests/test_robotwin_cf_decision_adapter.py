import itertools
import unittest
from collections import Counter

from scripts.train_robotwin_cf_decision_adapter import FOCUS, pair_stream, small_validation_set


class DecisionSamplingTest(unittest.TestCase):
    def test_training_keeps_all_scenes_and_excludes_replay_holdout(self):
        rows = [{"id": f"{p}/{d}/{s}", "pair_id": p, "task_config": d,
                 "replay_split": "train" if s < 3 else "replay_holdout"}
                for p in (*FOCUS, "delayed") for d in ("clean", "random") for s in range(4)]
        # Two focus pairs get two copies, delayed pair one: 5*2*3 draws.
        drawn = list(itertools.islice(pair_stream(rows, 42, 2), 30))
        counts = Counter(r["id"] for r in drawn)
        self.assertEqual(len(counts), 18)
        for row in rows:
            self.assertEqual(counts[row["id"]], 0 if row["replay_split"] != "train"
                             else 2 if row["pair_id"] in FOCUS else 1)
        self.assertEqual(len(small_validation_set(rows)), 6)
        self.assertEqual(drawn, list(itertools.islice(pair_stream(rows, 42, 2), 30)))


if __name__ == "__main__":
    unittest.main()
