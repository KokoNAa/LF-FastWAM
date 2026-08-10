import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.libero.init_state_utils import load_libero_task_init_states


class _FakeTaskSuite:
    def __init__(self, task):
        self.task = task

    def get_task(self, task_id):
        if task_id != 0:
            raise IndexError(task_id)
        return self.task


class LiberoInitStateCompatibilityTest(unittest.TestCase):
    def test_loads_numpy_backed_pruned_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "init_files"
            suite_dir = root / "libero_spatial"
            suite_dir.mkdir(parents=True)
            path = suite_dir / "task.pruned_init"
            expected = np.arange(12, dtype=np.float32).reshape(3, 4)
            torch.save(expected, path)
            suite = _FakeTaskSuite(
                SimpleNamespace(
                    problem_folder="libero_spatial",
                    init_states_file=path.name,
                )
            )

            actual = load_libero_task_init_states(suite, 0, root)

            np.testing.assert_array_equal(actual, expected)

    def test_rejects_path_outside_trusted_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            trusted_root = temp_root / "init_files"
            trusted_root.mkdir()
            outside = temp_root / "outside.pruned_init"
            torch.save(np.zeros(1, dtype=np.float32), outside)
            suite = _FakeTaskSuite(
                SimpleNamespace(
                    problem_folder="..",
                    init_states_file=outside.name,
                )
            )

            with self.assertRaisesRegex(ValueError, "escapes the configured root"):
                load_libero_task_init_states(suite, 0, trusted_root)


if __name__ == "__main__":
    unittest.main()
