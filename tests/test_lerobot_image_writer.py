import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset


class LeRobotImageWriterTest(unittest.TestCase):
    def test_synchronous_writer_accepts_hwc_and_chw_frames(self):
        dataset = LeRobotDataset.__new__(LeRobotDataset)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            hwc_path = root / "hwc.jpeg"
            chw_path = root / "nested/chw.jpeg"
            dataset._save_image(np.full((8, 12, 3), 127, dtype=np.uint8), hwc_path)
            dataset._save_image(np.ones((3, 8, 12), dtype=np.float32), chw_path)

            with Image.open(hwc_path) as image:
                self.assertEqual(image.size, (12, 8))
                self.assertEqual(image.mode, "RGB")
            with Image.open(chw_path) as image:
                self.assertEqual(image.size, (12, 8))
                self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
