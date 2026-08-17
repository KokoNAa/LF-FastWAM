import ast
import unittest
from pathlib import Path


class RobotVideoDatasetSourceTest(unittest.TestCase):
    def test_text_cache_hash_dependency_is_imported(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "src/fastwam/datasets/lerobot/robot_video_dataset.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_modules = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        hashlib_calls = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "hashlib"
        }

        self.assertIn("hashlib", imported_modules)
        self.assertIn("sha256", hashlib_calls)

    def test_text_precompute_scans_pgc_counterfactual_directories(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "scripts/precompute_text_embeds.py"
        )
        source = source_path.read_text(encoding="utf-8")
        self.assertIn("pgc_counterfactual_dataset_dirs", source)

    def test_v7_dataset_loads_training_only_mask_sidecars(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "src/fastwam/datasets/lerobot/robot_video_dataset.py"
        )
        source = source_path.read_text(encoding="utf-8")
        self.assertIn("pgc_target_mask_supervision_required", source)
        self.assertIn("load_pgc_target_mask_index", source)
        self.assertIn('"pgc_target_object_mask"', source)
        self.assertIn('"pgc_source_object_mask"', source)
        self.assertIn('"pgc_aux_object_mask"', source)
        self.assertIn('"pgc_aux_context"', source)
        processor_path = (
            Path(__file__).resolve().parents[1]
            / "src/fastwam/datasets/lerobot/processors/fastwam_processor.py"
        )
        processor_source = processor_path.read_text(encoding="utf-8")
        self.assertIn('sample["frame_index"]', processor_source)


if __name__ == "__main__":
    unittest.main()
