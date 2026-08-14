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


if __name__ == "__main__":
    unittest.main()
