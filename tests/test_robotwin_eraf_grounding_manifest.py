import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.robotwin.pgc_data import ROBOTWIN_ERAF_PAIR_IDS
from scripts.build_pgc_robotwin_eraf_grounding_manifest import (
    OUTPUT_FORMAT,
    build_grounding_manifest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _make_formal_tree(root: Path):
    indices = {}
    for task_config, episodes in (("demo_clean", 3), ("demo_randomized", 2)):
        manifest_dir = root / "formal" / task_config / "lerobot"
        manifest_dir.mkdir(parents=True)
        entries = []
        for pair_id in ROBOTWIN_ERAF_PAIR_IDS:
            for dataset_kind in ("native", "counterfactual"):
                dataset = manifest_dir / pair_id / dataset_kind
                sidecar = (
                    root / "formal" / task_config / "eraf" / pair_id / dataset_kind
                )
                (dataset / "meta").mkdir(parents=True)
                sidecar.mkdir(parents=True)
                (dataset / "meta" / "pgc_provenance.json").write_text(
                    json.dumps(
                        {
                            "artifact_role": "eraf_grounding_supervision",
                            "allowed_training_stages": ["grounding"],
                            "full_goal_verified": False,
                            "task_config": task_config,
                        }
                    ),
                    encoding="utf-8",
                )
                (sidecar / "index.json").write_text("{}", encoding="utf-8")
                records = {
                    episode: {
                        "initial_state_sha256": _digest(
                            f"{task_config}:{pair_id}:{episode}"
                        ),
                        "pair_id": pair_id,
                    }
                    for episode in range(episodes)
                }
                indices[str(sidecar.resolve())] = {
                    "dataset": str(dataset.resolve()),
                    "dataset_kind": dataset_kind,
                    "camera_count": 3,
                    "action_dim": 14,
                    "episode_count": episodes,
                    "artifact_role": "eraf_grounding_supervision",
                    "allowed_training_stages": ["grounding"],
                    "full_goal_verified": False,
                    "episodes_by_index": records,
                }
                entries.append(
                    {
                        "pair_id": pair_id,
                        "dataset_kind": dataset_kind,
                        "episodes": episodes,
                        "dataset": str(dataset),
                        "sidecar": str(sidecar),
                        "artifact_role": "eraf_grounding_supervision",
                        "full_goal_verified": False,
                        "valid": True,
                    }
                )
        (manifest_dir / "pgc_robotwin_eraf_prepared.json").write_text(
            json.dumps(
                {
                    "format": "pgc_robotwin_eraf_prepared_matrix_v1",
                    "complete": True,
                    "artifact_role": "eraf_grounding_supervision",
                    "allowed_training_stages": ["grounding"],
                    "full_goal_verified": False,
                    "pairs": list(ROBOTWIN_ERAF_PAIR_IDS),
                    "datasets": entries,
                }
            ),
            encoding="utf-8",
        )
    return indices


class RoboTwinERAFGroundingManifestTest(unittest.TestCase):
    def test_combines_clean_and_randomized_without_full_goal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            indices = _make_formal_tree(root)
            with mock.patch(
                "scripts.build_pgc_robotwin_eraf_grounding_manifest."
                "load_pgc_entity_relation_index",
                side_effect=lambda path: indices[str(Path(path).resolve())],
            ):
                payload = build_grounding_manifest(work_root=root)
            self.assertEqual(payload["format"], OUTPUT_FORMAT)
            self.assertEqual(payload["dataset_count"], 20)
            self.assertEqual(payload["total_successful_trajectories"], 50)
            self.assertFalse(payload["full_goal_verified"])
            self.assertEqual(payload["allowed_training_stages"], ["grounding"])
            for counts in payload["pair_episode_counts"].values():
                self.assertEqual(counts, {"native": 5, "counterfactual": 5})
            task_configs = [entry["task_config"] for entry in payload["datasets"]]
            self.assertEqual(task_configs[:2], ["demo_clean", "demo_randomized"])

    def test_rejects_full_goal_index_leakage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            indices = _make_formal_tree(root)
            first_dataset = (
                root
                / "formal"
                / "demo_clean"
                / "lerobot"
                / ROBOTWIN_ERAF_PAIR_IDS[0]
                / "native"
            )
            leaked = first_dataset / "meta" / "pgc_robotwin_full_goal" / "index.json"
            leaked.parent.mkdir(parents=True)
            leaked.write_text("{}", encoding="utf-8")
            with mock.patch(
                "scripts.build_pgc_robotwin_eraf_grounding_manifest."
                "load_pgc_entity_relation_index",
                side_effect=lambda path: indices[str(Path(path).resolve())],
            ):
                with self.assertRaisesRegex(ValueError, "Full-goal index leaked"):
                    build_grounding_manifest(work_root=root)


if __name__ == "__main__":
    unittest.main()
