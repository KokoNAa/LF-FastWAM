import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PolicyGuardTrainingScopeTest(unittest.TestCase):
    def test_joint_training_requires_explicit_unlock(self):
        source = (REPO_ROOT / "scripts/train_pgc_libero.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('PGC_ALLOW_JOINT_TRAINING:-false', source)
        self.assertIn('ALLOW_JOINT_TRAINING}" != "true"', source)
        self.assertIn("Joint four-suite training is locked", source)

    def test_isolated_launcher_accepts_only_official_suites(self):
        source = (REPO_ROOT / "scripts/train_pgc_libero_suite.sh").read_text(
            encoding="utf-8"
        )
        for suite in (
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
        ):
            self.assertIn(suite, source)
        self.assertIn('export PGC_TRAIN_SUITE="${SUITE}"', source)

    def test_common_launcher_enforces_provenance_scope(self):
        source = (REPO_ROOT / "scripts/train_pgc_libero.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('provenance.get("source_suites", [])', source)
        self.assertIn("outside training scope", source)
        self.assertIn("Counterfactual scope mismatch", source)


if __name__ == "__main__":
    unittest.main()
