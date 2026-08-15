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
        self.assertIn("++data.train.pretrained_norm_stats=", source)
        self.assertIn("PGC_BALANCE_NATIVE_COUNTERFACTUAL:-true", source)
        self.assertIn(
            "data.train.pgc_balance_native_counterfactual=", source
        )
        self.assertIn('model.policy_guard.version=${PGC_VERSION}', source)
        self.assertIn('PGC_VERSION="${PGC_VERSION:-2}"', source)
        self.assertIn("model.policy_guard.velocity_residual_max_abs=", source)

    def test_common_launcher_supports_explicit_weight_only_continuation(self):
        source = (REPO_ROOT / "scripts/train_pgc_libero.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("PGC_INIT_CHECKPOINT", source)
        self.assertIn("PGC_CONTINUE_FROM_STEP", source)
        self.assertIn("fastwam_policy_guard_v2", source)
        self.assertIn('expected_format = f"fastwam_policy_guard_v{expected_version}"', source)
        self.assertIn("PGC continuation step mismatch", source)
        self.assertIn(
            '"weight_only_start_step=${WEIGHT_ONLY_START_STEP}"',
            source,
        )

    def test_data_builder_can_be_restricted_to_one_suite(self):
        source = (REPO_ROOT / "scripts/build_pgc_libero_datasets.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("PGC_BUILD_SUITE:-all", source)
        self.assertIn('suites=("${BUILD_SUITE}")', source)
        self.assertIn('--source-suite "${suite}"', source)
        self.assertIn(
            'pgc_counterfactual_datasets.${BUILD_SUITE}.txt', source
        )

    def test_evaluation_supports_an_explicit_policy_horizon(self):
        source = (REPO_ROOT / "scripts/eval_pgc_libero.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("PGC_MAX_POLICY_STEPS", source)
        self.assertIn("EVALUATION.max_steps=${MAX_POLICY_STEPS}", source)

    def test_v3_launcher_selects_residual_only_training(self):
        source = (REPO_ROOT / "scripts/train_pgc_v3_libero_suite.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("export PGC_VERSION=3", source)
        self.assertIn("PGC_LEARNING_RATE", source)
        self.assertIn("PGC_VELOCITY_RESIDUAL_MAX_ABS", source)
        self.assertIn("PGC_VERIFIER_START_STEP", source)


if __name__ == "__main__":
    unittest.main()
