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
        self.assertIn("model.policy_guard.action_chunk_residual_max_abs=", source)
        self.assertIn("model.policy_guard.rollout_num_inference_steps=", source)

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

    def test_v4_launcher_selects_rollout_aligned_training(self):
        source = (REPO_ROOT / "scripts/train_pgc_v4_libero_suite.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("export PGC_VERSION=4", source)
        self.assertIn("PGC_ACTION_CHUNK_RESIDUAL_MAX_ABS", source)
        self.assertIn("PGC_ROLLOUT_INFERENCE_STEPS", source)
        self.assertIn("PGC_ADVANTAGE_TEMPERATURE", source)
        self.assertIn("PGC_CANDIDATE_MAX_DELTA_RMS", source)

    def test_v5_launcher_selects_paired_language_prefix_training(self):
        source = (REPO_ROOT / "scripts/train_pgc_v5_libero_suite.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("export PGC_VERSION=5", source)
        self.assertIn("PGC_EXECUTION_PREFIX_STEPS", source)
        self.assertIn("PGC_SAME_STATE_SOURCE_ZERO_WEIGHT", source)
        self.assertIn("PGC_GOAL_SEPARATION_WEIGHT", source)
        self.assertIn("PGC_VERIFIER_WRONG_LANGUAGE_WEIGHT", source)

    def test_v6_launcher_selects_visual_target_binding_and_v5_warm_start(self):
        source = (REPO_ROOT / "scripts/train_pgc_v6_libero_suite.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("export PGC_VERSION=6", source)
        self.assertIn("PGC_WARM_START_V5", source)
        self.assertIn("PGC_INIT_CHECKPOINT", source)
        self.assertIn("PGC_TARGET_BINDING_INTERACTION_WEIGHT", source)
        self.assertIn("PGC_TARGET_BINDING_PROTOTYPE_WEIGHT", source)
        self.assertIn("PGC_TARGET_BINDING_HARD_NEGATIVE_WEIGHT", source)
        self.assertIn("PGC_TARGET_BINDING_ACTION_START_STEP", source)
        self.assertIn('PGC_VERIFIER_START_STEP:-1500', source)

    def test_common_launcher_forwards_v6_target_binding_contract(self):
        source = (REPO_ROOT / "scripts/train_pgc_libero.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('2|3|4|5|6|7', source)
        self.assertIn("PGC v5 architecture warm start", source)
        self.assertIn(
            "model.policy_guard.target_binding_interaction_weight=", source
        )
        self.assertIn(
            "model.policy_guard.target_binding_prototype_weight=", source
        )
        self.assertIn(
            "model.policy_guard.target_binding_hard_negative_weight=", source
        )
        self.assertIn(
            "model.policy_guard.target_binding_action_start_step=", source
        )
        self.assertIn(
            "model.policy_guard.target_binding_action_ramp_steps=", source
        )
        self.assertIn("target_prototype_bank_persisted", source)

    def test_v7_launcher_selects_explicit_mask_object_tokens(self):
        source = (REPO_ROOT / "scripts/train_pgc_v7_libero_suite.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("export PGC_VERSION=7", source)
        self.assertIn("PGC_WARM_START_V5", source)
        self.assertIn("PGC_TARGET_BINDING_NUM_OBJECT_TOKENS", source)
        self.assertIn("PGC_TARGET_BINDING_CAMERA_COUNT", source)
        self.assertIn("PGC_TARGET_MASK_WEIGHT", source)
        self.assertIn("PGC_CROSS_OBJECT_WEIGHT", source)

    def test_common_launcher_requires_and_forwards_v7_mask_sidecars(self):
        source = (REPO_ROOT / "scripts/train_pgc_libero.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("load_pgc_target_mask_index", source)
        self.assertIn("pgc_target_mask_supervision_required=", source)
        self.assertIn("model.policy_guard.target_binding_num_object_tokens=", source)
        self.assertIn("model.policy_guard.target_mask_weight=", source)
        self.assertIn("model.policy_guard.cross_object_margin=", source)

    def test_evaluation_enforces_v4_rollout_step_alignment(self):
        source = (REPO_ROOT / "scripts/eval_pgc_libero.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("rollout/evaluation step alignment", source)
        self.assertIn("raw_fp32_pairwise_advantage", source)

    def test_v912_evaluation_restores_rebinding_checkpoint_contract(self):
        source = (REPO_ROOT / "scripts/eval_pgc_libero.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "PGC_V9_GROUNDING_OBJECTIVE_VERSION >= 13", source
        )
        self.assertIn("closed_loop_rebinding_hidden_dim=256", source)
        self.assertIn("closed_loop_query_residual_max_abs=1.0", source)
        self.assertIn("closed_loop_state_residual_max_abs=2.0", source)
        self.assertIn("phase_rebinding_energy_weight=0.01", source)


if __name__ == "__main__":
    unittest.main()
