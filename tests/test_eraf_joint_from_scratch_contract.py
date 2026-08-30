import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ERAFJointTrainingContractTest(unittest.TestCase):
    def test_safe_gain_objective_version_is_initialized_before_v928_defaults(self):
        source = (
            REPO_ROOT / "src/fastwam/models/wan22/fastwam.py"
        ).read_text(encoding="utf-8")
        objective_assignment = source.index(
            "self.policy_guard_eraf_grounding_objective_version = int("
        )
        wrong_gate_default = source.index(
            "self.policy_guard_eraf_safe_gain_wrong_gate_loss_weight = float("
        )
        self.assertLess(objective_assignment, wrong_gate_default)

    def test_safe_gain_launcher_and_task_preserve_the_control_contract(self):
        launcher = (REPO_ROOT / "scripts/train_libero_eraf_safe_gain.sh").read_text(
            encoding="utf-8"
        )
        task = (
            REPO_ROOT / "configs/task/libero_eraf_safe_gain_2cam224.yaml"
        ).read_text(encoding="utf-8")
        for contract in (
            "<no_eraf_lora_checkpoint>",
            "fastwam_lora_adapter_v1",
            "task=libero_eraf_safe_gain_2cam224",
            "pgc_bidirectional_language_supervision_required=true",
            "pgc_v9_closed_loop_native_dataset_count=1",
            "ERAF_SAFE_GAIN_MAX_STEPS:-10000",
        ):
            self.assertIn(contract, launcher)
        for contract in (
            "grounding_objective_version: 28",
            "pretrained_joint_training: true",
            "safe_gain_training: true",
            "grounding_aux_weight: 0.0",
            "safe_gain_num_tokens: 4",
            "safe_gain_gate_threshold: 0.80",
            "safe_gain_wrong_gate_loss_weight: 0.5",
            "safe_gain_gate_ranking_weight: 1.0",
            "safe_gain_gate_ranking_margin: 1.0",
            "safe_gain_non_regression_weight: 2.0",
            "pgc_bidirectional_language_supervision_required: true",
            "max_steps: 10000",
            "experts: [video, action]",
        ):
            self.assertIn(contract, task)

    def test_launcher_keeps_no_eraf_data_contract_and_runs_10k(self):
        source = (
            REPO_ROOT / "scripts/train_libero_eraf_joint_from_scratch.sh"
        ).read_text(encoding="utf-8")
        for contract in (
            'MAX_STEPS="${ERAF_JOINT_MAX_STEPS:-10000}"',
            "task=libero_eraf_joint_2cam224",
            "offline_native:closed_loop_native:historical_cf:strict_cf 1:1:1:1",
            "data.train.pgc_balance_native_counterfactual=true",
            "data.train.pgc_entity_relation_supervision_required=true",
            "data.train.pgc_bidirectional_language_supervision_required=true",
            "data.train.pgc_v9_balanced_sampling=true",
            "data.train.pgc_v9_phase_safe_memory=true",
            "data.train.pgc_v9_closed_loop_native_dataset_count=1",
            'ERAF_INITIALIZATION_CONTRACT="released_base_fresh_eraf"',
            "grounding_objective_version=26",
            'ERAF_FRESH_JOINT="true"',
            "bidirectional_supervision=true",
            "model.lora.rank=16",
            "model.lora.alpha=16",
            "model.lora.dropout=0.05",
            "'model.lora.experts=[video,action]'",
            "'model.lora.extra_trainable_patterns=[]'",
        ):
            self.assertIn(contract, source)

    def test_task_config_declares_pretrained_joint_scope_and_schedule(self):
        source = (
            REPO_ROOT / "configs/task/libero_eraf_joint_2cam224.yaml"
        ).read_text(encoding="utf-8")
        for contract in (
            "initialization_contract: released_base_pretrained_eraf",
            "grounding_objective_version: 26",
            "fresh_joint_training: false",
            "pretrained_joint_training: true",
            "pretrained_checkpoint: null",
            "bidirectional_supervision: true",
            "context_injection_warmup_steps: 0",
            "context_injection_ramp_steps: 1000",
            "learning_rate: 2.0e-5",
            "pgc_bidirectional_language_supervision_required: true",
            "max_steps: 10000",
            "experts: [video, action]",
            "extra_trainable_patterns: []",
        ):
            self.assertIn(contract, source)
        self.assertIn("role_attention_preservation_weight: 0.0", source)
        self.assertIn("phase_safe_memory_state_weight: 1.0", source)

    def test_pretrained_launcher_requires_separate_eraf_checkpoint(self):
        source = (
            REPO_ROOT / "scripts/train_libero_eraf_joint_pretrained.sh"
        ).read_text(encoding="utf-8")
        for contract in (
            "<pretrained_eraf_checkpoint>",
            'export ERAF_PRETRAINED_CHECKPOINT="${PRETRAINED_ERAF_CHECKPOINT}"',
            "train_libero_eraf_joint_from_scratch.sh",
        ):
            self.assertIn(contract, source)

    def test_legacy_libero_eraf_defaults_missing_camera_layout_to_horizontal(self):
        model_source = (
            REPO_ROOT / "src/fastwam/models/wan22/fastwam.py"
        ).read_text(encoding="utf-8")
        launcher_source = (
            REPO_ROOT / "scripts/train_libero_eraf_joint_from_scratch.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'metadata.get("eraf_camera_layout", "horizontal")',
            model_source,
        )
        self.assertIn(
            'eraf_metadata.get("eraf_camera_layout", "horizontal")',
            launcher_source,
        )

    def test_standard_pgc_evaluator_reconstructs_joint_metadata(self):
        source = (
            REPO_ROOT / "scripts/eval_pgc_libero.sh"
        ).read_text(encoding="utf-8")
        for contract in (
            'fresh_eraf_joint = bool(metadata.get("eraf_fresh_joint_training", False))',
            'metadata.get("eraf_pretrained_joint_training", False)',
            '"warm_start_contract": "initialization_contract"',
            '"eraf_fresh_joint_training": "fresh_joint_training"',
            '"eraf_pretrained_joint_training": "pretrained_joint_training"',
            '"eraf_bidirectional_supervision": "bidirectional_supervision"',
            '"eraf_context_injection_warmup_steps": (',
            '"eraf_context_injection_ramp_steps": "context_injection_ramp_steps"',
            "pretrained_eraf_plus_shared_video_action_lora_plus_fresh_eraf_",
            "exact_no_injection_warmup_then_append_bounded_eraf_tokens_to_",
        ):
            self.assertIn(contract, source)

    def test_standard_pgc_evaluator_accepts_v929_objective(self):
        source = (
            REPO_ROOT / "scripts/eval_pgc_libero.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("objective not in set(range(1, 30))", source)
        self.assertIn("if objective >= 28", source)
        self.assertIn("if objective >= 29", source)

    def test_base_model_schema_declares_safe_gain_metadata_overrides(self):
        source = (REPO_ROOT / "configs/model/fastwam.yaml").read_text(
            encoding="utf-8"
        )
        for contract in (
            "safe_gain_training: false",
            "safe_gain_num_tokens: 4",
            "safe_gain_gate_hidden_dim: 128",
            "safe_gain_gate_initial_probability: 0.10",
            "safe_gain_gate_threshold: 0.80",
            "safe_gain_deployment_threshold: null",
            "safe_gain_gate_loss_weight: 1.0",
            "safe_gain_wrong_gate_loss_weight: 0.0",
            "safe_gain_gate_ranking_weight: 0.0",
            "safe_gain_gate_ranking_margin: 1.0",
            "safe_gain_non_regression_weight: 2.0",
            "safe_gain_margin: 0.002",
            "safe_gain_injector_training_steps: 0",
            "safe_gain_gate_calibration_steps: 0",
            "safe_gain_noise_levels: 1",
        ):
            self.assertIn(contract, source)

    def test_v929_stages_injector_then_detached_gate_on_closed_loop_cf(self):
        task = (
            REPO_ROOT / "configs/task/libero_eraf_safe_gain_v929_2cam224.yaml"
        ).read_text(encoding="utf-8")
        launcher = (
            REPO_ROOT / "scripts/train_libero_eraf_safe_gain_v929.sh"
        ).read_text(encoding="utf-8")
        model = (
            REPO_ROOT / "src/fastwam/models/wan22/fastwam.py"
        ).read_text(encoding="utf-8")
        trainer = (REPO_ROOT / "src/fastwam/trainer.py").read_text(
            encoding="utf-8"
        )
        for contract in (
            "grounding_objective_version: 29",
            "safe_gain_injector_training_steps: 7000",
            "safe_gain_gate_calibration_steps: 3000",
            "safe_gain_noise_levels: 2",
            "safe_gain_closed_loop_action_weight: 2.0",
            "safe_gain_closed_loop_non_regression_weight: 4.0",
            "pgc_v9_safe_gain_counterfactual_replay: true",
        ):
            self.assertIn(contract, task)
        self.assertIn("meta/pgc_v8_closed_loop/index.json", launcher)
        self.assertIn("migrate_v928_to_v929_rollout_safe_gain", model)
        self.assertIn('detach_gate_inputs=training_phase == "gate"', model)
        self.assertIn('if training_phase == "injector":', model)
        self.assertIn('elif training_phase == "gate":', model)
        for contract in (
            "v929_full_policy_resume = bool(self.resume)",
            'getattr(self.model, "policy_guard_version", 0)',
            '"policy_guard_eraf_grounding_objective_version"',
            '"policy_guard_eraf_safe_gain_training"',
            "and not v929_full_policy_resume",
            "objective-29 safe-gain training restores a validated full",
        ):
            self.assertIn(contract, trainer)

    def test_checkpoint_loader_accepts_joint_role_scope_contracts(self):
        source = (
            REPO_ROOT / "src/fastwam/models/wan22/fastwam.py"
        ).read_text(encoding="utf-8")
        role_scope_validation = source[
            source.index("saved_action_stage = (") : source.index(
                "saved_scope = metadata.get(",
                source.index("saved_action_stage = ("),
            )
        ]
        for contract in (
            "saved_eraf_safe_gain_training",
            '"frozen_complete_eraf_plus_frozen_"',
            '"baseline_lora_plus_compressor_injector_"',
            '"gain_gate"',
            "if saved_eraf_pretrained_joint_training",
            "if saved_eraf_fresh_joint_training",
            '"pretrained_eraf_plus_shared_video_"',
            '"fresh_eraf_plus_shared_video_"',
            '"action_lora_plus_eraf_action_context_"',
        ):
            self.assertIn(contract, role_scope_validation)

    def test_safe_gain_checkpoint_contract_is_independent_of_eval_gate_override(self):
        source = (
            REPO_ROOT / "src/fastwam/models/wan22/fastwam.py"
        ).read_text(encoding="utf-8")
        start = source.index("if saved_grounding_objective >= 26:")
        gate_contract = source[
            start : source.index(
                "if saved_grounding_objective >= 27:",
                start,
            )
        ]
        self.assertIn('"gate_mode": (', gate_contract)
        self.assertIn('"guarded"', gate_contract)
        self.assertNotIn("self.policy_guard_gate_mode", gate_contract)

    def test_evaluator_accepts_eraf_single_path_without_legacy_gate_scores(self):
        source = (
            REPO_ROOT / "experiments/libero/eval_libero_single.py"
        ).read_text(encoding="utf-8")
        for contract in (
            '("policy_guard_base_score", "base_score", float)',
            '"policy_guard_eraf_single_path",',
            'item.get("eraf_single_path", False)',
            'if "base_score" in item',
            'if "counterfactual_score" in item',
            'if "score_margin" in item',
            'results["policy_guard_gated_decision_count"]',
        ):
            self.assertIn(contract, source)


if __name__ == "__main__":
    unittest.main()
