import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ERAFJointTrainingContractTest(unittest.TestCase):
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

    def test_checkpoint_loader_accepts_joint_role_scope_contracts(self):
        source = (
            REPO_ROOT / "src/fastwam/models/wan22/fastwam.py"
        ).read_text(encoding="utf-8")
        role_scope_validation = source[
            source.index("saved_action_stage = (") : source.index(
                'metadata.get("eraf_role_adapter_trainable_scope")',
                source.index("saved_action_stage = ("),
            )
        ]
        for contract in (
            "if saved_eraf_pretrained_joint_training",
            "if saved_eraf_fresh_joint_training",
            '"pretrained_eraf_plus_shared_video_"',
            '"fresh_eraf_plus_shared_video_"',
            '"action_lora_plus_eraf_action_context_"',
        ):
            self.assertIn(contract, role_scope_validation)


if __name__ == "__main__":
    unittest.main()
