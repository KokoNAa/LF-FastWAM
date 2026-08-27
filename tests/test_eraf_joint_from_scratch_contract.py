import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ERAFJointFromScratchContractTest(unittest.TestCase):
    def test_launcher_keeps_no_eraf_data_contract_and_runs_15k(self):
        source = (
            REPO_ROOT / "scripts/train_libero_eraf_joint_from_scratch.sh"
        ).read_text(encoding="utf-8")
        for contract in (
            'MAX_STEPS="${ERAF_JOINT_MAX_STEPS:-15000}"',
            "task=libero_eraf_joint_2cam224",
            "offline_native:closed_loop_native:historical_cf:strict_cf 1:1:1:1",
            "data.train.pgc_balance_native_counterfactual=true",
            "data.train.pgc_entity_relation_supervision_required=true",
            "data.train.pgc_bidirectional_language_supervision_required=true",
            "data.train.pgc_v9_balanced_sampling=true",
            "data.train.pgc_v9_phase_safe_memory=true",
            "data.train.pgc_v9_closed_loop_native_dataset_count=1",
            "initialization_contract=released_base_fresh_eraf",
            "grounding_objective_version=26",
            "fresh_joint_training=true",
            "bidirectional_supervision=true",
            "model.lora.rank=16",
            "model.lora.alpha=16",
            "model.lora.dropout=0.05",
            "'model.lora.experts=[video,action]'",
            "'model.lora.extra_trainable_patterns=[]'",
        ):
            self.assertIn(contract, source)

    def test_task_config_declares_fresh_joint_scope_and_schedule(self):
        source = (
            REPO_ROOT / "configs/task/libero_eraf_joint_2cam224.yaml"
        ).read_text(encoding="utf-8")
        for contract in (
            "initialization_contract: released_base_fresh_eraf",
            "grounding_objective_version: 26",
            "fresh_joint_training: true",
            "bidirectional_supervision: true",
            "context_injection_warmup_steps: 1500",
            "context_injection_ramp_steps: 1000",
            "pgc_bidirectional_language_supervision_required: true",
            "max_steps: 15000",
            "experts: [video, action]",
            "extra_trainable_patterns: []",
        ):
            self.assertIn(contract, source)
        self.assertIn("role_attention_preservation_weight: 0.0", source)
        self.assertIn("phase_safe_memory_state_weight: 1.0", source)

    def test_standard_pgc_evaluator_reconstructs_fresh_joint_metadata(self):
        source = (
            REPO_ROOT / "scripts/eval_pgc_libero.sh"
        ).read_text(encoding="utf-8")
        for contract in (
            'fresh_eraf_joint = bool(metadata.get("eraf_fresh_joint_training", False))',
            '"warm_start_contract": "initialization_contract"',
            '"eraf_fresh_joint_training": "fresh_joint_training"',
            '"eraf_bidirectional_supervision": "bidirectional_supervision"',
            '"eraf_context_injection_warmup_steps": (',
            '"eraf_context_injection_ramp_steps": "context_injection_ramp_steps"',
            "fresh_eraf_plus_shared_video_action_lora_plus_eraf_action_",
            "exact_no_injection_warmup_then_append_bounded_eraf_tokens_to_",
        ):
            self.assertIn(contract, source)


if __name__ == "__main__":
    unittest.main()
