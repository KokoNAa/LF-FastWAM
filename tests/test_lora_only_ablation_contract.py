import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class LoRAOnlyAblationContractTest(unittest.TestCase):
    def test_training_launcher_matches_formal_eraf_lora_controls(self):
        source = (
            REPO_ROOT / "scripts/train_libero_lora_only_ablation.sh"
        ).read_text(encoding="utf-8")
        for contract in (
            'MAX_STEPS="${LORA_ONLY_MAX_STEPS:-10000}"',
            'LEARNING_RATE="${LORA_ONLY_LEARNING_RATE:-5.0e-6}"',
            "data.train.pgc_v9_balanced_sampling=true",
            "data.train.pgc_v9_phase_safe_memory=true",
            "data.train.pgc_bidirectional_language_supervision_required=true",
            "data.train.pgc_v9_closed_loop_native_dataset_count=1",
            "model.policy_guard.enabled=false",
            "model.lora.rank=16",
            "model.lora.alpha=16",
            "model.lora.dropout=0.05",
            "model.lora.paired_language_control.enabled=true",
            "model.lora.paired_language_control.bidirectional_supervision=true",
            "lora-only-no-eraf-bidirectional-10k-seed",
        ):
            self.assertIn(contract, source)
        self.assertIn(
            "offline_native:closed_loop_native:historical_cf:strict_cf 1:1:1:1",
            source,
        )

    def test_task_config_constructs_no_eraf_modules(self):
        source = (
            REPO_ROOT / "configs/task/libero_lora_only_2cam224.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("policy_guard:\n    enabled: false", source)
        self.assertIn("experts: [video, action]", source)
        self.assertIn("extra_trainable_patterns: []", source)
        self.assertIn("paired_language_control:\n      enabled: true", source)
        self.assertIn("bidirectional_supervision: true", source)
        self.assertIn(
            "pgc_bidirectional_language_supervision_required: true", source
        )
        self.assertIn("max_steps: 10000", source)

    def test_evaluator_rejects_nonformal_adapter(self):
        source = (
            REPO_ROOT / "scripts/eval_libero_lora_only_ablation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('EXPECTED_STEP="${LORA_ONLY_EXPECTED_STEP:-10000}"', source)
        self.assertIn('payload.get("format") != "fastwam_lora_adapter_v1"', source)
        self.assertIn("Checkpoint is not the strict paired-language no-ERAF control.", source)
        self.assertIn('control.get("bidirectional_supervision") is not True', source)
        self.assertIn("lora_only_no_eraf_bidirectional_", source)
        self.assertIn(
            "model.lora.paired_language_control.bidirectional_supervision=false",
            source,
        )
        self.assertIn("model.policy_guard.enabled=false", source)
        self.assertIn(
            'if [[ "${CONDITION}" == "counterfactual" ]]', source
        )
        self.assertIn(
            'OVERRIDES+=("EVALUATION.counterfactual_diagnostics=true")',
            source,
        )

    def test_parallel_evaluations_use_distinct_tmux_sessions(self):
        evaluator = (
            REPO_ROOT / "scripts/eval_libero_lora_only_ablation.sh"
        ).read_text(encoding="utf-8")
        manager = (
            REPO_ROOT / "experiments/libero/run_libero_parallel_test.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("libero_lora_only_${EXPECTED_STEP}_${CONDITION}", evaluator)
        self.assertIn('LIBERO_TMUX_SESSION_NAME="${TMUX_SESSION_NAME}"', evaluator)
        self.assertIn(
            'SESSION_NAME="${LIBERO_TMUX_SESSION_NAME:-libero_test_v3}"',
            manager,
        )


if __name__ == "__main__":
    unittest.main()
