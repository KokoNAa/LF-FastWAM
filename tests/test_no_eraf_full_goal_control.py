import importlib.util
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_libero_no_eraf_full_goal_control_v938.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("no_eraf_fg25", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def formal_lora():
    return {
        "enabled": True,
        "rank": 16,
        "alpha": 16.0,
        "dropout": 0.05,
        "experts": ["video", "action"],
        "target_modules": ["q", "k", "v", "o"],
        "extra_trainable_patterns": [],
        "paired_language_control": {
            "enabled": True,
            "bidirectional_supervision": True,
            "world_language_weight": 0.1,
            "world_language_margin": 0.01,
            "native_action_weight": 1.0,
            "counterfactual_action_weight": 1.0,
            "action_language_weight": 1.0,
            "action_language_margin": 0.01,
            "regularization_weight": 1.0e-6,
        },
    }


def source_config(tmp_path):
    return OmegaConf.create(
        {
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "model": {
                "policy_guard": {
                    "enabled": True,
                    "entity_relation_grounding": {
                        "safe_gain_training": True,
                        "action_joint_training": True,
                        "pretrained_joint_training": True,
                        "bidirectional_supervision": True,
                    },
                },
                "lora": {"enabled": False},
            },
            "data": {
                "train": {
                    "pgc_closed_loop_corrective_dataset_dirs": ["old"],
                    "pgc_entity_relation_sidecar_dirs": [
                        f"sidecar-{index}" for index in range(5)
                    ],
                    "pgc_v9_safe_gain_counterfactual_replay": True,
                }
            },
            "resume": str(tmp_path / "v928.pt"),
            "weight_only_start_step": None,
            "max_steps": 250,
            "save_every": 50,
            "log_every": 10,
            "learning_rate": 2.0e-6,
            "wandb": {"name": "source"},
        }
    )


def test_no_eraf_full_goal_control_is_data_matched_and_eraf_free(tmp_path):
    module = load_runner()
    dataset = tmp_path / "full-goal"
    sidecar = tmp_path / "full-goal-sidecar"
    baseline = tmp_path / "step_008500.pt"
    output = tmp_path / "output"
    cfg = module.control_config(
        source_config(tmp_path),
        output,
        baseline,
        formal_lora(),
        dataset,
        sidecar,
        gpus=3,
    )
    assert cfg.model.policy_guard.enabled is False
    assert OmegaConf.to_container(cfg.model.policy_guard, resolve=True) == {
        "enabled": False
    }
    assert cfg.model.lora.enabled is True
    assert cfg.model.lora.paired_language_control.enabled is True
    assert cfg.model.lora.paired_language_control.bidirectional_supervision is True
    assert cfg.data.train.pgc_closed_loop_corrective_dataset_dirs == [
        str(dataset.resolve())
    ]
    assert cfg.data.train.pgc_entity_relation_sidecar_dirs[-1] == str(
        sidecar.resolve()
    )
    assert cfg.data.train.pgc_v9_safe_gain_counterfactual_replay is True
    assert cfg.resume == str(baseline.resolve())
    assert cfg.weight_only_start_step is None
    assert cfg.max_steps == 25
    assert cfg.save_every == 5
    assert cfg.learning_rate == 2.0e-6
    assert cfg.gradient_accumulation_steps == 4
    assert 3 * cfg.batch_size * cfg.gradient_accumulation_steps == 12


def test_no_eraf_full_goal_control_rejects_nonformal_lora():
    module = load_runner()
    config = formal_lora()
    config["experts"] = ["action"]
    try:
        module.validate_formal_lora_config(config)
    except ValueError as error:
        assert "formal bidirectional" in str(error)
    else:
        raise AssertionError("Expected nonformal LoRA config to be rejected.")


def test_no_eraf_full_goal_control_preserves_batch_twelve_on_four_gpus(tmp_path):
    module = load_runner()
    cfg = module.control_config(
        source_config(tmp_path),
        tmp_path / "output-4gpu",
        tmp_path / "step_008500.pt",
        formal_lora(),
        tmp_path / "full-goal",
        tmp_path / "full-goal-sidecar",
        gpus=4,
    )
    assert cfg.batch_size == 1
    assert cfg.gradient_accumulation_steps == 3
    assert 4 * cfg.batch_size * cfg.gradient_accumulation_steps == 12
    assert OmegaConf.to_container(cfg.model.policy_guard, resolve=True) == {
        "enabled": False
    }


def test_no_eraf_full_goal_control_records_exact_policy_identity_audit():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "torch.equal(baseline_state[name], v928_state[name])" in source
    assert '"eraf_forward_enabled": False' in source
    assert '"sidecar_usage": "sampling_and_language_provenance_only"' in source
    assert '"shared_video_action_lora_only"' in source
