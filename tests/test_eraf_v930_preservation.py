"""CPU regressions for the fixed teacher and actual FastWAM checkpoint path."""

import copy
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fastwam.models.wan22 import eraf_preservation as preservation
from test_policy_guard import tiny_pgc_fastwam


def model_for(objective=30, source=None):
    return tiny_pgc_fastwam(
        version=9,
        v9_stage="action",
        v9_grounding_objective_version=objective,
        v9_initialization_contract="released_base_pretrained_eraf",
        v9_completion_only_memory=True,
        v9_action_joint_training=True,
        v9_pretrained_joint_training=True,
        v9_safe_gain_training=True,
        v9_pretrained_checkpoint=str(source) if source else None,
        v9_bidirectional_supervision=True,
        v9_context_injection_warmup_steps=0,
        v9_context_injection_ramp_steps=0,
        v9_safe_gain_injector_training_steps=250 if objective == 30 else 0,
        v9_safe_gain_gate_calibration_steps=0,
        v9_safe_gain_noise_levels=2 if objective == 30 else 1,
    )


@pytest.fixture
def warm_checkpoint(tmp_path):
    torch.manual_seed(42)
    base_path, eraf_path, warm_path = (
        tmp_path / name for name in ("base.pt", "eraf.pt", "v928.pt")
    )
    base = tiny_pgc_fastwam(version=5)
    torch.save({"format": "fastwam_full_v1", "mot": base.mot.state_dict()}, base_path)
    eraf = tiny_pgc_fastwam(
        version=9, v9_stage="grounding", v9_grounding_objective_version=14
    )
    eraf.load_checkpoint(base_path)
    eraf.save_checkpoint(eraf_path, step=7250)
    warm = model_for(28, eraf_path)
    warm.load_checkpoint(base_path)
    warm.load_pretrained_eraf_checkpoint(eraf_path)
    warm.prepare_trainable_parameters()
    warm.save_checkpoint(warm_path, step=10000)
    return warm_path


def test_proxy_masks_padding_corrective_and_bad_teacher():
    student = torch.ones(5, 2, 3, requires_grad=True)
    teacher = torch.zeros_like(student, requires_grad=True)
    base = torch.ones_like(student)
    teacher.data[3].fill_(2)
    pad = torch.tensor(
        [[False, True], [False, False], [False, False], [False, False], [True, True]]
    )
    loss, eligible = preservation.preservation_loss(
        student=student,
        teacher=teacher,
        base=base,
        target=torch.zeros_like(student),
        action_is_pad=pad,
        direct_valid=torch.tensor([1, 0, 1, 1, 1]),
        corrective=torch.tensor([0, 0, 1, 0, 0]),
    )
    assert eligible.tolist() == [True, False, False, False, False]
    assert loss.item() == 1
    loss.backward()
    assert student.grad[0, 0].abs().sum() > 0
    assert student.grad[0, 1].abs().sum() == 0
    assert student.grad[1:].abs().sum() == 0
    assert teacher.grad is None
    empty, _ = preservation.preservation_loss(
        student=student,
        teacher=teacher,
        base=base,
        target=base,
        action_is_pad=torch.ones_like(pad),
        direct_valid=torch.ones(5),
        corrective=torch.zeros(5),
    )
    assert empty.item() == 0 and empty.requires_grad


def test_v930_hydra_composes_without_additive_overrides():
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
        version_base=None,
    ):
        cfg = compose(
            config_name="train", overrides=["task=libero_eraf_safe_gain_v930_2cam224"]
        )
    eraf = cfg.model.policy_guard.entity_relation_grounding
    assert eraf.grounding_objective_version == 30
    assert eraf.safe_gain_gate_calibration_steps == 0
    assert eraf.safe_gain_injector_training_steps == cfg.max_steps == 250
    assert eraf.preservation_weight == 1
    assert cfg.data.train.pgc_v9_safe_gain_counterfactual_replay
    assert cfg.learning_rate == eraf.action_geometry_learning_rate == 2e-6


def test_source_config_launcher_reuses_exact_bindings(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/train_libero_eraf_safe_gain_v930_from_config.py"
    spec = importlib.util.spec_from_file_location("v930_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(
            config_name="train", overrides=["task=libero_eraf_safe_gain_v929_2cam224"]
        )
    cfg.resume = str(tmp_path / "v928/checkpoints/weights/step_010000.pt")
    cfg.data.train.dataset_dirs = [
        str(tmp_path / "data/libero_10_no_noops_lerobot"),
        str(tmp_path / "closed_native"),
    ]
    cfg.data.train.pgc_counterfactual_dataset_dirs = [
        str(tmp_path / "historical"),
        str(tmp_path / "strict"),
    ]
    cfg.data.train.pgc_closed_loop_corrective_dataset_dirs = [
        str(tmp_path / "corrective")
    ]
    cfg.data.train.pgc_entity_relation_sidecar_dirs = [
        str(tmp_path / f"sidecar{i}") for i in range(5)
    ]
    cfg.data.train.pretrained_norm_stats = str(tmp_path / "stats.json")
    cfg.data.train.text_embedding_cache_dir = str(tmp_path / "cache")
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config_path)
    command, env = module.launch_spec(config_path, 4, {"RUN_TAG": "probe"})
    assert command[2:5] == ["libero_10", "4", cfg.resume]
    assert command[5:] == [
        str(tmp_path / name)
        for name in (
            "historical",
            "strict",
            "corrective",
            "sidecar0",
            "closed_native",
            "sidecar1",
            "sidecar2",
            "sidecar3",
            "sidecar4",
        )
    ] + ["42"]
    assert env["LIBERO_DATA_ROOT"] == str(tmp_path / "data")
    assert env["TEXT_CACHE_DIR"] == str(tmp_path / "cache")
    assert env["RUN_TAG"] == "probe"


def test_migration_optimizer_freeze_and_teacher_roundtrip(warm_checkpoint, tmp_path):
    student = model_for()
    # Same order as Trainer: modules/optimizer exist before checkpoint loading.
    student.prepare_trainable_parameters()
    groups = student.policy_guard_optimizer_groups(2e-6)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.2)
    student.load_checkpoint(warm_checkpoint)
    student.set_training_progress(0, 250)
    assert student._policy_guard_v929_training_phase() == "injector"
    assert student.policy_guard_action_expert is None
    assert {g["pgc_v9_group"] for g in groups} == set(preservation.INTERFACE_NAMES)
    teacher = student.eraf_preservation_teacher
    for name in preservation.INTERFACE_NAMES:
        for (key, current), old in zip(
            student.policy_guard_modules[name].named_parameters(),
            teacher[name].parameters(),
            strict=True,
        ):
            assert torch.equal(current, old), key
            assert current.data_ptr() != old.data_ptr()
    teacher_before = preservation.tensor_digest(teacher.state_dict())
    frozen_before = preservation.tensor_digest(student._v930_frozen_state())
    # Multiple real AdamW steps, including nonzero momentum and weight decay.
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(
            p.float().square().sum() + p.float().sum()
            for g in groups
            for p in g["params"]
        )
        loss.backward()
        optimizer.step()
    assert preservation.tensor_digest(student._v930_frozen_state()) == frozen_before
    assert preservation.tensor_digest(teacher.state_dict()) == teacher_before
    student._v930_audit_frozen()
    assert any(
        not torch.equal(p, old)
        for name in preservation.INTERFACE_NAMES
        for p, old in zip(
            student.policy_guard_modules[name].parameters(),
            teacher[name].parameters(),
            strict=True,
        )
    )
    saved = tmp_path / "v930.pt"
    student.save_checkpoint(saved, step=3)
    payload = torch.load(saved, weights_only=False)
    assert (
        payload["architecture_metadata"]["eraf_action_trainable_scope"]
        == preservation.ACTION_SCOPE
    )
    assert (
        payload["architecture_metadata"]["eraf_safe_gain_schedule_contract"]
        == preservation.SCHEDULE
    )
    assert (
        payload["architecture_metadata"]["eraf_expert_lora_training_contract"][
            "gain_gate_weight"
        ]
        == 0
    )
    resumed = model_for()
    resumed.prepare_trainable_parameters()
    resumed.load_checkpoint(saved)
    assert (
        preservation.tensor_digest(resumed.eraf_preservation_teacher.state_dict())
        == teacher_before
    )
    resumed._v930_audit_frozen()
    # Inference creates neither a teacher module nor a second expert.
    inference = model_for()
    inference.load_checkpoint(saved)
    assert inference.eraf_preservation_teacher is None
    assert inference.policy_guard_action_expert is None
    assert all(
        torch.equal(value, inference.policy_guard_modules.state_dict()[name])
        for name, value in student.policy_guard_modules.state_dict().items()
    )
    bad = copy.deepcopy(payload)
    key = next(iter(bad["eraf_preservation_teacher"]["state_dict"]))
    bad["eraf_preservation_teacher"]["state_dict"][key].add_(1)
    with pytest.raises(ValueError, match="checksum"):
        preservation.validate_teacher_payload(bad)
    with torch.no_grad():
        next(resumed.policy_guard_modules["eraf_gain_gate"].parameters()).add_(1)
    with pytest.raises(RuntimeError, match="tensors changed"):
        resumed._v930_audit_frozen()


def test_rejects_v929_and_wrong_v928_step(warm_checkpoint, tmp_path):
    payload = torch.load(warm_checkpoint, weights_only=False)
    payload["step"] = 8500
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="step10000"):
        model_for().load_checkpoint(path)
    payload["architecture_metadata"]["eraf_grounding_objective_version"] = 29
    torch.save(payload, path)
    with pytest.raises(ValueError, match="not V9.29"):
        model_for().load_checkpoint(path)


def test_real_action_forward_backward_and_eval_preflight(warm_checkpoint, tmp_path):
    student = model_for()
    student.prepare_trainable_parameters()
    student.load_checkpoint(warm_checkpoint)
    groups = student.policy_guard_optimizer_groups(2e-6)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    student.set_training_progress(0, 250)
    batch, clauses, patches = 4, 4, 4
    mask = torch.ones(batch, 3, dtype=torch.bool)
    inputs = {
        "action": torch.randn(batch, 2, 3),
        "action_is_pad": torch.zeros(batch, 2, dtype=torch.bool),
        "input_latents": torch.randn(batch, 2, 1, 2, 2),
        "fuse_vae_embedding_in_latents": True,
        "pgc_is_counterfactual": torch.tensor([False, True, True, True]),
        "pgc_is_closed_loop_corrective": torch.tensor([False, False, False, True]),
        "pgc_direct_action_valid": torch.ones(batch, dtype=torch.bool),
        "pgc_paired_language_valid": torch.ones(batch, dtype=torch.bool),
        "pgc_bidirectional_language_valid": torch.ones(batch, dtype=torch.bool),
        "pgc_source_context": torch.randn(batch, 3, 10),
        "pgc_source_context_mask": mask,
        "pgc_target_context": torch.randn(batch, 3, 10),
        "pgc_target_context_mask": mask,
        "language_context_len": 3,
    }
    labels = {
        "phase_safe_memory_previous_state_ids": torch.zeros(
            batch, clauses, dtype=torch.long
        ),
        "phase_safe_memory_state_valid": torch.ones(batch, clauses, dtype=torch.bool),
    }
    eraf_outputs = {
        "subject_attention": torch.ones(batch, clauses, patches) / patches,
        "reference_attention": torch.ones(batch, clauses, patches) / patches,
        "active_logits": torch.zeros(batch, clauses),
        "clause_execution_probability": torch.ones(batch, clauses) / clauses,
        "phase_safe_memory_state_probability": torch.ones(batch, clauses, 4) / 4,
        "phase_safe_memory_next_state_valid": torch.ones(
            batch, clauses, dtype=torch.bool
        ),
    }
    queries = torch.randn(
        batch,
        clauses,
        student.policy_guard_modules["eraf_action_token_compressor"].goal_dim,
    )
    # Only privileged labels and frozen ERAF extraction are synthetic. Actual
    # video cache, one Expert, student/teacher interfaces, multi-noise loss and
    # backward/optimizer paths run on CPU.
    with (
        patch.object(student, "_policy_guard_v9_labels", return_value=labels),
        patch.object(
            student,
            "_encode_policy_guard_eraf",
            return_value=(queries, None, eraf_outputs, {}),
        ),
    ):
        loss, metrics = student._training_loss_policy_guard_v927_safe_gain(
            inputs=inputs,
            full_context_mask=mask,
            state_only_context_mask=torch.zeros_like(mask),
        )
    assert torch.isfinite(loss) and loss.requires_grad
    assert metrics["loss_pgc_v930_teacher_preservation"] == pytest.approx(0, abs=1e-8)
    assert metrics["pgc_v930_gate_optimization_weight"] == 0
    loss.backward()
    for name in preservation.INTERFACE_NAMES:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in student.policy_guard_modules[name].parameters()
        )
    assert all(
        p.grad is None
        for p in student.policy_guard_modules["eraf_gain_gate"].parameters()
    )
    assert all(p.grad is None for p in student.eraf_preservation_teacher.parameters())
    optimizer.step()
    student._v930_audit_frozen()
    saved = tmp_path / "v930.pt"
    student.save_checkpoint(saved, step=1)
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/eval_pgc_libero.sh").read_text()
    blocks = [block.split("\nPY", 1)[0] for block in source.split("<<'PY'\n")[1:]]
    validator = next(
        block for block in blocks if "evaluation_inference_steps =" in block
    )
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [sys.executable, "-c", validator, str(saved), "2", "full"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    overrides_block = next(
        block for block in blocks if '"safe_gain_noise_levels"' in block
    )
    result = subprocess.run(
        [sys.executable, "-c", overrides_block, str(saved)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    overrides = result.stdout.splitlines()
    assert (
        "model.policy_guard.entity_relation_grounding.preservation_weight=1.0"
        in overrides
    )
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        evaluation = compose(
            config_name="sim_libero",
            overrides=[
                "task=libero_pgc_2cam224",
                "model.policy_guard.entity_relation_grounding.grounding_objective_version=30",
                *overrides,
            ],
        )
    assert (
        evaluation.model.policy_guard.entity_relation_grounding.safe_gain_gate_calibration_steps
        == 0
    )
