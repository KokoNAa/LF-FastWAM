"""CPU regressions for the fixed teacher and actual FastWAM checkpoint path."""

import copy
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fastwam.models.wan22 import eraf_preservation as preservation
from fastwam.trainer import _is_safe_gain_full_policy_resume
from fastwam.utils import cf_ablation as ablation
from test_policy_guard import tiny_pgc_fastwam


def model_for(objective=30, source=None, ablation="none"):
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
        v9_safe_gain_injector_training_steps=250 if objective in {30, 31, 32} else 0,
        v9_safe_gain_gate_calibration_steps=0,
        v9_safe_gain_noise_levels=2 if objective in {30, 31, 32} else 1,
        v9_cf_ablation=ablation,
        v9_full_goal_action_preservation_weight=1.0 if objective in {31, 32} else 0.0,
        v9_full_goal_token_preservation_weight=0.1 if objective in {31, 32} else 0.0,
        v9_full_goal_context_preservation_weight=1.0 if objective in {31, 32} else 0.0,
    )


@pytest.mark.parametrize("objective", [29, 30, 31, 32])
def test_safe_gain_full_policy_resume_accepts_supported_objectives(objective):
    model = SimpleNamespace(
        policy_guard_version=9,
        policy_guard_eraf_grounding_objective_version=objective,
        policy_guard_eraf_safe_gain_training=True,
    )
    assert _is_safe_gain_full_policy_resume(model, "/tmp/full-policy.pt")


@pytest.mark.parametrize(
    ("objective", "resume", "version", "safe_gain"),
    [
        (28, "/tmp/full-policy.pt", 9, True),
        (31, None, 9, True),
        (31, "/tmp/full-policy.pt", 8, True),
        (31, "/tmp/full-policy.pt", 9, False),
    ],
)
def test_safe_gain_full_policy_resume_rejects_invalid_contracts(
    objective, resume, version, safe_gain
):
    model = SimpleNamespace(
        policy_guard_version=version,
        policy_guard_eraf_grounding_objective_version=objective,
        policy_guard_eraf_safe_gain_training=safe_gain,
    )
    assert not _is_safe_gain_full_policy_resume(model, resume)


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


def test_v931_selective_full_goal_proxy_and_interface_losses():
    student = torch.ones(4, 2, 3, requires_grad=True)
    teacher = torch.zeros_like(student)
    base = torch.ones_like(student)
    # Row 2 is a target_lift row and is deliberately excluded by the caller's
    # audited full-goal candidate mask. Row 3 has a worse teacher than Base.
    teacher[3].fill_(2)
    full_goal = torch.tensor([False, True, False, True])
    loss, eligible = preservation.preservation_loss(
        student=student,
        teacher=teacher,
        target=torch.zeros_like(student),
        base=base,
        action_is_pad=torch.zeros(4, 2, dtype=torch.bool),
        direct_valid=torch.ones(4, dtype=torch.bool),
        corrective=torch.tensor([False, True, True, True]),
        candidate_mask=full_goal,
    )
    assert eligible.tolist() == [False, True, False, False]
    token_student = torch.ones(4, 2, 5, requires_grad=True)
    token_loss = preservation.interface_preservation_loss(
        student=token_student,
        teacher=torch.zeros_like(token_student),
        eligible=eligible,
    )
    (loss + token_loss).backward()
    assert student.grad[1].abs().sum() > 0
    assert student.grad[[0, 2, 3]].abs().sum() == 0
    assert token_student.grad[1].abs().sum() > 0
    assert token_student.grad[[0, 2, 3]].abs().sum() == 0


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


def test_v931_hydra_declares_selective_full_goal_contract():
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
        version_base=None,
    ):
        cfg = compose(
            config_name="train", overrides=["task=libero_eraf_safe_gain_v931_2cam224"]
        )
    eraf = cfg.model.policy_guard.entity_relation_grounding
    assert eraf.grounding_objective_version == 31
    assert eraf.cf_ablation == "mask_corrective_ranking"
    assert eraf.full_goal_action_preservation_weight == 1
    assert eraf.full_goal_token_preservation_weight == pytest.approx(0.1)
    assert eraf.full_goal_context_preservation_weight == 1
    assert eraf.full_goal_preservation_margin == 0
    assert eraf.safe_gain_gate_calibration_steps == 0
    assert eraf.safe_gain_injector_training_steps == cfg.max_steps == 250


def test_v932_hydra_changes_only_verified_ranking_route_and_short_schedule():
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
        version_base=None,
    ):
        cfg = compose(
            config_name="train", overrides=["task=libero_eraf_safe_gain_v932_2cam224"]
        )
    eraf = cfg.model.policy_guard.entity_relation_grounding
    assert eraf.grounding_objective_version == 32
    assert eraf.cf_ablation == "mask_lift_ranking"
    assert eraf.full_goal_action_preservation_weight == 1
    assert eraf.full_goal_token_preservation_weight == pytest.approx(0.1)
    assert eraf.full_goal_context_preservation_weight == 1
    assert eraf.safe_gain_injector_training_steps == cfg.max_steps == 50
    assert cfg.save_every == 25


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


def test_v931_runner_clones_actual_b_config_and_only_adds_declared_objective(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/train_libero_eraf_safe_gain_v931.py"
    spec = importlib.util.spec_from_file_location("v931_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        source = compose(
            config_name="train",
            overrides=["task=libero_eraf_safe_gain_v930_2cam224"],
        )
    eraf = source.model.policy_guard.entity_relation_grounding
    eraf.cf_ablation = "mask_corrective_ranking"
    source.resume = str(tmp_path / "v928/step_010000.pt")
    source.batch_size = 1
    source.gradient_accumulation_steps = 4
    source_path = tmp_path / "config.yaml"
    OmegaConf.save(source, source_path)
    loaded = module.load_v930_b(source_path)
    output = tmp_path / "v931"
    derived = module.training_config(loaded, output, 3)
    derived_eraf = derived.model.policy_guard.entity_relation_grounding
    assert derived.resume == loaded.resume
    assert derived.data == loaded.data
    assert derived.seed == loaded.seed
    assert derived.batch_size == loaded.batch_size
    assert derived.gradient_accumulation_steps == loaded.gradient_accumulation_steps
    assert derived.learning_rate == loaded.learning_rate
    assert derived.max_steps == loaded.max_steps == 250
    assert derived_eraf.grounding_objective_version == 31
    assert derived_eraf.cf_ablation == "mask_corrective_ranking"
    for name, value in module.FULL_GOAL_WEIGHTS.items():
        assert derived_eraf[name] == value
    assert module.training_command(output, 3, derived)[0:3] == [
        "bash", "scripts/train_zero1.sh", "3"
    ]
    four_gpu = module.training_config(loaded, tmp_path / "v931-4gpu", 4)
    assert four_gpu.batch_size == 1
    assert four_gpu.gradient_accumulation_steps == 3
    assert 4 * four_gpu.batch_size * four_gpu.gradient_accumulation_steps == 12


def test_v932_runner_keeps_batch_and_builds_25_50_schedule(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/train_libero_eraf_safe_gain_v932.py"
    spec = importlib.util.spec_from_file_location("v932_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        source = compose(
            config_name="train",
            overrides=["task=libero_eraf_safe_gain_v930_2cam224"],
        )
    eraf = source.model.policy_guard.entity_relation_grounding
    eraf.cf_ablation = "mask_corrective_ranking"
    source.resume = str(tmp_path / "v928/step_010000.pt")
    source.batch_size = 1
    source.gradient_accumulation_steps = 4
    source_path = tmp_path / "config.yaml"
    OmegaConf.save(source, source_path)
    loaded = module.load_v930_b(source_path)
    derived = module.training_config(loaded, tmp_path / "v932", 4)
    derived_eraf = derived.model.policy_guard.entity_relation_grounding
    assert derived.resume == loaded.resume
    assert derived.data == loaded.data
    assert derived.seed == loaded.seed
    assert derived.batch_size == 1
    assert derived.gradient_accumulation_steps == 3
    assert 4 * derived.batch_size * derived.gradient_accumulation_steps == 12
    assert derived.max_steps == derived_eraf.safe_gain_injector_training_steps == 50
    assert derived.save_every == 25
    assert derived_eraf.grounding_objective_version == 32
    assert derived_eraf.cf_ablation == "mask_lift_ranking"
    assert module.SAVE_STEPS == (25, 50)


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
    with pytest.raises(ValueError, match="another adapted ERAF"):
        model_for().load_checkpoint(path)


def test_v931_roundtrip_contract_and_rejects_v930_start(warm_checkpoint, tmp_path):
    v930 = model_for()
    v930.prepare_trainable_parameters()
    v930.load_checkpoint(warm_checkpoint)
    v930_path = tmp_path / "v930.pt"
    v930.save_checkpoint(v930_path, step=1)
    with pytest.raises(ValueError, match="another adapted ERAF"):
        model_for(31, ablation="mask_corrective_ranking").load_checkpoint(v930_path)

    v931 = model_for(31, ablation="mask_corrective_ranking")
    v931.prepare_trainable_parameters()
    v931.load_checkpoint(warm_checkpoint)
    v931_path = tmp_path / "v931.pt"
    v931.save_checkpoint(v931_path, step=1)
    payload = torch.load(v931_path, weights_only=False)
    metadata = payload["architecture_metadata"]
    assert (
        metadata["eraf_selective_full_goal_preservation_contract"]
        == preservation.SELECTIVE_FULL_GOAL_CONTRACT
    )
    assert metadata["eraf_full_goal_action_preservation_weight"] == 1
    assert metadata["eraf_full_goal_token_preservation_weight"] == pytest.approx(0.1)
    assert metadata["eraf_full_goal_context_preservation_weight"] == 1
    resumed = model_for(31, ablation="mask_corrective_ranking")
    resumed.prepare_trainable_parameters()
    resumed.load_checkpoint(v931_path)
    resumed._v930_audit_frozen()
    bad = copy.deepcopy(payload)
    bad["architecture_metadata"]["eraf_full_goal_token_preservation_weight"] = 1
    bad_path = tmp_path / "bad_v931.pt"
    torch.save(bad, bad_path)
    with pytest.raises(ValueError, match="full_goal_token"):
        model_for(31, ablation="mask_corrective_ranking").load_checkpoint(bad_path)


def test_v932_roundtrip_records_selective_ranking_mode(warm_checkpoint, tmp_path):
    model = model_for(32, ablation="mask_lift_ranking")
    model.prepare_trainable_parameters()
    model.load_checkpoint(warm_checkpoint)
    path = tmp_path / "v932.pt"
    model.save_checkpoint(path, step=50)
    payload = torch.load(path, weights_only=False)
    metadata = payload["architecture_metadata"]
    assert metadata["eraf_grounding_objective_version"] == 32
    assert ablation.checkpoint_mode(metadata) == "mask_lift_ranking"
    assert (
        metadata["eraf_selective_full_goal_preservation_contract"]
        == preservation.SELECTIVE_FULL_GOAL_CONTRACT
    )
    resumed = model_for(32, ablation="mask_lift_ranking")
    resumed.prepare_trainable_parameters()
    resumed.load_checkpoint(path)
    resumed._v930_audit_frozen()


@pytest.mark.parametrize("objective,ablation,all_lift", [
    (30, "none", False),
    (30, "mask_lift_corrective", False),
    (30, "mask_corrective_ranking", False),
    (30, "mask_lift_corrective", True),
    (31, "mask_corrective_ranking", False),
    (32, "mask_lift_ranking", False),
])
def test_real_action_forward_backward_and_eval_preflight(
    warm_checkpoint, tmp_path, objective, ablation, all_lift
):
    student = model_for(objective=objective, ablation=ablation)
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
        "pgc_is_closed_loop_corrective": torch.tensor([False, False, True, True]),
        "pgc_corrective_verification_kind": torch.tensor([0, 0, 1, 2]),
        "pgc_direct_action_valid": torch.ones(batch, dtype=torch.bool),
        "pgc_paired_language_valid": torch.ones(batch, dtype=torch.bool),
        "pgc_bidirectional_language_valid": torch.ones(batch, dtype=torch.bool),
        "pgc_source_context": torch.randn(batch, 3, 10),
        "pgc_source_context_mask": mask,
        "pgc_target_context": torch.randn(batch, 3, 10),
        "pgc_target_context_mask": mask,
        "language_context_len": 3,
    }
    if all_lift:
        inputs["pgc_is_counterfactual"].fill_(True)
        inputs["pgc_is_closed_loop_corrective"].fill_(True)
        inputs["pgc_corrective_verification_kind"].fill_(1)
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
        patch.object(
            student, "_forward_policy_guard_action_from_cache",
            wraps=student._forward_policy_guard_action_from_cache,
        ) as action_path,
    ):
        loss, metrics = student._training_loss_policy_guard_v927_safe_gain(
            inputs=inputs,
            full_context_mask=mask,
            state_only_context_mask=torch.zeros_like(mask),
        )
    assert torch.isfinite(loss) and loss.requires_grad
    # correct + wrong + two teacher draws + second correct-noise draw, even
    # when the whole microbatch contributes zero gradient in ablation A.
    assert action_path.call_count == 5
    assert metrics["loss_pgc_v930_teacher_preservation"] == pytest.approx(0, abs=1e-8)
    assert metrics["pgc_v930_gate_optimization_weight"] == 0
    if objective in {31, 32}:
        assert "loss_pgc_v931_full_goal_action_preservation" in metrics
        assert "loss_pgc_v931_full_goal_token_preservation" in metrics
        assert "loss_pgc_v931_full_goal_context_preservation" in metrics
        assert metrics["pgc_v931_target_lift_preservation_rate"] == 0
    if ablation == "mask_lift_corrective":
        for component in ("action", "ranking", "nonregression"):
            assert metrics[f"loss_pgc_v930_cf_lift_{component}_used"] == 0
    if ablation == "mask_corrective_ranking":
        for kind in ("lift", "goal"):
            assert metrics[f"loss_pgc_v930_cf_{kind}_ranking_used"] == 0
            assert metrics[f"loss_pgc_v930_cf_{kind}_action_used"] == metrics[f"loss_pgc_v930_cf_{kind}_action_raw"]
    if ablation == "mask_lift_ranking":
        assert metrics["loss_pgc_v930_cf_lift_ranking_used"] == 0
        assert metrics["loss_pgc_v930_cf_lift_action_used"] == metrics["loss_pgc_v930_cf_lift_action_raw"]
        assert metrics["loss_pgc_v930_cf_goal_ranking_used"] == metrics["loss_pgc_v930_cf_goal_ranking_raw"]
        assert metrics["loss_pgc_v930_cf_goal_action_used"] == metrics["loss_pgc_v930_cf_goal_action_raw"]
    if all_lift:
        assert loss.item() == 0
    loss.backward()
    for name in preservation.INTERFACE_NAMES:
        if all_lift:
            assert all(p.grad is None or p.grad.abs().sum() == 0 for p in student.policy_guard_modules[name].parameters())
        else:
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
    saved = tmp_path / f"v9{objective}.pt"
    student.save_checkpoint(saved, step=1)
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/eval_pgc_libero.sh").read_text()
    blocks = [block.split("\nPY", 1)[0] for block in source.split("<<'PY'\n")[1:]]
    validator = next(
        block for block in blocks if "evaluation_inference_steps =" in block
    )
    env = {**os.environ, "PYTHONPATH": str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
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
    assert f"model.policy_guard.entity_relation_grounding.cf_ablation={ablation}" in overrides
    assert (
        "model.policy_guard.entity_relation_grounding.preservation_weight=1.0"
        in overrides
    )
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        evaluation = compose(
            config_name="sim_libero",
            overrides=[
                "task=libero_pgc_2cam224",
                f"model.policy_guard.entity_relation_grounding.grounding_objective_version={objective}",
                *overrides,
            ],
        )
    assert (
        evaluation.model.policy_guard.entity_relation_grounding.safe_gain_gate_calibration_steps
        == 0
    )
    assert evaluation.model.policy_guard.entity_relation_grounding.cf_ablation == ablation
    reloaded = model_for(objective=objective, ablation=ablation)
    reloaded.load_checkpoint(saved)
    if objective == 30 and ablation != "none":
        with pytest.raises(ValueError, match="eraf_cf_ablation"):
            model_for().load_checkpoint(saved)
