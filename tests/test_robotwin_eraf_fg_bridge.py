"""Exercise the production attention and ERAF modules with small CPU tensors."""
import copy
from unittest.mock import patch

import pytest
import torch

from experiments.robotwin.eraf_fg_bridge import (
    build_cache, compatible_lora_config, masked_mse, predict, validate_model_geometry,
)
from experiments.robotwin.joint_adapter_repair import capture_inputs, predict as old_predict
from fastwam.models.wan22.lora import normalize_lora_config
from test_policy_guard import tiny_pgc_fastwam


@pytest.fixture
def case():
    torch.manual_seed(17)
    model = tiny_pgc_fastwam(
        version=9, v9_grounding_objective_version=26,
        v9_initialization_contract="released_base_fresh_eraf", v9_stage="action",
        v9_fresh_joint_training=True, v9_action_joint_training=True,
        v9_completion_only_memory=True, v9_bidirectional_supervision=True,
        v9_context_injection_warmup_steps=0, v9_context_injection_ramp_steps=0,
    ).eval()
    inputs = dict(prompt=None, input_image=torch.zeros(1, 3, 16, 16),
                  context=torch.randn(1, 3, 10), context_mask=torch.ones(1, 3, dtype=torch.bool),
                  action_horizon=4, num_inference_steps=2, seed=42,
                  proprio=torch.randn(1, 8))
    model.policy_guard_enabled = False
    with patch.object(model, "_encode_input_image_latents_tensor",
                      return_value=torch.randn(1, 2, 1, 2, 4)):
        _, captured, _ = capture_inputs(model, lambda: model.infer_action(**inputs))
    model.policy_guard_enabled = True
    captured["proprio"] = inputs["proprio"]
    return model, captured, torch.randn(1, 4, 3), torch.tensor([500.])


def test_no_eraf_matches_existing_shared_policy_exactly(case):
    model, captured, noisy, time = case
    a = old_predict(model, captured, noisy, time, checkpoint=False)
    b = predict(model, captured, noisy, time, eraf=False, checkpoint=False)
    assert torch.equal(a, b)


def test_eraf_production_path_has_action_and_video_and_interface_gradients(case):
    model, captured, noisy, time = case
    model.requires_grad_(True)
    value = predict(model, captured, noisy, time)
    assert torch.isfinite(value).all()
    value.square().mean().backward()
    for module in (model.video_expert, model.action_expert,
                   model.policy_guard_modules["eraf_action_context_injector"]):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_masked_full_goal_loss_ignores_padding_and_gradient():
    prediction = torch.zeros(1, 32, 14, requires_grad=True)
    target = torch.ones_like(prediction)
    target[:, 12:] = 99999
    loss = masked_mse(prediction, target, torch.arange(32) < 12)
    assert loss.item() == 1
    loss.backward()
    assert prediction.grad[:, :12].abs().sum() > 0
    assert prediction.grad[:, 12:].count_nonzero() == 0


def test_lora_training_flag_bridge_is_valid_and_does_not_mutate_source():
    original = {"enabled": True, "rank": 16, "experts": ["video", "action"],
                "paired_language_control": {"enabled": True, "bidirectional_supervision": True,
                  "deployment_matched_action_cache": True, "correct_branch_action_ranking": True}}
    snapshot = copy.deepcopy(original)
    normalized = normalize_lora_config(compatible_lora_config(original))
    assert not normalized["paired_language_control"]["enabled"]
    assert original == snapshot


def test_tiny_model_cannot_be_published_as_real_robotwin_checkpoint(case):
    with pytest.raises(ValueError, match="14-D"):
        validate_model_geometry(case[0])


def test_interface_stage_keeps_semantic_heads_and_experts_frozen(case):
    from experiments.robotwin.eraf_fg_bridge import trainable_parameters
    model, captured, noisy, time = case
    selected = trainable_parameters(model, "interface")
    assert not any(p.requires_grad for p in model.mot.parameters())
    assert not any("predicate_head" in n for n in selected)
    assert any(n.startswith("guard.eraf_action_grounding_bridge.") for n in selected)
    predict(model, captured, noisy, time).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in selected.values())


def test_master_optimizer_accumulates_updates_below_bfloat16_spacing():
    from experiments.robotwin.eraf_fg_bridge import MasterAdamW
    live = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    opt = MasterAdamW([live], lr=1e-5)
    for _ in range(300):
        opt.zero_grad()
        live.grad = torch.ones_like(live)
        opt.step()
    assert .996 < opt.master[0].item() < .998
    assert live.item() < 1


def test_semantic_pretraining_path_matches_production_eraf_heads(case):
    from experiments.robotwin.eraf_fg_data import grounding_outputs
    model, captured, _, _ = case
    seen = []
    hook = model.policy_guard_modules["entity_relation_affordance"].register_forward_hook(
        lambda module, inputs, output: seen.append(output[2]))
    try:
        build_cache(model, captured)
    finally:
        hook.remove()
    production = seen[0]
    actual, _ = grounding_outputs(model, captured)
    for key in ("subject_attention", "reference_attention", "predicate_logits",
                "predicate_truth_logits", "phase_logits", "clause_execution_probability"):
        assert torch.equal(actual[key], production[key]), key
