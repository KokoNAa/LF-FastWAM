"""CPU checks of the real one-expert inference path plus diagnostic isolation."""

import copy
import json
import random
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from fastwam.models.wan22.eraf_interface_probe import (
    COMBINATIONS, InterfaceProbe, file_sha256, isolated_rng, validate_checkpoints,
)
from fastwam.models.wan22.eraf_preservation import INTERFACE_NAMES, tensor_digest
from experiments.libero.eraf_interface_probe import observe_simulator
from scripts.eval_libero_eraf_interface_probe import job_config, load_cases, summarize
from scripts import eval_libero_eraf_interface_probe as runner
from test_eraf_v930_preservation import model_for, warm_checkpoint


@pytest.fixture
def candidate(warm_checkpoint, tmp_path):
    student = model_for()
    student.prepare_trainable_parameters()
    student.load_checkpoint(warm_checkpoint)
    with torch.no_grad():
        for name in INTERFACE_NAMES:
            for param in student.policy_guard_modules[name].parameters():
                param.add_(torch.randn_like(param) * 0.015)
    path = tmp_path / "candidate.pt"
    student.save_checkpoint(path, step=250)
    live = model_for()
    live.load_checkpoint(path)
    live.policy_guard_gate_mode = "counterfactual"
    live.eval()
    return live, path


def inference_fixture(model):
    queries = torch.randn(1, 4, model.policy_guard_modules[INTERFACE_NAMES[0]].goal_dim)
    output = {"role_attention": torch.randn(1, 4, 4)}
    state = {"phase_safe_memory_state_ids": torch.tensor([[3, 0, 0, 0]]),
             "phase_safe_memory_valid": torch.tensor([[True, False, False, False]])}

    def encode_goal(**kwargs):
        model._policy_guard_last_eraf_outputs = output
        model._policy_guard_last_eraf_diagnostics = {
            "phase_safe_memory_next_state_ids": state["phase_safe_memory_state_ids"].clone(),
            "phase_safe_memory_next_state_valid": state["phase_safe_memory_valid"].clone(),
        }
        return queries, queries.mean(1), {}

    inputs = dict(prompt=None, input_image=torch.zeros(1, 3, 16, 16),
                  context=torch.randn(1, 3, 10), context_mask=torch.ones(1, 3, dtype=torch.bool),
                  action_horizon=2, num_inference_steps=2, seed=42, policy_guard_state=state)
    return inputs, encode_goal


@pytest.mark.parametrize("driver", ["old_old", "new_new"])
def test_real_inference_shared_cache_no_state_rng_or_weight_change(candidate, warm_checkpoint, driver):
    model, path = candidate
    probe = InterfaceProbe(model, warm_checkpoint, path, driver)
    expert = model.action_expert
    inputs, encode_goal = inference_fixture(model)
    before_state = tensor_digest(inputs["policy_guard_state"])
    before_weights = tensor_digest(model.state_dict())
    gate_dim = model.policy_guard_modules["eraf_gain_gate"].diagnostic_dim
    with probe.driver_scope(), patch.object(
        model, "_encode_input_image_latents_tensor", return_value=torch.zeros(1, 2, 1, 2, 2)
    ), patch.object(model, "_encode_policy_guard_goal", side_effect=encode_goal) as upstream, patch.object(
        model, "_policy_guard_eraf_gain_features", return_value=torch.zeros(1, gate_dim)
    ), patch.object(model, "_forward_policy_guard_action_from_cache", wraps=model._forward_policy_guard_action_from_cache) as forward:
        ordinary = model.infer_action(**inputs)
        rng_before = torch.random.get_rng_state().clone()
        diagnosed = model.infer_action(**inputs, policy_guard_interface_probe=probe)
        assert torch.equal(rng_before, torch.random.get_rng_state())
        assert upstream.call_count == 2  # one upstream encoding per inference, NOT per variant
        assert forward.call_count == 12  # 2 ordinary + (2 driver + 4*2 probe)
        assert torch.equal(ordinary["action"], diagnosed["action"])
        assert tensor_digest(ordinary["policy_guard_state"]) == tensor_digest(diagnosed["policy_guard_state"])
        assert diagnosed["eraf_interface_probe"]["record"]["driver_repeat_validated"]
        arrays = diagnosed["eraf_interface_probe"]["arrays"]
        assert all(f"{variant}/action_normalized" in arrays for variant in COMBINATIONS)
        # Changing either real interface changes the conditioning in this fixture.
        assert not np.array_equal(arrays["new_old/tokens"], arrays["old_old/tokens"])
        assert not np.array_equal(arrays["old_new/injected_tokens"], arrays["old_old/injected_tokens"])
    assert model.action_expert is expert and model.policy_guard_action_expert is None
    assert tensor_digest(model.state_dict()) == before_weights
    assert tensor_digest(inputs["policy_guard_state"]) == before_state


def test_scope_and_rng_restore_after_exception(candidate, warm_checkpoint):
    model, path = candidate
    probe = InterfaceProbe(model, warm_checkpoint, path, "old_old")
    original = model.policy_guard_modules[INTERFACE_NAMES[0]]
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    with pytest.raises(RuntimeError, match="failure"):
        with probe.driver_scope(), isolated_rng() as reset:
            a = (random.random(), np.random.rand(), torch.rand(2))
            reset()
            assert random.random() == a[0] and np.random.rand() == a[1]
            assert torch.equal(torch.rand(2), a[2])
            raise RuntimeError("failure")
    assert model.policy_guard_modules[INTERFACE_NAMES[0]] is original
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


@pytest.mark.parametrize("part", ["guard", "lora", "teacher", "base"])
def test_checkpoint_contract_rejects_drift(candidate, warm_checkpoint, part):
    _, path = candidate
    warm = torch.load(warm_checkpoint, weights_only=False)
    current = torch.load(path, weights_only=False)
    if part == "guard":
        key = next(k for k in current["policy_guard"] if k.startswith("eraf_gain_gate."))
        current["policy_guard"][key] += 1
    elif part == "lora":
        current["eraf_shared_expert_lora"][next(iter(current["eraf_shared_expert_lora"]))] += 1
    elif part == "teacher":
        current["eraf_preservation_teacher"]["state_dict"][next(iter(current["eraf_preservation_teacher"]["state_dict"]))] += 1
    else:
        current["base_checkpoint"] = "different.pt"
    with pytest.raises(ValueError):
        validate_checkpoints(warm, current, file_sha256(warm_checkpoint))


def test_fixture_observer_records_joints_without_body_lookup():
    positions = {"microwave_1_hinge": 0.15, "white_cabinet_1_bottom_joint": -0.08}
    inner = SimpleNamespace(sim=SimpleNamespace(
        model=SimpleNamespace(joint_names=list(positions)),
        data=SimpleNamespace(get_joint_qpos=positions.__getitem__)),
        _eval_predicate=lambda p: p[0] == "close")
    metadata = {"source_goal_state": [["in", "bowl", "white_cabinet_1_bottom_region"]],
                "counterfactual_goal_state": [["close", "white_cabinet_1_bottom_region"]]}
    record = observe_simulator(SimpleNamespace(env=inner), metadata)
    assert record["joints_qpos"]["white_cabinet_1_bottom_joint"] == [-0.08]
    assert record["predicates"]["counterfactual_goal_state"][0]["holds"]
    assert not record["predicates"]["source_goal_state"][0]["holds"]


def test_config_preserves_protocol_and_all_trials(tmp_path):
    source = OmegaConf.create({
        "ckpt": "candidate.pt", "seed": 42,
        "model": {"policy_guard": {"gate_mode": "counterfactual",
                  "entity_relation_grounding": {"grounding_objective_version": 30}}},
        "EVALUATION": {"instruction_condition": "counterfactual", "counterfactual_diagnostics": True,
                       "num_trials": 5, "num_inference_steps": 10, "replan_steps": 10, "num_steps_wait": 30},
    })
    before = copy.deepcopy(source)
    cfg = job_config(source, warm="warm.pt", suite="libero_10", task=3, trials=[1, 4],
                     driver="old_old", output=tmp_path, gpu=2, stride=5)
    assert source == before
    for key in ("num_trials", "num_inference_steps", "replan_steps", "num_steps_wait"):
        assert cfg.EVALUATION[key] == source.EVALUATION[key]
    assert cfg.ckpt == source.ckpt and cfg.EVALUATION.interface_probe.trials == [1, 4]
    assert cfg.model == source.model
    source.EVALUATION.entity_relation_oracle = True
    with pytest.raises(ValueError, match="oracle"):
        job_config(source, warm="warm.pt", suite="libero_10", task=3, trials=[1],
                   driver="old_old", output=tmp_path, gpu=0, stride=5)


def test_cases_reject_duplicate_and_out_of_range(tmp_path):
    path = tmp_path / "cases.json"
    for cases in ([{"task_id": 3, "trial_id": 5}], [{"task_id": 3, "trial_id": 1}] * 2):
        path.write_text(json.dumps({"suite": "libero_10", "cases": cases}))
        with pytest.raises(ValueError):
            load_cases(path, 5)


@pytest.mark.parametrize("execute,mismatch", [(False, False), (True, False), (True, True)])
def test_runner_dry_run_and_warm_calibration_barrier(candidate, warm_checkpoint, tmp_path, monkeypatch, execute, mismatch):
    _, path = candidate
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"suite": "libero_10", "cases": [{"task_id": 3, "trial_id": 1}]}))
    reference = tmp_path / "reference"
    reference.mkdir()
    saved = {"task_suite": "libero_10", "task_id": 3, "total_episodes": 5,
             "success_episodes": [1], "language_intervention_manifest_sha256": file_sha256(manifest)}
    (reference / "gpu0_task3_results.json").write_text(json.dumps(saved))
    source = tmp_path / "manager_config.yaml"
    OmegaConf.save(OmegaConf.create({
        "ckpt": str(path), "seed": 42,
        "model": {"policy_guard": {"gate_mode": "counterfactual", "entity_relation_grounding": {"grounding_objective_version": 30}}},
        "EVALUATION": {"instruction_condition": "counterfactual", "counterfactual_diagnostics": True,
                       "num_trials": 5, "language_intervention_manifest": str(manifest)},
        "MULTIRUN": {"task_suite_names": ["libero_10"]},
    }), source)
    output = tmp_path / "new_run"
    argv = ["probe", "--source-config", str(source), "--warm-checkpoint", str(warm_checkpoint),
            "--warm-results", str(reference), "--candidate-results", str(reference),
            "--cases", str(cases), "--output", str(output)]
    if execute:
        argv.append("--execute")
    monkeypatch.setattr(sys, "argv", argv)
    calls = []

    def fake_worker(command, **kwargs):
        job_dir = runner.Path(command[command.index("--config-path") + 1])
        cfg = OmegaConf.load(job_dir / "probe_eval.yaml")
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == str(cfg.gpu_id)
        calls.append(cfg.EVALUATION.interface_probe.driver)
        result = dict(saved, success_episodes=[] if mismatch else [1])
        (job_dir / f"gpu{cfg.gpu_id}_task3_results.json").write_text(json.dumps(result))
        observer = job_dir / "interface_probe"
        observer.mkdir()
        record = {"kind": "probe", "trial": 1, "driver_repeat_validated": True,
                  "variants": {v: {"action_normalized_vs_old_old": {"rms": 0.0}} for v in COMBINATIONS}}
        (observer / "records.jsonl").write_text(json.dumps(record) + "\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_worker)
    if mismatch:
        with pytest.raises(SystemExit, match="CALIBRATION FAILED"):
            runner.main()
        assert calls == ["old_old"] and (output / "calibration_failure.json").is_file()
    else:
        runner.main()
        if execute:
            assert calls == ["old_old", "new_new"]
            assert json.loads((output / "summary.json").read_text())["all_driver_outcomes_reproduced"]
        else:
            assert calls == [] and not output.exists()
    output.mkdir(exist_ok=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.main()
