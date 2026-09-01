"""Single-factor numerator masks, historical metadata and config-cloned launch."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fastwam.utils import cf_ablation as ablation
from fastwam.models.wan22.fastwam import FastWAM
from scripts.train_libero_eraf_cf_ablation import load_control, training_config, training_command
import scripts.train_libero_eraf_cf_ablation as runner
from test_eraf_v930_preservation import model_for, warm_checkpoint


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode,actions,ranks", [
    ("none", [1, 1, 1, 1], [1, 1, 1, 1]),
    ("mask_lift_corrective", [1, 1, 0, 1], [1, 1, 0, 1]),
    ("mask_corrective_ranking", [1, 1, 1, 1], [1, 1, 0, 0]),
    ("mask_lift_ranking", [1, 1, 1, 1], [1, 1, 0, 1]),
])
def test_multipliers_preserve_denominators_and_gradient_isolation(mode, actions, ranks):
    corrective = torch.tensor([0, 0, 1, 1]).bool()
    kind = torch.tensor([0, 0, 1, 2])
    rng_before = torch.random.get_rng_state()
    action_mask, rank_mask, ids = ablation.loss_multipliers(mode, corrective, kind)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert action_mask.tolist() == list(map(bool, actions))
    assert rank_mask.tolist() == list(map(bool, ranks))
    assert torch.equal(ids, kind)
    values = torch.tensor([1., 2., 3., 4.], requires_grad=True)
    loss = FastWAM._masked_policy_guard_mean(values * action_mask, corrective)
    assert loss.item() == (3 * actions[2] + 4 * actions[3]) / 2
    loss.backward()
    assert values.grad.tolist() == [0, 0, actions[2] / 2, actions[3] / 2]
    ranking = values.detach().clone().requires_grad_()
    semantic_valid = torch.tensor([0, 1, 1, 1]).bool()
    loss = FastWAM._masked_policy_guard_mean(ranking * rank_mask, semantic_valid)
    assert loss.item() == pytest.approx((2 + 3 * ranks[2] + 4 * ranks[3]) / 3)
    loss.backward()
    assert ranking.grad.tolist() == pytest.approx([0, 1 / 3, ranks[2] / 3, ranks[3] / 3])


@pytest.mark.parametrize("positive_only", [False, True])
def test_causal_ranking_has_exactly_one_trainable_side(positive_only):
    correct = torch.tensor([2.0], requires_grad=True)
    wrong = torch.tensor([1.0], requires_grad=True)
    loss = ablation.causal_ranking_per_sample(
        0.5,
        correct,
        wrong,
        positive_only=positive_only,
    ).sum()
    loss.backward()
    if positive_only:
        assert correct.grad.tolist() == [1.0]
        assert wrong.grad is None
    else:
        assert correct.grad is None
        assert wrong.grad.tolist() == [-1.0]


@pytest.mark.parametrize("objective", [33, 34, 35, 36, 37])
def test_routed_ranking_gradient_contract(objective):
    correct = torch.tensor([2.0, 2.0], requires_grad=True)
    wrong = torch.tensor([1.0, 1.0], requires_grad=True)
    loss = ablation.routed_causal_ranking_per_sample(
        0.5,
        correct,
        wrong,
        objective=objective,
        corrective=torch.tensor([False, True]),
    ).sum()
    loss.backward()
    if objective in {34, 35, 36, 37}:
        assert correct.grad.tolist() == [1.0, 1.0]
        assert wrong.grad is None
    else:
        assert correct.grad.tolist() == [0.0, 1.0]
        assert wrong.grad.tolist() == [-1.0, 0.0]


def test_routed_ranking_rejects_wrong_corrective_shape():
    with pytest.raises(ValueError, match="Corrective ranking mask"):
        ablation.routed_causal_ranking_per_sample(
            0.5,
            torch.ones(2),
            torch.ones(2),
            objective=34,
            corrective=torch.ones(2, 1),
        )


def test_paired_semantic_contrast_only_trains_deployed_context():
    injected = torch.tensor(
        [[[1.0, 1.0], [1.0, 1.0]]], requires_grad=True
    )
    correct = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], requires_grad=True)
    wrong = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss, correct_similarity, wrong_similarity = (
        ablation.paired_semantic_contrast_per_sample(
            injected,
            correct,
            mask,
            wrong,
            mask,
            margin=0.1,
        )
    )
    assert correct_similarity.item() == pytest.approx(2**-0.5)
    assert wrong_similarity.item() == pytest.approx(2**-0.5)
    loss.sum().backward()
    assert injected.grad is not None and injected.grad.abs().sum() > 0
    assert correct.grad is None
    assert wrong.grad is None


@pytest.mark.parametrize("margin", [0.0, 2.0])
def test_paired_semantic_contrast_rejects_unsafe_margin(margin):
    tokens = torch.ones(1, 1, 2)
    mask = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="margin"):
        ablation.paired_semantic_contrast_per_sample(
            tokens, tokens, mask, tokens, mask, margin=margin
        )


def test_action_violation_semantic_mask_is_detached_and_respects_route():
    ranking = torch.tensor([0.2, 0.3, 0.0, 0.4], requires_grad=True)
    selected = ablation.action_violation_semantic_mask(
        torch.tensor([True, True, True, False]),
        torch.tensor([True, False, True, True]),
        ranking,
    )
    assert selected.tolist() == [True, False, False, False]
    assert not selected.requires_grad


def test_action_violation_semantic_weights_interpolate_without_route_gradient():
    ranking = torch.tensor([0.2, 0.3, 0.0, 0.4], requires_grad=True)
    weights = ablation.action_violation_semantic_weights(
        torch.tensor([True, True, True, False]),
        torch.tensor([True, False, True, True]),
        ranking,
        non_violation_weight=0.5,
    )
    assert weights.tolist() == [1.0, 0.5, 0.5, 0.0]
    assert not weights.requires_grad

    semantic_loss = torch.ones(4, requires_grad=True)
    (semantic_loss * weights).sum().backward()
    assert semantic_loss.grad.tolist() == [1.0, 0.5, 0.5, 0.0]
    assert ranking.grad is None


@pytest.mark.parametrize("weight", [-0.1, 0.0, 1.0, 1.1, float("nan")])
def test_action_violation_semantic_weights_reject_unsafe_interpolation(weight):
    with pytest.raises(ValueError, match="non-violation weight"):
        ablation.action_violation_semantic_weights(
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([0.1]),
            non_violation_weight=weight,
        )


@pytest.mark.parametrize("kind", [None, [0, 0], [0, 9], [0, 1.0], [[0, 1]]])
def test_bad_or_missing_verification_fails_closed(kind):
    with pytest.raises(ValueError):
        ablation.loss_multipliers("mask_lift_corrective", torch.tensor([0, 1]), kind)


def test_legacy_control_and_metadata():
    masks = ablation.loss_multipliers("none", torch.tensor([0, 1]))
    assert masks[0].all() and masks[1].all()
    assert masks[2].tolist() == [-1, -1]
    assert ablation.checkpoint_mode({}) == "none"
    for mode in ablation.MODES:
        assert ablation.checkpoint_mode({
            "eraf_cf_ablation": mode, "eraf_cf_ablation_contract": ablation.CONTRACT,
        }) == mode
    for metadata in (
        {"eraf_cf_ablation": "mask_lift_corrective"},
        {"eraf_cf_ablation_contract": ablation.CONTRACT},
        {"eraf_cf_ablation": "typo", "eraf_cf_ablation_contract": ablation.CONTRACT},
    ):
        with pytest.raises(ValueError):
            ablation.checkpoint_mode(metadata)
    assert ablation.verification_code("target_lift") == 1
    assert ablation.verification_code("counterfactual_goal") == 2
    with pytest.raises(ValueError):
        ablation.verification_code(None)
    with pytest.raises(ValueError, match="V9.30"):
        model_for(objective=28, ablation="mask_lift_corrective")


def test_historical_v930_checkpoint_without_ablation_metadata(warm_checkpoint, tmp_path):
    student = model_for()
    student.prepare_trainable_parameters()
    student.load_checkpoint(warm_checkpoint)
    saved = tmp_path / "control.pt"
    student.save_checkpoint(saved, step=250)
    payload = torch.load(saved, weights_only=False)
    payload["architecture_metadata"].pop("eraf_cf_ablation")
    payload["architecture_metadata"].pop("eraf_cf_ablation_contract")
    torch.save(payload, saved)
    model_for().load_checkpoint(saved)
    with pytest.raises(ValueError, match="eraf_cf_ablation"):
        model_for(ablation="mask_lift_corrective").load_checkpoint(saved)


def control_config(tmp_path):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=["task=libero_eraf_safe_gain_v930_2cam224"])
    cfg.resume = str(tmp_path / "v928/step_010000.pt")
    cfg.output_dir = str(tmp_path / "control")
    cfg.data.train.dataset_dirs = [str(tmp_path / "libero_10_no_noops_lerobot"), str(tmp_path / "closed_native")]
    cfg.data.train.pgc_counterfactual_dataset_dirs = [str(tmp_path / "historical"), str(tmp_path / "strict")]
    cfg.data.train.pgc_closed_loop_corrective_dataset_dirs = [str(tmp_path / "corrective")]
    cfg.data.train.pgc_entity_relation_sidecar_dirs = [str(tmp_path / f"sidecar{i}") for i in range(5)]
    cfg.data.train.pretrained_norm_stats = str(tmp_path / "stats.json")
    cfg.data.train.text_embedding_cache_dir = str(tmp_path / "cache")
    return cfg


def test_launch_clones_every_training_setting_and_keeps_same_warm_source(tmp_path):
    cfg = control_config(tmp_path)
    # Non-default values must survive; this runner must not silently reconstruct
    # a mostly matching task from shell defaults.
    cfg.weight_decay = 0.0042
    cfg.num_workers = 1
    cfg.model.policy_guard.entity_relation_grounding.action_causal_margin = 0.0123
    path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, path)
    original = OmegaConf.to_container(load_control(path), resolve=True)
    for mode in ablation.MODES:
        run = tmp_path / mode
        derived = training_config(load_control(path), mode, run)
        actual = OmegaConf.to_container(derived, resolve=True)
        actual.pop("hydra")
        expected = copy.deepcopy(original)
        expected["model"]["policy_guard"]["entity_relation_grounding"]["cf_ablation"] = mode
        expected["output_dir"] = str(run)
        expected["wandb"]["name"] = f"v930_cf_{mode}"
        assert actual == expected
        OmegaConf.save(derived, tmp_path / "ablation_train.yaml")
        # Exercise Hydra's real config-path loading with no task-group defaults.
        with initialize_config_dir(config_dir=str(tmp_path), version_base=None):
            composed = compose(config_name="ablation_train")
        assert composed.resume == cfg.resume
        assert composed.model.policy_guard.entity_relation_grounding.cf_ablation == mode
        cmd = training_command(run, 4, derived)
        assert cmd[:3] == ["bash", "scripts/train_zero1.sh", "4"]
        assert f"output_dir={run}" in cmd
    # Exercise the actual train_zero1 ordering: injected overrides, experiment
    # overrides, then --config-path/--config-name. No split positional groups.
    run.mkdir()
    OmegaConf.save(derived, run / "ablation_train.yaml")
    probe = tmp_path / "hydra_probe.py"
    probe.write_text(
        'import hydra\nfrom omegaconf import OmegaConf\n'
        '@hydra.main(config_path=None, config_name=None, version_base="1.3")\n'
        'def main(cfg):\n    print(OmegaConf.to_yaml(cfg))\n'
        'if __name__ == "__main__":\n    main()\n', encoding="utf-8",
    )
    cli = subprocess.run([
        sys.executable, str(probe), "output_dir=ignored", "wandb.name=ignored",
        *cmd[3:],
    ], cwd=tmp_path, capture_output=True, text=True)
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert f"cf_ablation: {mode}" in cli.stdout
    assert f"output_dir: {run}" in cli.stdout
    before = set(tmp_path.iterdir())
    env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "") + os.pathsep + str(ROOT / "src")}
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/train_libero_eraf_cf_ablation.py"),
        str(path), "4", "unit-test-dry-run", "--dry-run",
    ], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DRY_RUN" in result.stdout and "mask_corrective_ranking" in result.stdout
    assert set(tmp_path.iterdir()) == before
    assert not (ROOT / "runs/libero_eraf_cf_ablation/unit-test-dry-run").exists()
    cfg.model.policy_guard.entity_relation_grounding.cf_ablation = "mask_lift_corrective"
    OmegaConf.save(cfg, path)
    with pytest.raises(ValueError, match="unablated"):
        load_control(path)


@pytest.mark.parametrize("tamper_between_arms", [False, True])
def test_runner_sequence_keeps_warm_source_and_refuses_overwrite(tmp_path, monkeypatch, tamper_between_arms):
    """Mock GPU jobs only; exercise real config files, hashes and dispatch order."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    cfg = control_config(tmp_path)
    warm = Path(cfg.resume)
    warm.parent.mkdir(parents=True)
    warm.write_bytes(b"fixed warm checkpoint fixture")
    warm_sha = runner.file_sha256(warm)
    dataset = Path(cfg.data.train.pgc_closed_loop_corrective_dataset_dirs[0])
    index = dataset / "meta/pgc_v8_closed_loop/index.json"
    index.parent.mkdir(parents=True)
    index.write_text('{"fixture": true}', encoding="utf-8")
    source = tmp_path / "control.yaml"
    OmegaConf.save(cfg, source)
    monkeypatch.setattr(runner, "__file__", str(repo / "scripts/runner.py"))
    monkeypatch.chdir(repo)
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setattr(sys, "argv", ["runner", str(source), "4", "sequence"])
    from fastwam.datasets import pgc_libero
    monkeypatch.setattr(pgc_libero, "load_pgc_closed_loop_corrective_index", lambda _: {
        0: {"verification_kind": "target_lift"},
        1: {"verification_kind": "counterfactual_goal"},
    })
    seen = []

    def fake_job(command, *, env, check):
        assert check is True
        if env.get("ERAF_SAFE_GAIN_PREFLIGHT_ONLY") == "1":
            assert command[1] == "scripts/train_libero_eraf_safe_gain_v930.sh"
            assert command[4] == str(warm)
            seen.append("preflight")
            return subprocess.CompletedProcess(command, 0)
        directory = Path(command[command.index("--config-path") + 1])
        derived = OmegaConf.load(directory / "ablation_train.yaml")
        assert derived.resume == str(warm)
        mode = derived.model.policy_guard.entity_relation_grounding.cf_ablation
        seen.append(mode)
        output = directory / "checkpoints/weights/step_000250.pt"
        output.parent.mkdir(parents=True)
        torch.save({
            "step": 250,
            "architecture_metadata": {
                "eraf_cf_ablation": mode,
                "eraf_cf_ablation_contract": ablation.CONTRACT,
                "eraf_preservation_source": {"checkpoint_sha256": warm_sha},
            },
        }, output)
        if tamper_between_arms:
            index.write_text('{"fixture": "changed"}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_job)
    output_root = repo / "runs/libero_eraf_cf_ablation/sequence"
    if tamper_between_arms:
        with pytest.raises(RuntimeError, match="changed between experiment arms"):
            runner.main()
        assert seen == ["preflight", "mask_lift_corrective"]
        assert not (output_root / "mask_corrective_ranking").exists()
    else:
        runner.main()
        assert seen == ["preflight", *runner.DEFAULT_MODES]
        plan = json.loads((output_root / "experiment.json").read_text())
        assert plan["warm_checkpoint_sha256"] == warm_sha
        assert plan["effective_batch_size"] == 16
        assert plan["corrective_verification_counts"] == {"target_lift": 1, "counterfactual_goal": 1}
    with pytest.raises(FileExistsError):
        runner.main()
