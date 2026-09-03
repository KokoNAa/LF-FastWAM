#!/usr/bin/env python3
"""Train the matched V9.39 full-goal ERAF + shared-LoRA ablation.

This run changes one trainable boundary relative to V9.38: the shared
Video/Action LoRA is optimized together with the ERAF token compressor and
context injector.  The complete ERAF representation, GoalGraph, gain gate,
released Base, data mixture, fixed V9.28 interface teacher, objective, seed,
learning rate, effective batch, and single-path deployment remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from omegaconf import OmegaConf

from fastwam.models.wan22 import eraf_preservation as preservation
from fastwam.utils import cf_ablation

try:
    from scripts.train_libero_eraf_full_goal_refresh_v938 import (
        MODEL_OBJECTIVE,
        validate_full_goal_binding,
        refreshed_config,
    )
    from scripts.train_libero_eraf_safe_gain_v930_from_config import launch_spec
    from scripts.train_libero_eraf_safe_gain_v932 import (
        TARGET_EFFECTIVE_BATCH_SIZE,
        file_sha256,
        load_v930_b,
        source_cf_ablation_contract,
        training_command,
    )
except ModuleNotFoundError:
    from train_libero_eraf_full_goal_refresh_v938 import (
        MODEL_OBJECTIVE,
        validate_full_goal_binding,
        refreshed_config,
    )
    from train_libero_eraf_safe_gain_v930_from_config import launch_spec
    from train_libero_eraf_safe_gain_v932 import (
        TARGET_EFFECTIVE_BATCH_SIZE,
        file_sha256,
        load_v930_b,
        source_cf_ablation_contract,
        training_command,
    )


METHOD_VERSION = "V9.39"
MAX_STEPS = 25
SAVE_EVERY = 5
REQUIRED_SAVE_STEPS = (5, 10, 15, 20, 25)


def joint_config(source, output_dir, dataset, sidecar, gpus):
    cfg = refreshed_config(source, output_dir, dataset, sidecar, gpus)
    cfg.max_steps = MAX_STEPS
    cfg.save_every = SAVE_EVERY
    eraf = cfg.model.policy_guard.entity_relation_grounding
    eraf.safe_gain_injector_training_steps = MAX_STEPS
    eraf.safe_gain_gate_calibration_steps = 0
    eraf.safe_gain_lora_joint_training = True
    cfg.wandb.name = "v939_full_goal_eraf_shared_lora_joint_25steps"
    return cfg


def _changed_keys(before, after):
    import torch

    if set(before) != set(after):
        raise RuntimeError("V9.39 checkpoint tensor keys changed unexpectedly.")
    return sorted(
        name for name in before if not torch.equal(before[name], after[name])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="completed V9.30-B config.yaml")
    parser.add_argument("dataset", type=Path, help="35-episode full-goal dataset")
    parser.add_argument("sidecar", type=Path, help="sidecar bound to that dataset")
    parser.add_argument("gpus", type=int)
    parser.add_argument("run_tag")
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error("GPU count must be positive.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_tag):
        parser.error("run_tag must be a plain unique directory name.")
    if os.environ.get("NNODES", "1") != "1":
        parser.error("V9.39 supports one machine only.")

    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    source_path = args.config.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    sidecar = args.sidecar.expanduser().resolve()
    source = load_v930_b(source_path)
    binding = validate_full_goal_binding(source, dataset, sidecar, args.coverage)
    root = repo / "runs/libero_eraf_full_goal_lora_joint_v939_2cam224" / args.run_tag
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {root}")
    cfg = joint_config(source, root, dataset, sidecar, args.gpus)
    command = training_command(root, args.gpus, cfg, objective=MODEL_OBJECTIVE)
    plan = {
        "method_version": METHOD_VERSION,
        "model_objective_version": MODEL_OBJECTIVE,
        "change_scope": "unfreeze_shared_video_action_lora_only",
        "trainable_scope": preservation.LORA_JOINT_ACTION_SCOPE,
        "frozen_scope": "released_base_eraf_goalgraph_gain_gate_teacher",
        "warm_v928_checkpoint": str(Path(source.resume).resolve()),
        "source_v930_template_config": str(source_path),
        "source_v930_template_config_sha256": file_sha256(source_path),
        "source_v930_cf_ablation_contract": source_cf_ablation_contract(
            source.model.policy_guard.entity_relation_grounding
        ),
        "full_goal_binding": binding,
        "output": str(root),
        "gpus": args.gpus,
        "effective_batch_size": (
            args.gpus * int(cfg.batch_size) * int(cfg.gradient_accumulation_steps)
        ),
        "steps": MAX_STEPS,
        "save_steps": list(REQUIRED_SAVE_STEPS),
        "learning_rate": float(cfg.learning_rate),
        "future_video_flow_weight": 0.0,
        "video_lora_gradient_contract": (
            "action_objective_through_differentiable_video_cache_no_future_video_flow"
        ),
        "semantic_contract": (
            cf_ablation.SOFT_ACTION_VIOLATION_SEMANTIC_CONTRAST_CONTRACT
        ),
        "command": command,
    }
    if plan["effective_batch_size"] != TARGET_EFFECTIVE_BATCH_SIZE:
        raise ValueError("V9.39 must preserve effective batch size 12.")
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("DRY_RUN: validation complete; no CUDA or training job.", flush=True)
        return

    preflight, env = launch_spec(source_path, args.gpus, os.environ, source_objective=30)
    env.update(
        {
            "ERAF_SAFE_GAIN_PREFLIGHT_ONLY": "1",
            "PYTHON_BIN": sys.executable,
            "PATH": str(Path(sys.executable).parent)
            + os.pathsep
            + env.get("PATH", ""),
            "ERAF_SAFE_GAIN_MAX_STEPS": str(MAX_STEPS),
        }
    )
    subprocess.run(preflight, env=env, check=True)
    env.pop("ERAF_SAFE_GAIN_PREFLIGHT_ONLY")

    plan["warm_v928_checkpoint_sha256"] = file_sha256(source.resume)
    root.mkdir(parents=True, exist_ok=False)
    # The model objective remains 37; only the explicit V9.39 trainability flag
    # distinguishes this artifact from the V9.38 interface-only control.
    OmegaConf.save(cfg, root / "v937_train.yaml")
    (root / "experiment.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[TRAIN_START] output={root} trainable={plan['trainable_scope']}", flush=True)
    env["RUN_ID"] = args.run_tag
    subprocess.run(command, env=env, check=True)

    import torch

    checkpoint = root / "checkpoints/weights/step_000025.pt"
    initial = torch.load(source.resume, map_location="cpu", weights_only=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("architecture_metadata") or {}
    lora_contract = metadata.get("eraf_expert_lora_training_contract") or {}
    if (
        payload.get("step") != MAX_STEPS
        or metadata.get("eraf_grounding_objective_version") != MODEL_OBJECTIVE
        or metadata.get("eraf_safe_gain_lora_joint_training") is not True
        or metadata.get("eraf_action_trainable_scope")
        != preservation.LORA_JOINT_ACTION_SCOPE
        or metadata.get("eraf_role_adapter_trainable_scope")
        != preservation.LORA_JOINT_ROLE_SCOPE
        or lora_contract.get("baseline_lora_trainable") is not True
        or lora_contract.get("future_video_flow") != 0.0
        or lora_contract.get("video_lora_gradient_contract")
        != plan["video_lora_gradient_contract"]
        or metadata.get("eraf_preservation_source", {}).get("checkpoint_sha256")
        != plan["warm_v928_checkpoint_sha256"]
    ):
        raise RuntimeError(f"Final V9.39 checkpoint contract mismatch: {checkpoint}")
    preservation.validate_teacher_payload(payload)

    initial_lora = initial.get("eraf_shared_expert_lora") or {}
    final_lora = payload.get("eraf_shared_expert_lora") or {}
    changed_lora = _changed_keys(initial_lora, final_lora)
    if not changed_lora:
        raise RuntimeError("V9.39 shared Video/Action LoRA did not change.")
    changed_lora_by_expert = {
        expert: [
            name
            for name in changed_lora
            if name.startswith(f"mixtures.{expert}.")
        ]
        for expert in ("video", "action")
    }
    if any(not names for names in changed_lora_by_expert.values()):
        raise RuntimeError(
            "V9.39 must update both shared experts: "
            f"{ {name: len(keys) for name, keys in changed_lora_by_expert.items()} }"
        )
    initial_guard = initial.get("policy_guard") or {}
    final_guard = payload.get("policy_guard") or {}
    changed_guard = _changed_keys(initial_guard, final_guard)
    changed_guard_modules = sorted({name.split(".", 1)[0] for name in changed_guard})
    if not changed_guard or not set(changed_guard_modules).issubset(
        preservation.INTERFACE_NAMES
    ):
        raise RuntimeError(
            "V9.39 changed frozen guard tensors or failed to train its interface: "
            f"{changed_guard_modules}"
        )
    for step in REQUIRED_SAVE_STEPS:
        expected = root / "checkpoints/weights" / f"step_{step:06d}.pt"
        if not expected.is_file():
            raise FileNotFoundError(f"Missing V9.39 checkpoint: {expected}")
    print(
        "[TRAIN_DONE] "
        f"checkpoint={checkpoint} changed_lora="
        f"{ {name: len(keys) for name, keys in changed_lora_by_expert.items()} } "
        f"changed_guard_modules={changed_guard_modules}; evaluation not started",
        flush=True,
    )


if __name__ == "__main__":
    main()
