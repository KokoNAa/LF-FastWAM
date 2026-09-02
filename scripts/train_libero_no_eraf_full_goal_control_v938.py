#!/usr/bin/env python3
"""Train the data-matched V9.38 no-ERAF full-goal control for 25 steps.

The control starts from the exact formal no-ERAF LoRA policy frozen inside
the V9.28 teacher, disables the complete policy guard/ERAF path, and updates
only the shared Video+Action LoRA on the same four-way data recipe used by
V9.38.  ERAF sidecars are retained only for audited sampling and paired
language provenance; no sidecar tensor enters the model forward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping

from omegaconf import OmegaConf

try:
    from scripts.train_libero_eraf_full_goal_refresh_v938 import (
        validate_full_goal_binding,
    )
    from scripts.train_libero_eraf_safe_gain_v930_from_config import launch_spec
    from scripts.train_libero_eraf_safe_gain_v932 import (
        TARGET_EFFECTIVE_BATCH_SIZE,
        file_sha256,
        load_v930_b,
    )
except ModuleNotFoundError:
    from train_libero_eraf_full_goal_refresh_v938 import (
        validate_full_goal_binding,
    )
    from train_libero_eraf_safe_gain_v930_from_config import launch_spec
    from train_libero_eraf_safe_gain_v932 import (
        TARGET_EFFECTIVE_BATCH_SIZE,
        file_sha256,
        load_v930_b,
    )


METHOD_VERSION = "V9.38-no-ERAF-FG25"
EXPECTED_BASELINE_STEP = 8500
MAX_STEPS = 25
SAVE_EVERY = 5
LEARNING_RATE = 2.0e-6
REQUIRED_SAVE_STEPS = (5, 10, 15, 20, 25)
FORMAL_LORA = {
    "rank": 16,
    "alpha": 16.0,
    "dropout": 0.05,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_formal_lora_config(raw: Mapping) -> dict:
    config = OmegaConf.to_container(OmegaConf.create(raw), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("No-ERAF baseline has no LoRA configuration.")
    control = config.get("paired_language_control")
    if (
        config.get("enabled") is not True
        or set(config.get("experts") or []) != {"video", "action"}
        or list(config.get("extra_trainable_patterns") or [])
        or not isinstance(control, dict)
        or control.get("enabled") is not True
        or control.get("bidirectional_supervision") is not True
    ):
        raise ValueError(
            "No-ERAF full-goal control requires the formal bidirectional "
            "shared Video+Action LoRA configuration."
        )
    for name, expected in FORMAL_LORA.items():
        if not math.isclose(
            float(config.get(name, float("nan"))),
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"No-ERAF LoRA {name} mismatch: "
                f"expected={expected} got={config.get(name)!r}."
            )
    return config


def validate_policy_identity(source, baseline_path: Path) -> tuple[dict, dict]:
    import torch

    baseline_path = baseline_path.expanduser().resolve()
    v928_path = Path(str(source.resume)).expanduser().resolve()
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    v928 = torch.load(v928_path, map_location="cpu", weights_only=False)
    if (
        baseline.get("format") != "fastwam_lora_adapter_v1"
        or int(baseline.get("step", -1)) != EXPECTED_BASELINE_STEP
        or baseline.get("transition_contract") is not None
    ):
        raise ValueError(
            "The data-matched control must start from the formal no-ERAF "
            f"step{EXPECTED_BASELINE_STEP} adapter."
        )
    lora_config = validate_formal_lora_config(baseline.get("lora_config") or {})
    baseline_state = baseline.get("mot_trainable")
    v928_state = v928.get("eraf_shared_expert_lora")
    if not isinstance(baseline_state, dict) or not baseline_state:
        raise ValueError("No-ERAF baseline has no shared LoRA tensors.")
    if not isinstance(v928_state, dict) or not v928_state:
        raise ValueError("V9.28 teacher has no frozen shared LoRA tensors.")
    if set(baseline_state) != set(v928_state):
        raise ValueError("No-ERAF baseline and V9.28 shared LoRA keys differ.")
    mismatched = [
        name
        for name in baseline_state
        if tuple(baseline_state[name].shape) != tuple(v928_state[name].shape)
        or not torch.equal(baseline_state[name], v928_state[name])
    ]
    if mismatched:
        raise ValueError(
            "No-ERAF baseline is not the exact policy frozen in V9.28: "
            f"mismatched={mismatched[:20]}"
        )
    baseline_base = Path(str(baseline.get("base_checkpoint", ""))).expanduser()
    v928_base = Path(str(v928.get("base_checkpoint", ""))).expanduser()
    if (
        not baseline_base.is_absolute()
        or not v928_base.is_absolute()
        or baseline_base.resolve() != v928_base.resolve()
    ):
        raise ValueError(
            "No-ERAF baseline and V9.28 must reference the same absolute "
            "released FastWAM Base checkpoint."
        )
    identity = {
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_sha256": _sha256(baseline_path),
        "baseline_step": int(baseline["step"]),
        "v928_checkpoint": str(v928_path),
        "v928_checkpoint_sha256": _sha256(v928_path),
        "released_base": str(baseline_base.resolve()),
        "shared_lora_tensor_count": len(baseline_state),
        "shared_lora_exact_match": True,
    }
    return lora_config, identity


def control_config(
    source,
    output_dir: Path,
    baseline_path: Path,
    lora_config: Mapping,
    dataset: Path,
    sidecar: Path,
    gpus: int,
):
    cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
    per_accumulation_batch = int(gpus) * int(cfg.batch_size)
    if (
        gpus < 1
        or per_accumulation_batch <= 0
        or TARGET_EFFECTIVE_BATCH_SIZE % per_accumulation_batch
    ):
        raise ValueError(
            f"Cannot preserve effective batch {TARGET_EFFECTIVE_BATCH_SIZE} "
            f"with gpus={gpus} and per-GPU batch={cfg.batch_size}."
        )
    cfg.gradient_accumulation_steps = (
        TARGET_EFFECTIVE_BATCH_SIZE // per_accumulation_batch
    )
    cfg.data.train.pgc_closed_loop_corrective_dataset_dirs = [
        str(dataset.resolve())
    ]
    sidecars = list(cfg.data.train.pgc_entity_relation_sidecar_dirs)
    if len(sidecars) != 5:
        raise ValueError("No-ERAF FG25 expects the five-sidecar V9.30 recipe.")
    sidecars[-1] = str(sidecar.resolve())
    cfg.data.train.pgc_entity_relation_sidecar_dirs = sidecars
    cfg.data.train.pgc_v9_safe_gain_counterfactual_replay = True
    cfg.model.policy_guard.enabled = False
    cfg.model.lora = OmegaConf.create(
        OmegaConf.to_container(OmegaConf.create(lora_config), resolve=True)
    )
    cfg.resume = str(baseline_path.resolve())
    cfg.weight_only_start_step = None
    cfg.output_dir = str(output_dir.resolve())
    cfg.max_steps = MAX_STEPS
    cfg.save_every = SAVE_EVERY
    cfg.log_every = 5
    cfg.learning_rate = LEARNING_RATE
    cfg.wandb.name = "v938_no_eraf_full_goal_data_control_25steps"
    cfg.hydra = {
        "job": {"chdir": False},
        "run": {"dir": "."},
        "output_subdir": None,
    }
    return cfg


def training_command(root: Path, gpus: int, cfg) -> list[str]:
    return [
        "bash",
        "scripts/train_zero1.sh",
        str(gpus),
        f"output_dir={cfg.output_dir}",
        f"wandb.name={cfg.wandb.name}",
        "--config-path",
        str(root.resolve()),
        "--config-name",
        "no_eraf_fg25_train",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="completed V9.30-B config.yaml")
    parser.add_argument("baseline", type=Path, help="formal no-ERAF step8500 adapter")
    parser.add_argument("dataset", type=Path, help="35-episode full-goal dataset")
    parser.add_argument("sidecar", type=Path, help="sidecar bound to the dataset")
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
        parser.error("NoERAF-FG25 supports one machine only.")

    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    source_path = args.config.expanduser().resolve()
    baseline_path = args.baseline.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    sidecar = args.sidecar.expanduser().resolve()
    source = load_v930_b(source_path)
    binding = validate_full_goal_binding(
        source, dataset, sidecar, args.coverage
    )
    lora_config, identity = validate_policy_identity(source, baseline_path)
    root = repo / "runs/libero_no_eraf_full_goal_v938_2cam224" / args.run_tag
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {root}")
    cfg = control_config(
        source,
        root,
        baseline_path,
        lora_config,
        dataset,
        sidecar,
        args.gpus,
    )
    command = training_command(root, args.gpus, cfg)
    plan = {
        "method_version": METHOD_VERSION,
        "change_scope": "no_eraf_full_goal_data_matched_control",
        "policy_identity": identity,
        "source_v930_template_config": str(source_path),
        "source_v930_template_config_sha256": file_sha256(source_path),
        "full_goal_binding": binding,
        "output": str(root),
        "gpus": args.gpus,
        "effective_batch_size": (
            args.gpus * int(cfg.batch_size) * int(cfg.gradient_accumulation_steps)
        ),
        "steps": int(cfg.max_steps),
        "save_steps": list(REQUIRED_SAVE_STEPS),
        "learning_rate": float(cfg.learning_rate),
        "trainable_scope": "shared_video_action_lora_only",
        "eraf_forward_enabled": False,
        "sidecar_usage": "sampling_and_language_provenance_only",
        "command": command,
    }
    if plan["effective_batch_size"] != TARGET_EFFECTIVE_BATCH_SIZE:
        raise ValueError("NoERAF-FG25 must preserve effective batch size 12.")
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("DRY_RUN: validation complete; no training job.", flush=True)
        return

    preflight, env = launch_spec(
        source_path, args.gpus, os.environ, source_objective=30
    )
    env.update(
        {
            "ERAF_SAFE_GAIN_PREFLIGHT_ONLY": "1",
            "PYTHON_BIN": sys.executable,
            "PATH": str(Path(sys.executable).parent)
            + os.pathsep
            + env.get("PATH", ""),
        }
    )
    subprocess.run(preflight, env=env, check=True)
    env.pop("ERAF_SAFE_GAIN_PREFLIGHT_ONLY")

    root.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, root / "no_eraf_fg25_train.yaml")
    (root / "experiment.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "[TRAIN_START] no_eraf=true full_goal=35 steps=25 "
        f"output={root}",
        flush=True,
    )
    env["RUN_ID"] = args.run_tag
    subprocess.run(command, env=env, check=True)

    checkpoint = root / "checkpoints/weights/step_000025.pt"
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    final_config = validate_formal_lora_config(payload.get("lora_config") or {})
    initial = torch.load(
        baseline_path, map_location="cpu", weights_only=False
    ).get("mot_trainable")
    final = payload.get("mot_trainable")
    if (
        payload.get("format") != "fastwam_lora_adapter_v1"
        or int(payload.get("step", -1)) != MAX_STEPS
        or payload.get("transition_contract") is not None
        or Path(str(payload.get("base_checkpoint", ""))).resolve()
        != Path(identity["released_base"]).resolve()
        or set(initial or {}) != set(final or {})
        or not final_config["paired_language_control"]["enabled"]
    ):
        raise RuntimeError(
            f"Final no-ERAF FG25 checkpoint contract mismatch: {checkpoint}"
        )
    changed = sum(
        int(not torch.equal(initial[name], final[name])) for name in initial
    )
    if changed <= 0:
        raise RuntimeError("No no-ERAF LoRA tensors changed during training.")
    for step in REQUIRED_SAVE_STEPS:
        expected = root / "checkpoints/weights" / f"step_{step:06d}.pt"
        if not expected.is_file():
            raise FileNotFoundError(f"Missing no-ERAF FG25 checkpoint: {expected}")
    print(
        f"[TRAIN_DONE] checkpoint={checkpoint} changed_lora_tensors={changed}; "
        "evaluation not started",
        flush=True,
    )


if __name__ == "__main__":
    main()
