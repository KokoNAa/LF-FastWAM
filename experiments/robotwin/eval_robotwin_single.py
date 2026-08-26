"""
RobotWin single-task evaluation entrypoint (Hydra).

Features:
- Read `configs/sim_robotwin.yaml`.
- Check or create the symlink:
  `RoboTwin/policy/fastwam -> experiments/robotwin/fastwam`.
- Forward config overrides to the official RoboTwin entrypoint
  `script/eval_policy.py` and save logs.

Common arguments:
- `ckpt`: path to the FastWAM checkpoint (required).
- `EVALUATION.task_name`: task name to evaluate (required).
- `gpu_id`: sets `CUDA_VISIBLE_DEVICES`.

Examples:
1) Minimal run
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock

2) Run with more evaluation overrides
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock \
     EVALUATION.task_config=demo_randomized \
     EVALUATION.replan_steps=4 \
     EVALUATION.num_inference_steps=4 \
     gpu_id=0
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from experiments.robotwin.language_interventions import (
    load_intervention_manifest,
    normalize_condition,
    select_intervention_pair,
)

POLICY_NAME = "fastwam_policy"


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(cfg.EVALUATION.dataset_stats_path, base=PROJECT_ROOT)
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)

    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _ensure_policy_symlink(robotwin_root: Path, policy_source_dir: Path) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")

    policy_target = policy_root / POLICY_NAME
    source_resolved = policy_source_dir.resolve()

    if not policy_target.exists() and not policy_target.is_symlink():
        policy_target.symlink_to(source_resolved, target_is_directory=True)
        return policy_target

    if policy_target.is_symlink():
        target_resolved = policy_target.resolve()
        if target_resolved != source_resolved:
            raise RuntimeError(
                f"Policy symlink conflict: {policy_target} -> {target_resolved}, "
                f"expected -> {source_resolved}"
            )
        return policy_target

    raise RuntimeError(
        f"Path already exists and is not a symlink: {policy_target}. "
        "Please handle it manually to avoid overriding existing policy files."
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")

    condition = normalize_condition(cfg.EVALUATION.instruction_condition)

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    manifest_path = _resolve_optional_path(
        cfg.EVALUATION.language_intervention_manifest,
        base=PROJECT_ROOT,
    )
    if manifest_path is None and condition != "correct":
        raise ValueError(
            "RoboTwin shuffled/counterfactual evaluation requires "
            "EVALUATION.language_intervention_manifest."
        )
    if manifest_path is not None:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Language intervention manifest not found: {manifest_path}"
            )
        intervention_pairs = load_intervention_manifest(
            manifest_path, robotwin_root=robotwin_root
        )
        select_intervention_pair(
            intervention_pairs,
            source_task=str(cfg.EVALUATION.task_name),
        )

    policy_source_dir = (PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME).resolve()
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {policy_source_dir}")

    _ensure_policy_symlink(robotwin_root=robotwin_root, policy_source_dir=policy_source_dir)

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = (
        PROJECT_ROOT
        / "evaluate_results"
        / "robotwin"
        / ckpt_tag
        / run_ts
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_tag = f"{str(cfg.EVALUATION.task_name)}_{str(cfg.EVALUATION.task_config)}"
    if manifest_path is not None:
        log_tag += f"_{condition}"
    log_file = run_output_dir / (
        f"eval_{log_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    robotwin_eval_base = run_output_dir / str(cfg.EVALUATION.task_name)
    if manifest_path is not None:
        robotwin_eval_base = (
            robotwin_eval_base / str(cfg.EVALUATION.task_config) / condition
        )

    sim_cfg_path = (PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve()
    sim_task = HydraConfig.get().runtime.choices.get("task")
    resolved_sim_cfg_path = run_output_dir / f"eval_config_{log_tag}.yaml"
    # The RoboTwin policy runs in a child process.  Persist the already-composed
    # Hydra config so model overrides (B0/M1/TC/PGC) cannot be lost when that
    # process constructs FastWAM.
    OmegaConf.save(config=cfg, f=str(resolved_sim_cfg_path))

    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)

    overrides: list[str] = []
    _append_override(overrides, "task_name", cfg.EVALUATION.task_name)
    _append_override(overrides, "task_config", cfg.EVALUATION.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", cfg.seed)
    _append_override(overrides, "policy_name", cfg.EVALUATION.policy_name)
    _append_override(overrides, "instruction_type", cfg.EVALUATION.instruction_type)
    _append_override(overrides, "instruction_condition", condition)
    _append_override(
        overrides,
        "language_intervention_manifest",
        None if manifest_path is None else str(manifest_path),
    )
    _append_override(
        overrides,
        "matched_episode_records_path",
        cfg.EVALUATION.matched_episode_records_path,
    )
    _append_override(overrides, "project_root", str(PROJECT_ROOT))
    _append_override(overrides, "eval_num_episodes", cfg.EVALUATION.eval_num_episodes)

    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", sim_task)
    _append_override(
        overrides,
        "resolved_sim_cfg_path",
        str(resolved_sim_cfg_path),
    )
    _append_override(overrides, "eval_output_dir", str(robotwin_eval_base))
    _append_override(overrides, "mixed_precision", cfg.mixed_precision)
    _append_override(overrides, "device", cfg.EVALUATION.device)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "action_horizon", cfg.EVALUATION.action_horizon)
    _append_override(overrides, "replan_steps", cfg.EVALUATION.replan_steps)
    _append_override(overrides, "num_inference_steps", cfg.EVALUATION.num_inference_steps)
    _append_override(overrides, "sigma_shift", cfg.EVALUATION.sigma_shift)
    _append_override(overrides, "text_cfg_scale", cfg.EVALUATION.text_cfg_scale)
    _append_override(overrides, "negative_prompt", cfg.EVALUATION.negative_prompt)
    _append_override(overrides, "rand_device", cfg.EVALUATION.rand_device)
    _append_override(overrides, "tiled", cfg.EVALUATION.tiled)
    _append_override(overrides, "timing_enabled", cfg.EVALUATION.timing_enabled)
    _append_override(
        overrides,
        "skip_get_obs_within_replan",
        cfg.EVALUATION.skip_get_obs_within_replan,
    )

    cmd = [
        sys.executable,
        "-u",
        "script/eval_policy.py",
        "--config",
        f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        *overrides,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(robotwin_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"RoboTwin evaluation failed with return code {return_code}. Log: {log_file}")

    print(f"Evaluation finished successfully. Log saved to: {log_file}")


if __name__ == "__main__":
    main()
