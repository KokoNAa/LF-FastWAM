"""Multi-GPU matched Correct/Shuffle/Counterfactual manager for RoboTwin."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from experiments.robotwin.language_interventions import (
    load_intervention_manifest,
    normalize_condition,
    select_intervention_pair,
)
from scripts.summarize_robotwin_cis import load_job_output, summarize_run


SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
POLL_INTERVAL_SEC = 2
TERMINATE_TIMEOUT_SEC = 10
SUPPORTED_TASK_CONFIGS = ("demo_clean", "demo_randomized")


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                "`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., "
                f"got: {ckpt_path}"
            )
        return f"{parts[runs_idx + 1]}_{parts[runs_idx + 2]}"
    return ckpt_path.stem


def _unique_strings(values: Any, *, field: str) -> list[str]:
    result: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if not value:
            raise ValueError(f"{field} contains an empty value")
        if value in result:
            raise ValueError(f"{field} contains duplicate value {value!r}")
        result.append(value)
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
        "EVALUATION.instruction_condition",
        "EVALUATION.language_intervention_manifest",
        "EVALUATION.cis_tasks",
        "EVALUATION.cis_conditions",
        "EVALUATION.cis_task_configs",
        "EVALUATION.resume_completed",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [
        override
        for override in HydraConfig.get().overrides.task
        if not _is_blocked_override(override)
    ]


@dataclass(frozen=True)
class JobSpec:
    source_task: str
    task_config: str
    condition: str

    @property
    def key(self) -> str:
        return f"{self.source_task}/{self.task_config}/{self.condition}"


@dataclass
class RunningJob:
    spec: JobSpec
    gpu_id: str
    process: subprocess.Popen[str]


@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml"
)
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.is_file():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.is_dir():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")
    manifest_value = cfg.EVALUATION.language_intervention_manifest
    if manifest_value is None or not str(manifest_value).strip():
        raise ValueError(
            "EVALUATION.language_intervention_manifest is required by the CIS manager."
        )
    manifest_path = _resolve_path(str(manifest_value), base=PROJECT_ROOT)
    pairs = load_intervention_manifest(manifest_path, robotwin_root=robotwin_root)

    configured_tasks = [str(value) for value in cfg.EVALUATION.cis_tasks]
    if configured_tasks:
        tasks = _unique_strings(configured_tasks, field="EVALUATION.cis_tasks")
    elif cfg.EVALUATION.task_name is not None and str(cfg.EVALUATION.task_name).strip():
        tasks = [str(cfg.EVALUATION.task_name).strip()]
    else:
        tasks = [pair.source_task for pair in pairs]
    for task in tasks:
        select_intervention_pair(pairs, source_task=task)

    conditions = _unique_strings(
        [normalize_condition(value) for value in cfg.EVALUATION.cis_conditions],
        field="EVALUATION.cis_conditions",
    )
    task_configs = _unique_strings(
        cfg.EVALUATION.cis_task_configs,
        field="EVALUATION.cis_task_configs",
    )
    unsupported_configs = sorted(set(task_configs) - set(SUPPORTED_TASK_CONFIGS))
    if unsupported_configs:
        raise ValueError(
            f"Unsupported CIS task configs: {unsupported_configs}; "
            f"expected a subset of {list(SUPPORTED_TASK_CONFIGS)}"
        )

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if num_gpus <= 0 or max_tasks_per_gpu <= 0:
        raise ValueError("MULTIRUN.num_gpus and max_tasks_per_gpu must be positive")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        gpu_ids = [value.strip() for value in visible_devices.split(",")]
        if any(not value for value in gpu_ids) or len(gpu_ids) != num_gpus:
            raise ValueError(
                f"MULTIRUN.num_gpus={num_gpus}, but CUDA_VISIBLE_DEVICES="
                f"{visible_devices!r} exposes {len(gpu_ids)} entries"
            )
    else:
        gpu_ids = [str(index) for index in range(num_gpus)]
    expected_episodes = int(cfg.EVALUATION.eval_num_episodes)
    if expected_episodes <= 0:
        raise ValueError("EVALUATION.eval_num_episodes must be positive")

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_tag = output_dir.name
    if not run_tag:
        raise ValueError(f"Invalid EVALUATION.output_dir: {output_dir}")
    run_root = PROJECT_ROOT / "evaluate_results" / "robotwin" / ckpt_tag / run_tag
    run_root.mkdir(parents=True, exist_ok=True)
    manager_log = run_root / "cis_manager.log"
    state_path = run_root / "cis_manager_state.json"
    failed_path = run_root / "failed_jobs.txt"
    summary_prefix = run_root / "cis_summary"

    all_specs = [
        JobSpec(task, task_config, condition)
        for task in tasks
        for task_config in task_configs
        for condition in conditions
    ]
    resume_completed = bool(cfg.EVALUATION.resume_completed)
    pending: deque[JobSpec] = deque()
    completed: dict[str, dict[str, Any]] = {}
    failed: dict[str, dict[str, Any]] = {}
    running: list[RunningJob] = []
    extra_overrides = _collect_worker_overrides()

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    def job_dir(spec: JobSpec) -> Path:
        return run_root / spec.source_task / spec.task_config / spec.condition

    def load_complete_job(spec: JobSpec) -> dict[str, Any] | None:
        try:
            summary, _ = load_job_output(
                job_dir(spec),
                expected_episodes=expected_episodes,
                expected_source_task=spec.source_task,
                expected_task_config=spec.task_config,
                expected_condition=spec.condition,
                expected_checkpoint=ckpt_path,
            )
        except (FileNotFoundError, ValueError, KeyError):
            return None
        return summary

    def write_state() -> None:
        payload = {
            "format": "robotwin_cis_manager_state_v1",
            "checkpoint": str(ckpt_path),
            "manifest": str(manifest_path),
            "run_root": str(run_root),
            "expected_episodes_per_job": expected_episodes,
            "jobs": [
                {
                    **asdict(spec),
                    "key": spec.key,
                    "status": (
                        "completed"
                        if spec.key in completed
                        else (
                            "failed"
                            if spec.key in failed
                            else (
                                "running"
                                if any(item.spec == spec for item in running)
                                else "pending"
                            )
                        )
                    ),
                    "failure": failed.get(spec.key),
                }
                for spec in all_specs
            ],
        }
        state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with failed_path.open("w", encoding="utf-8") as handle:
            for spec in all_specs:
                if spec.key in failed:
                    handle.write(
                        f"{spec.key},return_code={failed[spec.key]['return_code']},"
                        f"reason={failed[spec.key]['reason']}\n"
                    )

    for spec in all_specs:
        cached = load_complete_job(spec) if resume_completed else None
        if cached is None:
            pending.append(spec)
        else:
            completed[spec.key] = cached
            log(f"resume skip complete job={spec.key}")

    def gpu_running_count(gpu_id: str) -> int:
        return sum(
            item.gpu_id == gpu_id and item.process.poll() is None for item in running
        )

    def build_command(spec: JobSpec, gpu_id: str) -> list[str]:
        command = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={ckpt_path}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={spec.source_task}",
            f"EVALUATION.task_config={spec.task_config}",
            f"EVALUATION.instruction_condition={spec.condition}",
            f"EVALUATION.language_intervention_manifest={manifest_path}",
            f"EVALUATION.output_dir={output_dir}",
        ]
        command.extend(extra_overrides)
        return command

    def launch(spec: JobSpec, gpu_id: str) -> RunningJob:
        command = build_command(spec, gpu_id)
        log(f"launch job={spec.key} gpu={gpu_id} cmd={' '.join(command)}")
        process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), text=True)
        return RunningJob(spec=spec, gpu_id=gpu_id, process=process)

    def fill_gpu(gpu_id: str) -> None:
        while pending and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            running.append(launch(pending.popleft(), gpu_id))

    def terminate_running() -> None:
        for item in running:
            if item.process.poll() is None:
                item.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for item in running:
            if item.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                item.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                item.process.kill()
                item.process.wait()

    log(
        f"CIS manager start jobs={len(all_specs)} pending={len(pending)} "
        f"gpus={gpu_ids} max_tasks_per_gpu={max_tasks_per_gpu} run_root={run_root}"
    )
    write_state()
    for gpu_id in gpu_ids:
        fill_gpu(gpu_id)

    try:
        while running:
            progressed = False
            for item in list(running):
                return_code = item.process.poll()
                if return_code is None:
                    continue
                progressed = True
                running.remove(item)
                if return_code != 0:
                    failed[item.spec.key] = {
                        "return_code": int(return_code),
                        "reason": "worker_failed",
                    }
                    log(
                        f"failed job={item.spec.key} gpu={item.gpu_id} "
                        f"return_code={return_code}"
                    )
                else:
                    summary = load_complete_job(item.spec)
                    if summary is None:
                        failed[item.spec.key] = {
                            "return_code": int(return_code),
                            "reason": "incomplete_or_invalid_output",
                        }
                        log(f"failed output validation job={item.spec.key}")
                    else:
                        completed[item.spec.key] = summary
                        log(
                            f"done job={item.spec.key} gpu={item.gpu_id} "
                            f"selected_goal_rate={summary['selected_goal_success_rate']:.4f}"
                        )
                write_state()
                fill_gpu(item.gpu_id)
            if not progressed:
                time.sleep(POLL_INTERVAL_SEC)
    except (KeyboardInterrupt, SystemExit):
        log("CIS manager interrupted; terminating active workers")
        terminate_running()
        write_state()
        raise

    write_state()
    if completed:
        payload = summarize_run(
            run_root,
            output_prefix=summary_prefix,
            expected_episodes=expected_episodes,
            expected_tasks=tasks,
            expected_task_configs=task_configs,
            expected_conditions=conditions,
            require_complete=not failed,
        )
        log(
            f"summary complete={payload['complete']} "
            f"jobs={payload['completed_jobs']}/{payload['expected_jobs']} "
            f"path={summary_prefix.with_suffix('.json')}"
        )

    if failed:
        raise RuntimeError(f"{len(failed)} RoboTwin CIS jobs failed; see {failed_path}")
    log("CIS manager finished successfully")


if __name__ == "__main__":
    main()
