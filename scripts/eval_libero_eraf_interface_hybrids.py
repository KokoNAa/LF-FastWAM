#!/usr/bin/env python3
"""Run actual new_old and old_new closed-loop CF rollouts on three GPUs.

This is an explicit causal interface intervention: one Action Expert is reused,
there is no training, no gate change, no simulator-truth policy input, and no
same-state probe overhead. Default is a read-only plan; --execute writes only a
new output directory.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from omegaconf import OmegaConf

from scripts.eval_libero_eraf_interface_probe import (
    REPO,
    job_config,
    load_cases,
    result_for,
    sha256,
)


HYBRID_DRIVERS = ("new_old", "old_new")


def _success_trials(root, suite, task, trials, manifest_sha256):
    result = result_for(root, suite, task)
    if (
        int(result["total_episodes"]) != trials
        or result.get("language_intervention_manifest_sha256") != manifest_sha256
    ):
        raise ValueError(f"Reference protocol mismatch: {root}, task{task}.")
    return sorted(int(x) for x in result["success_episodes"])


def build_summary(root, plan):
    summary = {
        "jobs": {},
        "selected_cases": [],
        "driver_scores": {
            driver: {
                "common_loss_recovered": 0,
                "common_loss_total": 0,
                "common_gain_retained": 0,
                "common_gain_total": 0,
            }
            for driver in HYBRID_DRIVERS
        },
        "interpretation": (
            "Hybrid drivers were executed closed-loop. Agreement is a paired "
            "protocol outcome, not a same-state trajectory claim."
        ),
    }
    outcomes = {}
    for job in plan["jobs"]:
        result = result_for(Path(root) / job["name"], plan["suite"], job["task_id"])
        if int(result["total_episodes"]) != plan["trials_per_task"]:
            raise RuntimeError(f"Incomplete hybrid task: {job['name']}")
        if result.get("language_intervention_manifest_sha256") != plan["manifest_sha256"]:
            raise RuntimeError(f"Manifest mismatch in hybrid task: {job['name']}")
        interface = result.get("interface_probe", {})
        if (
            interface.get("executed_driver") != job["driver"]
            or not interface.get("executed_driver_is_hybrid")
            or interface.get("probe_trials") != []
            or interface.get("latency_includes_diagnostic_predictions")
        ):
            raise RuntimeError(f"Hybrid execution provenance failed: {job['name']}")
        successes = sorted(int(x) for x in result["success_episodes"])
        outcome = {trial: trial in successes for trial in range(plan["trials_per_task"])}
        outcomes[(job["driver"], job["task_id"])] = outcome
        warm_set = set(job["warm_success_trials"])
        candidate_set = set(job["candidate_success_trials"])
        summary["jobs"][job["name"]] = {
            "success_trials": successes,
            "successes": len(successes),
            "trials": plan["trials_per_task"],
            "agreement_with_warm": sum(
                (trial in successes) == (trial in warm_set)
                for trial in range(plan["trials_per_task"])
            ),
            "agreement_with_candidate": sum(
                (trial in successes) == (trial in candidate_set)
                for trial in range(plan["trials_per_task"])
            ),
        }

    for case in plan["cases"]:
        task, trial, group = int(case["task_id"]), int(case["trial_id"]), case["group"]
        warm = bool(case["warm_success"])
        candidate = bool(case["candidate_success"])
        for driver in HYBRID_DRIVERS:
            value = outcomes[(driver, task)][trial]
            item = {
                "driver": driver,
                "task_id": task,
                "trial_id": trial,
                "group": group,
                "success": value,
                "warm_success": warm,
                "candidate_success": candidate,
                "matches_warm": value == warm,
                "matches_candidate": value == candidate,
            }
            summary["selected_cases"].append(item)
            score = summary["driver_scores"][driver]
            if group == "common_loss":
                score["common_loss_total"] += 1
                score["common_loss_recovered"] += int(value)
            elif group == "common_gain":
                score["common_gain_total"] += 1
                score["common_gain_retained"] += int(value)
            else:
                raise ValueError(f"Unknown diagnostic case group: {group!r}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--warm-checkpoint", type=Path, required=True)
    parser.add_argument("--warm-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        default=REPO / "configs/eval/libero10_cf_interface_probe_cases.json",
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("Use unique nonnegative GPU IDs.")
    root = args.output.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {root}")
    source = OmegaConf.load(args.source_config)
    candidate = Path(str(source.ckpt)).expanduser().resolve()
    warm = args.warm_checkpoint.expanduser().resolve()
    manifest = Path(
        str(source.EVALUATION.language_intervention_manifest)
    ).expanduser().resolve()
    for path in (candidate, warm, manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    trials = int(source.EVALUATION.num_trials)
    suite, raw_cases = load_cases(args.cases, trials)
    tasks = sorted({int(case["task_id"]) for case in raw_cases})
    if list(source.MULTIRUN.task_suite_names) != [suite]:
        raise ValueError("Hybrid case suite differs from source manager config.")
    manifest_sha256 = sha256(manifest)
    references = {}
    for task in tasks:
        references[("warm", task)] = _success_trials(
            args.warm_results, suite, task, trials, manifest_sha256
        )
        references[("candidate", task)] = _success_trials(
            args.candidate_results, suite, task, trials, manifest_sha256
        )
    cases = []
    for raw in raw_cases:
        case = dict(raw)
        task, trial = int(case["task_id"]), int(case["trial_id"])
        if case.get("group") not in {"common_loss", "common_gain"}:
            raise ValueError(f"Unknown diagnostic case group: {case.get('group')!r}")
        case["warm_success"] = trial in references[("warm", task)]
        case["candidate_success"] = trial in references[("candidate", task)]
        expected = (
            (True, False) if case["group"] == "common_loss" else (False, True)
        )
        if (case["warm_success"], case["candidate_success"]) != expected:
            raise ValueError(
                f"Historical selected-case contract changed: task{task}/trial{trial}."
            )
        cases.append(case)
    plan = {
        "contract": "explicit_closed_loop_hybrid_interface_causal_rollout_v1",
        "source_config": str(args.source_config.resolve()),
        "source_config_sha256": sha256(args.source_config),
        "warm_checkpoint": str(warm),
        "warm_sha256": sha256(warm),
        "candidate_checkpoint": str(candidate),
        "candidate_sha256": sha256(candidate),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "suite": suite,
        "cases": cases,
        "gpus": args.gpus,
        "trials_per_task": trials,
        "episodes": len(HYBRID_DRIVERS) * len(tasks) * trials,
        "jobs": [],
    }
    configs = {}
    for driver in HYBRID_DRIVERS:
        for task in tasks:
            name = f"{driver}_task{task}"
            gpu = args.gpus[len(plan["jobs"]) % len(args.gpus)]
            cfg = job_config(
                source,
                warm=warm,
                suite=suite,
                task=task,
                trials=[],
                driver=driver,
                output=root / name,
                gpu=gpu,
                stride=1,
            )
            cfg.ckpt = str(candidate)
            cfg.EVALUATION.interface_probe.execute_hybrid_driver = True
            command = [
                sys.executable,
                "experiments/libero/eval_libero_single.py",
                "--config-path",
                str(root / name),
                "--config-name",
                "hybrid_eval",
            ]
            plan["jobs"].append(
                {
                    "name": name,
                    "driver": driver,
                    "task_id": task,
                    "gpu": gpu,
                    "warm_success_trials": references[("warm", task)],
                    "candidate_success_trials": references[("candidate", task)],
                    "command": command,
                }
            )
            configs[name] = cfg
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print(
            "PLAN ONLY: no CUDA, subprocess, checkpoint load, or output writes. "
            "Add --execute to run."
        )
        return

    import torch
    from fastwam.models.wan22.eraf_interface_probe import validate_checkpoints

    validate_checkpoints(
        torch.load(warm, map_location="cpu", weights_only=False),
        torch.load(candidate, map_location="cpu", weights_only=False),
        plan["warm_sha256"],
    )
    root.mkdir(parents=True, exist_ok=False)
    (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name, cfg in configs.items():
        (root / name).mkdir()
        OmegaConf.save(cfg, root / name / "hybrid_eval.yaml")
    failed = threading.Event()

    def run_gpu(gpu, driver):
        for job in (
            item
            for item in plan["jobs"]
            if item["gpu"] == gpu and item["driver"] == driver
        ):
            if failed.is_set():
                return
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                PYTHONUNBUFFERED="1",
            )
            env["PYTHONPATH"] = (
                str(REPO / "src")
                + os.pathsep
                + str(REPO)
                + os.pathsep
                + env.get("PYTHONPATH", "")
            )
            print(f"[START] {job['name']} GPU{gpu}", flush=True)
            try:
                with (root / job["name"] / "worker.log").open("x") as log:
                    result = subprocess.run(
                        job["command"],
                        cwd=REPO,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            except Exception:
                failed.set()
                raise
            print(f"[EXIT] {job['name']} rc={result.returncode}", flush=True)
            if result.returncode:
                failed.set()
                return

    for driver in HYBRID_DRIVERS:
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            list(pool.map(lambda gpu: run_gpu(gpu, driver), args.gpus))
        if failed.is_set():
            raise SystemExit(
                f"FAILED: queued work stopped; inspect {root}/*/worker.log."
            )
    summary = build_summary(root, plan)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(
        f"[ALL_DONE] {len(plan['jobs'])} hybrid tasks; "
        f"{plan['episodes']} episodes; ROOT={root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
