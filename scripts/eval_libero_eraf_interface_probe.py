#!/usr/bin/env python3
"""Run same-cache interface probes on old/new driver trajectories, on 3 GPUs.

Default is a read-only plan. --execute creates a NEW output directory, never
trains or rewrites checkpoints, and schedules at most one process per GPU.
All original trials run in order; probes are limited to the requested cases.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cases(path, trials):
    payload = json.loads(Path(path).read_text())
    cases = payload["cases"]
    keys = [(int(c["task_id"]), int(c["trial_id"])) for c in cases]
    if not cases or len(keys) != len(set(keys)) or any(t < 0 or not 0 <= e < trials for t, e in keys):
        raise ValueError("Cases must be unique task/trial pairs inside the original trial range.")
    return str(payload["suite"]), cases


def job_config(source, *, warm, suite, task, trials, driver, output, gpu, stride):
    cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=False))
    evaluation = cfg.EVALUATION
    if (evaluation.instruction_condition != "counterfactual"
            or not evaluation.get("counterfactual_diagnostics", False)
            or cfg.model.policy_guard.gate_mode != "counterfactual"
            or cfg.model.policy_guard.entity_relation_grounding.grounding_objective_version != 30):
        raise ValueError("Source must be the completed forced-CF V9.30 candidate manager_config.yaml.")
    if evaluation.get("interface_probe") is not None:
        raise ValueError("Source must not already be an interface-probe config.")
    for name in ("visualize_future_video", "entity_relation_oracle", "entity_relation_shadow_audit",
                 "entity_relation_oracle_phase_servo", "entity_relation_stateless_replan_ablation",
                 "entity_relation_completion_only_memory_ablation", "entity_relation_diagnostics"):
        if evaluation.get(name, False):
            raise ValueError(f"Diagnostic config unexpectedly enables {name}.")
    for name in ("closed_loop_capture_dir", "entity_relation_closed_loop_capture_dir"):
        if evaluation.get(name) not in (None, "", "null"):
            raise ValueError(f"Do not mix interface probes with {name}.")
    evaluation.task_suite_name, evaluation.task_id = suite, task
    evaluation.output_dir = str(output)
    evaluation.interface_probe = {
        "warm_checkpoint": str(warm), "driver": driver,
        "trials": trials, "stride_replans": stride,
    }
    cfg.gpu_id = gpu
    # Operational destinations only. Model/data/protocol fields are inherited.
    cfg.output_dir = str(output)
    cfg.hydra = {"job": {"chdir": False}, "run": {"dir": str(output)}, "output_subdir": None}
    return cfg


def result_for(root, suite, task):
    files = list(Path(root).rglob(f"gpu*_task{task}_results.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one task{task} result under {root}, got {len(files)}.")
    result = json.loads(files[0].read_text())
    if result["task_suite"] != suite or int(result["task_id"]) != task:
        raise ValueError(f"Result identity mismatch: {files[0]}")
    return result


def summarize(root, plan):
    report = {"jobs": {}, "cases": [], "interpretation":
              "Only predictions within each probe share a state. Hybrid success is unmeasured."}
    for job in plan["jobs"]:
        result = result_for(Path(root) / job["name"], plan["suite"], job["task_id"])
        successes = sorted(int(x) for x in result["success_episodes"])
        report["jobs"][job["name"]] = {
            "success_trials": successes,
            "reference_success_trials": job["reference_success_trials"],
            "protocol_outcomes_reproduced": successes == job["reference_success_trials"],
        }
        if int(result["total_episodes"]) != plan["trials_per_task"]:
            raise RuntimeError(f"Incomplete task: {job['name']}")
        records = [json.loads(line) for line in (Path(root) / job["name"] / "interface_probe/records.jsonl").read_text().splitlines()]
        for trial in job["probe_trials"]:
            probes = [r for r in records if r["kind"] == "probe" and r["trial"] == trial]
            if not probes or not all(r["driver_repeat_validated"] for r in probes):
                raise RuntimeError(f"Missing/unvalidated probes: {job['name']} trial{trial}")
            state_records = [r for r in records if r["kind"] == "simulator" and r["trial"] == trial]
            first_goal = next((r["policy_step"] for r in state_records
                               if all(p["holds"] for p in r["predicates"]["counterfactual_goal_state"])), None)
            mean_delta = {variant: sum(r["variants"][variant]["action_normalized_vs_old_old"]["rms"] for r in probes) / len(probes)
                          for variant in ("new_old", "old_new", "new_new")}
            report["cases"].append({"job": job["name"], "trial": trial,
                                     "probe_count": len(probes), "first_cf_goal_step": first_goal,
                                     "mean_action_rms_vs_old_old": mean_delta})
    report["all_driver_outcomes_reproduced"] = all(x["protocol_outcomes_reproduced"] for x in report["jobs"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--warm-checkpoint", type=Path, required=True)
    parser.add_argument("--warm-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=REPO / "configs/eval/libero10_cf_interface_probe_cases.json")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--stride-replans", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if (not args.gpus or len(set(args.gpus)) != len(args.gpus)
            or min(args.gpus) < 0 or args.stride_replans < 1):
        parser.error("Use unique nonnegative GPU IDs and a positive probe stride.")
    root = args.output.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {root}")
    source = OmegaConf.load(args.source_config)
    candidate = Path(str(source.ckpt)).expanduser().resolve()
    warm = args.warm_checkpoint.expanduser().resolve()
    manifest = Path(str(source.EVALUATION.language_intervention_manifest)).expanduser().resolve()
    for path in (candidate, warm, manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    trials = int(source.EVALUATION.num_trials)
    suite, cases = load_cases(args.cases, trials)
    tasks = sorted({int(c["task_id"]) for c in cases})
    if list(source.MULTIRUN.task_suite_names) != [suite]:
        raise ValueError("Probe case suite differs from source manager config.")
    plan = {"source_config": str(args.source_config.resolve()),
            "source_config_sha256": sha256(args.source_config),
            "candidate_checkpoint": str(candidate), "candidate_sha256": sha256(candidate),
            "warm_checkpoint": str(warm), "warm_sha256": sha256(warm),
            "manifest": str(manifest), "manifest_sha256": sha256(manifest),
            "cases": cases, "suite": suite, "gpus": args.gpus,
            "trials_per_task": trials, "episodes": 2 * len(tasks) * trials,
            "jobs": []}
    configs = {}
    for driver, reference in (("old_old", args.warm_results), ("new_new", args.candidate_results)):
        for task in tasks:
            name = f"{driver}_task{task}"
            probe_trials = sorted(int(c["trial_id"]) for c in cases if int(c["task_id"]) == task)
            gpu = args.gpus[len(plan["jobs"]) % len(args.gpus)]
            cfg = job_config(source, warm=warm, suite=suite, task=task, trials=probe_trials,
                             driver=driver, output=root / name, gpu=gpu, stride=args.stride_replans)
            cfg.ckpt = str(candidate)
            previous = result_for(reference, suite, task)
            if (int(previous["total_episodes"]) != trials
                    or previous.get("language_intervention_manifest_sha256") != plan["manifest_sha256"]):
                raise ValueError(f"Reference trial count / manifest mismatch: {reference}, task{task}")
            command = [sys.executable, "experiments/libero/eval_libero_single.py",
                       "--config-path", str(root / name), "--config-name", "probe_eval"]
            plan["jobs"].append({"name": name, "gpu": gpu, "task_id": task,
                                 "probe_trials": probe_trials, "command": command,
                                 "reference_success_trials": sorted(int(x) for x in previous["success_episodes"])})
            configs[name] = cfg
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("PLAN ONLY: no CUDA, subprocess, checkpoint load, or output writes. Add --execute to run.")
        return
    # Validate both checkpoints once on CPU before starting any GPU worker.
    import torch
    from fastwam.models.wan22.eraf_interface_probe import validate_checkpoints
    validate_checkpoints(torch.load(warm, map_location="cpu", weights_only=False),
                         torch.load(candidate, map_location="cpu", weights_only=False), plan["warm_sha256"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name, cfg in configs.items():
        (root / name).mkdir()
        OmegaConf.save(cfg, root / name / "probe_eval.yaml")
    failed = threading.Event()

    def run_gpu(gpu, driver):
        for job in (j for j in plan["jobs"] if j["gpu"] == gpu and j["name"].startswith(driver + "_")):
            if failed.is_set():
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
            print(f"[START] {job['name']} GPU{gpu}", flush=True)
            try:
                with (root / job["name"] / "worker.log").open("x") as log:
                    result = subprocess.run(job["command"], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            except Exception:
                failed.set()
                raise
            print(f"[EXIT] {job['name']} rc={result.returncode}", flush=True)
            if result.returncode:
                failed.set()
                return

    for driver in ("old_old", "new_new"):
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            list(pool.map(lambda gpu: run_gpu(gpu, driver), args.gpus))
        if failed.is_set():
            raise SystemExit(f"FAILED: queued work stopped; inspect {root}/*/worker.log. Existing running workers were allowed to finish.")
        if driver == "old_old":
            calibration = {}
            for job in (j for j in plan["jobs"] if j["name"].startswith("old_old_")):
                result = result_for(root / job["name"], suite, job["task_id"])
                actual = sorted(int(x) for x in result["success_episodes"])
                if actual != job["reference_success_trials"] or int(result["total_episodes"]) != trials:
                    calibration[job["name"]] = {"expected": job["reference_success_trials"], "actual": actual}
            if calibration:
                (root / "calibration_failure.json").write_text(json.dumps(calibration, indent=2) + "\n")
                raise SystemExit("CALIBRATION FAILED: old_old did not reproduce warm trial outcomes; candidate wave NOT launched. Inspect calibration_failure.json before attributing module effects.")
    report = summarize(root, plan)
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[ALL_DONE] {len(plan['jobs'])} tasks; {plan['episodes']} driver episodes; ROOT={root}", flush=True)
    if not report["all_driver_outcomes_reproduced"]:
        print("WARNING: prior driver outcomes did not fully reproduce. Do NOT attribute prior lost trials to a specific module yet.", flush=True)


if __name__ == "__main__":
    main()
