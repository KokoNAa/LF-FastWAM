#!/usr/bin/env python3
"""Fit both languages on audited RoboTwin states, with an optional noise anchor.

Two independent Action-LoRA runs start from the same step1000 adapter. This is
a small repair experiment, not full four-pool training or a task-success claim.
The default is a read-only plan; --execute starts CUDA workers on fresh paths.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.probe_robotwin_denoising import DEFAULT_PAIRS, resolve_source_probe
from scripts.probe_robotwin_no_eraf import load_probe_policy, read_json, sha256, write_json

FORMAT = "robotwin_same_state_action_repair_v1"
ARMS = ("paired_flow", "paired_flow_anchor", "paired_flow_anchor_delta")


def state_key(state):
    return state["profile"], state["pair_id"], state["episode_index"]


def evaluation_steps(steps, every):
    return sorted({0, steps, *range(every, steps + 1, every)})


def build_plan(args):
    from experiments.robotwin.no_eraf_probe import require_pair

    source_repair = getattr(args, "source_repair", None)
    if source_repair:
        previous_root = Path(source_repair).expanduser().resolve()
        previous = read_json(previous_root / "plan.json")
        previous_summary = read_json(previous_root / "summary.json")
        if (previous.get("format") != FORMAT or previous_summary.get("format") != FORMAT
                or previous_summary.get("complete") is not True
                or previous["initial_model"] != args.initial_model):
            raise ValueError("source-repair must be complete and use the same initial model.")
        root = resolve_source_probe(previous["source_probe"])
    else:
        root = resolve_source_probe(args.source_probe)
    source = read_json(root / "plan.json")
    summary = read_json(root / "summary.json")
    if (source.get("format") != "robotwin_no_eraf_same_state_probe_v1"
            or summary.get("format") != source["format"] or summary.get("complete") is not True):
        raise ValueError("Expected a completed, audited same-state probe.")
    if source["action_horizon"] != 32 or source["inference_steps"] != 10:
        raise ValueError("Repair requires the audited 32-action / 10-denoising-step protocol.")
    if args.initial_model not in source["checkpoints"] or not args.initial_model.startswith("step"):
        raise ValueError("Repair must start from an audited LoRA checkpoint.")
    all_states = read_json(root / "states.json")
    if len({s["id"] for s in all_states}) != len(all_states):
        raise ValueError("Duplicate source states.")
    paired = [s for s in all_states if s["frame_index"] == 0 and s["dual_reference_valid"]
              and s["initial_observations_exactly_equal"] and s["expert_kind"] == "native"]
    fit = [s for s in paired if s["profile"] == "historical" and s["pair_id"] in args.pairs]
    if {s["pair_id"] for s in fit} != set(args.pairs) or not fit:
        raise ValueError("Each requested repair pair needs an identical historical initial observation.")
    pairs = {state_key(p): p for p in source["pairs"]}
    if len(pairs) != len(source["pairs"]):
        raise ValueError("Duplicate source pair provenance.")
    for state in fit:
        pair = pairs[state_key(state)]
        require_pair(pair["native"]["audit"], pair["counterfactual"]["audit"])
        if not (pair["native_in_training_split"] and pair["counterfactual_in_training_split"]
                and state["own_reference_in_training_split"]):
            raise ValueError("Both repair references must belong to the authorized historical training split.")
        for field in ("source_instruction", "counterfactual_instruction", "scene_seed"):
            if state[field] != pair[field]:
                raise ValueError(f"Repair state/provenance mismatch: {field}")
    fit_ids = {s["id"] for s in fit}
    fit_hashes = {s["observation_sha256"] for s in fit}
    guard = [s for s in paired if s["id"] not in fit_ids and s["observation_sha256"] not in fit_hashes]
    selected = [{**s, "split": "repair" if s["id"] in fit_ids else "guard"} for s in fit + guard]
    record_path = root / args.initial_model / "records.jsonl"
    records = [json.loads(line) for line in record_path.read_text().splitlines()]
    if (len(records) != len(all_states) or {r["id"] for r in records} != {s["id"] for s in all_states}
            or read_json(root / args.initial_model / "complete.json")["states"] != len(all_states)):
        raise ValueError("Incomplete initial checkpoint predictions.")
    by_id = {row["id"]: row for row in records}
    saved = {}
    for state in selected:
        row = by_id[state["id"]]
        if row["observation_sha256"] != state["observation_sha256"] or not row["dual_reference_valid"]:
            raise ValueError("Initial prediction/observation mismatch.")
        saved[state["id"]] = {"path": row["actions_file"], "sha256": sha256(row["actions_file"])}
    training_draws = {}
    normalization_states = len(fit)
    if source_repair:
        if args.train_seed != previous["train_seed"] or args.steps > previous["steps"]:
            raise ValueError("source-repair requires the original train seed and no more than its recorded steps.")
        previous_fit = {s["id"] for s in previous["states"] if s["split"] == "repair"}
        if not fit_ids <= previous_fit:
            raise ValueError("source-repair can only isolate a subset of its original repair states.")
        previous_arm = "paired_flow_anchor" if "paired_flow_anchor" in previous["arms"] else previous["arms"][0]
        training_path = previous_root / previous_arm / "training.jsonl"
        records = [json.loads(line) for line in training_path.read_text().splitlines()]
        if [r["step"] for r in records] != list(range(1, previous["steps"] + 1)):
            raise ValueError("Incomplete source-repair training draws.")
        for row in records[:args.steps]:
            draws = {d["id"]: d for d in row["draws"] if d["id"] in fit_ids}
            if set(draws) != fit_ids or len([d for d in row["draws"] if d["id"] in fit_ids]) != len(fit_ids):
                raise ValueError("Missing/duplicate source-repair state draws.")
            for draw in draws.values():
                if (not {"id", "noise_seed", "time", "scheduler_weight", "noise_sha256"} <= set(draw)
                        or not isinstance(draw["noise_seed"], int) or draw["noise_seed"] < 0
                        or any(not math.isfinite(draw[k]) or draw[k] < 0 for k in ("time", "scheduler_weight"))
                        or not isinstance(draw["noise_sha256"], str) or len(draw["noise_sha256"]) != 64):
                    raise ValueError("Invalid source-repair training draw.")
            training_draws[str(row["step"])] = draws
        normalization_states = previous.get("normalization_state_count", len(previous_fit))
        if not isinstance(normalization_states, int) or normalization_states < len(fit):
            raise ValueError("Invalid source-repair loss normalization count.")
    noise_seeds = ({d["noise_seed"] for draws in training_draws.values() for d in draws.values()}
                   if training_draws else {args.train_seed + step * len(fit) + i
                       for step in range(1, args.steps + 1) for i in range(len(fit))})
    if noise_seeds.intersection(args.eval_seeds):
        raise ValueError("Training noise seeds overlap evaluation noise seeds.")
    paths = [root / name for name in ("plan.json", "states.json", "summary.json")]
    paths += [record_path, root / args.initial_model / "complete.json"]
    if source_repair:
        for path, expected_hash in previous["source_artifact_sha256"].items():
            if sha256(path) != expected_hash:
                raise ValueError(f"Source artifact changed since the earlier repair: {path}")
        paths += [previous_root / "plan.json", previous_root / "summary.json", training_path]
    return {"format": FORMAT, "source_probe": str(root), "source_plan": source,
            "source_repair": str(previous_root) if source_repair else None,
            "source_artifact_sha256": {str(path): sha256(path) for path in paths},
            "output": str(Path(args.output).expanduser().resolve()), "initial_model": args.initial_model,
            "initial_step": int(args.initial_model[4:]), "states": selected,
            "saved_predictions": saved, "arms": args.arms, "gpus": args.gpus,
            "steps": args.steps, "eval_every": args.eval_every,
            "evaluation_steps": evaluation_steps(args.steps, args.eval_every),
            "train_seed": args.train_seed, "eval_seeds": args.eval_seeds,
            "training_draws": training_draws, "normalization_state_count": normalization_states,
            "learning_rate": args.learning_rate, "anchor_weight": args.anchor_weight,
            "conditional_anchor_gain": getattr(args, "conditional_anchor_gain", 4.),
            "anchor_objective": "paired_common_conditional_v1",
            "fixed_flow_sigmas": getattr(args, "fixed_flow_sigmas", []),
            "audit_anchor_gradients": getattr(args, "audit_anchor_gradients", False),
            "minimum_improvement": args.minimum_improvement,
            "guard_relative": args.guard_relative, "guard_absolute": args.guard_absolute,
            "scope": "Action LoRA A/B only; frozen production Video/context caches; dropout off; no ranking, Video loss, ERAF or full-goal data.",
            "interpretation": "Repair-state learnability with evaluation noise seeds excluded from training and separate regression states. Not held-out-scene generalization or closed-loop success. Both anchor arms have the same Action calls per optimizer step, twice that of paired_flow. Conditional gain reweights existing positive supervision; it adds no new labels."}


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table.")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(output="latest", search_root=None):
    if output == "latest":
        search_root = Path(search_root) if search_root is not None else REPO / "runs/robotwin_same_state_repair"
        candidates = []
        for path in search_root.glob("*/summary.json"):
            try:
                summary = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if (summary.get("format") == FORMAT and summary.get("complete") is True
                    and (path.parent / "repair_summary.csv").is_file()):
                candidates.append(path)
        if not candidates:
            raise ValueError(f"No completed repair experiment under {search_root}.")
        root = max(candidates, key=lambda path: path.stat().st_mtime).parent
    else:
        root = Path(output).expanduser().resolve()
    summary = read_json(root / "summary.json")
    if summary.get("format") != FORMAT or summary.get("complete") is not True:
        raise ValueError("Repair experiment is not complete.")
    print(f"[report] {root}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print((root / "repair_summary.csv").read_text(), end="")
    if summary.get("fixed_flow_summary"):
        print("[fixed_flow_summary] Fixed noise/timesteps; only sigma=1 has identical noisy inputs across languages.")
        print((root / "fixed_flow_summary.csv").read_text(), end="")
    return root


def worker(args):
    plan = read_json(args.plan)
    root = Path(plan["output"]) / args.arm
    root.mkdir(exist_ok=False)
    for path, expected in plan["source_artifact_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Source artifact changed after planning: {path}")
    policy, audit = load_probe_policy(plan["source_plan"], plan["initial_model"], args.gpu)
    import numpy as np
    import torch
    from experiments.robotwin.denoising_probe import capture_action_cache
    from experiments.robotwin.no_eraf_probe import CAMERAS, difference, observation_hash, typed_hash
    from experiments.robotwin.same_state_repair import (
        action_rows, anchor_gradient_audit, audit_frozen, backward_paired_anchor,
        backward_paired_flow, checkpoint_score, fixed_flow_rows,
        configure_action_lora, frozen_versions, move_cache, noise_tensor,
        predict_with_grad, repair_payload, sample_cached_actions,
    )

    model = policy.model
    initial_payload = torch.load(plan["source_plan"]["checkpoints"][plan["initial_model"]],
                                 map_location="cpu", weights_only=False)
    if initial_payload["lora_config"].get("extra_trainable_patterns"):
        raise ValueError("This experiment requires a pure inherited Video/Action LoRA adapter.")
    selected = configure_action_lora(model)
    protected = frozen_versions(model)
    audit.update(trainable_scope=plan["scope"], trainable_names=list(selected),
                 trainable_elements=sum(p.numel() for p in selected.values()),
                 parameter_dtype="float32", module_mode="eval_with_autograd")
    write_json(root / "checkpoint_audit.json", audit)
    key = policy.processor.shape_meta["action"][0]["key"]
    normalizer = policy.processor.normalizer.normalizers["action"][key]
    cached, references, observations = {}, {}, {}
    replay_rows = []

    def numpy_first(value):
        return value.detach().float().cpu().numpy()[0]

    def normalized_raw(raw):
        value = torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0)
        return numpy_first(value * normalizer.scale + normalizer.offset)

    for state in plan["states"]:
        if sha256(state["file"]) != state["sha256"]:
            raise ValueError("Fixed state changed.")
        with np.load(state["file"], allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        if observation_hash(arrays) != state["observation_sha256"]:
            raise ValueError("Fixed observation fingerprint changed.")
        previous = plan["saved_predictions"][state["id"]]
        if sha256(previous["path"]) != previous["sha256"]:
            raise ValueError("Initial saved prediction changed.")
        with np.load(previous["path"], allow_pickle=False) as data:
            saved = {key: data[key] for key in data.files}
        refs = {kind: numpy_first(normalizer.forward(torch.as_tensor(arrays[kind + "_reference"]).unsqueeze(0)))
                for kind in ("source", "target")}
        for kind in refs:
            if difference(refs[kind], saved[kind + "_reference"])["max_abs"] > 1e-5:
                raise ValueError("Expert normalization changed from the audited probe.")
        if state["split"] == "repair" and difference(refs["source"][:24], refs["target"][:24])["rms"] <= 1e-6:
            raise ValueError("Repair references must differ within the executed 24-action prefix.")
        obs = {"joint_action": {"vector": arrays["state"]}, "observation": {
            camera: {"rgb": arrays[camera]} for camera in CAMERAS}}
        observations[state["id"]], references[state["id"]] = obs, refs
        caches = {}
        for language, field in (("source", "source_instruction"), ("target", "counterfactual_instruction")):
            policy.policy_guard_state = None
            raw, capture, calls = capture_action_cache(model, lambda: policy._infer_action_chunk(obs, state[field]))
            error = difference(normalized_raw(raw), saved[language])
            if calls != 10 or error["max_abs"] > 1e-5:
                raise ValueError(f"Initial production replay differs: {state['id']}/{language}: {error}")
            # Keep CPU copies; only the current state's two caches occupy GPU memory.
            caches[language] = move_cache(capture, "cpu", clone=True)
            cached_prediction = sample_cached_actions(model, capture, policy.seed, 10)
            cache_error = difference(cached_prediction, normalized_raw(raw))
            if cache_error["max_abs"] > 1e-5:
                raise ValueError("Cached sampler differs from the production wrapper.")
            replay_rows.append({"id": state["id"], "language": language,
                                "initial_replay": error, "cached_replay": cache_error})
            del capture
        cached[state["id"]] = caches
        print(f"[cache] {args.arm} {state['split']} {state['id']}", flush=True)
    write_json(root / "initial_replay.json", replay_rows)

    fit_states = [state for state in plan["states"] if state["split"] == "repair"]
    normalization_states = plan.get("normalization_state_count", len(fit_states))
    # Confirm that removing only no_grad preserves the production predictor value.
    first = fit_states[0]
    probe_cache = move_cache(cached[first["id"]]["source"], model.device)
    probe_noise = noise_tensor((1, 32, 14), policy.seed, model)
    probe_time = torch.tensor([model.train_action_scheduler.num_train_timesteps],
                              device=model.device, dtype=model.torch_dtype)
    production = model._predict_action_noise_with_cache(
        latents_action=probe_noise, timestep_action=probe_time, **probe_cache)
    differentiable = predict_with_grad(model, probe_cache, probe_noise, probe_time)
    predictor_error = difference(numpy_first(production), numpy_first(differentiable))
    if predictor_error["max_abs"] > 1e-5:
        raise ValueError("Autograd predictor differs from production.")
    write_json(root / "autograd_forward_check.json", predictor_error)
    del probe_cache, production, differentiable, probe_noise, probe_time

    if plan.get("audit_anchor_gradients"):
        gradient_rows = []
        for state in fit_states:
            caches = move_cache(cached[state["id"]], model.device)
            refs = {kind: torch.as_tensor(value, device=model.device, dtype=model.torch_dtype).unsqueeze(0)
                    for kind, value in references[state["id"]].items()}
            noise = noise_tensor((1, 32, 14), policy.seed, model)
            gradient_rows.append({"id": state["id"], "noise_seed": policy.seed,
                                  **anchor_gradient_audit(model, caches, refs, noise)})
            if any(p.grad is not None for p in selected.values()):
                raise ValueError("Read-only gradient audit populated optimizer gradients.")
            audit_frozen(model, protected)
            del caches, refs, noise
        write_json(root / "anchor_gradient_audit.json", gradient_rows)

    optimizer = torch.optim.AdamW(list(selected.values()), lr=plan["learning_rate"], weight_decay=0.)
    baseline, scores, latest_predictions = {}, [], {}
    (root / "actions").mkdir()
    (root / "checkpoints").mkdir()

    def evaluate(step):
        rows, flow_rows = [], []
        model.eval()
        for state in plan["states"]:
            caches = move_cache(cached[state["id"]], model.device)
            refs = references[state["id"]]
            if state["split"] == "repair" and plan.get("fixed_flow_sigmas"):
                refs_tensor = {k: torch.as_tensor(v, device=model.device, dtype=model.torch_dtype).unsqueeze(0)
                               for k, v in refs.items()}
                flow_rows.extend({"arm": args.arm, "repair_step": step, "id": state["id"], **row}
                                 for row in fixed_flow_rows(model, caches, refs_tensor,
                                                           plan["eval_seeds"], plan["fixed_flow_sigmas"]))
                del refs_tensor
            for seed in plan["eval_seeds"]:
                predictions = {kind: sample_cached_actions(model, caches[kind], seed, 10)
                               for kind in ("source", "target")}
                index = (state["id"], seed)
                if step == 0:
                    baseline[index] = predictions
                latest_predictions[index] = predictions
                np.savez_compressed(root / "actions" / f"repair{step:06d}_{state['id']}_seed{seed}.npz",
                                    **predictions, source_reference=refs["source"], target_reference=refs["target"])
                for metrics in action_rows(predictions["source"], predictions["target"], refs["source"],
                                           refs["target"], baseline[index]["source"], baseline[index]["target"]):
                    rows.append({"arm": args.arm, "repair_step": step, "id": state["id"],
                                 "pair_id": state["pair_id"], "profile": state["profile"],
                                 "split": state["split"], "noise_seed": seed, **metrics})
            del caches
        score = checkpoint_score(rows, minimum_improvement=plan["minimum_improvement"],
                                 guard_relative=plan["guard_relative"], guard_absolute=plan["guard_absolute"])
        score.update(repair_step=step, checkpoint=None)
        if step:
            path = root / "checkpoints" / f"repair_{step:06d}.pt"
            payload = repair_payload(model, initial_payload, step, {
                "arm": args.arm, "source_probe": plan["source_probe"],
                "initial_checkpoint": plan["source_plan"]["checkpoints"][plan["initial_model"]],
                "initial_checkpoint_sha256": plan["source_plan"]["checkpoint_sha256"][plan["initial_model"]],
                "repair_state_ids": [s["id"] for s in fit_states], "scope": plan["scope"],
                "learning_rate": plan["learning_rate"],
                "anchor_weight": plan["anchor_weight"] if args.arm != "paired_flow" else 0.,
                "conditional_anchor_gain": plan["conditional_anchor_gain"] if args.arm == "paired_flow_anchor_delta" else 1.,
                "anchor_objective": plan.get("anchor_objective", "paired_positive_mse"),
                "normalization_state_count": normalization_states,
                "source_training_draws_reused": bool(plan.get("training_draws")),
                "source_plan": str(Path(plan["output"]) / "plan.json"), "git_commit": plan["git_commit"]})
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if set(saved["mot_trainable"]) != set(payload["mot_trainable"]) or any(
                    not torch.equal(saved["mot_trainable"][key], tensor) for key, tensor in payload["mot_trainable"].items()):
                raise ValueError("Adapter serialization round trip failed.")
            score.update(checkpoint=str(path), checkpoint_sha256=sha256(path))
            del payload, saved
        audit_frozen(model, protected)
        write_json(root / f"evaluation_{step:06d}.json", {"rows": rows, "score": score, "fixed_flow_rows": flow_rows})
        scores.append(score)
        print(f"[eval] {args.arm} repair_step={step} both_correct={score['fit_both_correct']}/{score['fit_rows']} "
              f"fit_pass={score['fit_pass']} guard_pass={score['guard_pass']} "
              f"fit_rmse={score['fit_mean_correct_rmse']:.6g}", flush=True)

    evaluate(0)
    with (root / "training.jsonl").open("x") as log:
        for step in range(1, plan["steps"] + 1):
            optimizer.zero_grad(set_to_none=True)
            draws, terms = [], []
            for index, state in enumerate(fit_states):
                caches = move_cache(cached[state["id"]], model.device)
                refs = {kind: torch.as_tensor(value, device=model.device, dtype=model.torch_dtype).unsqueeze(0)
                        for kind, value in references[state["id"]].items()}
                previous_draw = plan.get("training_draws", {}).get(str(step), {}).get(state["id"])
                seed = previous_draw["noise_seed"] if previous_draw else plan["train_seed"] + step * len(fit_states) + index
                noise = noise_tensor((1, 32, 14), seed, model)
                # Same distribution as sample_training_t, but an explicit CPU
                # generator makes draws independent of extra anchor forwards.
                u = torch.rand((1,), generator=torch.Generator(device="cpu").manual_seed(1_000_000_000 + seed))
                scheduler = model.train_action_scheduler
                time = (scheduler._phi(u, scheduler.shift) * scheduler.num_train_timesteps).to(
                    device=model.device, dtype=model.torch_dtype)
                weight = float(scheduler.training_weight(time).float().item())
                draw = {"id": state["id"], "noise_seed": seed, "time": float(time.float().item()),
                        "scheduler_weight": weight, "noise_sha256": typed_hash(numpy_first(noise))}
                if previous_draw and (any(draw[k] != previous_draw[k] for k in ("id", "noise_seed", "noise_sha256"))
                        or any(abs(draw[k] - previous_draw[k]) > 1e-7 for k in ("time", "scheduler_weight"))):
                    raise ValueError("Reused training noise/timestep differs from the original repair.")
                errors = backward_paired_flow(model, caches, refs, noise, time, weight / normalization_states)
                item = {"id": state["id"], "flow_source_mse": errors["source"], "flow_target_mse": errors["target"]}
                if args.arm != "paired_flow":
                    gain = plan["conditional_anchor_gain"] if args.arm == "paired_flow_anchor_delta" else 1.
                    anchor = backward_paired_anchor(model, caches, refs, noise,
                        plan["anchor_weight"] / normalization_states, conditional_gain=gain)
                    item.update(anchor_source_mse=anchor["source"], anchor_target_mse=anchor["target"],
                                anchor_common_mse=anchor["common_mse"],
                                anchor_conditional_mse=anchor["conditional_mse"], conditional_anchor_gain=gain)
                terms.append(item)
                draws.append(draw)
                del caches, refs, noise
            grads = [p.grad for p in selected.values() if p.grad is not None]
            if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
                raise RuntimeError("Missing or nonfinite Action LoRA gradients.")
            norm = float(torch.nn.utils.clip_grad_norm_(list(selected.values()), 1., error_if_nonfinite=True))
            if norm == 0:
                raise RuntimeError("All Action LoRA gradients are zero.")
            optimizer.step()
            audit_frozen(model, protected)
            log.write(json.dumps({"step": step, "draws": draws, "terms": terms,
                                  "gradient_norm_before_clip": norm, "parameters_with_grad": len(grads)}, allow_nan=False) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0:
                print(f"[train] {args.arm} optimizer_step={step}/{plan['steps']} grad_norm={norm:.6g}", flush=True)
            if step in plan["evaluation_steps"]:
                evaluate(step)

    # Recompute observations through the untouched production wrapper after the
    # intervention, rather than trusting a cached-only improvement.
    final_replay = []
    for state in plan["states"]:
        for language, field in (("source", "source_instruction"), ("target", "counterfactual_instruction")):
            policy.policy_guard_state = None
            raw = policy._infer_action_chunk(observations[state["id"]], state[field])
            expected = latest_predictions[(state["id"], policy.seed)][language]
            error = difference(normalized_raw(raw), expected)
            if error["max_abs"] > 1e-5:
                raise ValueError(f"Repaired production/cached replay mismatch: {state['id']}/{language}: {error}")
            final_replay.append({"id": state["id"], "language": language, **error})
    audit_frozen(model, protected)
    write_json(root / "final_production_replay.json", final_replay)
    eligible = [score for score in scores if score["checkpoint"] and score["eligible_for_closed_loop_check"]]
    best = min(eligible, key=lambda score: score["fit_mean_correct_rmse"]) if eligible else None
    write_json(root / "complete.json", {"format": FORMAT, "complete": True, "arm": args.arm,
               "steps": plan["steps"], "scores": scores, "best": best,
               "last_checkpoint": scores[-1]["checkpoint"], "frozen_parameter_versions_unchanged": True,
               "final_production_replay_passed": True, "interpretation": plan["interpretation"]})


def summarize(output):
    from statistics import fmean

    root = Path(output)
    plan = read_json(root / "plan.json")
    expected = {(state["id"], seed, horizon) for state in plan["states"]
                for seed in plan["eval_seeds"] for horizon in (24, 32)}
    summaries, results, draws_by_step, initial = [], {}, {}, {}
    flow_summary, flow_initial, gradient_audits = [], {}, {}
    fit_ids = {s["id"] for s in plan["states"] if s["split"] == "repair"}
    for arm in plan["arms"]:
        complete = read_json(root / arm / "complete.json")
        if (complete.get("format") != FORMAT or complete.get("complete") is not True
                or complete["steps"] != plan["steps"] or not complete["final_production_replay_passed"]
                or not complete["frozen_parameter_versions_unchanged"]):
            raise ValueError(f"Incomplete repair arm: {arm}")
        training = [json.loads(line) for line in (root / arm / "training.jsonl").read_text().splitlines()]
        if [row["step"] for row in training] != list(range(1, plan["steps"] + 1)):
            raise ValueError("Incomplete/duplicate optimizer steps.")
        for row in training:
            previous = draws_by_step.setdefault(row["step"], row["draws"])
            if previous != row["draws"]:
                raise ValueError("Training noise/sigma draws differ across arms.")
            if plan.get("anchor_objective") and arm != "paired_flow":
                gain = plan["conditional_anchor_gain"] if arm == "paired_flow_anchor_delta" else 1.
                for term in row["terms"]:
                    if term.get("conditional_anchor_gain") != gain:
                        raise ValueError("Logged conditional gain differs from the planned objective.")
        for step in plan["evaluation_steps"]:
            evaluation = read_json(root / arm / f"evaluation_{step:06d}.json")
            if plan.get("fixed_flow_sigmas"):
                expected_flow = {(key, seed, sigma, horizon) for key in fit_ids for seed in plan["eval_seeds"]
                                 for sigma in plan["fixed_flow_sigmas"] for horizon in (24, 32)}
                flow = evaluation["fixed_flow_rows"]
                if (len(flow) != len(expected_flow) or
                        {(r["id"], r["noise_seed"], r["sigma"], r["horizon"]) for r in flow} != expected_flow):
                    raise ValueError("Incomplete/duplicate fixed-flow evaluations.")
                fields = ("source_mse", "target_mse", "common_mse", "conditional_mse")
                for row in flow:
                    if row["arm"] != arm or row["repair_step"] != step:
                        raise ValueError("Fixed-flow evaluation identity mismatch.")
                    if any(not math.isfinite(row[f]) or row[f] < 0 for f in fields):
                        raise ValueError("Invalid fixed-flow error.")
                    if not math.isclose((row["source_mse"] + row["target_mse"]) / 2,
                                        row["common_mse"] + row["conditional_mse"], rel_tol=1e-5, abs_tol=1e-6):
                        raise ValueError("Fixed-flow error decomposition mismatch.")
                    if step == 0:
                        key = row["id"], row["noise_seed"], row["sigma"], row["horizon"]
                        previous = flow_initial.setdefault(key, row)
                        if any(abs(previous[f] - row[f]) > 1e-5 for f in (*fields, "actual_time")):
                            raise ValueError("Initial fixed-flow evaluations differ across arms.")
                groups = {}
                for row in flow:
                    groups.setdefault((row["id"], row["sigma"], row["horizon"]), []).append(row)
                for (key, sigma, horizon), group in sorted(groups.items()):
                    projections = [r["delta_projection"] for r in group]
                    flow_summary.append({"arm": arm, "repair_step": step, "id": key, "sigma": sigma,
                        "horizon": horizon, "seeds": len(group),
                        **{f: fmean(r[f] for r in group) for f in fields},
                        "delta_projection": fmean(projections) if all(x is not None for x in projections) else None})
            rows = evaluation["rows"]
            if len(rows) != len(expected) or {(r["id"], r["noise_seed"], r["horizon"]) for r in rows} != expected:
                raise ValueError("Incomplete/duplicate evaluations.")
            for row in rows:
                if row["arm"] != arm or row["repair_step"] != step:
                    raise ValueError("Evaluation belongs to another arm or checkpoint.")
                if step == 0:
                    key = row["id"], row["noise_seed"], row["horizon"]
                    previous = initial.setdefault(key, row)
                    for field in ("source_correct_rmse", "target_correct_rmse", "source_language_margin", "target_language_margin"):
                        if abs(previous[field] - row[field]) > 1e-5:
                            raise ValueError("Initial models differ across repair arms.")
            groups = {}
            for row in rows:
                groups.setdefault((row["split"], row["horizon"]), []).append(row)
            for (split, horizon), group in sorted(groups.items()):
                summaries.append({"arm": arm, "repair_step": step, "split": split, "horizon": horizon,
                    "rows": len(group), "both_correct": sum(row["both_correct"] for row in group),
                    **{field: fmean(row[field] for row in group) for field in (
                        "source_correct_rmse", "target_correct_rmse", "initial_source_rmse", "initial_target_rmse",
                        "source_language_margin", "target_language_margin")},
                    "fit_pass": evaluation["score"]["fit_pass"], "guard_pass": evaluation["score"]["guard_pass"]})
        results[arm] = {"best": complete["best"], "last_checkpoint": complete["last_checkpoint"],
                        "last_score": complete["scores"][-1]}
        if plan.get("audit_anchor_gradients"):
            gradient_audits[arm] = read_json(root / arm / "anchor_gradient_audit.json")
            if (len(gradient_audits[arm]) != len(fit_ids)
                    or {r["id"] for r in gradient_audits[arm]} != fit_ids):
                raise ValueError("Incomplete/duplicate initial anchor gradient audit.")
    if flow_summary:
        write_csv(root / "fixed_flow_summary.csv", flow_summary)
    write_csv(root / "repair_summary.csv", summaries)
    write_json(root / "summary.json", {"format": FORMAT, "complete": True, "arms": results,
               "source_training_draws_reused": bool(plan.get("training_draws")),
               "normalization_state_count": plan.get("normalization_state_count", len(fit_ids)),
               "fixed_flow_summary": str(root / "fixed_flow_summary.csv") if flow_summary else None,
               "anchor_gradient_audit": gradient_audits,
               "anchor_gradient_audit_interpretation": f"Unweighted common/difference endpoint gradients at initial weights and seed {plan['source_plan']['probe_seed']}; not whole-training or cross-task gradients." if gradient_audits else None,
               "interpretation": plan["interpretation"]})
    print(f"[complete] {root / 'summary.json'}\n[table] {root / 'repair_summary.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    source_args = run.add_mutually_exclusive_group()
    source_args.add_argument("--source-probe", default="latest")
    source_args.add_argument("--source-repair", help="Reuse a completed repair's original probe, not its repaired weights.")
    run.add_argument("--output", required=True)
    run.add_argument("--initial-model", choices=["step500", "step1000"], default="step1000")
    run.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    run.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS[:2]))
    run.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    run.add_argument("--steps", type=int, default=300)
    run.add_argument("--eval-every", type=int, default=50)
    run.add_argument("--train-seed", type=int, default=17000)
    run.add_argument("--eval-seeds", type=int, nargs="+", default=[42, 43, 44])
    run.add_argument("--learning-rate", type=float, default=5e-6)
    run.add_argument("--anchor-weight", type=float, default=.25)
    run.add_argument("--conditional-anchor-gain", type=float, default=4.)
    run.add_argument("--fixed-flow-sigmas", type=float, nargs="+", default=[])
    run.add_argument("--audit-anchor-gradients", action="store_true")
    run.add_argument("--minimum-improvement", type=float, default=.05)
    run.add_argument("--guard-relative", type=float, default=.10)
    run.add_argument("--guard-absolute", type=float, default=.005)
    run.add_argument("--execute", action="store_true")
    work = sub.add_parser("worker")
    work.add_argument("--plan", required=True)
    work.add_argument("--arm", choices=ARMS, required=True)
    work.add_argument("--gpu", type=int, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("output")
    show = sub.add_parser("report")
    show.add_argument("output", nargs="?", default="latest")
    args = parser.parse_args()
    if args.command == "worker":
        return worker(args)
    if args.command == "summarize":
        return summarize(args.output)
    if args.command == "report":
        return report(args.output)
    if (args.steps <= 0 or args.eval_every <= 0 or args.train_seed < 0
            or min(args.gpus) < 0 or min(args.eval_seeds) < 0
            or any(len(set(values)) != len(values) for values in (args.arms, args.gpus, args.pairs, args.eval_seeds))
            or any(not math.isfinite(value) or value < 0 for value in (
                args.learning_rate, args.anchor_weight, args.minimum_improvement, args.guard_relative, args.guard_absolute))
            or not math.isfinite(args.conditional_anchor_gain) or args.conditional_anchor_gain < 1
            or any(not math.isfinite(s) or not 0 < s <= 1 for s in args.fixed_flow_sigmas)
            or len(set(args.fixed_flow_sigmas)) != len(args.fixed_flow_sigmas)
            or args.learning_rate == 0 or args.anchor_weight == 0 or args.minimum_improvement >= 1):
        parser.error("Invalid steps, seeds, GPUs, repeated selections or loss/check thresholds.")
    if "paired_flow_anchor_delta" in args.arms and 1. not in args.fixed_flow_sigmas:
        parser.error("The conditional-difference arm requires --fixed-flow-sigmas including 1.")
    plan = build_plan(args)
    if plan["source_plan"]["probe_seed"] not in args.eval_seeds:
        parser.error("eval-seeds must include the original probe seed for production replay checks.")
    root = Path(plan["output"])
    if root.exists():
        raise FileExistsError(f"Use a fresh output directory: {root}")
    print(json.dumps({"source_probe": plan["source_probe"], "output": str(root),
                      "initial_model": args.initial_model, "steps": args.steps, "arms": args.arms,
                      "conditional_anchor_gain": args.conditional_anchor_gain,
                      "states": [{key: s[key] for key in ("id", "split")} for s in plan["states"]],
                      "scope": plan["scope"]}, indent=2), flush=True)
    if not args.execute:
        print("PLAN ONLY. Add --execute to run the CUDA repair experiment.")
        return
    root.mkdir(parents=True)
    plan["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    write_json(root / "plan.json", plan)

    def gpu_queue(index):
        codes = []
        for arm in args.arms[index::len(args.gpus)]:
            print(f"[start] {arm} physical_gpu={args.gpus[index]}", flush=True)
            path = root / f"{arm}.log"
            with path.open("x") as log:
                code = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                    "--plan", str(root / "plan.json"), "--arm", arm, "--gpu", str(args.gpus[index])],
                    cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"[exit] {arm} code={code}", flush=True)
            codes.append(code)
            if code:
                print("\n".join(path.read_text(errors="replace").splitlines()[-45:]), flush=True)
                break
        return codes

    count = min(len(args.arms), len(args.gpus))
    with ThreadPoolExecutor(max_workers=count) as pool:
        codes = list(pool.map(gpu_queue, range(count)))
    if any(code for queue in codes for code in queue):
        raise SystemExit("Repair experiment failed; worker error tails are printed above. Partial results are not summarized.")
    summarize(root)


if __name__ == "__main__":
    main()
