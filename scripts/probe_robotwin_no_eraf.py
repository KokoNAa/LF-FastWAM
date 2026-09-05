#!/usr/bin/env python3
"""Probe Base/LoRA language-conditioned generation on fixed RoboTwin states.

run defaults to planning; --execute writes a fresh directory and starts one
worker per listed GPU. No training or simulation. Reads audited raw experts,
uses the production inference wrapper, and saves every predicted action.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaping audited path: {relative} under {root}")
    return path


def build_plan(args):
    from omegaconf import OmegaConf
    from experiments.robotwin.no_eraf_probe import FORMAT, training_episode_ids, require_pair

    train = Path(args.train_run).expanduser().resolve()
    cfg_path = train / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg_model = cfg.model
    if (cfg_model.policy_guard.enabled or cfg_model.langforce_mvp.enabled
            or cfg_model.transition_contract.enabled
            or cfg_model.action_dit_config.use_latent_action_queries):
        raise ValueError("Expected the query-free no-ERAF training run.")
    control = cfg_model.lora.paired_language_control
    if not all(control.get(key, False) for key in (
        "enabled", "bidirectional_supervision", "deployment_matched_action_cache",
        "correct_branch_action_ranking",
    )):
        raise ValueError("Training config lacks the temporal-v2 cache/ranking contract.")
    horizon = int(cfg.data.train.num_frames) - 1
    if horizon != 32 or int(cfg.data.train.processor.action_output_dim) != 14:
        raise ValueError("Probe currently requires native 32x14 RoboTwin actions.")
    if int(cfg.data.train.get("action_video_freq_ratio", 4)) != 4:
        raise ValueError("Probe requires the released 32-action/9-video-frame alignment.")
    base, stats = [Path(p).expanduser().resolve() for p in (args.base_checkpoint, args.stats_path)]
    checkpoints = {"base": base, **{
        f"step{step}": train / "checkpoints/weights" / f"step_{step:06d}.pt"
        for step in args.steps
    }}
    for path in (*checkpoints.values(), stats):
        if not path.is_file():
            raise FileNotFoundError(path)
    if Path(str(cfg.resume)).resolve() != base:
        raise ValueError("Requested Base differs from the training run's resume checkpoint.")
    if Path(str(cfg.data.train.pretrained_norm_stats)).resolve() != stats:
        raise ValueError("Requested action statistics differ from the training run.")
    training_dirs = {str(Path(p).resolve()) for p in list(cfg.data.train.dataset_dirs)
                     + list(cfg.data.train.pgc_counterfactual_dataset_dirs)}
    # RobotVideoDataset defaults to .05; its BaseLerobotDataset uses seed=42.
    val_proportion = float(cfg.data.train.get("val_set_proportion", .05))
    pairs = []
    expert = Path(args.expert_root).expanduser().resolve()
    for profile in args.profiles:
        domain = expert / profile / args.task_config
        audits = sorted((domain / "raw").glob("*/counterfactual/meta/pgc_episodes.jsonl"))
        if not audits:
            raise ValueError(f"No expert pair audits under {domain}")
        for target_audit in audits:
            pair_id = target_audit.parents[2].name
            if args.pairs and pair_id not in args.pairs:
                continue
            raw_pair = target_audit.parents[2]
            native_audit = raw_pair / "native/meta/pgc_episodes.jsonl"
            records = {}
            train_ids = {}
            for kind, audit in (("native", native_audit), ("counterfactual", target_audit)):
                rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
                records[kind] = {int(row["episode_index"]): row for row in rows}
                if len(rows) != len(records[kind]):
                    raise ValueError(f"Duplicate episode indices: {audit}")
                # The strict source replay is raw audit data only: preparation
                # deliberately never converts it to a LeRobot training dataset.
                if profile == "strict" and kind == "native":
                    train_ids[kind] = set(records[kind])
                    continue
                dataset = domain / "lerobot" / pair_id / kind
                info = read_json(dataset / "meta/info.json")
                if set(records[kind]) != set(range(int(info["total_episodes"]))):
                    raise ValueError(f"Raw/LeRobot episode coverage mismatch: {audit}")
                converted = {int(r["episode_index"]): r for r in (
                    json.loads(line) for line in (dataset / "meta/pgc_episodes.jsonl").read_text().splitlines() if line.strip()
                )}
                if set(converted) != set(records[kind]):
                    raise ValueError(f"Converted provenance coverage mismatch: {dataset}")
                for episode, raw_record in records[kind].items():
                    for field in ("pair_id", "scene_seed", "initial_state_sha256", "action_sha256"):
                        if converted[episode].get(field) != raw_record.get(field):
                            raise ValueError(f"Raw/converted provenance mismatch: {dataset}/{episode}/{field}")
                if str(dataset.resolve()) not in training_dirs:
                    raise ValueError(f"Dataset absent from the selected training config: {dataset}")
                train_ids[kind] = training_episode_ids(int(info["total_episodes"]), val_proportion)
            eligible = sorted(train_ids["native"] & train_ids["counterfactual"])
            if len(eligible) < args.episodes_per_pair:
                raise ValueError(f"Only {len(eligible)} eligible episodes for {profile}/{pair_id}")
            for episode in eligible[:args.episodes_per_pair]:
                native, target = records["native"][episode], records["counterfactual"][episode]
                require_pair(native, target)
                pair = {"id": f"{profile}_{pair_id}_ep{episode:04d}", "profile": profile,
                        "pair_id": pair_id, "episode_index": episode,
                        "scene_seed": target["scene_seed"],
                        "source_instruction": target["source_instruction"],
                        "counterfactual_instruction": target["counterfactual_instruction"],
                        "native_in_training_split": profile == "historical",
                        "counterfactual_in_training_split": True}
                for kind, record in (("native", native), ("counterfactual", target)):
                    root = raw_pair / kind
                    pair[kind] = {"audit": record,
                                  "hdf5": str(inside(root, record["raw_hdf5"])),
                                  "initial_state": str(inside(root, record["source_initial_state_catalog"]))}
                pairs.append(pair)
    if not pairs or (args.pairs and set(args.pairs) != {p["pair_id"] for p in pairs}):
        raise ValueError("No pairs selected or requested pairs were not found.")
    return {"format": FORMAT, "train_config": str(cfg_path),
            "train_config_sha256": sha256(cfg_path), "base_checkpoint": str(base),
            "stats_path": str(stats), "stats_sha256": sha256(stats),
            "checkpoints": {key: str(value) for key, value in checkpoints.items()},
            "output": str(Path(args.output).expanduser().resolve()),
            "action_horizon": horizon, "num_video_frames": 9, "task_config": args.task_config,
            "inference_steps": args.inference_steps, "probe_seed": args.seed,
            "fractions": args.fractions, "pairs": pairs, "gpus": args.gpus,
            "split": {"validation_proportion": val_proportion, "dataset_split_seed": 42},
            "interpretation": "Offline deployment-sampled action probe; no CIS/goal success measured."}


def prepare_states(plan):
    import h5py
    import numpy as np
    from PIL import Image
    from experiments.robotwin.no_eraf_probe import (
        CAMERAS, frame_positions, observations_equal, observation_hash, typed_hash,
        last_equal_qpos_prefix,
    )
    output = Path(plan["output"])
    (output / "states").mkdir()
    states = []
    horizon = plan["action_horizon"]

    def observation(handle, actions, frame):
        obs = {"state": actions[frame]}
        for camera in CAMERAS:
            raw = bytes(handle[f"observation/{camera}/rgb"][frame]).rstrip(b"\0")
            with Image.open(io.BytesIO(raw)) as im:
                obs[camera] = np.asarray(im.convert("RGB"), dtype=np.uint8)
        return obs

    for pair in plan["pairs"]:
        with h5py.File(pair["native"]["hdf5"], "r") as native, h5py.File(pair["counterfactual"]["hdf5"], "r") as target:
            handles = {"native": native, "counterfactual": target}
            actions = {}
            for kind, handle in handles.items():
                raw = np.asarray(handle["joint_action/vector"])
                audit = pair[kind]["audit"]
                # The collector's action validation casts to float32.
                raw = np.ascontiguousarray(raw, dtype=np.float32)
                if raw.ndim != 2 or raw.shape[1] != 14 or not np.isfinite(raw).all():
                    raise ValueError(f"Invalid qpos actions: {pair['id']}/{kind}")
                if typed_hash(raw) != audit["action_sha256"] or len(raw) != audit["action_count"]:
                    raise ValueError(f"Audited expert action hash/count mismatch: {pair['id']}/{kind}")
                initial = np.load(pair[kind]["initial_state"], allow_pickle=False)
                if typed_hash(initial) != audit["initial_state_sha256"]:
                    raise ValueError(f"Initial-state hash mismatch: {pair['id']}/{kind}")
                if any(len(handle[f"observation/{camera}/rgb"]) != len(raw) for camera in CAMERAS):
                    raise ValueError("RGB/action lengths differ.")
                actions[kind] = raw
            initial_obs = {kind: observation(handles[kind], actions[kind], 0) for kind in handles}
            matched = observations_equal(initial_obs["native"], initial_obs["counterfactual"])
            prefix_frame = last_equal_qpos_prefix(actions["native"], actions["counterfactual"], horizon)
            positions = {kind: frame_positions(len(actions[kind]), horizon, plan["fractions"]) for kind in handles}
            if prefix_frame is not None:
                for kind in positions:
                    positions[kind] = sorted({*positions[kind], prefix_frame})
            # Independent trajectory fractions are NOT paired decision states.
            # Only identical complete observations admit the dual-reference test.
            same_frames = set()
            for frame in set(positions["native"]) & set(positions["counterfactual"]):
                if observations_equal(observation(native, actions["native"], frame),
                                      observation(target, actions["counterfactual"], frame)):
                    same_frames.add(frame)
            for kind in handles:
                for frame in positions[kind]:
                    if kind == "counterfactual" and frame in same_frames:
                        continue
                    obs = observation(handles[kind], actions[kind], frame)
                    state_id = f"{pair['id']}_{kind}_f{frame:05d}"
                    dual = frame in same_frames
                    arrays = {**obs, "own_reference": actions[kind][frame:frame + horizon]}
                    if dual:
                        arrays.update(source_reference=actions["native"][frame:frame + horizon],
                                      target_reference=actions["counterfactual"][frame:frame + horizon])
                    path = output / "states" / f"{state_id}.npz"
                    np.savez_compressed(path, **arrays)
                    states.append({"id": state_id, "file": str(path), "sha256": sha256(path),
                                   "observation_sha256": observation_hash(obs),
                                   "pair_id": pair["pair_id"], "profile": pair["profile"],
                                   "episode_index": pair["episode_index"], "scene_seed": pair["scene_seed"],
                                   "expert_kind": kind, "frame_index": frame,
                                   "last_equal_qpos_prefix_frame": prefix_frame,
                                   "is_prefix_boundary_candidate": frame == prefix_frame,
                                   "own_reference_in_training_split": pair[f"{kind}_in_training_split"],
                                   "source_instruction": pair["source_instruction"],
                                   "counterfactual_instruction": pair["counterfactual_instruction"],
                                   "dual_reference_valid": dual, "initial_observations_exactly_equal": matched})
        print(f"[states] {pair['id']} initial_observations_equal={matched}", flush=True)
    write_json(output / "states.json", states)
    return states


def validate_loaded_adapter(model, payload, base, expected_step):
    import torch
    if payload.get("format") != "fastwam_lora_adapter_v1" or int(payload.get("step", -1)) != expected_step:
        raise ValueError("Unexpected adapter format/step.")
    if Path(str(model.lora_base_checkpoint)).resolve() != Path(base).resolve():
        raise ValueError("Runtime adapter Base differs from requested Base.")
    if payload.get("policy_guard") or payload.get("transition_contract"):
        raise ValueError("Not a no-ERAF adapter.")
    current = model.mot.state_dict()
    saved = payload["mot_trainable"]
    if not saved or not model.lora_enabled:
        raise ValueError("Empty or disabled LoRA.")
    moments = {}
    for name, value in saved.items():
        if name not in current or not torch.equal(current[name].detach().cpu(), value.to(dtype=current[name].dtype).cpu()):
            raise ValueError(f"Runtime adapter tensor differs from checkpoint: {name}")
        if name.endswith((".lora_A", ".lora_B")):
            expert = "video" if ".video." in name else "action" if ".action." in name else "unknown"
            key = expert + "/" + name.rsplit(".", 1)[-1]
            data = value.float()
            item = moments.setdefault(key, {"tensors": 0, "elements": 0, "sum_sq": 0., "max_abs": 0.})
            item["tensors"] += 1
            item["elements"] += data.numel()
            item["sum_sq"] += float(data.square().sum())
            item["max_abs"] = max(item["max_abs"], float(data.abs().max()))
    for key, item in moments.items():
        item["rms"] = (item.pop("sum_sq") / item["elements"]) ** .5
    for expert in ("video", "action"):
        if moments.get(expert + "/lora_B", {}).get("rms", 0.) == 0:
            raise ValueError(f"Missing or all-zero {expert} LoRA-B weights.")
    return {"runtime_exact_match_after_dtype_cast": True, "tensor_count": len(saved),
            "moments": moments, "saved_lora_config": payload["lora_config"]}


def inference_bootstrap_configs(cfg):
    """Build Base-first configs; the adapter loader restores saved LoRA later."""
    from omegaconf import OmegaConf

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    processor_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train.processor, resolve=True))
    model_cfg.load_text_encoder = True
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.action_dit_pretrained_path = None
    model_cfg.lora.enabled = False
    # All dependent switches must be disabled even when LoRA itself is off:
    # normalize_lora_config validates the paired-control contract first.
    for key in ("enabled", "bidirectional_supervision",
                "deployment_matched_action_cache", "correct_branch_action_ranking"):
        model_cfg.lora.paired_language_control[key] = False
    return model_cfg, processor_cfg


def worker(args):
    # Set physical GPU visibility before importing torch or the policy wrapper.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for key in list(os.environ):
        if key.startswith("PGC_ROBOTWIN_CLOSED_LOOP_CAPTURE"):
            del os.environ[key]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy
    from experiments.robotwin.no_eraf_probe import difference, reference_metrics, observation_hash

    plan = read_json(args.plan)
    output = Path(plan["output"]) / args.model
    output.mkdir(exist_ok=False)
    checkpoint = plan["checkpoints"][args.model]
    if sha256(checkpoint) != plan["checkpoint_sha256"][args.model]:
        raise ValueError("Checkpoint changed after planning.")
    if sha256(plan["stats_path"]) != plan["stats_sha256"]:
        raise ValueError("Action statistics changed after planning.")
    if sha256(plan["train_config"]) != plan["train_config_sha256"]:
        raise ValueError("Training config changed after planning.")
    cfg = OmegaConf.load(plan["train_config"])
    model_cfg, processor_cfg = inference_bootstrap_configs(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU.")
    policy = WorldActionRobotWinPolicy(
        model_cfg=model_cfg, processor_cfg=processor_cfg,
        checkpoint_path=checkpoint, dataset_stats_path=Path(plan["stats_path"]),
        device="cuda:0", model_dtype=torch.bfloat16,
        action_horizon=plan["action_horizon"], replan_steps=24,
        num_inference_steps=plan["inference_steps"], sigma_shift=None,
        seed=plan["probe_seed"], text_cfg_scale=1., negative_prompt="",
        rand_device="cpu", tiled=False, timing_enabled=False,
        num_video_frames=plan["num_video_frames"], task_name="same_state_probe", task_config=plan["task_config"],
    )
    audit = {"model": args.model, "checkpoint": checkpoint,
             "checkpoint_sha256": plan["checkpoint_sha256"][args.model],
             "physical_gpu": args.gpu, "lora_enabled": policy.model.lora_enabled}
    if args.model == "base":
        if policy.model.lora_enabled:
            raise ValueError("Base unexpectedly enabled LoRA.")
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        audit.update(validate_loaded_adapter(policy.model, payload, plan["base_checkpoint"], int(args.model[4:])))
        del payload
    if (policy.model.policy_guard_enabled or policy.model.transition_contract_enabled
            or policy.model.action_expert.use_latent_action_queries):
        raise ValueError("Runtime is not query-free no-ERAF.")
    write_json(output / "checkpoint_audit.json", audit)
    key = policy.processor.shape_meta["action"][0]["key"]
    normalizer = policy.processor.normalizer.normalizers["action"][key]

    def normalized(actions, *, reference=False):
        value = torch.as_tensor(actions, dtype=torch.float32).unsqueeze(0)
        # References use the training clamp. Generated outputs must not be
        # clipped again: clipping would conceal unstable or differing models.
        result = normalizer.forward(value) if reference else value * normalizer.scale + normalizer.offset
        return result.squeeze(0).cpu().numpy()

    def predict(obs, text):
        torch.manual_seed(plan["probe_seed"])
        np.random.seed(plan["probe_seed"])
        policy.policy_guard_state = None
        return policy._infer_action_chunk(obs, text)

    states = read_json(Path(plan["output"]) / "states.json")
    with (output / "records.jsonl").open("x") as log:
        for number, state in enumerate(states):
            if sha256(state["file"]) != state["sha256"]:
                raise ValueError("Probe state changed between model workers.")
            with np.load(state["file"], allow_pickle=False) as stored:
                arrays = {key: stored[key] for key in stored.files}
            if observation_hash(arrays) != state["observation_sha256"]:
                raise ValueError("Observation fingerprint mismatch.")
            obs = {"joint_action": {"vector": arrays["state"]}, "observation": {
                camera: {"rgb": arrays[camera]} for camera in ("head_camera", "left_camera", "right_camera")
            }}
            raw_source = predict(obs, state["source_instruction"])
            raw_target = predict(obs, state["counterfactual_instruction"])
            source, target = normalized(raw_source), normalized(raw_target)
            repeat = None
            if number == 0:
                repeat = difference(source, normalized(predict(obs, state["source_instruction"])))
                if repeat["max_abs"] > 1e-5:
                    write_json(output / "repeat_failure.json", repeat)
                    raise RuntimeError("Same-model/same-state/same-seed repeat failed.")
            refs = {key: normalized(arrays[key], reference=True) for key in (
                "own_reference", "source_reference", "target_reference"
            ) if key in arrays}
            record = {**state, "model": args.model, "repeat_check": repeat,
                      "metrics_normalized": reference_metrics(source, target, refs["own_reference"],
                          state["expert_kind"], refs.get("source_reference"), refs.get("target_reference"))}
            record["metrics_executed_prefix24"] = reference_metrics(
                source[:24], target[:24], refs["own_reference"][:24], state["expert_kind"],
                refs["source_reference"][:24] if "source_reference" in refs else None,
                refs["target_reference"][:24] if "target_reference" in refs else None)
            action_path = output / f"{state['id']}.npz"
            np.savez_compressed(action_path, source=source, target=target,
                                source_qpos=raw_source, target_qpos=raw_target, **refs)
            record["actions_file"] = str(action_path)
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            print(f"[probe] {args.model} {number + 1}/{len(states)} {state['id']} "
                  f"language_rms={record['metrics_normalized']['language_delta']['rms']:.6g}", flush=True)
    write_json(output / "complete.json", {"model": args.model, "states": len(states)})


def summarize(root):
    import numpy as np
    from experiments.robotwin.no_eraf_probe import difference
    root = Path(root)
    plan, states = read_json(root / "plan.json"), read_json(root / "states.json")
    expected = {s["id"] for s in states}
    by_model = {}
    for model in plan["checkpoints"]:
        if read_json(root / model / "complete.json")["states"] != len(states):
            raise ValueError(f"Incomplete model: {model}")
        rows = [json.loads(line) for line in (root / model / "records.jsonl").read_text().splitlines()]
        if len(rows) != len(expected) or {r["id"] for r in rows} != expected:
            raise ValueError(f"Duplicate or incomplete state records: {model}")
        by_model[model] = {r["id"]: r for r in rows}
    flat, models = [], {}
    for model, rows in by_model.items():
        for state_id, row in rows.items():
            metrics = row["metrics_normalized"]
            item = {key: row[key] for key in ("id", "model", "profile", "pair_id", "expert_kind", "frame_index", "own_reference_in_training_split", "is_prefix_boundary_candidate")}
            item.update(language_delta_rms=metrics["language_delta"]["rms"],
                        correct_language_reference_rmse=metrics["correct_language_reference_rmse"],
                        wrong_minus_correct_reference_rmse=metrics["wrong_minus_correct_reference_rmse"],
                        executed_prefix24_language_delta_rms=row["metrics_executed_prefix24"]["language_delta"]["rms"])
            with np.load(row["actions_file"]) as prediction, np.load(by_model["base"][state_id]["actions_file"]) as base:
                if row["observation_sha256"] != by_model["base"][state_id]["observation_sha256"]:
                    raise ValueError("Cross-model observation mismatch.")
                for language in ("source", "target"):
                    item[language + "_delta_vs_base_rms"] = difference(prediction[language], base[language])["rms"]
            dual = metrics["dual_reference"]
            for field in ("expert_separation_rms", "language_delta_projection_on_expert_delta",
                          "language_delta_cosine_with_expert_delta", "source_language_prefers_source_expert",
                          "target_language_prefers_target_expert"):
                item[field] = dual[field] if dual else None
            flat.append(item)
        own = [r for r in flat if r["model"] == model]
        mean_fields = ("language_delta_rms", "correct_language_reference_rmse", "wrong_minus_correct_reference_rmse",
                       "source_delta_vs_base_rms", "target_delta_vs_base_rms")
        models[model] = {"states": len(own), **{key: float(np.mean([r[key] for r in own])) for key in mean_fields}}
        models[model]["groups"] = []
        for profile, kind in sorted({(r["profile"], r["expert_kind"]) for r in own}):
            subset = [r for r in own if r["profile"] == profile and r["expert_kind"] == kind]
            models[model]["groups"].append({"profile": profile, "expert_kind": kind, "states": len(subset),
                **{key: float(np.mean([r[key] for r in subset])) for key in mean_fields}})
        dual_rows = [r for r in own if r["target_language_prefers_target_expert"] is not None]
        models[model]["distinguishable_matched_reference_states"] = len(dual_rows)
        models[model]["target_language_prefers_target_expert_count"] = sum(bool(r["target_language_prefers_target_expert"]) for r in dual_rows)
    summary = {"format": plan["format"], "complete": True, "models": models,
               "interpretation": "Action-space diagnostics only. Expert RMSE/projection is not CIS, Cartesian goal direction, or task success. Mid-trajectory states share observations across language/model, not across expert trajectories."}
    write_json(root / "summary.json", summary)
    with (root / "comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("train-run", "expert-root", "base-checkpoint", "stats-path", "output"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--steps", nargs="+", type=int, default=[500, 1000])
    run.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    run.add_argument("--profiles", nargs="+", choices=["historical", "strict"], default=["historical"])
    run.add_argument("--task-config", choices=["demo_clean", "demo_randomized"], default="demo_clean")
    run.add_argument("--pairs", nargs="+")
    run.add_argument("--episodes-per-pair", type=int, default=1)
    run.add_argument("--fractions", nargs="+", type=float, default=[.25, .5, .75])
    run.add_argument("--inference-steps", type=int, default=10)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--execute", action="store_true")
    work = sub.add_parser("worker")
    work.add_argument("--plan", required=True)
    work.add_argument("--model", required=True)
    work.add_argument("--gpu", type=int, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("output")
    args = parser.parse_args()
    if args.command == "worker":
        return worker(args)
    if args.command == "summarize":
        return summarize(args.output)
    if (not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0
            or len(set(args.steps)) != len(args.steps) or min(args.steps) < 1
            or args.episodes_per_pair < 1 or args.inference_steps < 1 or args.seed < 0
            or len(set(args.profiles)) != len(args.profiles)
            or any(not 0 <= f <= 1 for f in args.fractions)):
        parser.error("Invalid GPUs, steps, profiles, fractions, or sample count.")
    plan = build_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    root = Path(plan["output"])
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {root}")
    if not args.execute:
        print("PLAN ONLY. Add --execute to capture fixed states and run inference.")
        return
    root.mkdir(parents=True, exist_ok=False)
    plan["checkpoint_sha256"] = {}
    for model, path in plan["checkpoints"].items():
        print(f"[hash] {model} {path}", flush=True)
        plan["checkpoint_sha256"][model] = sha256(path)
    plan["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    write_json(root / "plan.json", plan)
    states = prepare_states(plan)
    print(f"[ready] {len(states)} fixed states, {len(plan['checkpoints'])} models", flush=True)
    jobs = list(plan["checkpoints"])

    def gpu_queue(index):
        codes = []
        for model in jobs[index::len(args.gpus)]:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                       "--plan", str(root / "plan.json"), "--model", model, "--gpu", str(args.gpus[index])]
            print(f"[start] {model} physical_gpu={args.gpus[index]}", flush=True)
            with (root / f"{model}.log").open("x") as log:
                code = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"[exit] {model} code={code}", flush=True)
            codes.append(code)
            if code:
                break
        return codes

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        results = list(pool.map(gpu_queue, range(min(len(jobs), len(args.gpus)))))
    if any(code for codes in results for code in codes):
        raise SystemExit(f"Probe failed; inspect {root}/*.log. Partial results are not summarized.")
    summarize(root)


if __name__ == "__main__":
    main()
