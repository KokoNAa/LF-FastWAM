#!/usr/bin/env python3
"""Cold-load a repair adapter through the ordinary deployment policy and replay."""
import argparse
from collections import Counter
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]


def main():
    from scripts.probe_robotwin_no_eraf import read_json, write_json, sha256, load_probe_policy
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repair-run", type=Path, required=True)
    ap.add_argument("--arm", choices=["action", "joint"], default="joint")
    ap.add_argument("--step", type=int, default=200)
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    root = args.repair_run.resolve()
    output = root / f"cold_reload_{args.arm}_{args.step:06d}.json"
    if output.exists():
        raise ValueError("Cold-reload output already exists.")
    plan = read_json(root / "plan.json")
    if not read_json(root / args.arm / "complete.json")["complete"]:
        raise ValueError("Repair arm is incomplete.")
    source = read_json(Path(plan["source_probe"]) / "plan.json")
    checkpoint = root / args.arm / f"step_{args.step:06d}.pt"
    key = f"step{1000 + args.step}"
    source["checkpoints"][key] = str(checkpoint)
    source["checkpoint_sha256"][key] = sha256(checkpoint)
    policy, audit = load_probe_policy(source, key, args.gpu)
    import numpy as np
    import torch
    from experiments.robotwin.no_eraf_probe import CAMERAS, observation_hash, difference
    norm = policy.processor.normalizer.normalizers["action"][policy.processor.shape_meta["action"][0]["key"]]
    audit["loaded_adapter_dtypes"] = dict(Counter(str(p.dtype) for n, p in policy.model.mot.named_parameters()
                                                 if n.endswith((".lora_A", ".lora_B"))))
    rows = []
    for state in plan["states"]:
        if sha256(state["file"]) != state["sha256"]:
            raise ValueError("Observation file changed.")
        with np.load(state["file"], allow_pickle=False) as f:
            arrays = dict(f)
        if observation_hash(arrays) != state["observation_sha256"]:
            raise ValueError("Observation content changed.")
        obs = {"joint_action": {"vector": arrays["state"]}, "observation": {k: {"rgb": arrays[k]} for k in CAMERAS}}
        for seed in plan["eval_seeds"]:
            path = root / args.arm / f"train{args.step:06d}_{state['id']}_seed{seed}.npz"
            with np.load(path, allow_pickle=False) as f:
                expected = dict(f)
            for language, field in (("source", "source_instruction"), ("target", "counterfactual_instruction")):
                policy.seed = seed
                policy.policy_guard_state = None
                torch.manual_seed(seed)
                np.random.seed(seed)
                raw = policy._infer_action_chunk(obs, state[field])
                result = (torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0) * norm.scale + norm.offset)[0].numpy()
                error = difference(result, expected[language])
                rows.append({"id": state["id"], "seed": seed, "language": language, **error})
                if error["max_abs"] > 1e-5:
                    write_json(output.with_suffix(".failure.json"), {"complete": False, "audit": audit, "rows": rows})
                    raise ValueError("Cold-loaded checkpoint differs from training evaluation.")
        print(f"[cold-reload] {state['id']}", flush=True)
    write_json(output, {"complete": True, "audit": audit, "rows": rows,
                       "maximum_error": max(r["max_abs"] for r in rows), "repair_plan_sha256": sha256(root / "plan.json")})
    print("[complete]", output, flush=True)


if __name__ == "__main__":
    main()
