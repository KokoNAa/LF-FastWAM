#!/usr/bin/env python3
"""Bootstrap, inspect, and audit the explicit warm-policy ERAF/FG protocol."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["bootstrap", "inspect", "acceptance"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--manifest")
    ap.add_argument("--output")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.mode == "acceptance":
        from experiments.robotwin.eraf_fg_contract import acceptance
        print(json.dumps(acceptance(json.loads(Path(args.manifest).read_text())), indent=2))
        return
    if not args.checkpoint:
        ap.error("--checkpoint required")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/root/gpufree-data/fastwam/FastWAM/checkpoints")
    import torch
    from experiments.robotwin.eraf_fg_bridge import (
        file_sha256, load_policy, save_repair_checkpoint, validate_payload)
    if args.mode == "inspect":
        payload = validate_payload(torch.load(args.checkpoint, map_location="cpu", weights_only=False))
        print(json.dumps({k: v for k, v in payload.items() if k not in {"mot_trainable", "policy_guard"}}, indent=2))
        return
    if not args.manifest or not args.output:
        ap.error("bootstrap requires --manifest and --output")
    torch.manual_seed(args.seed)
    manifest = json.loads(Path(args.manifest).read_text())
    policy = load_policy(args.checkpoint, manifest, seed=args.seed, bootstrap=True)
    path = Path(args.output).resolve()
    save_repair_checkpoint(policy.model, path, stage="bootstrap", steps=0,
        parent=args.checkpoint, fg_supervision="off",
        provenance={"initialization_seed": args.seed, "source_manifest": str(Path(args.manifest).resolve()),
                    "source_manifest_sha256": file_sha256(args.manifest),
                    "policy_adapter_exact_copy_verified": True, "fresh_eraf": True})
    print(json.dumps({"complete": True, "checkpoint": str(path), "sha256": file_sha256(path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
