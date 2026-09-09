#!/usr/bin/env python3
"""Initialize ordinary ERAF on current no-ERAF; never create a separate FG parent."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from scripts.run_robotwin_formal_five40 import read, write, sha
from experiments.robotwin.current_noeraf import branch_parent, current_baseline


def prepare(source, output, *, storage_root=Path('/root/gpufree-data/LF-FastWAM/runs')):
    import torch
    torch.set_num_threads(2)
    source, output = Path(source).resolve(), Path(output).resolve()
    if not output.is_relative_to(storage_root.resolve()):
        raise ValueError('Model outputs must stay on the server data disk.')
    models, audit, protocol = (read(source / n) for n in
                              ('final_models.json', 'completion_audit.json', 'protocol.json'))
    path, digest = current_baseline(source, models, audit)
    if sha(path) != digest:
        raise ValueError('Actual current no-ERAF checkpoint hash changed.')
    policy = torch.load(path, map_location='cpu', weights_only=False)
    output.mkdir(parents=True, exist_ok=False)
    arms, inputs = {}, {path: digest}
    for arm, fg in (('eraf_only', 'off'),):
        donor_path = protocol['arms'][arm]['parent']
        donor_sha = sha(donor_path)
        if donor_sha != protocol['input_sha256'][donor_path]:
            raise ValueError('Audited semantic donor changed.')
        donor = torch.load(donor_path, map_location='cpu', weights_only=False)
        result = branch_parent(policy, donor, policy_path=path, policy_sha256=digest,
            semantic_path=donor_path, semantic_sha256=donor_sha, fg=fg)
        target = output / (arm + '.pt')
        with target.open('xb') as stream:
            torch.save(result, stream)
        loaded = torch.load(target, map_location='cpu', weights_only=False)
        if any(not torch.equal(v, loaded['mot_trainable'][k]) for k, v in policy['mot_trainable'].items()):
            raise ValueError('Readback policy adapters do not match the current no-ERAF baseline.')
        if any(not torch.equal(v, loaded['policy_guard'][k]) for k, v in donor['policy_guard'].items()):
            raise ValueError('Readback guard does not match the declared semantic donor.')
        inputs[donor_path] = donor_sha
        arms[arm] = dict(path=str(target), sha256=sha(target), policy_adapters_equal=True,
            baseline_sha256=digest, semantic_donor_sha256=donor_sha,
            semantic_pretraining_steps=donor['optimizer_steps'], optimizer_updates=0)
        del donor, result, loaded
    if any(sha(p) != h for p, h in inputs.items()):
        raise ValueError('A source checkpoint changed during preparation.')
    receipt = dict(complete=True, baseline_checkpoint=path, baseline_sha256=digest,
        baseline_retrained=False, input_sha256=inputs, arms=arms,
        deployed_identity_verified=False, optimizer_updates=0)
    write(output / 'preparation.json', receipt)
    return receipt


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-trial', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    prepare(args.source_trial, args.output)
