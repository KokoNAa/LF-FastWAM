#!/usr/bin/env python3
"""Audit complete new FG windows against retained parent rows and raw actions."""
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('parent','manifest','collection','output'):ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args()
    import torch
    torch.set_num_threads(2)
    from experiments.robotwin.expanded_fg_preparation import audit_extension,audit_cached_targets
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    source,target,collection=[json.loads(p.read_text()) for p in (args.parent,args.manifest,args.collection)]
    bindings={str(p):file_sha256(p) for p in (args.parent,args.manifest,args.collection)}
    report=audit_extension(source,target,collection)
    report['cached_target_audit']=audit_cached_targets(source,target,collection)
    if bindings!={p:file_sha256(p) for p in bindings}:raise ValueError('Audit inputs changed.')
    report['input_sha256']=bindings
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
