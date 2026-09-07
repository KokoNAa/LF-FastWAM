#!/usr/bin/env python3
"""Initialize a server-only residual context checkpoint without optimizer steps."""
import argparse
import json
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args()
    import torch
    from experiments.robotwin.context_residual import initialize, ZEROED
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    source=Path(args.checkpoint).resolve();out=Path(args.output).resolve()
    if out.exists(): raise FileExistsError(out)
    payload=torch.load(source,map_location='cpu',weights_only=False)
    result=initialize(payload,parent_path=source,parent_sha256=file_sha256(source))
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=out.with_suffix('.tmp');torch.save(result,temporary);temporary.replace(out)
    saved=torch.load(out,map_location='cpu',weights_only=False)
    changed=[]
    for section in ('mot_trainable','policy_guard'):
        assert saved[section].keys()==payload[section].keys()
        for k,v in payload[section].items():
            if not torch.equal(v,saved[section][k]):
                assert section=='policy_guard' and k in ZEROED
                changed.append(k)
    assert all(saved['policy_guard'][k].count_nonzero()==0 for k in ZEROED)
    report={'complete':True,'checkpoint':str(out),'checkpoint_sha256':file_sha256(out),
        'parent':str(source),'parent_sha256':result['parent_sha256'],'context_injection_mode':saved['context_injection_mode'],
        'optimizer_updates':0,'changed_tensors':changed,'policy_and_other_guard_tensors_identical':True,
        'zero_context_output_verified':True,'deployed_identity_checked':False}
    out.with_suffix('.initialization.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
