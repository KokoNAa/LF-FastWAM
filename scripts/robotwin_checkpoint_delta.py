#!/usr/bin/env python3
"""Transfer exact RoboTwin checkpoint deltas against an already archived parent."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SECTIONS = ('mot_trainable', 'policy_guard')
FORMAT = 'robotwin_checkpoint_delta_v1'


def fingerprint(payload):
    """Canonical metadata and tensor bytes; no second scan of checkpoint files."""
    import torch
    metadata = {k: v for k, v in payload.items() if k not in SECTIONS}
    header = json.dumps(metadata, sort_keys=True, ensure_ascii=False, allow_nan=False,
                        separators=(',', ':')).encode()
    digest = hashlib.sha256(len(header).to_bytes(8, 'big') + header)
    for section in SECTIONS:
        for name, value in sorted(payload[section].items()):
            if not isinstance(value, torch.Tensor):
                raise ValueError('Only tensor state dictionaries are supported.')
            label = json.dumps([section, name, str(value.dtype), list(value.shape)],
                               separators=(',', ':')).encode()
            digest.update(len(label).to_bytes(8, 'big')); digest.update(label)
            # View BF16 and scalar tensors as bytes without numeric conversion.
            array = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(array))
    return digest.hexdigest()


def make_delta(parent, checkpoint):
    import torch
    changed = {}
    for section in SECTIONS:
        if parent[section].keys() != checkpoint[section].keys():
            raise ValueError('Changed parameter sets require a full checkpoint archive.')
        changed[section] = {name: value for name, value in checkpoint[section].items()
            if value.dtype != parent[section][name].dtype or not torch.equal(value, parent[section][name])}
    return dict(format=FORMAT, parent_fingerprint=fingerprint(parent),
        checkpoint_fingerprint=fingerprint(checkpoint), changed=changed,
        checkpoint_metadata={k: v for k, v in checkpoint.items() if k not in SECTIONS})


def restore(parent, delta):
    if delta.get('format') != FORMAT or fingerprint(parent) != delta['parent_fingerprint']:
        raise ValueError('Delta does not match the archived parent checkpoint.')
    result = dict(delta['checkpoint_metadata'])
    for section in SECTIONS:
        if not set(delta['changed'][section]) <= set(parent[section]):
            raise ValueError('Delta contains an unexpected parameter.')
        result[section] = dict(parent[section]) | delta['changed'][section]
    if fingerprint(result) != delta['checkpoint_fingerprint']:
        raise ValueError('Reconstructed checkpoint does not match the source tensors and metadata.')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['pack', 'restore'])
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    parent = torch.load(args.parent, map_location='cpu', weights_only=False)
    source = torch.load(args.input, map_location='cpu', weights_only=False)
    if args.mode == 'pack':
        result = make_delta(parent, source)
        # Round-trip validation occurs before writing a transferable archive.
        restore(parent, result)
        receipt = dict(mode='pack', changed_tensors={k: len(v) for k, v in result['changed'].items()},
                       parent_fingerprint=result['parent_fingerprint'], checkpoint_fingerprint=result['checkpoint_fingerprint'])
    else:
        result = restore(parent, source)
        receipt = dict(mode='restore', checkpoint_fingerprint=source['checkpoint_fingerprint'],
                       exact_tensor_and_metadata_reconstruction=True,
                       byte_identical_serialization_claim=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + '.tmp')
    torch.save(result, temp); temp.replace(args.output)
    receipt.update(input=str(args.input.resolve()), parent=str(args.parent.resolve()),
                   output=str(args.output.resolve()), bytes=args.output.stat().st_size)
    args.output.with_suffix(args.output.suffix + '.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
