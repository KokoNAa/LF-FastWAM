#!/usr/bin/env python3
"""Reuse verified FG masks for an expanded manifest with identical corrections."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source-index', 'source-manifest', 'target-manifest', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    from experiments.robotwin.fg_role_masks import build_expanded_mask_binding, VerifiedCorrectionMasks
    report = build_expanded_mask_binding(args.source_index, args.source_manifest, args.target_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    reader = VerifiedCorrectionMasks(args.output, args.target_manifest)
    reader.validate_coverage(json.loads(args.target_manifest.read_text())['states'])
    print(json.dumps(dict(complete=True, binding=str(args.output.resolve()),
        binding_sha256=reader.index_sha256, verified_correction_archives=len(reader.records),
        source_replay_unchanged=True, new_labels_created=False)))


if __name__ == '__main__':
    main()
