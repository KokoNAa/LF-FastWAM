#!/usr/bin/env python3
"""Bind existing training scenes to seen descriptions of their actual objects."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]
from experiments.robotwin.decision_language_replay import bound_spatial_instruction_pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--source-bank', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    records = {}
    for domain in ('demo_clean', 'demo_randomized'):
        plan = json.loads((Path(args.source_bank) / domain / 'plan.json').read_text())
        for pair in plan['pairs']:
            key = domain, pair['pair_id']
            if key in records or not pair['pair_id'].startswith('place_a2b_'):
                continue
            path = Path(pair['native']['hdf5']).parent.parent / 'meta/pgc_episodes.jsonl'
            records[key] = {r['episode_index']: r for r in map(json.loads, path.read_text().splitlines())}
    specs, rows = {}, []
    for row in manifest['states']:
        row = dict(row)
        if row['replay_split'] == 'train' and row['pair_id'].startswith('place_a2b_'):
            key = (row['task_config'], row['pair_id'], row['episode_index'])
            if key not in specs:
                record = records[key[:2]][key[2]]
                specs[key] = bound_spatial_instruction_pairs(REPO, row['pair_id'], record['scene_info']['info'])
            row['language_replay_key'] = ':'.join(map(str, key))
            row['seen_instruction_pairs'] = specs[key]
        rows.append(row)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = manifest | {'states': rows, 'language_bound_scenes': len(specs),
                         'language_scope': 'Seen task templates and seen object descriptions; original training scenes only.'}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'[complete] bound_scenes={len(specs)} manifest={output}', flush=True)


if __name__ == '__main__':
    main()
