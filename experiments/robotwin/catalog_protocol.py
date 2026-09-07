"""Reserved, auditable scene namespaces for policy-independent catalogs."""
import hashlib
import json
from pathlib import Path

NAMESPACES = {'dev': (91300000, 91400000), 'test': (91700000, 91800000)}


def validate_namespace(split, start_seed, max_attempts):
    if split not in NAMESPACES:
        raise ValueError('Unknown catalog split.')
    lower, upper = NAMESPACES[split]
    if max_attempts < 1 or not lower <= start_seed < start_seed + max_attempts <= upper:
        raise ValueError(f'{split} catalog attempts must remain in [{lower},{upper}).')


def excluded_scene_records(paths, *, split):
    if split == 'test' and not paths:
        raise ValueError('Independent test catalogs require explicit training/development exclusion records.')
    seeds, receipts = set(), []
    for name in paths:
        path = Path(name).resolve()
        if path.suffix not in {'.json', '.jsonl'}:
            raise ValueError('Exclusions must be JSON manifests or JSONL episode records.')
        content = path.read_bytes()
        if path.suffix == '.jsonl':
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        else:
            value = json.loads(content)
            rows = value.get('states') if isinstance(value, dict) else value
        if not isinstance(rows, list) or not rows:
            raise ValueError('Exclusion source contains no scene records.')
        for row in rows:
            seed = row.get('scene_seed')
            if type(seed) is not int:
                raise ValueError('Every excluded record needs an integer scene_seed.')
            seeds.add(seed)
        receipts.append({'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(),
                         'records': len(rows)})
    return seeds, receipts
