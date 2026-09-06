#!/usr/bin/env python3
"""Assemble completed incremental caches and audit the formal scene budget."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_rows(rows, *, formal=True, fg_train_scenes=24):
    from scripts.collect_robotwin_eraf_fg import historical_scene_keys
    from experiments.robotwin.eraf_fg_contract import scene_key, validate_correction, TARGET_TASKS
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate replay IDs.')
    train = historical_scene_keys([r for r in rows if r['replay_split'] == 'train'])
    held = historical_scene_keys([r for r in rows if r['replay_split'] == 'replay_holdout'])
    if train & held:
        raise ValueError('Training and data-holdout scenes overlap.')
    scenes = defaultdict(list)
    retention, native = defaultdict(set), defaultdict(set)
    for row in rows:
        if row.get('fg_correction'):
            scenes[scene_key(row)].append(row)
        if row.get('cf_retention'):
            if not row.get('full_cf_episode_success'):
                raise ValueError('Unaudited CF retention.')
            retention[row['source_task']].add(scene_key(row))
        if row.get('native_retention'):
            native[row['source_task']].add(scene_key(row))
    counts, windows, origins, prefixes = Counter(), Counter(), Counter(), defaultdict(list)
    for key, group in scenes.items():
        record = validate_correction(group[0])
        split = record['replay_split']
        count = int(record['recorded_action_count'])
        if sorted(r['frame_index'] for r in group) != list(range(0, count, 8)):
            raise ValueError(f'Missing, duplicated or unexpected full-goal windows: {key}')
        for row in group:
            validate_correction(row)
            if any(row[field] != value for field, value in record.items()
                   if field not in {'id', 'payload', 'frame_index', 'reference_valid_actions'}):
                raise ValueError(f'Inconsistent correction provenance: {key}')
            if row['reference_valid_actions'] != min(32, count - row['frame_index']):
                raise ValueError(f'Incorrect real-action tail mask: {key}')
        counts[key[0], split] += 1
        windows[key[0], split] += len(group)
        origins[key[0], record['failure_kind']] += 1
        prefixes[key[0]].append(record['prefix_action_count'])
    expected = {(task, split): n for task in TARGET_TASKS
                for split, n in [('train', fg_train_scenes), ('replay_holdout', 6)]}
    if formal and counts != expected:
        raise ValueError(f'Expected exactly{fg_train_scenes} train +6 holdout FG scenes per task, got {dict(counts)}')
    retention_counts = {task: len(keys) for task, keys in retention.items()}
    if formal and retention_counts != {t: 10 for t in ('place_a2b_right', 'place_burger_fries', 'stack_blocks_two')}:
        raise ValueError(f'Expected ten successful CF scenes per preserved task: {retention_counts}')
    return {'states': len(rows), 'train_scenes': len(train), 'data_holdout_scenes': len(held),
            'train_holdout_overlap': 0,
            'fg_scenes': [{'task': k[0], 'split': k[1], 'scenes': v, 'windows': windows[k]}
                          for k, v in sorted(counts.items())],
            'cf_retention_scenes': retention_counts,
            'native_retention_scenes': {task: len(keys) for task, keys in native.items()},
            'failure_origins': [{'task': k[0], 'kind': k[1], 'scenes': v} for k, v in sorted(origins.items())],
            'policy_prefix_actions': {task: sorted(values) for task, values in prefixes.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', required=True)
    ap.add_argument('--batches', nargs='+', required=True)
    ap.add_argument('--catalog-roots', nargs='+', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--fg-train-scenes', type=int, choices=[24, 48], default=24)
    ap.add_argument('--added-native-scenes-per-task', type=int, choices=[0, 10], default=0)
    args = ap.parse_args()
    from experiments.robotwin.eraf_fg_contract import scene_key, validate_correction
    from scripts.collect_robotwin_eraf_fg import historical_scene_keys
    from experiments.robotwin.eraf_fg_data import retention_language, validate_retention_scene
    base = read(args.base)
    if base.get('complete') is not True:
        raise ValueError('Incomplete base bank.')
    rows = list(base['states'])
    seen = historical_scene_keys(rows)
    inputs = {str(Path(args.base).resolve()): sha(args.base)}
    archive_hashes, added_native = {}, defaultdict(set)
    for batch in args.batches:
        batch_rows, records, retention_records = [], {}, {}
        reports = sorted(Path(batch).glob('shard*/complete.json'))
        if not reports:
            raise ValueError(f'No completed cache shard: {batch}')
        if {read(path)['shard'] for path in reports} != set(range(len(reports))):
            raise ValueError(f'Missing or duplicated cache shard indices: {batch}')
        for path in reports:
            report = read(path)
            if report.get('complete') is not True or len(reports) != report['shards']:
                raise ValueError(f'Incomplete cache batch: {batch}')
            inputs[str(path.resolve())] = sha(path)
            for collection in report['collections']:
                if collection in inputs:
                    continue
                inputs[collection] = sha(collection)
                manifest = read(collection)
                if manifest.get('complete') is not True:
                    raise ValueError(f'Incomplete input record set: {collection}')
                if 'records' in manifest:
                    for item in manifest['records']:
                        record = validate_correction(item)
                        key = scene_key(record)
                        if key in records or key in seen:
                            raise ValueError(f'Duplicate source correction scene: {key}')
                        records[key] = record
                        archive_hashes[record['frame_path']] = record['frame_sha256']
                else:
                    grouped = defaultdict(list)
                    for record in manifest['states']:
                        retention_language(record)
                        if record['id'] in retention_records:
                            raise ValueError('Duplicate retention record ID.')
                        retention_records[record['id']] = record
                        grouped[scene_key(record)].append(record)
                        archive_hashes[record['capture_path']] = record['capture_sha256']
                    for key, group in grouped.items():
                        validate_retention_scene(group)
                        if key in records or key in seen:
                            raise ValueError(f'Duplicate source retention scene: {key}')
                        records[key] = group[0]
            state_path = path.parent / 'states.jsonl'
            shard_rows = [json.loads(line) for line in state_path.read_text().splitlines()]
            if len(shard_rows) != report['states']:
                raise ValueError(f'Cache journal count mismatch: {path}')
            inputs[str(state_path.resolve())] = sha(state_path)
            batch_rows.extend(shard_rows)
        if {scene_key(row) for row in batch_rows} != set(records):
            raise ValueError(f'Prepared/source scene mismatch: {batch}')
        for row in batch_rows:
            source = (retention_records[row['id']] if row.get('native_retention') or row.get('cf_retention')
                      else records[scene_key(row)])
            if any(row.get(key) != value for key, value in source.items()):
                raise ValueError('Prepared row differs from its verified source record.')
            if row.get('native_retention'):
                added_native[row['source_task']].add(scene_key(row))
        seen.update(records)
        rows.extend(batch_rows)
    for row in rows:
        if not Path(row['payload']).is_file():
            raise ValueError(f'Missing replay payload: {row["payload"]}')
        if row.get('fg_correction'):
            archive_hashes[row['frame_path']] = row['frame_sha256']
            archive_hashes[str(Path(row['frame_path']).parent / 'controls.pkl')] = row['controls_sha256']
        elif row.get('capture_sha256'):
            archive_hashes[row['capture_path']] = row['capture_sha256']
    for path, digest in archive_hashes.items():
        if sha(path) != digest:
            raise ValueError(f'Corrective archive identity changed: {path}')
    expected_native = ({task: args.added_native_scenes_per_task for task in
        ('place_a2b_left', 'place_a2b_right', 'place_burger_fries', 'stack_blocks_two', 'blocks_ranking_rgb')}
        if args.added_native_scenes_per_task else {})
    if {task: len(keys) for task, keys in added_native.items()} != expected_native:
        raise ValueError('Added native-retention scenes do not match the declared exact quota.')
    report = audit_rows(rows, fg_train_scenes=args.fg_train_scenes)
    report['added_native_retention_scenes'] = expected_native
    catalog_counts = {}
    for catalog in args.catalog_roots:
        paths = sorted(Path(catalog).glob('*/demo_clean/correct/episodes.jsonl'))
        if len(paths) != 5:
            raise ValueError(f'Expected all five task catalogs: {catalog}')
        catalog_rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
        overlap = historical_scene_keys(rows) & {scene_key(row) for row in catalog_rows}
        if overlap:
            raise ValueError(f'Training/evaluation scene overlap: {overlap}')
        for path in paths:
            inputs[str(path.resolve())] = sha(path)
        catalog_counts[catalog] = len(catalog_rows)
    report.update(complete=True, inputs_sha256=inputs, verified_frame_and_control_archives=len(archive_hashes),
                  evaluation_catalog_scenes=catalog_counts, evaluation_scene_overlap=0)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(base, states=rows, complete=True, parent_manifest=args.base,
                  prepared_batches=args.batches, formal_data_audit=str(output / 'audit.json'))
    (output / 'manifest.json').write_text(json.dumps(result, indent=2))
    report['manifest_sha256'] = sha(output / 'manifest.json')
    (output / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'inputs_sha256'}, indent=2))


if __name__ == '__main__':
    main()
