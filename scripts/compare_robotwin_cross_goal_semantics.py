#!/usr/bin/env python3
"""Compare semantic probes only after matching actual observations and labels."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def identity(row):
    fields = ('pair_id', 'task_config', 'scene_seed', 'observation_language',
              'instruction_language', 'raw_path', 'frame', 'instruction',
              'state_sha256', 'rgb_sha256', 'crossed')
    result = {key: row[key] for key in fields}
    result['clauses'] = [{k: c[k] for k in ('clause', 'truth_label')} for c in row['clauses']]
    return result


def compare(reports, *, expected_queries=80, expected_scenes=20, allow_distinct_manifests=False):
    baseline = None
    models = {}
    manifests = {}
    for name, report in reports.items():
        if not report['complete'] or report['action_quality_evaluated']:
            raise ValueError('Require complete semantic-only reports.')
        rows = report['records']
        keys = [(r['pair_id'], r['task_config'], r['scene_seed'], r['observation_language'],
                 r['instruction_language'], r['frame']) for r in rows]
        if len(rows) != expected_queries or len(set(keys)) != len(rows):
            raise ValueError('Missing or duplicate queries.')
        if len({k[:3] for k in keys}) != expected_scenes:
            raise ValueError('Unexpected scene count.')
        ordered = sorted(rows, key=lambda r: json.dumps(identity(r), sort_keys=True))
        manifests[name] = report['manifest_sha256']
        inputs = dict(records=[identity(r) for r in ordered])
        if not allow_distinct_manifests:
            inputs['manifest_sha256'] = report['manifest_sha256']
        if baseline is None:
            baseline = inputs
        elif inputs != baseline:
            raise ValueError('Actual observations, instructions, manifest, or labels differ.')
        groups = defaultdict(lambda: dict(queries=0, clauses=0, positives=0, negatives=0,
            true_positives=0, false_positives=0, hits=0, goal_error_cm_sum=0.))
        for row in rows:
            if row['crossed'] != (row['observation_language'] != row['instruction_language']):
                raise ValueError('Cross-goal flag contradicts language branches.')
            if len({c['clause'] for c in row['clauses']}) != len(row['clauses']):
                raise ValueError('Duplicate clauses.')
            for group in (groups[('all', row['crossed'])], groups[(row['pair_id'], row['crossed'])]):
                group['queries'] += 1
                for c in row['clauses']:
                    p, error = c['truth_probability'], c['goal_error_cm']
                    if not math.isfinite(p) or not 0 <= p <= 1 or not math.isfinite(error) or error < 0:
                        raise ValueError('Invalid prediction or error.')
                    truth, pred = bool(c['truth_label']), p >= .5
                    group['clauses'] += 1
                    group['positives'] += truth
                    group['negatives'] += not truth
                    group['true_positives'] += truth and pred
                    group['false_positives'] += not truth and pred
                    group['hits'] += truth == pred
                    group['goal_error_cm_sum'] += error
        own, cross = groups[('all', False)], groups[('all', True)]
        # This diagnostic is specifically for own-true/opposite-false endpoints.
        if own['negatives'] or cross['positives'] or not own['positives'] or not cross['negatives']:
            raise ValueError('Endpoint labels do not meet this diagnostic contract.')
        models[name] = dict(checkpoint_sha256=report['checkpoint_sha256'], own=own, crossed=cross,
            balanced_clause_accuracy=(own['true_positives']/own['positives'] +
                1-cross['false_positives']/cross['negatives'])/2,
            cells=[dict(pair_id=k[0], crossed=k[1], **v) for k, v in sorted(groups.items()) if k[0] != 'all'])
    if baseline is None:
        raise ValueError('No reports supplied.')
    return dict(complete=True, format='robotwin_cross_goal_terminal_comparison_v3' if allow_distinct_manifests else 'robotwin_cross_goal_terminal_comparison_v2',
        matched_queries=expected_queries, matched_scenes=expected_scenes, inputs_and_labels_match=True,
        manifest_sha256_by_model=manifests, distinct_training_manifests_allowed=allow_distinct_manifests,
        query_sha256=hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest(),
        models=models, action_quality_evaluated=False,
        scope='Fixed held-out own/opposite-goal semantic endpoints. Report recall and specificity together; this is not a CF rollout score.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reports', type=Path, required=True, help='JSON mapping of model names to report paths')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--allow-distinct-manifests', action='store_true',
                    help='Permit changed training banks while still matching every actual diagnostic input and label.')
    args = ap.parse_args()
    result = compare({k: json.loads(Path(v).read_text()) for k, v in json.loads(args.reports.read_text()).items()},
                     allow_distinct_manifests=args.allow_distinct_manifests)
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
