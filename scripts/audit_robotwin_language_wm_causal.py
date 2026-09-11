#!/usr/bin/env python3
"""Independently check frozen artifacts and paired controls, without GPU inference."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    repo = Path(__file__).resolve().parents[1]
    plan = read(root / 'plan.json')
    ph = sha(root / 'plan.json')
    status = read(root / 'status.json')
    assert status['compute_complete'] and not status['active_jobs']
    assert len(status['jobs']) == 6
    processes = []
    for label, job in status['jobs'].items():
        assert job['exit_code'] == 0 and job['verified_states'] == 10
        stat = Path(f"/proc/{job['pid']}/stat")
        same = stat.exists() and stat.read_text().rsplit(') ', 1)[1].split()[19] == job['start_time']
        assert not same, f'Owned job still alive: {label}'
        processes.append(dict(job=label, exit_code=0, owned_process_alive=False))
    for path, expected in plan['input_sha256'].items():
        assert sha(path) == expected, path
    assert sha(plan['source_plan']) == plan['source_plan_sha256']
    for path, expected in plan['source_sha256'].items():
        assert sha(repo / path) == expected, path
    inputs = defaultdict(set)
    fixed_noises = defaultdict(set)
    gen_noises = defaultdict(set)
    gen_clamps = defaultdict(set)
    schedule = set()
    rows = []
    checks = []
    generations = []
    archived = {root / 'plan.json', root / 'status.json'}
    output_files = 0
    state_count = 0
    latent_equalities = []
    max_joint = 0.
    for model in plan['models']:
        for seed in plan['noise_seeds']:
            for state in plan['states']:
                folder = root / 'results' / model / f'seed{seed}' / state['id']
                proof = read(folder / 'complete.json')
                assert proof['complete'] and proof['plan_sha256'] == ph and not proof['smoke']
                for name, expected in proof['output_sha256'].items():
                    assert sha(folder / name) == expected, str(folder / name)
                    output_files += 1
                    if name.endswith('.json') or name == 'cross_attention_spatial.npz':
                        archived.add(folder / name)
                archived.add(folder / 'complete.json')
                inp = read(folder / 'inputs.json')
                assert inp['plan_sha256'] == ph
                inputs[state['id']].add(inp['observation_sha256'])
                assert all(v['shape'] == [1, 129, 4096] for v in inp['contexts'].values())
                fixed = read(folder / 'fixed.json')
                assert fixed['complete'] and len(fixed['rows']) == 84 and len(fixed['checks']) == 4
                for check in fixed['checks']:
                    assert check['repeat']['max_abs'] == check['self_patch']['max_abs'] == 0
                    assert check['full_patch_recovered'] and check['all_mask_language_equal']
                    if check['joint_equivalence'] is not None:
                        max_joint = max(max_joint, check['joint_equivalence']['max_abs'])
                    fixed_noises[state['id'], seed, check['reference'], check['sigma']].add(
                        (check['noise_sha256'], check['noisy_sha256']))
                    checks.append(check)
                for ref in ['source', 'target']:
                    for sigma in plan['sigmas']:
                        group = [r for r in fixed['rows'] if r['reference'] == ref and r['sigma'] == sigma]
                        assert len(group) == 21 and len({r['noisy_sha256'] for r in group}) == 1
                rows.extend(dict(model=model, task=state['task'], scene=state['scene_seed'], **r)
                            for r in fixed['rows'])
                state_count += 1
                if seed not in plan['generation_seeds'] or state['id'] not in plan['generation_states']:
                    continue
                gf = folder / 'generated'
                gen = read(gf / 'generations.json')
                assert gen['complete'] and gen['clips'] == len(gen['rows']) == 27
                assert {r['condition']['name'] for r in gen['rows']} == {c['name'] for c in plan['generation_conditions']}
                for g in gen['rows']:
                    assert g['frames'] == 9
                    pr = g['proof']
                    gen_noises[state['id'], seed].add(pr['initial_noise_sha256'])
                    gen_clamps[state['id'], seed].add(pr['initial_clamped_sha256'])
                    assert len(pr['steps']) == 20
                    schedule.add(tuple((s['timestep'], s['delta']) for s in pr['steps']))
                    c = g['condition']
                    for i, step in enumerate(pr['steps']):
                        stage = c['stage']
                        active = stage == 'all' or (stage == 'early' and i < 6) or (stage == 'middle' and 6 <= i < 13) or (stage == 'late' and i >= 13)
                        assert step['step'] == i and step['intervention_active'] == active
                        assert step['donor_same_input'] == (c['mode'] == 'patch' and active)
                    generations.append(g)
                for a, b in [('source_patch_target_layer_all', 'target'),
                             ('target_patch_source_layer_all', 'source'),
                             ('source_mask_layer_all', 'target_mask_layer_all')]:
                    with np.load(gf / a / 'latent.npz') as fa, np.load(gf / b / 'latent.npz') as fb:
                        error = float(np.max(np.abs(fa['latent'] - fb['latent'])))
                    assert error == 0, (model, state['id'], a, b, error)
                    latent_equalities.append(dict(model=model, state=state['id'], first=a, second=b, max_abs=error))
    assert state_count == 60 and len(rows) == 5040 and len(checks) == 240 and len(generations) == 270
    assert all(len(v) == 1 for v in inputs.values())
    assert all(len(v) == 1 for v in fixed_noises.values())
    assert all(len(v) == 1 for v in gen_noises.values()) and all(len(v) == 1 for v in gen_clamps.values())
    assert len(schedule) == 1 and max_joint == 0
    # Recompute every reported wrong-minus-correct scene mean from the immutable rows.
    grouped = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ['model', 'task', 'scene', 'seed', 'reference', 'sigma'])
        assert row['condition'] not in grouped[key]
        grouped[key][row['condition']] = row
    q = read(root / 'report' / 'quantitative.json')
    reconstructed = 0
    for s in q['statistics']:
        if not s['metric'].startswith('wrong_minus_correct_'):
            continue
        region = s['metric'].removeprefix('wrong_minus_correct_')
        scene_values = defaultdict(list)
        for (model, task, scene, seed, ref, sigma), group in grouped.items():
            if (model, task, ref, sigma) != (s['model'], s['task'], s['reference'], s['sigma']):
                continue
            suffix = '' if s['condition'] == 'baseline' else '_' + s['condition']
            wrong = 'target' if ref == 'source' else 'source'
            scene_values[str(scene)].append(group[wrong + suffix]['errors'][region] - group[ref + suffix]['errors'][region])
        means = {k: float(np.mean(v)) for k, v in scene_values.items()}
        assert means == s['scene_means']
        assert float(np.mean(list(means.values()))) == s['mean']
        reconstructed += 1
    result = dict(format='robotwin_language_wm_runtime_data_audit_v1', complete=True,
                  completed_at=datetime.now(timezone.utc).isoformat(), plan_sha256=ph,
                  states=state_count, fixed_rows=len(rows), fixed_controls=len(checks), generated_clips=len(generations),
                  hashed_output_files=output_files, hashed_input_files=len(plan['input_sha256']),
                  processes=processes, paired_observations_equal=True, paired_noise_equal=True,
                  joint_video_max_abs=max_joint, generation_latent_equalities=latent_equalities,
                  reconstructed_margin_statistics=reconstructed, schedule=list(next(iter(schedule))),
                  gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
                  scope='Artifact and numeric controls only. Visual interpretation is recorded separately.',
                  audit_script_sha256=sha(__file__))
    audit = root / 'report' / 'runtime_data_audit.json'
    audit.write_text(json.dumps(result, indent=2) + '\n')
    archived.add(audit)
    archive = root / 'report' / 'raw_measurements.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(archived):
            tar.add(path, arcname=str(path.relative_to(root)))
    manifest = dict(path=str(archive), sha256=sha(archive), bytes=archive.stat().st_size,
                    files=len(archived), checkpoints_included=False)
    (root / 'report' / 'raw_archive.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['complete','states','fixed_rows','generated_clips','reconstructed_margin_statistics']}))
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
