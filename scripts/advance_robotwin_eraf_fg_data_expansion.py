#!/usr/bin/env python3
"""Finish the predeclared data expansion, audit its caches, and stop before training."""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from scripts.advance_robotwin_eraf_fg_campaign import read, write
from scripts.assemble_robotwin_eraf_fg_bank import sha

SMOKES = ('native_left_smoke', 'native_ranking_smoke')


def audit_native_cache(collection, cache):
    """Exercise compact restoration and condition checks on every smoke state."""
    import torch
    from experiments.robotwin.compact_replay import ReplayPayloads
    from experiments.robotwin.eraf_fg_data import validate_retention_scene
    source = read(collection)
    if not source['complete'] or source['condition'] != 'correct' or source['successful_scenes'] != 1:
        raise ValueError('Native smoke is not one complete Correct success.')
    validate_retention_scene(source['states'])
    report = read(cache / 'shard0/complete.json')
    rows = [json.loads(line) for line in (cache / 'shard0/states.jsonl').read_text().splitlines()]
    if not report['complete'] or report['states'] != len(rows) or len(rows) != len(source['states']):
        raise ValueError('Native smoke cache count mismatch.')
    original = {row['id']: row for row in source['states']}
    if set(original) != {row['id'] for row in rows}:
        raise ValueError('Native smoke cache identities changed.')
    payloads = ReplayPayloads(rows, 'cpu', capacity=1)
    for row in rows:
        if any(row.get(k) != v for k, v in original[row['id']].items()):
            raise ValueError('Prepared native row differs from successful capture.')
        payload = payloads[row['id']]
        if any(set(payload[key]) != {'source'} for key in ('captured', 'references', 'valid')):
            raise ValueError('Native retention was cached under a CF language.')
        ref, valid = payload['references']['source'], payload['valid']['source']
        captured = payload['captured']['source']
        if ref.shape != (1, 32, 14) or not torch.isfinite(ref).all() or not valid.all():
            raise ValueError('Invalid full teacher action horizon.')
        if captured['policy_guard_state'] is not None or captured['proprio'].shape != (1, 14):
            raise ValueError('Native capture has invalid proprio or carried memory.')
        if row['policy_memory'] != 'none' or row['fg_correction']:
            raise ValueError('Native retention was mislabeled as an FG correction.')
    return {'complete': True, 'states': len(rows), 'collection_sha256': sha(collection),
            'state_journal_sha256': sha(cache / 'shard0/states.jsonl'),
            'language': 'source', 'compact_payloads_restored': len(rows)}


def verified_snapshot_done(data, name, job):
    """Admit complete individual FG records without relabeling a failed parent."""
    if not job.get('verified_snapshot'):
        return False
    from experiments.robotwin.eraf_fg_contract import validate_correction, scene_key
    path = data / name / 'manifest.json'
    source = read(path)
    if sha(path) != job['snapshot_sha256'] or not source.get('complete'):
        raise ValueError('Verified-record recovery snapshot changed.')
    records = [validate_correction(r) for r in source['records']]
    if len(records) != job['scenes'] or len({scene_key(r) for r in records}) != len(records):
        raise ValueError('Verified-record recovery quota differs.')
    originals = []
    for path, digest in source['record_paths_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Recovered source record changed.')
        originals.append(validate_correction(read(path)))
    if len(source['record_paths_sha256']) != len(records):
        raise ValueError('Missing independent source record provenance.')
    if sorted(records, key=scene_key) != sorted(originals, key=scene_key):
        raise ValueError('Recovery snapshot differs from its verified source records.')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    data = root / 'data_expansion_v2'
    lock = (data / 'coordinator.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(data / 'plan.json')
    if plan['policy_training_started'] or sha(plan['base']) != plan['base_sha256']:
        raise ValueError('Expansion plan parent changed or training already started.')
    deadline = datetime.fromisoformat(plan['stop_experiments_hkt'])
    jobs = plan['jobs']
    if (sum(j['scenes'] for j in jobs.values() if j['kind'] == 'fg') != 48
            or sum(j['scenes'] for j in jobs.values() if j['kind'] == 'native') != 50):
        raise ValueError('Expansion exceeds or differs from the declared budget.')
    plan.setdefault('cache_jobs', {})
    caches = plan['cache_jobs']
    plan['coordinator_code'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    plan['coordinator_pid'] = os.getpid()
    write(data / 'plan.json', plan)
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + ':' + os.environ['PATH'],
        PYTHONPATH=str(REPO / 'src') + ':' + str(REPO), OMP_NUM_THREADS='2',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
    wrapper = ('import subprocess,sys,json,pathlib; r=subprocess.run(json.loads(sys.argv[1])); '
        'pathlib.Path(sys.argv[2]).write_text(json.dumps(dict(exit_code=r.returncode))); sys.exit(r.returncode)')

    def owned(job):
        path = Path(f'/proc/{job["pid"]}/cmdline')
        return (path.exists() and str(data).encode() in path.read_bytes()
                and os.getpgid(job['pid']) == job['pid'])

    def done(name, job):
        marker = data / (name + '_exit.json')
        if marker.exists():
            if read(marker)['exit_code'] != 0:
                raise RuntimeError(f'{name} failed; retain partial data and inspect its log.')
            return True
        if not owned(job):
            raise RuntimeError(f'{name} vanished without a successful exit marker.')
        return False

    def budget():
        if datetime.now() >= deadline:
            raise RuntimeError('Archive/shutdown reserve reached.')

    def start(name, command, gpu, container, seconds=7200):
        budget()
        seconds = min(seconds, int((deadline - datetime.now()).total_seconds()))
        command = ['timeout', '--foreground', '--signal=TERM', '--kill-after=30s', f'{seconds}s'] + command
        with (data / (name + '.log')).open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', '-c', wrapper, json.dumps(command),
                str(data / (name + '_exit.json'))], cwd=REPO, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        container.setdefault(name, {}).update(pid=proc.pid, gpu=gpu, command=command,
            started_at=datetime.now().isoformat())
        write(data / 'plan.json', plan)
        print(f'[start] {name} gpu={gpu} pid={proc.pid}', flush=True)

    def stop_owned():
        for name, job in list(jobs.items()) + list(caches.items()):
            if 'pid' not in job or (data / (name + '_exit.json')).exists():
                continue
            try:
                if owned(job):
                    os.killpg(job['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass

    def interrupt(signum, frame):
        raise RuntimeError(f'Coordinator received signal {signum}.')

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        while True:
            budget()
            finished = {name for name, j in jobs.items() if verified_snapshot_done(data, name, j)
                        or ('pid' in j and done(name, j))}
            cached = {name for name, j in caches.items() if done(name, j)}
            for name in SMOKES:
                marker = data / (name + '_cache_audit.json')
                if 'cache_' + name in cached and not marker.exists():
                    report = audit_native_cache(data / name / 'manifest.json', data / ('cache_' + name))
                    write(marker, report)
                    print(f'[verified] {name} Correct cache: {report["states"]} states', flush=True)
            smoke_ok = all((data / (name + '_cache_audit.json')).exists() for name in SMOKES)
            if len(cached) == len(jobs):
                break
            # Acquire the second pool only after its complete evaluation handover.
            pools = {3, 4, 5}
            prior = root / 'arm_eraf_geometry_full200_retain4cf2/complete.json'
            if prior.exists() and read(prior)['complete']:
                pools.update({0, 1, 2})
            gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                '--format=csv,noheader,nounits'], text=True)
            gpu_map = {u.strip(): int(i) for i, u in (s.split(',') for s in gpu_rows.splitlines())}
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                '--format=csv,noheader,nounits'], text=True)
            busy = {gpu_map[s.split(',')[0].strip()] for s in apps.splitlines() if s.strip()}
            busy.update(j['gpu'] for n, j in jobs.items() if 'pid' in j and n not in finished)
            busy.update(j['gpu'] for n, j in caches.items() if n not in cached)
            ready_caches = sorted(finished - {name.removeprefix('cache_') for name in caches},
                                  key=lambda n: (n not in SMOKES, n))
            ready_collections = [name for name, j in jobs.items() if 'pid' not in j and name not in finished
                                 and (j['kind'] == 'fg' or smoke_ok or name in SMOKES)]
            for gpu in sorted(pools - busy):
                if ready_caches:
                    source = ready_caches.pop(0)
                    name = 'cache_' + source
                    collection = data / source / 'manifest.json'
                    if not read(collection)['complete']:
                        raise ValueError('Collector exited without complete requested quota.')
                    command = [sys.executable, '-u', 'scripts/prepare_robotwin_eraf_fg_replay.py', 'worker',
                        '--manifest', plan['base'], '--checkpoint', str(root / 'grounding3000/step_002250.pt'),
                        '--collections', str(collection), '--output', str(data / name), '--gpu', str(gpu)]
                    start(name, command, gpu, caches)
                elif ready_collections:
                    name = ready_collections.pop(0)
                    job = jobs[name]
                    fg = job['kind'] == 'fg'
                    command = [sys.executable, '-u', 'scripts/collect_robotwin_eraf_fg.py' if fg else
                        'scripts/collect_robotwin_cf_retention.py', '--manifest', plan['base'],
                        '--checkpoint', plan['fg_teacher' if fg else 'native_teacher'],
                        '--output', str(data / name), '--robotwin-root',
                        '/root/gpufree-data/LF-FastWAM/third_party/RoboTwin', '--gpu', str(gpu)]
                    for flag in ('task', 'start_seed', 'scenes', 'max_attempts'):
                        command += ['--' + flag.replace('_', '-'), str(job[flag])]
                    command += (['--holdout-scenes', '0', '--candidate-order', 'early_first'] if fg
                                else ['--condition', 'correct'])
                    start(name, command, gpu, jobs, seconds=7200)
            time.sleep(15)
        budget()
        bank = root / 'bank_full_v2'
        if bank.exists():
            raise FileExistsError('Preserve an existing expanded bank; review instead of overwriting.')
        fixed = '/root/gpufree-data/LF-FastWAM/evaluate_results/robotwin/robotwin_uncond_3cam_384/cf-improvement-20260905-235608-base'
        subprocess.run([sys.executable, '-u', 'scripts/assemble_robotwin_eraf_fg_bank.py',
            '--base', plan['base'], '--batches', *[str(data / n) for n in sorted(caches)],
            '--catalog-roots', fixed, str(root / 'catalog_dev'), str(root / 'catalog_test'),
            '--fg-train-scenes', '48', '--added-native-scenes-per-task', '10', '--output', str(bank)],
            cwd=REPO, env=env, check=True, timeout=max(1, int((deadline - datetime.now()).total_seconds())))
        audit = read(bank / 'audit.json')
        if not audit['complete'] or sha(bank / 'manifest.json') != audit['manifest_sha256']:
            raise ValueError('Expanded bank identity audit failed.')
        write(data / 'complete.json', {'complete': True, 'bank': str(bank / 'manifest.json'),
            'manifest_sha256': audit['manifest_sha256'], 'finished_at': datetime.now().isoformat(),
            'next_action': 'Review expanded data audit before any fresh policy training.'})
        print('[complete] Bounded expanded bank audited; no action training started.', flush=True)
    finally:
        stop_owned()


if __name__ == '__main__':
    main()
