#!/usr/bin/env python3
"""Advance the authorized warm400 campaign through its first matched candidate.

The campaign's launch.json and cache_batches/ledger.json are authoritative.
This bounded driver collects no new seed ranges, chooses no checkpoint from
evaluation outcomes, and stops after interface100 + joint200 and both dev sets.
"""
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
PYTHON = sys.executable
COLLECTORS = {'left_collection28': 0, 'ranking_collection_part0': 1,
              'ranking_collection_part1': 2, 'ranking_collection_part2': 4}
TASKS = ['blocks_ranking_rgb', 'place_a2b_left', 'stack_blocks_two',
         'place_a2b_right', 'place_burger_fries']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    lock = (root / 'first_candidate_driver.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    launch = read(root / 'launch.json')
    deadline = datetime.fromisoformat(launch['stop_experiments_hkt'])
    if datetime.now() >= deadline:
        raise RuntimeError('The experimental budget has expired.')
    env = os.environ.copy()
    env.update(PATH=str(Path(PYTHON).parent) + ':' + env['PATH'],
               PYTHONPATH=str(REPO / 'src') + ':' + str(REPO), OMP_NUM_THREADS='2',
               DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
               VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
    wrapper = ('import subprocess,sys,json,pathlib; r=subprocess.run(json.loads(sys.argv[1])); '
               'pathlib.Path(sys.argv[2]).write_text(json.dumps(dict(exit_code=r.returncode))); '
               'sys.exit(r.returncode)')

    def budget():
        if datetime.now() >= deadline:
            for name, job in launch['jobs'].items():
                if (name not in COLLECTORS and not name.startswith('cache_fg_batch')
                        and job.get('driver') != 'first_candidate'):
                    continue
                if (root / (name + '_exit.json')).exists():
                    continue
                pid = job['pid']
                command_file = Path(f'/proc/{pid}/cmdline')
                if command_file.exists() and str(root).encode() in command_file.read_bytes():
                    if os.getpgid(pid) == pid:
                        os.killpg(pid, signal.SIGTERM)
            raise RuntimeError('Reserved archive/shutdown time reached; no new work may launch.')

    def done(name):
        path = root / (name + '_exit.json')
        if path.exists():
            status = read(path)
            if status['exit_code'] != 0:
                raise RuntimeError(f'{name} failed: {status}')
            return True
        pid = launch['jobs'][name]['pid']
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise RuntimeError(f'{name} disappeared without an exit marker.') from None
        return False

    def start(name, command, gpus):
        budget()
        if name in launch['jobs']:
            return
        job_env = dict(env, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)))
        with (root / (name + '.log')).open('x') as log:
            process = subprocess.Popen([PYTHON, '-u', '-c', wrapper, json.dumps(command),
                                        str(root / (name + '_exit.json'))],
                cwd=REPO, env=job_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch['jobs'][name] = {'pid': process.pid, 'command': command,
            'gpus': ','.join(map(str, gpus)), 'started_at': datetime.now().isoformat(),
            'code': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'driver': 'first_candidate'}
        write(root / 'launch.json', launch)
        print(f'[start] {name} gpu={gpus} pid={process.pid}', flush=True)

    def wait_for(names):
        while not all(done(name) for name in names):
            budget()
            time.sleep(20)
        print('[complete] ' + ','.join(names), flush=True)

    from experiments.robotwin.eraf_fg_contract import FORMAT, validate_correction
    ledger_path = root / 'cache_batches/ledger.json'
    ledger = read(ledger_path)
    cache_gpus = [3, 5, 0, 1, 2, 4]
    last_snapshot = None
    while True:
        budget()
        finished = {name: done(name) for name in COLLECTORS}
        batches = ledger['batches']
        cache_finished = {b['name']: done(b['name']) for b in batches}
        busy = {gpu for name, gpu in COLLECTORS.items() if not finished[name]}
        for name, complete in cache_finished.items():
            if not complete:
                busy.update(map(int, launch['jobs'][name]['gpus'].split(',')))
        used = {path for batch in batches for path in batch['record_paths']}
        sources = [root / name for name in COLLECTORS] + [root / 'ranking_collection28']
        paths = sorted(path for folder in sources for path in folder.glob('scene_*/record.json')
                       if str(path) not in used)
        near_finish = sum(not value for value in finished.values()) <= 1
        available = [gpu for gpu in cache_gpus if gpu not in busy]
        for position, gpu in enumerate(available):
            if not paths or (len(paths) < 6 and not near_finish):
                continue
            # Once only the final collector remains, spread its small tail
            # across free GPUs instead of waiting for another six-scene batch.
            workers_left = len(available) - position
            count = min(12, (len(paths) + workers_left - 1) // workers_left) if near_finish else 12
            selected, paths = paths[:count], paths[count:]
            records = [validate_correction(read(path)) for path in selected]
            index = len(ledger['batches'])
            name = f'cache_fg_batch{index}'
            collection = root / f'cache_batches/batch{index}_records.json'
            if collection.exists():
                raise ValueError(f'Untracked cache snapshot already exists: {collection}')
            write(collection, {'format': FORMAT, 'complete': True,
                'requested_scenes': len(records), 'records': records,
                'provenance': 'Complete bounded verified-record snapshot; parent job status is separate',
                'record_paths': list(map(str, selected))})
            command = list(launch['jobs']['cache_fg_batch0']['command'])
            for flag, value in {'--output': root / name, '--collections': collection, '--gpu': gpu}.items():
                command[command.index(flag) + 1] = str(value)
            start(name, command, [gpu])
            ledger['batches'].append({'name': name, 'collection': str(collection),
                'record_paths': list(map(str, selected)), 'records': len(records)})
            write(ledger_path, ledger)
        snapshot = {name: len(list((root / name).glob('scene_*/record.json'))) for name in COLLECTORS}
        snapshot['cache_batches'] = len(ledger['batches'])
        snapshot['completed_cache_batches'] = sum(cache_finished.values())
        if snapshot != last_snapshot:
            print('[data] ' + json.dumps(snapshot), flush=True)
            last_snapshot = snapshot
        if all(finished.values()) and not paths and all(done(b['name']) for b in ledger['batches']):
            break
        time.sleep(20)

    original = Path('/root/gpufree-data/LF-FastWAM')
    fixed_catalog = original / 'evaluate_results/robotwin/robotwin_uncond_3cam_384/cf-improvement-20260905-235608-base'
    bank = root / 'bank_full_v1/manifest.json'
    if not bank.exists():
        budget()
        subprocess.run([PYTHON, '-u', 'scripts/assemble_robotwin_eraf_fg_bank.py',
            '--base', str(root / 'bank_smoke_retention.json'), '--batches',
            *[str(root / b['name']) for b in ledger['batches']], '--catalog-roots',
            str(fixed_catalog), str(root / 'catalog_dev'), str(root / 'catalog_test'),
            '--output', str(bank.parent)], cwd=REPO, env=env, check=True)
    from scripts.assemble_robotwin_eraf_fg_bank import sha
    audit = read(bank.parent / 'audit.json')
    if audit['complete'] is not True or sha(bank) != audit['manifest_sha256']:
        raise ValueError('Formal data audit incomplete.')

    template = launch['jobs']['interface_smoke']['command']
    for name, stage, steps, lr, checkpoint in [
        ('interface_full100', 'interface', 100, '5e-5', root / 'grounding3000/step_002250.pt'),
        ('joint_full200', 'joint', 200, '1e-5', root / 'interface_full100/step_000100.pt'),
    ]:
        command = list(template)
        for flag, value in {'--manifest': bank, '--checkpoint': checkpoint, '--output': root / name,
                             '--stage': stage, '--steps': steps, '--save-every': steps,
                             '--learning-rate': lr}.items():
            command[command.index(flag) + 1] = str(value)
        command = [PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=6'] + command[2:]
        start(name, command, list(range(6)))
        wait_for([name])

    checkpoint = root / 'joint_full200/step_000200.pt'
    evaluations = [(kind, task, episodes, catalog) for task in TASKS
                   for kind, episodes, catalog in [('reg', 3, fixed_catalog), ('dev', 6, root / 'catalog_dev')]]
    running = {}
    while evaluations or running:
        budget()
        for gpu, name in list(running.items()):
            if done(name):
                print(f'[complete] {name}', flush=True)
                del running[gpu]
        for gpu in range(6):
            if gpu in running or not evaluations:
                continue
            kind, task, episodes, catalog = evaluations.pop(0)
            name = f'{kind}_full200_{task}'
            command = [PYTHON, '-u', 'scripts/eval_robotwin_eraf_fg.py', 'worker',
                '--manifest', str(bank), '--checkpoint', str(checkpoint), '--eraf', 'on',
                '--catalog-root', str(catalog), '--output', str(root / f'{kind}_full200'),
                '--tasks', task, '--episodes', str(episodes), '--gpu', str(gpu), '--videos']
            start(name, command, [gpu])
            running[gpu] = name
        if running:
            time.sleep(20)
    summaries = {}
    for kind, episodes, catalog in [('reg', 3, fixed_catalog), ('dev', 6, root / 'catalog_dev')]:
        output = root / f'{kind}_full200'
        subprocess.run([PYTHON, '-u', 'scripts/eval_robotwin_eraf_fg.py', 'summarize',
            '--output', str(output), '--checkpoint', str(checkpoint),
            '--catalog-root', str(catalog), '--episodes', str(episodes)], cwd=REPO, env=env, check=True)
        summaries[kind] = str(output / 'summary.json')
    write(root / 'first_candidate_complete.json', {'complete': True, 'checkpoint': str(checkpoint),
          'summaries': summaries, 'finished_at': datetime.now().isoformat(),
          'next_action': 'Review fixed regression and development results before continuation or ablations.'})
    print('[complete] First candidate is ready for review; locked test remains untouched.', flush=True)


if __name__ == '__main__':
    main()
