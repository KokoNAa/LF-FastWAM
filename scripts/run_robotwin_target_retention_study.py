#!/usr/bin/env python3
"""Complete the authorized target-retention development study; never open test."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
TASKS = ['place_a2b_left', 'place_a2b_right', 'place_burger_fries', 'stack_blocks_two', 'blocks_ranking_rgb']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--collection', type=Path, required=True)
    ap.add_argument('--deadline', required=True)
    ap.add_argument('--ranking-scenes', type=int, choices=[10, 11, 12], default=12)
    ap.add_argument('--driver-name', default='remaining_driver')
    args = ap.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        ap.error('Use the explicitly authorized absolute deadline with timezone')
    D, C = args.root.resolve(), args.collection.resolve()
    R = D.parent / 'robotwin_eraf_fg/20260906-warm400-v1'
    parent = R / 'grounding3000/step_002250.pt'
    source = D.parent / 'robotwin_cf_cause_audit/decision-bank-20260905-220457'
    sources = [C / name / 'manifest.json' for name in ('shard3', 'shard4', 'shard5', 'ranking_backfill4')]
    if Path(args.driver_name).name != args.driver_name or args.driver_name in {'', '.', '..'}:
        ap.error('Use a plain new driver directory name')
    root = D / args.driver_name
    root.mkdir(exist_ok=False)
    state = dict(complete=False, deadline=args.deadline, phase='waiting_for_ranking_successes',
                 jobs={}, test_opened=False, ranking_scene_quota=args.ranking_scenes)
    processes = {}
    env = os.environ.copy()
    env.update(PATH='/opt/conda/bin:' + env['PATH'], PYTHONPATH=str(REPO / 'src') + ':' + str(REPO),
               DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
               VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json', OMP_NUM_THREADS='2')

    def save():
        temporary = root / 'driver.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(root / 'driver.json')

    def budget():
        if time.time() >= deadline.timestamp():
            raise TimeoutError('Authorized work cutoff reached')

    def interrupted(signum, frame):
        raise InterruptedError(f'Study interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def start(name, command, gpus=()):
        budget()
        with (root / (name + '.log')).open('x') as log:
            process = subprocess.Popen(list(map(str, command)), cwd=REPO,
                env=env | {'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus))},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = process
        state['jobs'][name] = dict(pid=process.pid, command=list(map(str, command)), gpus=list(gpus))
        save()
        print('[start]', name, process.pid, flush=True)

    def done(name):
        rc = processes[name].poll()
        if rc is None:
            return False
        state['jobs'][name]['exit_code'] = rc
        save()
        if rc:
            raise RuntimeError(f'{name} failed with exit {rc}; see preserved log')
        return True

    def dependency(name):
        path = D / (name + '.exit.json')
        if not path.exists():
            return False
        if json.loads(path.read_text())['exit_code'] != 0:
            raise RuntimeError(f'Dependency {name} failed')
        if not json.loads((D / name / 'driver.json').read_text())['complete']:
            raise RuntimeError(f'Dependency {name} lacks a complete audit')
        return True

    def train(arm, fg, gpus):
        start(arm, [sys.executable, '-u', 'scripts/run_robotwin_cup_ablation.py',
            '--root', D, '--output', D / arm, '--manifest', D / 'cache_all5/manifest.json',
            '--checkpoint', parent, '--source-bank', source, '--fg', fg, '--gpus', *gpus,
            '--steps', 200, '--deadline', args.deadline, '--training-only',
            '--target-tasks', 'place_a2b_left', 'blocks_ranking_rgb',
            '--cf-retention-tasks', *TASKS], gpus)

    def evaluate(arm, gpus):
        plan = json.loads((D / 'baseline_eval_plan.json').read_text())
        plan['models'] = {arm: dict(checkpoint=str(D / arm / 'joint/step_000200.pt'),
                                   manifest=str(D / 'cache_all5/manifest.json'), policy_kind='repair')}
        path = D / (arm + '_eval_plan.json')
        path.write_text(json.dumps(plan, indent=2) + '\n')
        start('eval_' + arm, [sys.executable, '-u', 'scripts/run_robotwin_fixed_eval_matrix.py',
            '--plan', path, '--output', D / ('eval_' + arm), '--gpus', *gpus,
            '--deadline', args.deadline], gpus)

    try:
        save()
        while True:
            budget()
            count = sum(json.loads(p.read_text())['successful_scenes'] for p in sources if p.exists())
            state['available_ranking_scenes'] = count
            save()
            if count >= args.ranking_scenes and all(p.exists() for p in sources):
                break
            if all(p.exists() for p in sources) and all(
                    json.loads(p.read_text()).get('complete') for p in sources):
                raise RuntimeError('Collectors finished below the declared quota')
            time.sleep(5)
        # Stop only timeout supervisors for this collection. Their successful
        # captures are atomically published; incomplete attempts stay recorded.
        stopped = []
        directories = {str(p.parent) for p in sources}
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                argv = (entry / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
                if (Path(argv[0]).name == 'timeout' and 'scripts/collect_robotwin_cf_retention.py' in argv
                        and '--output' in argv and argv[argv.index('--output') + 1] in directories):
                    pid = int(entry.name)
                    os.kill(pid, signal.SIGTERM)
                    stopped.append(pid)
            except (FileNotFoundError, ProcessLookupError):
                pass
        state.update(phase='freeze_success_pool', collectors_stopped_after_pool_quota=stopped)
        save()
        for _ in range(30):
            budget()
            if all(not (Path('/proc') / str(pid)).exists() for pid in stopped):
                break
            time.sleep(1)
        else:
            raise RuntimeError('Owned collector supervisors did not exit after quota stop')
        start('curate', [sys.executable, 'scripts/curate_robotwin_retention_pool.py',
            '--collections', *sources, '--parent-manifest', D / 'cache_left_cf/manifest.json',
            '--output', D / 'ranking_pool/manifest.json', '--task', 'blocks_ranking_rgb', '--scenes', args.ranking_scenes])
        while not done('curate'):
            budget(); time.sleep(1)
        state['phase'] = 'cache_ranking'; save()
        base = ['--manifest', D / 'cache_left_cf/manifest.json', '--checkpoint', parent,
                '--output', D / 'cache_all5', '--collections', D / 'ranking_pool/manifest.json', '--shards', 3]
        for shard, gpu in enumerate((3, 4, 5)):
            start(f'cache{shard}', [sys.executable, '-u', 'scripts/prepare_robotwin_eraf_fg_replay.py',
                'worker', *base, '--shard', shard, '--gpu', gpu], [gpu])
        while not all(done(f'cache{i}') for i in range(3)):
            budget(); time.sleep(5)
        start('merge', [sys.executable, 'scripts/prepare_robotwin_eraf_fg_replay.py', 'merge', *base])
        while not done('merge'):
            budget(); time.sleep(1)
        from experiments.robotwin.eraf_fg_data import validate_cf_retention_coverage
        rows = json.loads((D / 'cache_all5/manifest.json').read_text())['states']
        state['cf_retention_scenes'] = validate_cf_retention_coverage(rows, TASKS)
        state['phase'] = 'train_and_evaluate'; save()
        train('all5_off200', 'off', [3, 4])
        while True:
            budget()
            if 'all5_full200' not in processes and dependency('eval_old3'):
                train('all5_full200', 'full', [0, 1])
            if 'eval_all5_off200' not in processes and done('all5_off200'):
                evaluate('all5_off200', [3, 4, 5])
            if ('all5_full200' in processes and 'eval_all5_full200' not in processes
                    and done('all5_full200')):
                evaluate('all5_full200', [0, 1, 2] if dependency('eval_baseline') else [0, 1])
            if all(name in processes and done(name) for name in ('eval_all5_off200', 'eval_all5_full200')):
                break
            time.sleep(5)
        state.update(complete=True, phase='development_complete_test_unopened')
    except BaseException as error:
        state['error'] = repr(error)
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
        save()


if __name__ == '__main__':
    sys.path[:0] = [str(REPO), str(REPO / 'src')]
    main()
