#!/usr/bin/env python3
"""Evaluate frozen ERAF candidates after primary workers release their GPUs."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--deadline', required=True)
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None:
        ap.error('The authorized absolute cutoff needs a timezone')
    D = args.root.resolve()
    root = D / 'eraf_followup'
    root.mkdir(exist_ok=False)
    interface = D / 'eraf_interface_pilot10/interface/step_000010.pt'
    interface_audit = json.loads((D / 'eraf_interface_pilot10/driver.json').read_text())['weight_audit']
    if not interface_audit['policy_adapters_equal'] or not interface_audit['only_declared_interface_parameters_changed']:
        raise ValueError('Interface pilot has no verified frozen-policy checkpoint')
    plan = dict(complete=False, deadline=args.deadline, jobs={}, test_opened=False,
                optimizer_updates=0, interface_audit=interface_audit,
                purpose='Development evaluation of fixed interface and composed FG+interface candidates')
    processes = {}
    env = os.environ.copy()
    env.update(PATH='/opt/conda/bin:' + env['PATH'], PYTHONPATH=str(REPO / 'src') + ':' + str(REPO),
               DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
               VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json', OMP_NUM_THREADS='2')

    def save():
        temp = root / 'driver.json.tmp'
        temp.write_text(json.dumps(plan, indent=2) + '\n')
        temp.replace(root / 'driver.json')

    def budget():
        if time.time() >= cutoff.timestamp():
            raise TimeoutError('Authorized work cutoff reached')

    def stop(signum, frame):
        raise InterruptedError(f'Followup interrupted by {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def start(name, command, gpus=()):
        budget()
        with (root / (name + '.log')).open('x') as log:
            p = subprocess.Popen(list(map(str, command)), cwd=REPO,
                env=env | {'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus))},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = p
        plan['jobs'][name] = dict(pid=p.pid, command=list(map(str, command)), gpus=list(gpus))
        save(); print('[start]', name, p.pid, flush=True)

    def done(name):
        rc = processes[name].poll()
        if rc is None:
            return False
        plan['jobs'][name]['exit_code'] = rc
        save()
        if rc:
            raise RuntimeError(f'{name} failed with {rc}; preserve partial results')
        return True

    def primary_done(name):
        primary = json.loads((D / 'remaining_driver_v2/driver.json').read_text())
        if primary.get('error'):
            raise RuntimeError('Primary experiment failed: ' + primary['error'])
        value = primary['jobs'].get(name, {}).get('exit_code')
        if value not in (None, 0):
            raise RuntimeError(f'Primary dependency {name} failed')
        return value == 0

    def evaluate(name, checkpoint, gpus):
        matrix = json.loads((D / 'baseline_eval_plan.json').read_text())
        matrix['models'] = {name: dict(checkpoint=str(checkpoint), policy_kind='repair', eraf='on',
                                     manifest=str(D / 'cache_all5/manifest.json'))}
        path = root / (name + '_plan.json')
        path.write_text(json.dumps(matrix, indent=2) + '\n')
        start(name, [sys.executable, '-u', 'scripts/run_robotwin_fixed_eval_matrix.py',
            '--plan', path, '--output', D / ('eval_' + name), '--gpus', *gpus,
            '--deadline', args.deadline], gpus)

    try:
        save()
        while True:
            budget()
            if 'compose' not in processes and primary_done('all5_full200'):
                start('compose', [sys.executable, 'scripts/compose_robotwin_eraf_fg_checkpoint.py',
                    '--policy', D / 'all5_full200/joint/step_000200.pt', '--interface', interface,
                    '--interface-plan', D / 'eraf_interface_pilot10/interface/plan.json',
                    '--output', D / 'eraf_composed/full200_interface10.pt'])
            if 'interface10_eraf_on' not in processes and primary_done('eval_all5_off200'):
                evaluate('interface10_eraf_on', interface, [3, 4, 5])
            if ('full200_eraf_on' not in processes and 'compose' in processes and done('compose')
                    and primary_done('eval_all5_full200')):
                baseline_exit = D / 'eval_baseline.exit.json'
                if baseline_exit.exists() and json.loads(baseline_exit.read_text())['exit_code'] == 0:
                    evaluate('full200_eraf_on', D / 'eraf_composed/full200_interface10.pt', [0, 1, 2])
            if all(name in processes and done(name) for name in ('interface10_eraf_on', 'full200_eraf_on')):
                break
            time.sleep(5)
        plan['complete'] = True
    except BaseException as error:
        plan['error'] = repr(error)
        raise
    finally:
        for p in processes.values():
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in processes.values():
            if p.poll() is None:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
        save()


if __name__ == '__main__':
    main()
