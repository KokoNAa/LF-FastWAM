#!/usr/bin/env python3
"""Run the predeclared local12 diagnostic after both saved-model evaluations."""
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
    ap.add_argument('--phase', type=Path, required=True)
    ap.add_argument('--deadline', default='2026-09-07T03:00:00+08:00')
    args = ap.parse_args()
    phase = args.phase.resolve()
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    root = phase/'local12_followup'
    root.mkdir(exist_ok=False)
    plan = dict(phase=str(phase), deadline=args.deadline, complete=False,
                status='waiting_for_100_evaluations', jobs={}, hash_scans=False,
                purpose='Full versus first12 correction diagnostic; original admission gates unchanged')
    processes = {}

    def save():
        (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')

    def interrupted(signum, frame):
        raise InterruptedError(f'Followup interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def budget():
        if time.time() >= deadline:
            raise TimeoutError('Predeclared work deadline reached')

    def command(output, *, evaluate=False):
        base = json.loads((phase/'arm_full200/driver.json').read_text())
        checkpoint = phase/'arm_local200/joint/step_000100.pt' if evaluate else base['checkpoint']
        cmd = [sys.executable, '-u', 'scripts/run_robotwin_cup_ablation.py',
               '--root', base['root'], '--output', str(output), '--manifest', base['manifest'],
               '--checkpoint', str(checkpoint), '--source-bank', base['source_bank'],
               '--correct-teacher', base['correct_teacher'], '--cf-teacher', base['cf_teacher'],
               '--fg', 'local', '--gpus', *(['3', '4', '5'] if evaluate else ['0', '1', '2']),
               '--steps', '100' if evaluate else '200', '--deadline', args.deadline]
        if evaluate:
            cmd.append('--evaluate-only')
        return cmd

    def start(name, cmd):
        budget()
        with (root/(name+'.log')).open('x') as log:
            process = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        processes[name] = process
        plan['jobs'][name] = dict(pid=process.pid, command=cmd)
        save()

    try:
        save()
        while True:
            budget()
            ready = True
            for arm in ('off', 'full'):
                path = phase/f'eval_{arm}100.exit.json'
                if not path.exists():
                    ready = False
                    continue
                if json.loads(path.read_text())['exit_code'] != 0:
                    raise RuntimeError(f'Dependent {arm}100 evaluation failed')
                driver = json.loads((phase/f'eval_{arm}100/driver.json').read_text())
                if not driver['complete']:
                    raise RuntimeError('Dependent evaluation has no complete driver result')
            if ready:
                break
            time.sleep(10)
        if deadline-time.time() < 3600:
            raise TimeoutError('Less than one hour remains; do not start a new training arm')
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                          '--format=csv,noheader,nounits'], text=True)
        if len(memory.splitlines()) != 6 or any(int(n) > 1500 for n in memory.splitlines()):
            raise RuntimeError('Expected six idle GPUs after the owned dependencies')
        plan['status'] = 'running_local12'
        start('local200', command(phase/'arm_local200'))
        while True:
            budget()
            # The trainer publishes this path by atomic rename after closing it.
            checkpoint = phase/'arm_local200/joint/step_000100.pt'
            if checkpoint.exists() and 'local100' not in processes:
                start('local100', command(phase/'eval_local100', evaluate=True))
            for name, process in processes.items():
                rc = process.poll()
                if rc is not None:
                    plan['jobs'][name]['exit_code'] = rc
                    if rc != 0:
                        raise RuntimeError(f'{name} failed with {rc}')
            save()
            if len(processes) == 2 and all(p.poll() == 0 for p in processes.values()):
                break
            time.sleep(10)
        plan.update(complete=True, status='complete')
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
        save()


if __name__ == '__main__':
    main()
