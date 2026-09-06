#!/usr/bin/env python3
"""Final fixed four-model target assessment, without changing admission gates."""
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
TASKS = ['blocks_ranking_rgb', 'place_a2b_left']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', type=Path, required=True)
    ap.add_argument('--deadline', default='2026-09-07T03:00:00+08:00')
    args = ap.parse_args()
    phase = args.phase.resolve()
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    output = phase/'locked_target_diagnostic'
    output.mkdir(exist_ok=False)
    catalog = phase.parent/'catalog_test'
    models = {'shared400': dict(checkpoint='/root/gpufree-data/LF-FastWAM/runs/robotwin_cf_dense/20260906-native-retention-v1/repair-shared-decisions/step_000400.pt', policy_kind='legacy')}
    models.update({arm+'200': dict(checkpoint=str(phase/f'arm_{arm}200/joint/step_000200.pt'),
                                  policy_kind='repair') for arm in ('off', 'local', 'full')})
    plan = dict(complete=False, status='waiting_for_local12', models=models, tasks=TASKS,
                conditions=['correct', 'counterfactual'], episodes_per_cell=10, expected_episodes=160,
                catalog=str(catalog), deadline=args.deadline, eraf='off', hash_scans=False,
                evaluation_kind='final_locked_target_subset_diagnostic',
                formal_five_task_admission=False, additional_updates_after_test=0, jobs={})
    processes = {}
    env = os.environ.copy()
    env.update(PATH='/opt/conda/bin:'+env['PATH'], PYTHONPATH=str(REPO/'src')+':'+str(REPO),
               DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
               VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json', OMP_NUM_THREADS='2')

    def save():
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')

    def interrupted(signum, frame):
        raise InterruptedError(f'Final assessment interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def budget():
        if time.time() >= deadline:
            raise TimeoutError('Predeclared work deadline reached')

    def command(arm, mode, tasks, gpu=None):
        model = models[arm]
        cmd = [sys.executable, '-u', 'scripts/eval_robotwin_eraf_fg.py', mode,
               '--output', str(output/arm), '--manifest', str(phase/'bank_matched/manifest.json'),
               '--checkpoint', model['checkpoint'], '--policy-kind', model['policy_kind'],
               '--catalog-root', str(catalog), '--tasks', *tasks, '--episodes', '10',
               '--eraf', 'off', '--skip-file-hashes']
        if gpu is not None:
            cmd += ['--gpu', str(gpu), '--videos']
        return cmd

    def start(name, cmd, gpu=None):
        budget()
        with (output/(name+'.log')).open('x') as log:
            process = subprocess.Popen(cmd, cwd=REPO,
                env=env | {'CUDA_VISIBLE_DEVICES': '' if gpu is None else str(gpu)},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = process
        plan['jobs'][name] = dict(pid=process.pid, gpu=gpu, command=cmd)
        save()

    def done(name):
        rc = processes[name].poll()
        if rc is None:
            return False
        plan['jobs'][name]['exit_code'] = rc
        save()
        if rc:
            raise RuntimeError(f'{name} failed with exit code {rc}')
        return True

    try:
        save()
        dependency = phase/'local12_followup.exit.json'
        while not dependency.exists():
            budget()
            time.sleep(10)
        if json.loads(dependency.read_text())['exit_code'] != 0:
            raise RuntimeError('Local12 prerequisite did not complete')
        if deadline-time.time() < 2700:
            raise TimeoutError('Less than45 minutes remain; do not open the target test')
        for arm in ('off', 'local', 'full'):
            driver = json.loads((phase/f'arm_{arm}200/driver.json').read_text())
            audit = json.loads((phase/f'arm_{arm}200/training_audit.json').read_text())
            if not driver['complete'] or not audit['complete'] or audit['steps'] != 200:
                raise ValueError('Unfinished or mismatched fixed200 arm')
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                          '--format=csv,noheader,nounits'], text=True)
        if len(memory.splitlines()) != 6 or any(int(n) > 1500 for n in memory.splitlines()):
            raise RuntimeError('Expected six idle GPUs after the owned dependencies')
        plan['status'] = 'evaluating_fixed_models'
        plan['test_opened_at'] = datetime.now().astimezone().isoformat()
        save()
        queue = [(arm, task) for task in TASKS for arm in models]
        running = {}
        while queue or running:
            budget()
            for gpu, name in list(running.items()):
                if done(name):
                    del running[gpu]
            for gpu in range(6):
                if gpu in running or not queue:
                    continue
                arm, task = queue.pop(0)
                name = arm+'_'+task
                start(name, command(arm, 'worker', [task], gpu), gpu)
                running[gpu] = name
            if running:
                time.sleep(5)
        for arm in models:
            name = 'summarize_'+arm
            start(name, command(arm, 'summarize', TASKS))
            while not done(name):
                budget()
                time.sleep(1)
        summaries = {arm: json.loads((output/arm/'summary.json').read_text()) for arm in models}
        for task in TASKS:
            for condition in plan['conditions']:
                states = [json.loads((output/arm/task/'demo_clean'/condition/'initial_states.json').read_text()) for arm in models]
                if any(value != states[0] for value in states[1:]):
                    raise ValueError('Physical initial states differ across fixed models')
        if any(not s['complete'] or s['episodes'] != 40 for s in summaries.values()):
            raise ValueError('Incomplete final target matrix')
        report = dict(complete=True, evaluation_kind=plan['evaluation_kind'],
                      formal_five_task_admission=False, episodes=160, models=summaries)
        (output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
        plan.update(complete=True, status='complete')
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
    main()
