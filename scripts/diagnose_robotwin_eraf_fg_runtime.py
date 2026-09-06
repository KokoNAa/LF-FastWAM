#!/usr/bin/env python3
"""Diagnose ERAF or cross-replan memory on an unchanged trained checkpoint.

Wait for all ten first-candidate evaluations to be dispatched, then use only
their released GPUs. This is an inference toggle, not a trained FG-only arm.
"""
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
TASKS = ['place_a2b_left', 'place_a2b_right', 'place_burger_fries',
         'stack_blocks_two', 'blocks_ranking_rgb']


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--eraf', choices=['on', 'off'], default='off')
    ap.add_argument('--memory-mode', choices=['carry', 'reset'], default='carry')
    ap.add_argument('--gpus', nargs='+', type=int, choices=range(6), default=list(range(6)))
    ap.add_argument('--tasks', nargs='+', choices=TASKS,
                    default=['place_a2b_left', 'stack_blocks_two'])
    args = ap.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    if not args.tasks or len(set(args.tasks)) != len(args.tasks):
        raise ValueError('Nonempty unique task list required.')
    if args.memory_mode == 'reset' and args.eraf != 'on':
        raise ValueError('Memory reset requires ERAF on.')
    output.mkdir(parents=True, exist_ok=False)
    deadline = datetime.fromisoformat(read(root / 'launch.json')['stop_experiments_hkt'])
    checkpoint = root / 'joint_full200/step_000200.pt'
    expected = {f'{kind}_full200_{task}' for kind in ('reg', 'dev') for task in TASKS}
    jobs, running, pending = {}, {}, list(args.tasks)
    python = sys.executable
    env = dict(os.environ, PYTHONPATH=str(REPO / 'src') + ':' + str(REPO),
        OMP_NUM_THREADS='2', DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
    wrapper = ('import subprocess,sys,json,pathlib; r=subprocess.run(json.loads(sys.argv[1])); '
               'pathlib.Path(sys.argv[2]).write_text(json.dumps(dict(exit_code=r.returncode))); '
               'sys.exit(r.returncode)')
    plan = {'kind': 'unchanged_checkpoint_inference_toggle', 'eraf': args.eraf,
        'memory_mode': args.memory_mode, 'allowed_gpus': args.gpus,
        'checkpoint': str(checkpoint), 'tasks': args.tasks, 'episodes_per_condition': 3,
        'interpretation': 'Diagnostic only; this is not an independently trained FG-only ablation.',
        'code': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'stop_experiments_hkt': deadline.isoformat(), 'jobs': jobs}
    (output / 'diagnostic_plan.json').write_text(json.dumps(plan, indent=2))
    try:
        while pending or running:
            if datetime.now() >= deadline:
                raise RuntimeError('Reserved archive/shutdown time reached.')
            primary = read(root / 'launch.json')['jobs']
            for gpu, task in list(running.items()):
                status = output / (task + '_exit.json')
                if status.exists():
                    if read(status)['exit_code'] != 0:
                        raise RuntimeError(f'Diagnostic failed: {task}')
                    del running[gpu]
                    print(f'[complete] {task}', flush=True)
                elif not Path(f'/proc/{jobs[task]["pid"]}').exists():
                    raise RuntimeError(f'Diagnostic disappeared without exit status: {task}')
            if expected.issubset(primary):
                busy = {int(gpu) for name in expected
                    if not (root / (name + '_exit.json')).exists()
                    for gpu in primary[name]['gpus'].split(',')}
                memory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                    '--format=csv,noheader,nounits'], text=True)
                free = [int(line.split(',')[0]) for line in memory.splitlines()
                    if int(line.split(',')[1]) < 500
                    and int(line.split(',')[0]) in args.gpus
                    and int(line.split(',')[0]) not in busy | set(running)]
                for gpu in free:
                    if not pending:
                        break
                    task = pending.pop(0)
                    command = list(primary[f'reg_full200_{task}']['command'])
                    for flag, value in {'--output': output, '--eraf': args.eraf, '--gpu': gpu}.items():
                        command[command.index(flag) + 1] = str(value)
                    command += ['--memory-mode', args.memory_mode]
                    if Path(command[command.index('--checkpoint') + 1]) != checkpoint:
                        raise ValueError('The primary checkpoint changed.')
                    with (output / (task + '.log')).open('x') as log:
                        process = subprocess.Popen([python, '-u', '-c', wrapper, json.dumps(command),
                            str(output / (task + '_exit.json'))], cwd=REPO,
                            env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
                    jobs[task] = {'pid': process.pid, 'gpu': gpu, 'command': command,
                                  'started_at': datetime.now().isoformat()}
                    running[gpu] = task
                    (output / 'diagnostic_plan.json').write_text(json.dumps(plan, indent=2))
                    print(f'[start] {task} eraf={args.eraf} memory={args.memory_mode} gpu={gpu} pid={process.pid}', flush=True)
            if pending or running:
                time.sleep(20)
        template = primary['reg_full200_place_a2b_left']['command']
        catalog = template[template.index('--catalog-root') + 1]
        subprocess.run([python, '-u', 'scripts/eval_robotwin_eraf_fg.py', 'summarize',
            '--output', str(output), '--checkpoint', str(checkpoint), '--catalog-root', catalog,
            '--episodes', '3', '--tasks', *args.tasks], cwd=REPO, env=env, check=True)
        (output / 'diagnostic_complete.json').write_text(json.dumps({'complete': True,
            'kind': plan['kind'], 'episodes': len(args.tasks) * 6,
            'finished_at': datetime.now().isoformat()}))
        print('[complete] Inference toggle is ready for comparison.', flush=True)
    finally:
        for task in running.values():
            pid = jobs[task]['pid']
            command_file = Path(f'/proc/{pid}/cmdline')
            if command_file.exists() and str(output).encode() in command_file.read_bytes():
                if os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGTERM)


if __name__ == '__main__':
    main()
