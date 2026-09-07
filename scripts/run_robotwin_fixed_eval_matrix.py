#!/usr/bin/env python3
"""Evaluate declared checkpoints on fixed catalogs within an absolute deadline."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpus', type=int, nargs='+', required=True)
    ap.add_argument('--deadline', required=True)
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None:
        ap.error('Deadline must include the authorized timezone')
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(g not in range(6) for g in args.gpus):
        ap.error('Assign distinct GPUs in 0..5')
    plan = json.loads(args.plan.read_text())
    models, catalogs = plan['models'], plan['catalogs']
    if not models or not catalogs:
        ap.error('Specify at least one model and catalog')
    for name in [*models, *catalogs]:
        if not name or Path(name).name != name or name in {'.', '..'}:
            ap.error('Model and catalog names must be plain directory names')
    for model in models.values():
        if model['policy_kind'] not in {'legacy', 'repair'}:
            ap.error('Unknown policy kind')
        model.setdefault('eraf', 'off')
        if model['eraf'] not in {'on', 'off'} or (model['eraf'] == 'on' and model['policy_kind'] != 'repair'):
            ap.error('ERAF-on evaluation requires a repair checkpoint')
        for key in ('checkpoint', 'manifest'):
            model[key] = str(Path(model[key]).resolve(strict=True))
    for catalog in catalogs.values():
        catalog['path'] = str(Path(catalog['path']).resolve(strict=True))
        if catalog['episodes'] <= 0:
            ap.error('Episode count must be positive')
        for task in TASKS:
            path = Path(catalog['path']) / task / 'demo_clean/correct/episodes.jsonl'
            if len(path.read_text().splitlines()) != catalog['episodes']:
                ap.error('Catalog count differs from the declared fixed budget')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PATH='/opt/conda/bin:' + env['PATH'],
               PYTHONPATH=str(REPO / 'src') + ':' + str(REPO),
               DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
               VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json', OMP_NUM_THREADS='2')
    report = dict(plan=plan, gpus=args.gpus, deadline=args.deadline, complete=False,
                  optimizer_updates=0, eraf_by_model={name: model['eraf'] for name, model in models.items()}, jobs={})
    processes = {}

    def save():
        temp = output / 'driver.json.tmp'
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(output / 'driver.json')

    def budget():
        if time.time() >= cutoff.timestamp():
            raise TimeoutError('Authorized work cutoff reached')

    def interrupted(signum, frame):
        raise InterruptedError(f'Evaluation interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def start(name, command, gpu):
        budget()
        with (output / (name + '.log')).open('x') as log:
            process = subprocess.Popen(list(map(str, command)), cwd=REPO,
                env=env | {'CUDA_VISIBLE_DEVICES': str(gpu) if gpu is not None else ''},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = process
        report['jobs'][name] = dict(pid=process.pid, gpu=gpu, command=list(map(str, command)))
        save()
        print('[start]', name, process.pid, flush=True)

    def done(name):
        rc = processes[name].poll()
        if rc is None:
            return False
        report['jobs'][name]['exit_code'] = rc
        save()
        if rc != 0:
            raise RuntimeError(f'{name} failed with exit {rc}; partial results preserved')
        return True

    try:
        queue = [(m, c, t) for m in models for t in TASKS for c in catalogs]
        running = {}
        while queue or running:
            budget()
            for gpu, name in list(running.items()):
                if done(name):
                    del running[gpu]
            for gpu in args.gpus:
                if gpu in running or not queue:
                    continue
                model_name, catalog_name, task = queue.pop(0)
                model, catalog = models[model_name], catalogs[catalog_name]
                name = f'{model_name}_{catalog_name}_{task}'
                start(name, [sys.executable, '-u', 'scripts/eval_robotwin_eraf_fg.py', 'worker',
                    '--output', output / model_name / catalog_name,
                    '--checkpoint', model['checkpoint'], '--manifest', model['manifest'],
                    '--catalog-root', catalog['path'], '--episodes', catalog['episodes'],
                    '--tasks', task, '--policy-kind', model['policy_kind'], '--eraf', model['eraf'],
                    '--gpu', gpu, '--videos', '--skip-file-hashes'], gpu)
                running[gpu] = name
            if running:
                time.sleep(5)
        for model_name, model in models.items():
            for catalog_name, catalog in catalogs.items():
                name = f'summarize_{model_name}_{catalog_name}'
                start(name, [sys.executable, 'scripts/eval_robotwin_eraf_fg.py', 'summarize',
                    '--output', output / model_name / catalog_name,
                    '--checkpoint', model['checkpoint'], '--catalog-root', catalog['path'],
                    '--episodes', catalog['episodes'], '--skip-file-hashes'], None)
                while not done(name):
                    budget()
                    time.sleep(1)
        # Confirm identical physical initialization across every model/condition.
        for catalog_name in catalogs:
            for task in TASKS:
                initial = []
                for model_name in models:
                    for condition in ('correct', 'counterfactual'):
                        path = output / model_name / catalog_name / task / 'demo_clean' / condition / 'initial_states.json'
                        initial.append(json.loads(path.read_text()))
                if any(value != initial[0] for value in initial[1:]):
                    raise ValueError('Initial states differ across models or instructions')
        report.update(complete=True, matched_initial_states=True)
    except BaseException as error:
        report['error'] = repr(error)
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
    main()
