#!/usr/bin/env python3
"""Bounded Action-LoRA cup/left pilot with fixed regression and development sets."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]
TASKS = ['place_a2b_left', 'place_a2b_right', 'place_burger_fries', 'stack_blocks_two', 'blocks_ranking_rgb']


def training_audit(output, parent, steps, world):
    """Inspect actual updates and samples, without whole-file hashing."""
    import torch
    before = torch.load(parent, map_location='cpu', weights_only=False)
    after = torch.load(output/f'joint/step_{steps:06d}.pt', map_location='cpu', weights_only=False)
    changed = []
    for key in before['mot_trainable']:
        if not torch.equal(before['mot_trainable'][key], after['mot_trainable'][key]):
            if '.action.' not in key:
                raise ValueError('A frozen non-action adapter changed: '+key)
            changed.append(key)
    if not changed or set(before['mot_trainable']) != set(after['mot_trainable']):
        raise ValueError('Missing real action updates or changed adapter geometry')
    if set(before['policy_guard']) != set(after['policy_guard']) or any(
            not torch.equal(v, after['policy_guard'][key]) for key, v in before['policy_guard'].items()):
        raise ValueError('Frozen ERAF parameters changed')
    if after['optimizer_steps'] != steps or after['parent_checkpoint'] != str(Path(parent).resolve()):
        raise ValueError('Wrong update budget or policy initialization')
    cup_examples = ordinary_controls = examples = 0
    all_norms = []
    for rank in range(world):
        rows = [json.loads(line) for line in (output/f'joint/rank{rank}.jsonl').read_text().splitlines()]
        if [r['step'] for r in rows] != list(range(1, steps+1)):
            raise ValueError('Incomplete optimizer-step journal')
        norms = [r['grad_norm'] for r in rows]
        if not all(math.isfinite(n) for n in norms): raise ValueError('Nonfinite optimization')
        all_norms.append(norms)
        for row in rows:
            if len(row['examples']) != 12//world: raise ValueError('Changed global batch budget')
            for item in row['examples']:
                examples += 1
                cup_examples += item['id'].startswith('cup_')
                ordinary_controls += bool(item['ordinary_cf_control'])
    if any(norms != all_norms[0] for norms in all_norms[1:]):
        raise ValueError('Ranks disagree on global gradients')
    if not cup_examples: raise ValueError('No new cup examples actually trained')
    return {'complete': True, 'steps': steps, 'examples': examples, 'cup_examples': cup_examples,
            'ordinary_cf_control_examples': ordinary_controls, 'changed_action_tensors': len(changed),
            'frozen_video_and_guard_equal': True, 'hash_scans': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'output', 'manifest', 'checkpoint', 'source-bank'):
        ap.add_argument('--'+key, type=Path, required=True)
    teacher_root = '/root/gpufree-data/LF-FastWAM/runs/robotwin_cf_dense/20260906-native-retention-v1'
    ap.add_argument('--correct-teacher', type=Path, default=teacher_root+'/repair-dense-native/step_000600.pt')
    ap.add_argument('--cf-teacher', type=Path, default=teacher_root+'/repair-shared-decisions/step_000400.pt')
    ap.add_argument('--fg', choices=['off', 'local', 'full'], required=True)
    ap.add_argument('--gpus', type=int, nargs='+', required=True)
    ap.add_argument('--steps', type=int, choices=[2, 200], default=200)
    ap.add_argument('--deadline', default='2026-09-07T03:00:00+08:00')
    ap.add_argument('--training-only', action='store_true')
    args = ap.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(g not in range(6) for g in args.gpus) or 12 % len(args.gpus):
        ap.error('Use distinct assigned GPUs and a world size dividing12')
    for key in ('root', 'output', 'manifest', 'checkpoint', 'source_bank', 'correct_teacher', 'cf_teacher'):
        setattr(args, key, getattr(args, key).resolve())
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PATH='/opt/conda/bin:'+env['PATH'], PYTHONPATH=str(REPO/'src')+':'+str(REPO),
               DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
               VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json', OMP_NUM_THREADS='2')
    plan = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    plan.update(jobs={}, complete=False, hash_scans=False, eraf='off', policy_scope='action',
                correction_weight=.25, global_batch=12, learning_rate=1e-5, correct_weight=4., cf_weight=2.)
    processes = {}

    def save():
        (args.output/'driver.json').write_text(json.dumps(plan, indent=2)+'\n')

    def budget():
        if time.time() >= deadline: raise TimeoutError('Predeclared experiment deadline reached')

    def start(name, command, gpus):
        budget()
        with (args.output/(name+'.log')).open('x') as log:
            process = subprocess.Popen([str(v) for v in command], cwd=REPO,
                env=env | {'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus))},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = process
        plan['jobs'][name] = {'pid': process.pid, 'gpus': gpus, 'command': list(map(str, command))}
        save(); print('[start]', name, process.pid, flush=True)

    def done(name):
        rc = processes[name].poll()
        if rc is None: return False
        plan['jobs'][name]['exit_code'] = rc; save()
        if rc != 0: raise RuntimeError(f'{name} exited with {rc}; see its preserved log')
        return True

    try:
        command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={len(args.gpus)}', 'scripts/train_robotwin_eraf_fg_action.py',
            '--stage', 'joint', '--steps', args.steps, '--save-every', 2 if args.steps == 2 else 100,
            '--learning-rate', '1e-5', '--manifest', args.manifest, '--checkpoint', args.checkpoint,
            '--source-bank', args.source_bank, '--output', args.output/'joint', '--eraf', 'off',
            '--correct-teacher', args.correct_teacher, '--cf-teacher', args.cf_teacher,
            '--fg', args.fg, '--correct-weight', '4', '--cf-weight', '2', '--policy-scope', 'action',
            '--correction-weight', '.25', '--target-tasks', 'place_a2b_left', 'place_empty_cup',
            '--skip-file-hashes']
        start('train', command, args.gpus)
        while not done('train'): budget(); time.sleep(5)
        audit = training_audit(args.output, args.checkpoint, args.steps, len(args.gpus))
        (args.output/'training_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        if args.training_only:
            plan['complete'] = True; save(); return
        checkpoint = args.output/f'joint/step_{args.steps:06d}.pt'
        catalogs = {'reg': (3, Path('/root/gpufree-data/LF-FastWAM/evaluate_results/robotwin/robotwin_uncond_3cam_384/cf-improvement-20260905-235608-base')),
                    'dev': (6, args.root/'catalog_dev')}
        cup_catalog = args.root/'cup_baseline_20260906_2221/catalog_dev/catalog.json'
        queue = [('cup', None)] + [(kind, task) for task in TASKS for kind in catalogs]
        running = {}
        while queue or running:
            budget()
            for gpu, name in list(running.items()):
                if done(name): del running[gpu]
            for gpu in args.gpus:
                if gpu in running or not queue: continue
                kind, task = queue.pop(0)
                name = kind if task is None else kind+'_'+task
                if kind == 'cup':
                    command = [sys.executable, '-u', 'scripts/eval_robotwin_cup_cf.py', 'worker',
                        '--output', args.output/'cup', '--manifest', args.manifest, '--checkpoint', checkpoint,
                        '--catalog', cup_catalog, '--policy-kind', 'repair', '--episode-count', '6',
                        '--robotwin-root', '/root/gpufree-data/LF-FastWAM/third_party/RoboTwin']
                else:
                    episodes, catalog = catalogs[kind]
                    command = [sys.executable, '-u', 'scripts/eval_robotwin_eraf_fg.py', 'worker',
                        '--output', args.output/kind, '--manifest', args.manifest, '--checkpoint', checkpoint,
                        '--catalog-root', catalog, '--tasks', task, '--episodes', episodes,
                        '--eraf', 'off', '--gpu', gpu, '--videos', '--skip-file-hashes']
                start(name, command, [gpu]); running[gpu] = name
            if running: time.sleep(5)
        for kind, (episodes, catalog) in catalogs.items():
            start('summarize_'+kind, [sys.executable, 'scripts/eval_robotwin_eraf_fg.py', 'summarize',
                '--output', args.output/kind, '--checkpoint', checkpoint, '--catalog-root', catalog,
                '--episodes', episodes, '--skip-file-hashes'], [])
            while not done('summarize_'+kind): budget(); time.sleep(1)
        start('summarize_cup', [sys.executable, 'scripts/eval_robotwin_cup_cf.py', 'summarize',
            '--output', args.output/'cup_summary.json', '--catalog', cup_catalog,
            '--workers', args.output/'cup/worker.json'], [])
        while not done('summarize_cup'): budget(); time.sleep(1)
        plan['complete'] = True; save()
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes.values():
            if process.poll() is None:
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)
        save()


if __name__ == '__main__': main()
