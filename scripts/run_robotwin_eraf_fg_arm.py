#!/usr/bin/env python3
"""Train one independent arm and evaluate both declared development catalogs.

Each arm owns a disjoint GPU pool, output directory, and process ledger.
It never chooses a checkpoint from the locked test or starts an automatic
continuation. An explicit continuation needs its exact optimizer companion.
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
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from scripts.advance_robotwin_eraf_fg_campaign import read, write, TASKS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'output', 'checkpoint'):
        ap.add_argument('--' + key, required=True)
    ap.add_argument('--eraf', choices=['on', 'off'], required=True)
    ap.add_argument('--fg', choices=['off', 'local', 'full'], required=True)
    ap.add_argument('--gpus', type=int, choices=range(6), nargs='+', required=True)
    ap.add_argument('--correct-weight', type=float, default=4.)
    ap.add_argument('--cf-weight', type=float, default=2.)
    ap.add_argument('--steps', type=int, choices=[200, 400, 800], default=200)
    ap.add_argument('--resume-state', help='Exact optimizer companion for an explicitly selected continuation.')
    ap.add_argument('--manifest', help='Audited expanded bank; defaults to the original formal bank.')
    args = ap.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or 12 % len(args.gpus):
        ap.error('Distinct GPUs dividing global batch12 required.')
    if (args.steps > 200) != bool(args.resume_state):
        ap.error('400/800-step continuation requires its optimizer companion; fresh arms use200.')
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    if output.parent != root or output == root:
        ap.error('Each arm must own a direct child of the campaign root.')
    launch = read(root / 'launch.json')
    deadline = datetime.fromisoformat(launch['stop_experiments_hkt'])
    if datetime.now() >= deadline:
        raise RuntimeError('Experimental budget expired.')
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    manifest = Path(args.manifest).resolve() if args.manifest else root / 'bank_full_v1/manifest.json'
    audit = read(manifest.parent / 'audit.json')
    if not audit['complete'] or file_sha256(manifest) != audit['manifest_sha256']:
        raise ValueError('Formal replay audit does not match.')
    output.mkdir(exist_ok=False)
    plan = vars(args) | {'manifest': str(manifest), 'checkpoint_sha256': file_sha256(args.checkpoint),
        'manifest_sha256': audit['manifest_sha256'],
        'interface_steps': 100 if args.eraf == 'on' and not args.resume_state else 0,
        'joint_steps': args.steps, 'global_batch': 12, 'seed': 42,
        'code': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'stop_experiments_hkt': deadline.isoformat(), 'jobs': {}}
    write(output / 'plan.json', plan)
    env = os.environ.copy()
    env.update(PATH=str(Path(sys.executable).parent) + ':' + env['PATH'],
        PYTHONPATH=str(REPO / 'src') + ':' + str(REPO), OMP_NUM_THREADS='2',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
    wrapper = ('import subprocess,sys,json,pathlib; r=subprocess.run(json.loads(sys.argv[1])); '
        'pathlib.Path(sys.argv[2]).write_text(json.dumps(dict(exit_code=r.returncode))); sys.exit(r.returncode)')
    processes = {}

    def budget():
        if datetime.now() >= deadline:
            raise RuntimeError('Archive/shutdown reserve reached.')

    def start(name, command, gpus):
        budget()
        with (output / (name + '.log')).open('x') as log:
            process = subprocess.Popen([sys.executable, '-u', '-c', wrapper, json.dumps(command),
                str(output / (name + '_exit.json'))], cwd=REPO,
                env=dict(env, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus))),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = process
        plan['jobs'][name] = {'pid': process.pid, 'gpus': gpus, 'command': command,
                              'started_at': datetime.now().isoformat()}
        write(output / 'plan.json', plan)
        print(f'[start] {name} gpus={gpus} pid={process.pid}', flush=True)

    def done(name):
        code = processes[name].poll()
        if code is None:
            return False
        if code or read(output / (name + '_exit.json'))['exit_code']:
            raise RuntimeError(f'{name} failed: exit {code}.')
        return True

    def train(stage, checkpoint):
        template = list(launch['jobs']['interface_smoke']['command'])
        for flag, value in {'--manifest': manifest, '--checkpoint': checkpoint,
            '--output': output / stage, '--stage': stage, '--steps': 100 if stage == 'interface' else args.steps,
            '--save-every': 100 if stage == 'interface' else args.steps,
            '--learning-rate': '5e-5' if stage == 'interface' else '1e-5',
            '--fg': args.fg, '--eraf': args.eraf,
            '--correct-weight': args.correct_weight, '--cf-weight': args.cf_weight}.items():
            if flag in template:
                template[template.index(flag) + 1] = str(value)
            else:
                template += [flag, str(value)]
        if stage == 'joint' and args.resume_state:
            template += ['--resume-state', str(Path(args.resume_state).resolve())]
        command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={len(args.gpus)}'] + template[2:]
        start(stage, command, args.gpus)
        while not done(stage):
            budget()
            time.sleep(15)
        return output / stage / ('step_000100.pt' if stage == 'interface' else f'step_{args.steps:06d}.pt')

    try:
        parent = Path(args.checkpoint).resolve()
        if args.eraf == 'on' and not args.resume_state:
            parent = train('interface', parent)
        checkpoint = train('joint', parent)
        fixed = Path('/root/gpufree-data/LF-FastWAM/evaluate_results/robotwin/robotwin_uncond_3cam_384/cf-improvement-20260905-235608-base')
        catalogs = {'reg': (3, fixed), 'dev': (6, root / 'catalog_dev')}
        queue = [(kind, task) for task in TASKS for kind in catalogs]
        running = {}
        while queue or running:
            budget()
            for gpu, name in list(running.items()):
                if done(name):
                    print(f'[complete] {name}', flush=True)
                    del running[gpu]
            for gpu in args.gpus:
                if gpu in running or not queue:
                    continue
                kind, task = queue.pop(0)
                episodes, catalog = catalogs[kind]
                name = kind + '_' + task
                start(name, [sys.executable, '-u', 'scripts/eval_robotwin_eraf_fg.py', 'worker',
                    '--manifest', str(manifest), '--checkpoint', str(checkpoint), '--eraf', args.eraf,
                    '--catalog-root', str(catalog), '--output', str(output / kind),
                    '--tasks', task, '--episodes', str(episodes), '--gpu', str(gpu), '--videos'], [gpu])
                running[gpu] = name
            if running:
                time.sleep(15)
        for kind, (episodes, catalog) in catalogs.items():
            budget()
            subprocess.run([sys.executable, '-u', 'scripts/eval_robotwin_eraf_fg.py', 'summarize',
                '--output', str(output / kind), '--checkpoint', str(checkpoint),
                '--catalog-root', str(catalog), '--episodes', str(episodes)], cwd=REPO, env=env, check=True)
        write(output / 'complete.json', {'complete': True, 'checkpoint': str(checkpoint),
            'checkpoint_sha256': file_sha256(checkpoint), 'finished_at': datetime.now().isoformat(),
            'next_action': 'Review both development sets. Locked test untouched.'})
    finally:
        for name, process in processes.items():
            if process.poll() is not None:
                continue
            path = Path(f'/proc/{process.pid}/cmdline')
            if path.exists() and str(output).encode() in path.read_bytes() and os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGTERM)


if __name__ == '__main__':
    main()
