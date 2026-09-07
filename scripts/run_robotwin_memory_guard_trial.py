#!/usr/bin/env python3
"""After a complete baseline, evaluate the memory fix with identical weights."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from scripts.compare_robotwin_ten_task_methods import build_report


def source_process(pid, root, *, proc_root=Path('/proc')):
    """Bind the actual source process, including its kernel start time."""
    path = proc_root / str(pid)
    try:
        fields = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
        argv = (path / 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return None
    if fields[0] == 'Z':
        return None
    if not any(a.endswith(b'/run_robotwin_calibrated_joint_trial.py') for a in argv) or str(root).encode() not in argv:
        raise ValueError('Source PID belongs to another command; do not touch it.')
    return fields[19]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--source-pid', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--deadline', required=True)
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None:
        ap.error('Require a timezone-aware deadline.')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Commit code before execution.')
    source = args.source_root.resolve()
    launch = json.loads(source.with_name(source.name + '-launch.json').read_text())
    if launch['pid'] != args.source_pid:
        raise ValueError('Source launch receipt and actual process do not agree.')
    start_time = source_process(args.source_pid, source)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = dict(complete=False, stage='waiting_for_baseline', jobs={})
    processes = {}
    env = os.environ | dict(PATH='/opt/conda/bin:' + os.environ['PATH'], PYTHONPATH=str(REPO / 'src') + ':' + str(REPO),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')

    def write(name, value):
        temp = root / (name + '.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(root / name)

    protocol = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        source_root=str(source), source_pid=args.source_pid, source_process_start_time=start_time,
        source_code_commit=launch['code_commit'], deadline=args.deadline, platform_shutdown=None,
        activation='Wait for actual baseline process termination and complete90CF. Recompute all baseline comparisons. Skip if FG+ERAF already leads all; independent verification takes priority.',
        scope='Fixed-checkpoint comparison: only the rejected-memory-completion fallback changes relative to bb3bcce. Both ERAF arms use their identical baseline weights, task catalogs, instructions and inference recipe. No training or new checkpoint selection.',
        total_eval_episodes=90, correct_evaluated=False, independent_test=False, goal_achievement_claim=False,
        model_storage='server_only', disk_reserve_GiB=5)
    write('protocol.json', protocol)
    write('driver.json', state)

    def budget():
        if time.time() >= cutoff.timestamp():
            raise TimeoutError('Memory trial cutoff; stop only this operator\'s jobs. Server remains on.')
        if shutil.disk_usage(root).free < 5 * 1024**3:
            raise RuntimeError('Five GiB disk reserve reached.')

    def stop(signum, frame):
        raise InterruptedError(f'Signal{signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while True:
            budget()
            current = source_process(args.source_pid, source)
            if current is None:
                break
            if start_time is None or current != start_time:
                raise ValueError('Source process identity changed.')
            time.sleep(10)
        previous = json.loads((source / 'driver.json').read_text())
        if not previous['complete'] or previous['stage'] != 'complete' or any(j.get('exit_code') != 0 for j in previous['jobs'].values()):
            raise ValueError('Baseline stopped without a complete successful matrix; do not infer a result.')
        config = json.loads((source / 'comparison_config.json').read_text())
        baseline = build_report(config)
        write('baseline_comparison.json', baseline)
        if baseline['target_strictly_exceeds_all_listed_dev_controls']:
            state.update(complete=True, stage='skipped_baseline_leads', independent_validation_required=True)
            return
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise ValueError('Another process owns the GPUs; do not overlap it.')
        if shutil.disk_usage(root).free < 5 * 1024**3 + 160 * 1024**2:
            raise RuntimeError('Reserve evaluation media space before starting.')
        from experiments.robotwin.eraf_fg_bridge import file_sha256
        old = json.loads((source / 'protocol.json').read_text())
        arms = ('eraf_only', 'eraf_fg')
        checkpoints = {a: source / a / 'joint/step_000200.pt' for a in arms}
        for arm, checkpoint in checkpoints.items():
            audit = json.loads(checkpoint.with_name('freeze_audit.json').read_text())
            if not audit['complete'] or audit['unexpected_changes'] or not audit['optimizer_checkpoint_and_contract_match']:
                raise ValueError('Source action checkpoint lacks a completed training audit.')
        protocol.update(manifest=old['manifest'], groups=old['groups'],
            checkpoints={a: str(p) for a, p in checkpoints.items()},
            checkpoint_sha256={a: file_sha256(p) for a, p in checkpoints.items()},
            manifest_sha256=file_sha256(old['manifest']), inference='Same evaluator and arguments as baseline; memory fallback fix only.')
        if protocol['manifest_sha256'] != old['manifest_sha256']:
            raise ValueError('Training/evaluation manifest changed.')
        write('protocol.json', protocol)
        state['stage'] = 'evaluating'
        pending = [(group, task, arm) for group, spec in old['groups'].items() for task in spec['tasks'] for arm in arms]
        active = {}
        while pending or active:
            budget()
            for name, gpu in list(active.items()):
                rc = processes[name].poll()
                if rc is not None:
                    state['jobs'][name]['exit_code'] = rc
                    del active[name]
                    write('driver.json', state)
                    if rc:
                        raise RuntimeError(f'{name} failed: {rc}')
            for gpu in sorted(set(range(6)) - set(active.values())):
                if not pending:
                    break
                group, task, arm = pending.pop(0)
                spec = old['groups'][group]
                name = f'eval_{group}_{arm}_{task}'
                command = ['/opt/conda/bin/python', '-u', str(REPO / 'scripts/eval_robotwin_eraf_fg.py'), 'worker',
                    '--output', str(root / ('eval_' + group) / arm / 'dev'), '--checkpoint', str(checkpoints[arm]),
                    '--manifest', old['manifest'], '--catalog-root', spec['catalog'], '--episodes', str(spec['episodes']),
                    '--tasks', task, '--policy-kind', 'repair', '--eraf', 'on', '--conditions', 'counterfactual',
                    '--gpu', str(gpu), '--videos', '--skip-file-hashes', '--interventions', str(REPO / 'configs/eval/robotwin_cis_ten_tasks.json')]
                with (root / (name + '.log')).open('x') as log:
                    p = subprocess.Popen(command, cwd=REPO, env=env | {'CUDA_VISIBLE_DEVICES': str(gpu)},
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes[name] = p
                state['jobs'][name] = dict(pid=p.pid, gpu=gpu, command=command)
                active[name] = gpu
                write('driver.json', state)
            if active:
                time.sleep(5)
        for group, spec in old['groups'].items():
            for arm in arms:
                subprocess.run(['/opt/conda/bin/python', str(REPO / 'scripts/eval_robotwin_eraf_fg.py'), 'summarize',
                    '--output', str(root / ('eval_' + group) / arm / 'dev'), '--checkpoint', str(checkpoints[arm]),
                    '--catalog-root', spec['catalog'], '--episodes', str(spec['episodes']), '--tasks', *spec['tasks'],
                    '--conditions', 'counterfactual', '--skip-file-hashes'], cwd=REPO, env=env, check=True,
                    timeout=60, stdout=subprocess.DEVNULL)
        for arm, checkpoint in checkpoints.items():
            if file_sha256(checkpoint) != protocol['checkpoint_sha256'][arm]:
                raise ValueError('Checkpoint changed during fixed-weight evaluation.')
            name = 'pre_memory_guard_' + arm
            if name in config['methods']:
                raise ValueError('Historical comparison name collision.')
            config['methods'][name] = config['methods'][arm]
            config['methods'][arm] = {g: str(root / ('eval_' + g) / arm / 'dev') for g in old['groups']}
        write('comparison_config.json', config)
        write('comparison.json', build_report(config))
        state.update(complete=True, stage='complete', independent_validation_required=True)
    except BaseException as error:
        state['error'] = repr(error)
        raise
    finally:
        for p in processes.values():
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for name, p in processes.items():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            state['jobs'][name]['exit_code'] = p.returncode
        state['finished'] = datetime.now().astimezone().isoformat()
        write('driver.json', state)


if __name__ == '__main__':
    main()
