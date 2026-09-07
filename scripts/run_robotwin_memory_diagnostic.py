#!/usr/bin/env python3
"""Compare reset-memory and ERAF bypass against a completed, unchanged CF run."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-eval', type=Path, required=True)
    ap.add_argument('--model', default='fg_semantic_fg')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--deadline', required=True)
    args = ap.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        ap.error('The work deadline requires an explicit timezone.')
    source, root = args.source_eval.resolve(), args.output.resolve()
    source_plan = json.loads((source / 'protocol.json').read_text())
    checkpoint = Path(source_plan['models'][args.model])
    spec = source_plan['groups']['original_five']
    source_commit = source_plan['code_commit']
    subprocess.run(['git', 'diff', '--exit-code', source_commit, '--', 'src',
                    'experiments', 'scripts/eval_robotwin_eraf_fg.py'], cwd=REPO, check=True)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Commit the diagnostic code before execution.')
    root.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        target = root / name
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(target)

    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    modes = {'reset': {'eraf': 'on', 'memory': 'reset'},
             'bypass': {'eraf': 'off', 'memory': 'carry'}}
    plan = dict(format='robotwin_memory_diagnostic_v1', source_eval=str(source),
        model=args.model, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        source_code_commit=source_commit, model_and_evaluator_code_identical_to_source=True,
        modes=modes, original_carry_results=str(source / 'eval_original_five' / args.model / 'dev'),
        groups=spec, total_episodes=2 * len(spec['tasks']) * spec['episodes'],
        optimizer_updates=0, independent_test=False, correct_evaluated=False,
        primary_success_definition_unchanged=True, deadline=args.deadline, platform_shutdown=None,
        hypothesis='Determine whether cross-replan completion memory causes observed CF losses. '
                   'Bypass uses the identical candidate checkpoint to check that its frozen action policy retains baseline performance. '
                   'No new training, goal labels, or task-specific routing are introduced.')
    write('protocol.json', plan)
    report = dict(complete=False, stage='waiting_for_source', jobs={})
    processes = {}

    def budget():
        if time.time() >= deadline.timestamp():
            raise TimeoutError('Diagnostic work deadline reached; no platform shutdown.')
        if shutil.disk_usage(root).free < 5 * 1024**3:
            raise RuntimeError('Five GiB disk reserve reached.')

    def stop(signum, _frame):
        raise InterruptedError(f'Signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = os.environ | dict(PATH='/opt/conda/bin:' + os.environ['PATH'],
        PYTHONPATH=str(REPO / 'src') + ':' + str(REPO), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
    try:
        write('driver.json', report)
        launch = json.loads(source.with_name(source.name + '-launch.json').read_text())
        while True:
            budget()
            driver = json.loads((source / 'driver.json').read_text())
            proc = Path('/proc') / str(launch['pid'])
            try:
                running = (proc / 'stat').read_text().split(') ')[1][0] != 'Z' and \
                    b'run_context_residual_full_eval.py' in (proc / 'cmdline').read_bytes()
            except FileNotFoundError:
                running = False
            if driver.get('error'):
                raise RuntimeError('Source evaluation failed: ' + driver['error'])
            if driver['complete'] and not running:
                break
            if not running:
                raise RuntimeError('Source producer absent without complete results.')
            time.sleep(5)
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPUs are still occupied; do not overlap experiments.')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_sha:
            raise ValueError('Candidate checkpoint changed while waiting.')
        report['stage'] = 'evaluating'
        pending = [(mode, task) for task in spec['tasks'] for mode in modes]
        active = {}
        while pending or active:
            budget()
            for name, gpu in list(active.items()):
                rc = processes[name].poll()
                if rc is not None:
                    report['jobs'][name]['exit_code'] = rc
                    del active[name]
                    write('driver.json', report)
                    if rc:
                        raise RuntimeError(f'{name} failed: {rc}')
            for gpu in sorted(set(range(6)) - set(active.values())):
                if not pending:
                    break
                mode, task = pending.pop(0)
                name = mode + '_' + task
                cmd = ['/opt/conda/bin/python', '-u', str(REPO / 'scripts/eval_robotwin_eraf_fg.py'),
                    'worker', '--output', str(root / mode / 'dev'), '--checkpoint', str(checkpoint),
                    '--manifest', source_plan['manifest'], '--catalog-root', spec['catalog'],
                    '--episodes', str(spec['episodes']), '--tasks', task, '--policy-kind', 'repair',
                    '--eraf', modes[mode]['eraf'], '--memory-mode', modes[mode]['memory'],
                    '--conditions', 'counterfactual', '--gpu', str(gpu), '--videos', '--skip-file-hashes',
                    '--interventions', str(REPO / 'configs/eval/robotwin_cis_ten_tasks.json')]
                with (root / (name + '.log')).open('x') as log:
                    p = subprocess.Popen(cmd, cwd=REPO, env=env | {'CUDA_VISIBLE_DEVICES': str(gpu)},
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes[name] = p
                active[name] = gpu
                report['jobs'][name] = dict(pid=p.pid, gpu=gpu, command=cmd)
                write('driver.json', report)
            if active:
                time.sleep(5)
        scores = {}
        for mode in modes:
            dest = root / mode / 'dev'
            subprocess.run(['/opt/conda/bin/python', str(REPO / 'scripts/eval_robotwin_eraf_fg.py'),
                'summarize', '--output', str(dest), '--checkpoint', str(checkpoint),
                '--catalog-root', spec['catalog'], '--episodes', str(spec['episodes']),
                '--tasks', *spec['tasks'], '--conditions', 'counterfactual', '--skip-file-hashes'],
                cwd=REPO, env=env, check=True, timeout=60, stdout=subprocess.DEVNULL)
            summary = json.loads((dest / 'summary.json').read_text())
            assert summary['complete'] and summary['episodes'] == len(spec['tasks']) * spec['episodes']
            scores[mode] = {cell['source_task']: cell['selected_goal_successes'] for cell in summary['cells']}
            for task in spec['tasks']:
                suffix = Path(task) / 'demo_clean/counterfactual/initial_states.json'
                expected = Path(plan['original_carry_results']) / suffix
                assert json.loads((dest / suffix).read_text()) == json.loads(expected.read_text())
        scores['carry'] = json.loads((source / 'comparison.json').read_text())['models'][args.model]['original_five']
        write('comparison.json', dict(complete=True, models=scores, matched_initial_states=True,
            scope='Original five development tasks, six CF episodes each; no ten-task or independent-test claim.',
            checkpoint_sha256=checkpoint_sha, optimizer_updates=0))
        report.update(complete=True, stage='complete')
    except BaseException as error:
        report['error'] = repr(error)
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
            report['jobs'][name]['exit_code'] = p.returncode
        report['finished'] = datetime.now().astimezone().isoformat()
        write('driver.json', report)


if __name__ == '__main__':
    main()
