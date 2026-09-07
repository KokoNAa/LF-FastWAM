#!/usr/bin/env python3
"""Run the four-arm FG weight/routing mechanism trial with a shared GPU queue."""
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
from scripts.run_robotwin_cf_priority_campaign import OLD_TASKS, NEW_TASKS, training_command, write


def protocol(primary, primary_root, output, deadline, code):
    plan = dict(primary)
    plan.update(format='robotwin_fg_weight_routing_trial_v1', code_commit=code,
        deadline=deadline, joint_steps=200, checkpoints=[50, 200], correction_weight=0.1,
        interface_parent=str(primary_root / 'eraf_fg/interface/step_000100.pt'),
        primary_root=str(primary_root), output=str(output),
        selection='Evaluate the predeclared final step200 only; save every50 steps for recovery across bounded power windows.',
        arms={
            'ordinary_cf': dict(eraf='off', fg='off', route='joint', gpus=[0]),
            'fg_only': dict(eraf='off', fg='full', route='joint', gpus=[1]),
            'eraf_fg': dict(eraf='on', fg='full', route='joint', gpus=[2, 3]),
            'eraf_fg_routed': dict(eraf='on', fg='full', route='eraf_only', gpus=[4, 5])},
        causal_scope='Both ERAF arms reuse the identical archived policy-frozen interface100 parent and differ only in FG gradient routing. All arms use correction coefficient0.1 and fresh optimizers. Controls use one GPU and ERAF arms two; global sample/noise schedule is unchanged but floating-point reduction order can differ. No equal-FLOP claim.',
        strongest_comparator=dict(checkpoint=str(primary_root / 'ordinary_cf/joint/step_000200.pt'),
            original_five_cf='21/30', ten_task_macro_cf=0.35,
            evidence_root=str(primary_root), correction_weight=1.0),
        hypothesis='Current-recipe three-batch gradients show large opposing FG gradients in policy and ERAF interfaces. Lower FG coefficient and an ERAF-only FG gradient route require closed-loop verification.',
        data_scope='Original five-task bank remains fixed. The separately prepared formal80-scene bank is not used in this mechanism trial.',
        independent_test_opened=False)
    return plan


def command(plan, output, arm):
    spec = plan['arms'][arm]
    parent = plan['interface_parent'] if spec['eraf'] == 'on' else plan['checkpoint']
    result = training_command(plan, output, arm, 'joint', parent, warm=spec['eraf'] == 'off')
    result[result.index('--nproc_per_node=2')] = f"--nproc_per_node={len(spec['gpus'])}"
    return result + ['--fg-gradient-route', spec['route']]


def cells_for(arm):
    return [(arm, 'original_five', task, 6) for task in OLD_TASKS] + [
        (arm, 'additional_five', task, 3) for task in NEW_TASKS]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--primary-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--deadline', required=True)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None:
        ap.error('The cutoff needs an explicit timezone')
    primary_root, output = args.primary_root.resolve(), args.output.resolve()
    primary = json.loads((primary_root / 'protocol.json').read_text())
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    plan = protocol(primary, primary_root, output, args.deadline, code)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'protocol.json', plan)
    if not args.execute:
        return
    for key in ('checkpoint', 'interface_parent', 'manifest', 'source_bank', 'correct_teacher', 'old_catalog', 'new_catalog'):
        if not Path(plan[key]).exists():
            raise FileNotFoundError(plan[key])
    if not json.loads((primary_root / 'driver.json').read_text())['complete']:
        raise ValueError('The archived primary experiment must be complete')
    interface = json.loads((primary_root / 'eraf_fg/interface/plan.json').read_text())
    if interface['checkpoint'] != plan['checkpoint'] or interface['stage'] != 'interface' or interface['steps'] != 100:
        raise ValueError('Unexpected shared interface parent')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO, text=True).strip():
        raise RuntimeError('Commit tracked changes before running')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('The six GPUs must be free before initial assignment')
    env = os.environ | dict(PATH='/opt/conda/bin:' + os.environ['PATH'],
        PYTHONPATH=str(REPO / 'src') + ':' + str(REPO), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
    report = dict(complete=False, stage='training_and_fixed_cf_evaluation', jobs={},
                  started=datetime.now().astimezone().isoformat(), code_commit=code)
    processes, running, pending = {}, {}, []
    completed_training = set()

    def save():
        write(output / 'driver.json', report)

    def budget():
        if time.time() >= cutoff.timestamp():
            raise TimeoutError('Work cutoff reached; retain all checkpoints and partial evaluations')
        if shutil.disk_usage(output).free < 5 * 1024**3:
            raise RuntimeError('Five GiB disk reserve reached')

    def start(name, cmd, gpus, kind, arm=None):
        budget()
        with (output / (name + '.log')).open('x') as log:
            proc = subprocess.Popen(list(map(str, cmd)), cwd=REPO,
                env=env | {'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus))},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = proc
        running[name] = dict(gpus=gpus, kind=kind, arm=arm)
        report['jobs'][name] = dict(pid=proc.pid, command=list(map(str, cmd)), gpus=gpus,
            kind=kind, arm=arm, started=datetime.now().astimezone().isoformat())
        save()

    def stop(signum, frame):
        raise InterruptedError(f'Trial received signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    try:
        for arm, spec in plan['arms'].items():
            (output / arm).mkdir()
            start(arm + '_joint', command(plan, output, arm), spec['gpus'], 'train', arm)
        while running or pending:
            budget()
            for name, job in list(running.items()):
                rc = processes[name].poll()
                if rc is None:
                    continue
                report['jobs'][name].update(exit_code=rc, completed=datetime.now().astimezone().isoformat())
                del running[name]
                save()
                if rc:
                    raise RuntimeError(f'{name} failed with exit {rc}')
                if job['kind'] == 'train':
                    arm = job['arm']
                    done = json.loads((output / arm / 'joint/complete.json').read_text())
                    if not done['complete'] or done['optimizer_steps'] != 200:
                        raise ValueError('Training did not complete the declared budget')
                    completed_training.add(arm)
                    pending.extend(cells_for(arm))
            # Evaluate expensive original tasks first as each trained model becomes ready.
            pending.sort(key=lambda c: (c[1] != 'original_five', (OLD_TASKS + NEW_TASKS).index(c[2])))
            busy = {gpu for job in running.values() for gpu in job['gpus']}
            for gpu in sorted(set(range(6)) - busy):
                if not pending:
                    break
                arm, group, task, episodes = pending.pop(0)
                catalog = plan['old_catalog' if group == 'original_five' else 'new_catalog']
                dest = output / ('eval_' + group) / (arm + '_200') / 'dev'
                cmd = [sys.executable, '-u', REPO / 'scripts/eval_robotwin_eraf_fg.py', 'worker',
                    '--output', dest, '--checkpoint', output / arm / 'joint/step_000200.pt',
                    '--manifest', plan['manifest'], '--catalog-root', catalog, '--episodes', str(episodes),
                    '--tasks', task, '--policy-kind', 'repair', '--eraf', plan['arms'][arm]['eraf'],
                    '--conditions', 'counterfactual', '--gpu', str(gpu), '--videos', '--skip-file-hashes',
                    '--interventions', REPO / 'configs/eval/robotwin_cis_ten_tasks.json']
                start(f'eval_{group}_{arm}_{task}', cmd, [gpu], 'eval', arm)
            if running:
                time.sleep(5)
        if completed_training != set(plan['arms']):
            raise ValueError('Missing completed arm')
        report['stage'] = 'summarizing'
        save()
        results = {}
        for arm in plan['arms']:
            results[arm] = {}
            for group, tasks, episodes, catalog in [('original_five', OLD_TASKS, 6, plan['old_catalog']),
                                                   ('additional_five', NEW_TASKS, 3, plan['new_catalog'])]:
                dest = output / ('eval_' + group) / (arm + '_200') / 'dev'
                name = f'summarize_{group}_{arm}'
                start(name, [sys.executable, REPO / 'scripts/eval_robotwin_eraf_fg.py', 'summarize',
                    '--output', dest, '--checkpoint', output / arm / 'joint/step_000200.pt',
                    '--catalog-root', catalog, '--episodes', str(episodes), '--tasks', *tasks,
                    '--conditions', 'counterfactual', '--skip-file-hashes'], [], 'summary', arm)
                while processes[name].poll() is None:
                    budget(); time.sleep(1)
                report['jobs'][name]['exit_code'] = processes[name].returncode
                del running[name]
                if processes[name].returncode:
                    raise RuntimeError(f'{name} failed')
                summary = json.loads((dest / 'summary.json').read_text())
                cells = summary['cells']
                if (not summary['complete'] or summary['episodes'] != len(tasks) * episodes
                        or len(cells) != len(tasks) or {c['source_task'] for c in cells} != set(tasks)
                        or any(c['condition'] != 'counterfactual' or c['total_episodes'] != episodes for c in cells)):
                    raise ValueError('Incomplete fixed CF evaluation')
                results[arm][group] = {c['source_task']: c['selected_goal_successes'] for c in cells}
        for group, tasks in [('original_five', OLD_TASKS), ('additional_five', NEW_TASKS)]:
            for task in tasks:
                states = [json.loads((output / ('eval_' + group) / (arm + '_200') / 'dev' /
                    task / 'demo_clean/counterfactual/initial_states.json').read_text()) for arm in plan['arms']]
                reference = json.loads((primary_root / ('eval_' + group) / 'ordinary_cf_200/dev' /
                    task / 'demo_clean/counterfactual/initial_states.json').read_text())
                if any(state != reference for state in states):
                    raise ValueError('Physical initial states differ from the strongest archived control')
        for value in results.values():
            value['ten_task_macro_cf'] = (sum(value['original_five'].values()) / 6 +
                                         sum(value['additional_five'].values()) / 3) / 10
        write(output / 'comparison.json', dict(complete=True, models=results,
            strongest_archived_control=plan['strongest_comparator'], matched_initial_states=True,
            independent_test=False, correct_evaluated=False))
        report.update(complete=True, stage='complete', completed=datetime.now().astimezone().isoformat())
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        for name, process in processes.items():
            if process.poll() is None:
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
            report['jobs'][name]['exit_code'] = process.returncode
        save()


if __name__ == '__main__':
    main()
