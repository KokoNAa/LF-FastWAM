#!/usr/bin/env python3
"""Change only FG retention teacher; evaluate all existing paired DEV scenes."""
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
from scripts.run_robotwin_five_task_repair import training_command
from scripts.run_robotwin_formal_five40 import read, write, sha, records
from experiments.robotwin.five_task_action import TASKS

PARENT_SHA = '26aea72064d7c17c39992ce2ece74ae63a7496d13f0931bcee3e2ba39e98fdcd'
SMOKE_CODE = '94200fc251fff5199ba5d0afd5599e8f41c0cb26'
REPORT_CODE = '460cdcbf98361fcaedbde7e19f3861c1f89fbd3b'


def candidate_command(source_plan, root, parent):
    """Preserve the original 200-step recipe; change only the frozen CF teacher."""
    plan = json.loads(json.dumps(source_plan))
    plan['steps'] = 200
    plan['arms']['eraf_fg'].update(parent=str(parent), parent_sha256=PARENT_SHA, gpus=[0, 1, 2])
    cmd = training_command(plan, root, 'eraf_fg')
    cmd[cmd.index('--cf-teacher') + 1] = str(parent)
    return cmd + ['--cf-teacher-mode', 'parent_eraf']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source-trial', 'smoke-root', 'report-worktree', 'output', 'deadline'):
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or cutoff.timestamp() <= time.time():
        ap.error('A future authorized absolute deadline is required.')
    source, smoke, validation, root = map(Path, (args.source_trial, args.smoke_root, args.report_worktree, args.output))
    if root.exists():
        raise FileExistsError(root)
    prior, smoke_proof = read(source / 'completion_audit.json'), read(smoke / 'complete.json')
    source_plan, models = read(source / 'protocol.json'), read(source / 'final_models.json')
    parent = Path(models['eraf_only']['final_checkpoint'])
    if (not prior['complete'] or not smoke_proof['complete'] or smoke_proof['code_commit'] != SMOKE_CODE
            or smoke_proof['parent_sha256'] != PARENT_SHA or sha(parent) != PARENT_SHA
            or not smoke_proof['production_identity']['all_five_tasks_exact']
            or smoke_proof['training_audit']['full_parent_eraf_teacher_sha256'] != PARENT_SHA):
        raise ValueError('Complete source and parent-teacher smoke evidence required.')
    git = lambda *cmd, cwd=REPO: subprocess.check_output(['git', *cmd], cwd=cwd, text=True).strip()
    code = git('rev-parse', 'HEAD')
    if git('status', '--porcelain') or git('rev-parse', 'HEAD', cwd=validation) != REPORT_CODE or git('status', '--porcelain', cwd=validation):
        raise ValueError('Clean pinned source and validation worktrees required.')
    tested = ['experiments/robotwin/native_teacher.py', 'experiments/robotwin/deployed_action_objective.py',
              'experiments/robotwin/parent_eraf_teacher.py', 'scripts/train_robotwin_eraf_fg_action.py',
              'scripts/audit_robotwin_eraf_fg_action_stage.py']
    if git('diff', SMOKE_CODE, 'HEAD', '--', *tested):
        raise ValueError('Training implementation differs from the tested smoke.')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('GPUs must be idle before starting.')
    if shutil.disk_usage(source).free < 8 * 1024**3:
        raise ValueError('Need 8 GiB free on the data disk.')
    bindings = {str(source / n): sha(source / n) for n in
                ('protocol.json', 'final_models.json', 'completion_audit.json', 'catalog_frozen.json')}
    bindings.update({str(smoke / 'complete.json'): sha(smoke / 'complete.json')})
    for arm, spec in models.items():
        if sha(spec['final_checkpoint']) != spec['final_sha256']:
            raise ValueError('Source model changed: ' + arm)
        bindings[spec['final_checkpoint']] = spec['final_sha256']
    for rel, digest in prior['episode_sha256'].items():
        if sha(source / rel) != digest:
            raise ValueError('Source episode evidence changed: ' + rel)
        bindings[str(source / rel)] = digest
    for path in (source_plan['manifest'], REPO / 'configs/eval/robotwin_cis_ten_tasks.json'):
        bindings[str(path)] = sha(path)
    root.mkdir(parents=True)
    (root / 'catalog').symlink_to(source / 'catalog', target_is_directory=True)
    (root / 'evaluation').mkdir()
    for arm in ('no_eraf', 'eraf_only'):
        (root / 'evaluation' / arm).symlink_to(source / 'evaluation' / arm, target_is_directory=True)
    cmd = candidate_command(source_plan, root, parent)
    models['eraf_fg'].update(parent=str(parent), parent_sha256=PARENT_SHA,
                            final_checkpoint=str(root / 'eraf_fg/joint/step_000200.pt'), final_sha256=None)
    plan = dict(format='robotwin_parent_eraf_retention_trial_v1', complete=False, status='admitted', jobs={},
                source_trial=str(source), arms=models, tasks=list(TASKS), dev_episodes=12, steps=200,
                code_commit=code, deadline=args.deadline, input_sha256=bindings, training_command=cmd,
                training_comparison=dict(source_format='robotwin_current_no_eraf_serial_trial_v1',
                    cumulative_optimizer_steps_since_baseline=dict(no_eraf=0, eraf_only=200, eraf_fg=400),
                    equal_additional_training_budget=False),
                cf_teacher_mode='parent_eraf', new_evaluation_episodes=60, reused_reference_episodes=120,
                independent_test=False, correct_evaluated=False, goal_achieved=False, platform_shutdown=None,
                checkpoint_selection='Predeclared final200 only; no partial outcome selection.')
    write(root / 'protocol.json', plan)
    env = os.environ | dict(PATH='/opt/conda/bin:' + os.environ['PATH'], PYTHONPATH=str(REPO / 'src') + ':' + str(REPO),
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    processes = {}
    def save(): write(root / 'status.json', plan)
    def budget():
        if time.time() >= cutoff.timestamp(): raise TimeoutError('Authorized work deadline reached; platform stays on.')
        if shutil.disk_usage(root).free < 3 * 1024**3: raise RuntimeError('Data-disk reserve reached.')
    def launch(name, command, gpus):
        budget()
        with (root / (name + '.log')).open('x') as log:
            p = subprocess.Popen(command, cwd=REPO, env=env | {'CUDA_VISIBLE_DEVICES': gpus},
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = p
        stat = (Path('/proc') / str(p.pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
        plan['jobs'][name] = dict(pid=p.pid, start_time=stat[19], command=command, gpus=gpus)
        save()
    def done(name):
        rc = processes[name].poll()
        if rc is None: return False
        plan['jobs'][name]['exit_code'] = rc; save()
        if rc: raise RuntimeError(name + ' failed: ' + str(rc))
        return True
    def run(name, command, gpus, *, publish_status=True):
        if publish_status: plan['status'] = name
        launch(name, command, gpus)
        while not done(name): budget(); time.sleep(3)
    def stop(sig, frame): raise InterruptedError('Signal ' + str(sig))
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    save()
    try:
        run('train_fg200', cmd, '0,1,2')
        run('audit_training', [sys.executable, str(REPO / 'scripts/audit_robotwin_eraf_fg_action_stage.py'),
                               '--output', str(root / 'eraf_fg/joint')], '')
        proof = read(root / 'eraf_fg/joint/freeze_audit.json')
        if not proof['changed_lora_tensors'] or not proof['changed_guard_tensors'] or proof['full_parent_eraf_teacher_sha256'] != PARENT_SHA:
            raise ValueError('Joint parent-teacher training audit failed.')
        models['eraf_fg']['final_sha256'] = sha(models['eraf_fg']['final_checkpoint'])
        write(root / 'final_models.json', models)
        pending, active = list(TASKS), {}
        plan['status'] = 'evaluating_all_60_dev'; save()
        while pending or active:
            budget()
            for gpu, name in list(active.items()):
                if done(name): del active[gpu]
            for gpu in (0, 1, 2):
                if gpu in active or not pending: continue
                task = pending.pop(0); name = 'eval_' + task
                command = [sys.executable, str(REPO / 'scripts/eval_robotwin_eraf_fg.py'), 'worker',
                    '--output', str(root / 'evaluation/eraf_fg'), '--manifest', source_plan['manifest'],
                    '--checkpoint', models['eraf_fg']['final_checkpoint'], '--catalog-root', str(root / 'catalog'),
                    '--interventions', str(REPO / 'configs/eval/robotwin_cis_ten_tasks.json'), '--tasks', task,
                    '--conditions', 'counterfactual', '--episodes', '12', '--eraf', 'on', '--policy-kind', 'repair',
                    '--memory-mode', 'carry', '--manipulation-metrics', '--videos', '--gpu', str(gpu)]
                launch(name, command, str(gpu)); active[gpu] = name
            if active: time.sleep(3)
        for path, digest in bindings.items():
            if sha(path) != digest: raise ValueError('Bound input changed: ' + path)
        for task in TASKS:
            folder = root / 'evaluation/eraf_fg' / task / 'demo_clean/counterfactual'
            cell = read(folder / 'complete.json')
            if not cell['complete'] or len(records(folder / 'episodes.jsonl')) != 12:
                raise ValueError('Incomplete new DEV cell: ' + task)
        plan.update(complete=True, status='complete', finished_at=datetime.now().astimezone().isoformat()); save()
        write(root / 'terminal_verification.json', dict(complete=True, episodes=180, new_episodes=60,
              reused_reference_episodes=120, all_jobs_exit_zero=True, inputs_unchanged=True,
              independent_test=False, goal_achieved=False))
        run('paired_report', [sys.executable, str(validation / 'scripts/report_robotwin_five_task_comparison.py'),
                             '--root', str(root), '--output', str(root / 'paired_report')], '', publish_status=False)
        plan['status'] = 'complete'; save()
        report = read(root / 'paired_report/PAIRED_COMPARISON.json')
        report['reference_reuse'] = dict(source_trial=str(source), new_episodes=60, reused_reference_episodes=120)
        teacher_cells = []
        for task in TASKS:
            relative = Path('evaluation/eraf_fg') / task / 'demo_clean/counterfactual/episodes.jsonl'
            old_rows, new_rows = records(source / relative), records(root / relative)
            if [r['scene_seed'] for r in old_rows] != [r['scene_seed'] for r in new_rows]:
                raise ValueError('Old/new teacher comparison scenes differ.')
            old = [r['counterfactual_goal_ever_success'] for r in old_rows]
            new = [r['counterfactual_goal_ever_success'] for r in new_rows]
            teacher_cells.append(dict(task=task, episodes=12, old_teacher_cf=sum(old), parent_teacher_cf=sum(new),
                                      gains=sum(not a and b for a,b in zip(old,new,strict=True)),
                                      losses=sum(a and not b for a,b in zip(old,new,strict=True))))
        report['matched_teacher_comparison'] = dict(old_trial=str(source), steps_each=200,
            same_parent_sha256=PARENT_SHA, changed_factor='CF retention teacher only', cells=teacher_cells)
        write(root / 'paired_report/PAIRED_COMPARISON.json', report)
        md = root / 'paired_report/PAIRED_COMPARISON.md'
        extra = ['\n本轮新增 FG 60 回合；no-eraf 和 ERAF 共 120 回合沿用同一固定开发场景的原始记录。属于开发集复测，非独立测试。',
                 '', '同一 ERAF 父模型各续训 200 步，只改变保留教师：', '',
                 '| 任务 | 原 no-eraf 教师 FG200 | 完整父 ERAF 教师 FG200 | 新增/丢失成功 |',
                 '|---|---:|---:|---:|']
        extra += [f"| {c['task']} | {c['old_teacher_cf']}/12 | {c['parent_teacher_cf']}/12 | {c['gains']}/{c['losses']} |" for c in teacher_cells]
        md.write_text(md.read_text() + '\n'.join(extra) + '\n')
    except BaseException as error:
        plan.update(complete=False, status='stopped', error=repr(error)); save(); raise
    finally:
        for p in processes.values():
            if p.poll() is None: os.killpg(p.pid, signal.SIGTERM)
        for p in processes.values():
            if p.poll() is None:
                try: p.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()


if __name__ == '__main__': main()
