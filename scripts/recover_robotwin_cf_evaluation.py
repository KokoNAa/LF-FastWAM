#!/usr/bin/env python3
"""Recover interrupted CF evaluation, preserving complete cells and trained weights."""
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
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ARMS = ('no_eraf', 'fg_only', 'eraf_only', 'eraf_fg')
DEPLOYMENT = dict(action_horizon=32, replan_steps=24, inference_steps=10)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def metadata(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def assert_source_idle(source, *, proc_root=Path('/proc')):
    for process in proc_root.iterdir():
        if not process.name.isdecimal():
            continue
        try:
            state = (process / 'stat').read_text().rsplit(') ', 1)[1].split()[0]
            argv = (process / 'cmdline').read_bytes()
        except FileNotFoundError:
            continue
        if state != 'Z' and str(source).encode() in argv and int(process.name) != os.getpid():
            # Exclude this controller's launcher; require no training/evaluation owner.
            if any(token in argv for token in (b'run_robotwin_', b'eval_robotwin_', b'train_robotwin_')):
                raise ValueError('Source training/evaluation process is still live: ' + process.name)


def validate_complete_cell(directory, *, checkpoint, catalog, episodes, eraf):
    marker = directory / 'complete.json'
    if not marker.exists():
        return False
    report = read(marker)
    if not report['complete']:
        raise ValueError('A completed cell marker declares failure.')
    if (report['checkpoint'] != str(checkpoint) or report['checkpoint_metadata'] != metadata(checkpoint)
            or report['canonical_metadata'] != metadata(catalog) or not report['skip_file_hashes']
            or report['deployment'] != DEPLOYMENT or report['eraf'] != eraf
            or report['policy_kind'] != 'repair' or report['memory_mode'] != 'carry'):
        raise ValueError('Completed evaluation identity or deployment differs.')
    rows = [json.loads(line) for line in (directory / 'episodes.jsonl').read_text().splitlines()]
    canonical = [json.loads(line) for line in catalog.read_text().splitlines()]
    initial = read(directory / 'initial_states.json')
    if len(rows) != episodes or len(canonical) != episodes or len(initial) != episodes:
        raise ValueError('Completed evaluation has the wrong episode count.')
    for row, expected, state in zip(rows, canonical, initial, strict=True):
        for field in ('episode_index', 'scene_seed', 'source_instruction', 'counterfactual_instruction'):
            if row[field] != expected[field]:
                raise ValueError('Completed episode does not match catalog: ' + field)
        if (row['checkpoint'] != str(checkpoint) or row['condition'] != 'counterfactual'
                or row['selected_goal'] != 'counterfactual'
                or row['policy_instruction'] != expected['counterfactual_instruction']
                or state['scene_seed'] != row['scene_seed']
                or state['sha256'] != expected['initial_physical_state_sha256']):
            raise ValueError('Completed episode goal, model or physical state differs.')
    if read(directory / 'summary.json')['total_episodes'] != episodes:
        raise ValueError('Completed cell summary is incomplete.')
    return True


def worker_command(runtime, plan, source, output, group, task, arm, gpu):
    spec = plan['groups'][group]
    return [sys.executable, '-u', str(runtime / 'scripts/eval_robotwin_eraf_fg.py'), 'worker',
            '--output', str(output / ('eval_' + group) / arm / 'dev'),
            '--checkpoint', str(source / arm / 'joint/step_000200.pt'),
            '--manifest', plan['manifest'], '--catalog-root', spec['catalog'],
            '--episodes', str(spec['episodes']), '--tasks', task, '--policy-kind', 'repair',
            '--eraf', plan['arms'][arm]['eraf'], '--conditions', 'counterfactual',
            '--gpu', str(gpu), '--videos', '--skip-file-hashes',
            '--interventions', str(runtime / 'configs/eval/robotwin_cis_ten_tasks.json')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--deadline', required=True)
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()
    source, runtime, output = (p.resolve() for p in (args.source_root, args.runtime, args.output))
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or cutoff.timestamp() <= time.time():
        raise ValueError('A future timezone-aware process deadline is required.')
    if output.exists():
        raise FileExistsError(output)
    assert_source_idle(source)
    plan = read(source / 'protocol.json')
    driver = read(source / 'driver.json')
    if driver['complete']:
        raise ValueError('Source trial already complete; do not repeat evaluation.')
    for arm in ARMS:
        if any(driver['jobs'][name]['exit_code'] != 0 for name in (arm, arm + '_audit')):
            raise ValueError('All training and audits must have completed before recovery.')
    paired = read(source / 'paired_action_audit.json')
    if not (paired['complete'] and paired['actual_optimizer_steps_per_arm'] == 200
            and paired['actual_sample_ids_and_rank_norms_verified']):
        raise ValueError('Actual paired training audit is incomplete.')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime, text=True).strip() != plan['code_commit']:
        raise ValueError('Original evaluation runtime changed.')
    for checkout in (runtime, REPO):
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=checkout, text=True).strip():
            raise ValueError('Commit runtime/controller code before launch.')
    if sha(plan['manifest']) != plan['manifest_sha256']:
        raise ValueError('Training/evaluation manifest changed.')
    if sha(plan['prior_comparison_config']) != plan['prior_comparison_sha256']:
        raise ValueError('Historical comparison configuration changed.')
    ledger = read(source / 'final_checkpoint_hashes.json')
    if not ledger['complete'] or set(ledger['models']) != set(ARMS):
        raise ValueError('Final weight ledger incomplete.')
    for arm, record in ledger['models'].items():
        path = source / arm / 'joint/step_000200.pt'
        ident = metadata(path)
        if (ident != dict(path=record['path'], size=record['bytes'], mtime_ns=record['mtime_ns'])
                or sha(path) != record['sha256']):
            raise ValueError('Final weight no longer matches archived audit: ' + arm)
    retained, pending = [], []
    for group, spec in plan['groups'].items():
        for task in spec['tasks']:
            catalog = Path(spec['catalog']) / task / 'demo_clean/correct/episodes.jsonl'
            for arm in ARMS:
                rel = Path('eval_' + group) / arm / 'dev' / task / 'demo_clean/counterfactual'
                complete = validate_complete_cell(source / rel, checkpoint=source / arm / 'joint/step_000200.pt',
                    catalog=catalog, episodes=spec['episodes'], eraf=plan['arms'][arm]['eraf'])
                if complete:
                    retained.append(dict(relative_path=str(rel), episodes_sha256=sha(source/rel/'episodes.jsonl')))
                else:
                    pending.append((group, task, arm))
    receipt = dict(format='robotwin_cf_evaluation_recovery_v1', source_root=str(source), runtime=str(runtime),
        controller_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        runtime_commit=plan['code_commit'], retained_cells=retained, pending_cells=pending,
        source_driver_sha256=sha(source/'driver.json'), final_weights=ledger, deadline=args.deadline,
        partial_policy='Retain partial source directories unchanged; rerun entire incomplete cells in new output. No cherry picking or duplicate rows in final matrix.',
        training_steps_this_recovery=0, platform_shutdown=None)
    if args.plan_only:
        print(json.dumps(receipt, indent=2)); return
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('GPUs already owned by a process.')
    if shutil.disk_usage(source).free < 3 * 1024**3 + 100 * 1024**2:
        raise ValueError('Need three-GiB reserve plus evaluation media allowance.')
    output.mkdir(parents=True)
    def write(name, value):
        target = output / name
        temp = target.with_name(target.name + '.tmp')
        temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(target)
    write('recovery_plan.json', receipt)
    write('protocol.json', plan | dict(recovery_controller_commit=receipt['controller_commit'],
        interrupted_trial=str(source), evaluation_recovery=True, deadline=args.deadline))
    write('final_checkpoint_hashes.json', ledger)
    write('paired_action_audit.json', paired)
    for arm in ARMS:
        (output / arm).symlink_to(source / arm, target_is_directory=True)
    for record in retained:
        dest = output / record['relative_path']; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(source / record['relative_path'], target_is_directory=True)
    state = dict(complete=False, terminal=False, stage='evaluating', jobs={}, retained_cells=len(retained))
    processes = {}
    env = os.environ | dict(PATH='/opt/conda/bin:' + os.environ.get('PATH',''),
        PYTHONPATH=str(runtime/'src') + ':' + str(runtime), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
    def budget():
        if time.time() >= cutoff.timestamp(): raise TimeoutError('Own process deadline reached.')
        if shutil.disk_usage(output).free < 3 * 1024**3: raise RuntimeError('Three-GiB reserve reached.')
    def stop(signum, frame): raise InterruptedError('Signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def launch(name, command, gpu):
        budget()
        with (output / (name + '.log')).open('x') as log:
            proc = subprocess.Popen(command, cwd=runtime,
                env=env | {'CUDA_VISIBLE_DEVICES': '' if gpu is None else str(gpu)},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = proc; state['jobs'][name] = dict(pid=proc.pid, command=command, gpu=gpu, exit_code=None)
        write('driver.json', state); return proc
    try:
        active = {}
        while pending or active:
            budget()
            for name, gpu in list(active.items()):
                rc = processes[name].poll()
                if rc is not None:
                    state['jobs'][name]['exit_code'] = rc; del active[name]; write('driver.json', state)
                    if rc: raise RuntimeError(name + ' failed: ' + str(rc))
            for gpu in sorted(set(range(6)) - set(active.values())):
                if not pending: break
                group, task, arm = pending.pop(0); name = f'eval_{group}_{arm}_{task}'
                launch(name, worker_command(runtime, plan, source, output, group, task, arm, gpu), gpu)
                active[name] = gpu
            if active: time.sleep(5)
        state['stage'] = 'summarizing'; write('driver.json', state)
        for group, spec in plan['groups'].items():
            for arm in ARMS:
                name = f'summarize_{group}_{arm}'
                proc = launch(name, [sys.executable, str(runtime/'scripts/eval_robotwin_eraf_fg.py'), 'summarize',
                    '--output', str(output/('eval_'+group)/arm/'dev'), '--checkpoint', str(source/arm/'joint/step_000200.pt'),
                    '--catalog-root', spec['catalog'], '--episodes', str(spec['episodes']), '--tasks', *spec['tasks'],
                    '--conditions', 'counterfactual', '--skip-file-hashes'], None)
                while proc.poll() is None: budget(); time.sleep(2)
                state['jobs'][name]['exit_code'] = proc.returncode; write('driver.json', state)
                if proc.returncode: raise RuntimeError(name + ' failed.')
        sys.path[:0] = [str(runtime), str(runtime/'src')]
        from scripts.run_robotwin_expanded_fg_trial import comparison_config
        from scripts.compare_robotwin_ten_task_methods import build_report
        config = comparison_config(read(plan['prior_comparison_config']), output, plan['groups'], plan['comparison_history_prefix'])
        write('comparison_config.json', config); write('comparison.json', build_report(config))
        for arm, record in ledger['models'].items():
            if sha(record['path']) != record['sha256']: raise ValueError('Weight changed during evaluation: ' + arm)
        for record in retained:
            if sha(output/record['relative_path']/'episodes.jsonl') != record['episodes_sha256']:
                raise ValueError('Retained completed cell changed during recovery.')
        write('recovery_audit.json', dict(complete=True, actual_weights_unchanged=True,
            retained_cells_unchanged=True, no_training_performed=True, final_matrix_complete=True))
        state.update(complete=True, stage='complete')
    except BaseException as error:
        state.update(stage='stopped', error=repr(error)); raise
    finally:
        for proc in processes.values():
            if proc.poll() is None:
                try: os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        for name, proc in processes.items():
            try: proc.wait(timeout=8)
            except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            state['jobs'][name]['exit_code'] = proc.returncode
        state.update(terminal=True, finished_at=datetime.now().astimezone().isoformat()); write('driver.json', state)


if __name__ == '__main__':
    main()
