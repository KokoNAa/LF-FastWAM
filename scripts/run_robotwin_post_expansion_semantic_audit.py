#!/usr/bin/env python3
"""After complete CF comparisons, inspect frozen semantics only if FG+ERAF does not lead."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from scripts.compare_robotwin_ten_task_methods import build_report


def audit_command(plan, arm):
    return ['/opt/conda/bin/python', '-u', str(REPO / 'scripts/audit_robotwin_eraf_fg_semantics.py'),
            '--manifest', plan['manifest'], '--checkpoint', plan['checkpoints'][arm],
            '--source-bank', plan['source_bank'], '--output', str(Path(plan['output']) / (arm + '.json')),
            '--trajectory-holdout', '--include-truth-reversals']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--wait-for', type=Path, required=True)
    ap.add_argument('--owner-pid', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--deadline', required=True)
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or args.owner_pid <= 0:
        ap.error('Require absolute timezone-aware deadline and positive owner PID.')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Commit the audit code before execution.')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        temporary = root / (name + '.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(root / name)

    owner = args.wait_for.resolve()
    training = json.loads((owner / 'protocol.json').read_text())
    runs = owner.parents[1]
    methods = {arm: {group: str(owner / ('eval_' + group) / arm / 'dev')
                     for group in training['groups']} for arm in training['arms']}
    historical = runs / 'robotwin_cf_priority/20260907-1543-warmoff200-cf4'
    for name, folder in [('historical_strongest_no_eraf', 'ordinary_cf_200'),
                         ('historical_eraf_fg200', 'eraf_fg_200')]:
        methods[name] = {group: str(historical / ('eval_' + group) / folder / 'dev')
                         for group in training['groups']}
    fg_only = runs / 'robotwin_fg_routing/20260907-1949-resume200-fixed'
    methods['historical_best_fg_only'] = {
        group: str(fg_only / ('eval_' + group) / 'fg_only_200/dev') for group in training['groups']}
    preflight_path = runs / 'robotwin_ten_task_grounding/20260908-mask-binding/reversal_query_preflight.json'
    preflight = json.loads(preflight_path.read_text())
    if not preflight['complete'] or preflight['manifest_sha256'] != training['manifest_sha256']:
        raise ValueError('Missing matching held-out query preflight.')
    plan = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                wait_for=str(owner), owner_pid=args.owner_pid, output=str(root), deadline=args.deadline,
                manifest=training['manifest'], manifest_sha256=training['manifest_sha256'],
                source_bank=training['source_bank'], comparison=dict(target='eraf_fg', methods=methods),
                query_preflight=str(preflight_path), query_sha256=preflight['query_sha256'],
                expected_queries=preflight['phase_and_reversal_queries'],
                checkpoints={arm: str(owner / arm / 'joint/step_000200.pt')
                             for arm in ('no_eraf', 'eraf_only', 'eraf_fg')},
                gpus=dict(no_eraf=0, eraf_only=1, eraf_fg=2), platform_shutdown=None,
                scope='Preflight expert-holdout queries per model; no training or policy test. All models remain server-side.',
                trigger='Complete paired ten-task CF comparison does not show FG+ERAF strictly ahead of every listed control.')
    write('plan.json', plan)
    state = dict(complete=False, stage='waiting_for_cf', jobs={})
    write('driver.json', state)
    processes = {}

    def budget():
        if time.time() >= cutoff.timestamp():
            raise TimeoutError('Audit cutoff; no server shutdown.')

    try:
        while True:
            budget()
            proc = Path('/proc') / str(args.owner_pid)
            try:
                live = (proc / 'stat').read_text().split(') ')[1][0] != 'Z'
                command = (proc / 'cmdline').read_bytes()
            except FileNotFoundError:
                live = False
            if not live:
                break
            if b'run_robotwin_ten_task_action_expansion.py' not in command or str(owner).encode() not in command:
                raise ValueError('Owner PID was reused by a different command.')
            time.sleep(10)
        finished = json.loads((owner / 'driver.json').read_text())
        if not finished['complete'] or finished['stage'] != 'complete':
            raise ValueError('Owner stopped without complete CF evidence; do not launch diagnostics.')
        comparison = build_report(plan['comparison'])
        write('paired_comparison.json', comparison)
        if comparison['target_strictly_exceeds_all_listed_dev_controls']:
            state.update(complete=True, stage='dev_lead_requires_independent_validation', audits_launched=False)
            return
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise ValueError('GPUs are occupied after owner exit; do not overlap unrelated work.')
        if hashlib.sha256(Path(plan['manifest']).read_bytes()).hexdigest() != plan['manifest_sha256']:
            raise ValueError('Expert manifest changed.')
        state['stage'] = 'auditing_frozen_semantics'
        env = os.environ | dict(PATH='/opt/conda/bin:' + os.environ['PATH'],
            PYTHONPATH=str(REPO / 'src') + ':' + str(REPO), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
            DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
            VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
        for arm, gpu in plan['gpus'].items():
            budget()
            command = audit_command(plan, arm)
            with (root / (arm + '.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=REPO, env=env | {'CUDA_VISIBLE_DEVICES': str(gpu)},
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes[arm] = process
            state['jobs'][arm] = dict(pid=process.pid, gpu=gpu, command=command)
            write('driver.json', state)
        while any(p.poll() is None for p in processes.values()):
            budget()
            for arm, process in processes.items():
                rc = process.poll()
                if rc is not None:
                    state['jobs'][arm]['exit_code'] = rc
                    if rc:
                        raise RuntimeError(f'{arm} audit failed: {rc}')
            write('driver.json', state)
            time.sleep(5)
        reports = {}
        for arm, process in processes.items():
            state['jobs'][arm]['exit_code'] = process.returncode
            if process.returncode:
                raise RuntimeError(f'{arm} audit failed: {process.returncode}')
            reports[arm] = json.loads((root / (arm + '.json')).read_text())
            if not reports[arm]['complete'] or reports[arm]['manifest_sha256'] != plan['manifest_sha256']:
                raise ValueError('Incomplete or mismatched semantic report.')
            records = [{**{k: row[k] for k in ('pair_id', 'scene_seed', 'language', 'path', 'frame', 'instruction')},
                        'replay_split': 'replay_holdout'} for row in reports[arm]['trajectory_queries']]
            if (len(records) != plan['expected_queries'] or
                    hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest() != plan['query_sha256']):
                raise ValueError('Observed semantic queries differ from the held-out preflight.')
        keys = ('pair_id', 'scene_seed', 'language', 'path', 'frame', 'instruction', 'state_sha256', 'rgb_sha256')
        identities = [[{k: row[k] for k in keys} for row in r['trajectory_queries']] for r in reports.values()]
        if any(x != identities[0] for x in identities[1:]):
            raise ValueError('Semantic audits used different states or instructions.')
        write('comparison.json', dict(complete=True, models={arm: r['cells'] for arm, r in reports.items()},
            matched_queries=len(identities[0]), training_performed=False, goal_achievement_claim=False))
        state.update(complete=True, stage='complete')
    except BaseException as error:
        state['error'] = repr(error)
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        write('driver.json', state)


if __name__ == '__main__':
    main()
