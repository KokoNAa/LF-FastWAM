#!/usr/bin/env python3
"""Run five frozen-model diagnostics only after the complete expanded CF trial."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--deadline', required=True)
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None:
        ap.error('Deadline needs timezone.')
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    source = args.source_root.resolve(); root = args.output.resolve()
    read = lambda p: json.loads(p.read_text())
    driver = read(source/'driver.json')
    if not (driver.get('complete') and driver.get('terminal') and driver.get('stage') == 'complete'):
        raise ValueError('Source trial is not complete.')
    launch = read(source.with_name(source.name+'-launch.json'))
    for pid in [launch['pid'], *[v['pid'] for v in driver['jobs'].values()]]:
        proc = Path('/proc')/str(pid)
        try:
            if (proc/'stat').read_text().rsplit(') ', 1)[1][0] != 'Z' and str(source).encode() in (proc/'cmdline').read_bytes():
                raise ValueError('A source process is still live.')
        except FileNotFoundError:
            pass
    if any(v.get('exit_code') != 0 for v in driver['jobs'].values()):
        raise ValueError('Source trial contains nonzero exits.')
    plan = read(source/'plan.json'); hashes = read(source/'final_checkpoint_hashes.json')
    if not hashes['complete'] or not read(source/'comparison.json')['complete']:
        raise ValueError('Missing completed comparison or hash ledger.')
    models = {'strongest': dict(path=plan['strongest_checkpoint'], sha256=plan['source_policy_sha256'], eraf='off')}
    models.update({a:dict(path=h['path'], sha256=h['sha256'], eraf=plan['arms'][a]['eraf']) for a,h in hashes['models'].items()})
    if len(models) != 5 or file_sha256(plan['manifest']) != plan['manifest_sha256']:
        raise ValueError('Invalid model set or changed manifest.')
    for m in models.values():
        if file_sha256(m['path']) != m['sha256']:
            raise ValueError('A completed source model changed.')
    root.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        p = root/name; temp = p.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2)+'\n');temp.replace(p)
    state = dict(complete=False, terminal=False, stage='diagnostic', jobs={})
    write('protocol.json', dict(models=models, source_root=str(source), source_plan_sha256=file_sha256(source/'plan.json'),
        source_comparison_sha256=file_sha256(source/'comparison.json'), manifest=plan['manifest'],
        manifest_sha256=plan['manifest_sha256'], deadline=args.deadline, optimizer_updates=0,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(), platform_shutdown=None))
    processes = {}
    try:
        for gpu, (name, m) in enumerate(models.items()):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       PYTHONPATH=str(REPO/'src')+':'+str(REPO),
                       DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
            cmd = [sys.executable, '-u', str(REPO/'scripts/probe_robotwin_expanded_fg_actions.py'),
                   '--manifest',plan['manifest'],'--source-bank',plan['source_bank'], '--checkpoint',m['path'],
                   '--output',str(root/name),'--eraf',m['eraf']]
            with (root/(name+'.log')).open('x') as log:
                p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes[name] = p
            state['jobs'][name] = dict(pid=p.pid, gpu=gpu, command=cmd)
            write('driver.json', state)
        while True:
            if time.time() >= cutoff.timestamp():
                raise TimeoutError('Diagnostic-only process deadline reached; platform stays on.')
            for name,p in processes.items():
                rc=p.poll()
                if rc is not None:
                    state['jobs'][name]['exit_code']=rc
                    if rc:raise RuntimeError(name+' diagnostic failed: '+str(rc))
            write('driver.json',state)
            if all(p.poll() is not None for p in processes.values()):break
            time.sleep(5)
        reports={k:read(root/k/'summary.json') for k in models}
        reference=None; aggregates={}
        keys=('id','task','kind','split','scene_seed','frame','raw_path','source_instruction','counterfactual_instruction',
              'state_sha256','rgb_sha256','reference_sha256','valid_sha256')
        for name,report in reports.items():
            if not report['complete'] or not report['input_hashes_stable'] or report['protocol']['checkpoint_sha256'] != models[name]['sha256']:
                raise ValueError('Incomplete or changed diagnostic inputs.')
            identity=[{k:r[k] for k in keys} for r in report['records']]
            if reference is None:reference=identity
            if identity != reference:raise ValueError('Actual cross-model observations/references differ.')
            groups=defaultdict(list)
            for r in report['records']:groups[r['task']+'|'+r['split']+'|'+r['kind']].append(r)
            aggregates[name]={key:dict(observations=len(rs), **{metric:sum(r[metric] for r in rs)/len(rs)
                for metric in ('deployed_cf_mse_first24','deployed_source_to_cf_reference_mse_first24','deployed_language_delta_mse_first24')},
                mean_flow_velocity_mse={s:sum(r['flow_velocity_mse'][s] for r in rs)/len(rs) for s in rs[0]['flow_velocity_mse']})
                for key,rs in groups.items()}
        write('comparison.json',dict(complete=True, matched_actual_inputs=True, observations_per_model=len(reference),
                                    aggregates=aggregates, scope='Diagnostic action errors only; not closed-loop CF success.'))
        state.update(complete=True,terminal=True,stage='complete')
    except BaseException as error:
        state.update(complete=False,terminal=True,stage='failed',error=repr(error))
        raise
    finally:
        for name,p in processes.items():
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=8)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            state['jobs'][name]['exit_code']=p.returncode
        state['finished_at']=datetime.now().astimezone().isoformat();write('driver.json',state)


if __name__ == '__main__':main()
