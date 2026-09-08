#!/usr/bin/env python3
"""Use released GPUs for bounded fixed-state action checks after the ERAF action arms finish."""
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

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]


def eligible(driver,steps):
    if driver.get('stage')!='joint_training' or driver.get('terminal'):
        raise ValueError('Primary controller must still be in joint training.')
    for arm in ('eraf_only','eraf_fg'):
        if driver['jobs'][arm].get('exit_code')!=0:raise ValueError('Both ERAF action workers must finish first.')
    for arm in ('no_eraf','fg_only'):
        if driver['jobs'][arm].get('exit_code') is not None or not 1<=steps[arm]<=170:
            raise ValueError('Need at least thirty pending updates in each slower arm.')


def live(pid,source):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text().rsplit(') ',1)[1].split()
        cmd=(p/'cmdline').read_bytes()
    except FileNotFoundError:return False
    return stat[0]!='Z' and str(source).encode() in cmd and b'run_robotwin_initial_anchor_trial.py' in cmd


def compare_actions(previous,current):
    if not previous['complete'] or not current['complete']:
        raise ValueError('Incomplete action diagnostic.')
    keys=('id','task','kind','split','scene_seed','frame','raw_path','source_instruction',
          'counterfactual_instruction','state_sha256','rgb_sha256','reference_sha256','valid_sha256')
    old=previous['records'];new=current['records']
    if len(old)!=18 or len(new)!=18 or [{k:r[k] for k in keys} for r in old]!=[{k:r[k] for k in keys} for r in new]:
        raise ValueError('Actual diagnostic observations or references differ.')
    groups={}
    for rows,label in [(old,'previous'),(new,'current')]:
        for task in sorted({r['task'] for r in rows}):
            for split in ('train','replay_holdout'):
                for kind in ('ordinary','fg'):
                    selected=[r for r in rows if (r['task'],r['split'],r['kind'])==(task,split,kind)]
                    key=task+'|'+split+'|'+kind
                    groups.setdefault(key,{})[label]=dict(observations=len(selected),
                        mean_deployed_cf_mse_first24=sum(r['deployed_cf_mse_first24'] for r in selected)/len(selected),
                        mean_endpoint_mse_first24=sum(r['flow_velocity_mse']['endpoint_action_mse_first24'] for r in selected)/len(selected))
    return dict(actual_inputs_equal=True,groups=groups)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seconds',type=int,default=240)
    args=ap.parse_args()
    if not 1<=args.seconds<=240:ap.error('At most240seconds of diagnostic work.')
    source=args.source_root.resolve();root=args.output.resolve();read=lambda p:json.loads(Path(p).read_text())
    plan=read(source/'protocol.json');driver=read(source/'driver.json')
    pid=read(source.with_name(source.name+'-launch.json'))['pid']
    if not live(pid,source):raise ValueError('Primary controller is not live.')
    steps={}
    for arm in ('no_eraf','fg_only'):
        rows=[]
        for line in (source/arm/'joint/rank0.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
        steps[arm]=rows[-1]['step'] if rows else 0
    eligible(driver,steps)
    for arm in ('eraf_only','eraf_fg'):
        if not read(source/arm/'joint/complete.json')['complete'] or read(source/arm/'joint/plan.json')['steps']!=200:
            raise ValueError('Missing final200action stage.')
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    uuids={u.strip() for line in gpu_rows.splitlines() for i,u in [line.split(',')] if int(i.strip()) in (0,2)}
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if len(uuids)!=2 or any(line.split(',')[0].strip() in uuids for line in active.splitlines()):
        raise ValueError('Diagnostic GPUs0and2 are not free.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Commit first.')
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    models={m:source/a/'joint/step_000200.pt' for m,a in [('ordinary','eraf_only'),('fg','eraf_fg')]}
    bindings={str(p):file_sha256(p) for p in [Path(plan['manifest']),*models.values()]}
    root.mkdir(parents=True,exist_ok=False);processes={};cutoff=time.monotonic()+args.seconds
    state=dict(complete=False,terminal=False,jobs={})
    def write(name,value):
        p=root/(name+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(root/name)
    write('protocol.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_root=str(source),source_pid=pid,source_pending_steps=steps,seconds=args.seconds,gpus=[0,2],
        input_sha256=bindings,started_at=datetime.now().astimezone().isoformat(),
        scope='Fixed18train/holdout action queries per finished action model; no model updates, CF scoring or checkpoint selection. Stop own probes if primary leaves joint training.'))
    write('driver.json',state)
    from scripts.audit_robotwin_eraf_fg_action_stage import audit
    for mode,model_path in models.items():
        write(mode+'_freeze_audit.json',audit(model_path.parent))
    env=os.environ|dict(PYTHONPATH=str(REPO/'src')+':'+str(REPO),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
    def stop(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for mode,gpu in [('ordinary',0),('fg',2)]:
            cmd=[sys.executable,'-u',str(REPO/'scripts/probe_robotwin_expanded_fg_actions.py'),
                '--manifest',plan['manifest'],'--source-bank',plan['source_bank'],'--checkpoint',str(models[mode]),
                '--output',str(root/mode),'--eraf','on']
            with (root/(mode+'.log')).open('x') as log:
                p=subprocess.Popen(cmd,cwd=REPO,env=env|dict(CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[mode]=p;state['jobs'][mode]=dict(pid=p.pid,command=cmd,gpu=gpu,exit_code=None);write('driver.json',state)
        while any(p.poll() is None for p in processes.values()):
            if time.monotonic()>=cutoff:raise TimeoutError('Diagnostic budget reached.')
            if not live(pid,source) or read(source/'driver.json')['stage']!='joint_training':
                raise RuntimeError('Yield diagnostic GPUs to the primary experiment.')
            for mode,p in processes.items():
                rc=p.poll()
                if rc is not None and rc!=0:raise RuntimeError(mode+' probe failed:'+str(rc))
            time.sleep(1)
        if any(p.returncode!=0 for p in processes.values()):raise RuntimeError('An action probe failed.')
        if any(file_sha256(p)!=digest for p,digest in bindings.items()):raise ValueError('An input changed.')
        prior=Path(plan['source_action_diagnostic'])
        compared={}
        for mode,arm in [('ordinary','eraf_only'),('fg','eraf_fg')]:
            previous=read(prior/arm/'summary.json');current=read(root/mode/'summary.json')
            compared[arm]=compare_actions(previous,current)
        write('comparison.json',dict(complete=True,comparisons=compared,
            prior_diagnostic=str(prior),source_diagnostic_sha256=file_sha256(prior/'comparison.json'),
            scope='Fixed-state action error comparisons only, not CF success or independent-test results.'))
        state['complete']=True
    except BaseException as error:
        state['error']=repr(error);raise
    finally:
        for p in processes.values():
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for mode,p in processes.items():
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            state['jobs'][mode]['exit_code']=p.returncode
        state.update(terminal=True,finished_at=datetime.now().astimezone().isoformat());write('driver.json',state)


if __name__=='__main__':main()
