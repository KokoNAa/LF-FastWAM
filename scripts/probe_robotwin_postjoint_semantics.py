#!/usr/bin/env python3
"""Use released GPUs for bounded fixed-query semantics after the ERAF action arms finish."""
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
        if driver['jobs'][arm].get('exit_code') is not None or not 1<=steps[arm]<=150:
            raise ValueError('Need at least fifty pending updates in each slower arm.')


def live(pid,source):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text().rsplit(') ',1)[1].split()
        cmd=(p/'cmdline').read_bytes()
    except FileNotFoundError:return False
    return stat[0]!='Z' and str(source).encode() in cmd and b'continue_robotwin_expanded_fg_actions.py' in cmd


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
        scope='Fixed80held-out semantic queries per finished action model; no model updates, CF scoring or checkpoint selection. Stop own probes if primary leaves joint training.'))
    write('driver.json',state)
    env=os.environ|dict(PYTHONPATH=str(REPO/'src')+':'+str(REPO),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
    def stop(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for mode,gpu in [('ordinary',0),('fg',2)]:
            cmd=[sys.executable,'-u',str(REPO/'scripts/probe_robotwin_cross_goal_semantics.py'),
                '--manifest',plan['manifest'],'--source-bank',plan['source_bank'],'--checkpoint',str(models[mode]),
                '--output',str(root/(mode+'.json'))]
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
        if any(p.returncode!=0 for p in processes.values()):raise RuntimeError('A semantic probe failed.')
        if any(file_sha256(p)!=digest for p,digest in bindings.items()):raise ValueError('An input changed.')
        from scripts.compare_robotwin_cross_goal_semantics import compare
        prior=Path(plan['continuation_source'])/'terminal'
        reports={m+'_semantic1500':read(prior/(m+'.json')) for m in models}
        reports.update({m+'_joint200':read(root/(m+'.json')) for m in models})
        write('comparison.json',compare(reports));state['complete']=True
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
