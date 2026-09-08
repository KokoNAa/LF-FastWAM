#!/usr/bin/env python3
"""Bounded production identity and full-unroll gradient feasibility on proposed parents."""
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

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
ARMS=('eraf_only','eraf_fg')
read=lambda p:json.loads(Path(p).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def live(pid,root):
    proc=Path('/proc')/str(pid)
    try:
        return proc.joinpath('stat').read_text().rsplit(') ',1)[1].split()[0]!='Z' and str(root).encode() in proc.joinpath('cmdline').read_bytes()
    except FileNotFoundError:return False


def compare(summaries):
    if set(summaries)!=set(ARMS) or any(s.get('complete') is not True or s.get('parameters_unchanged') is not True or s.get('input_hashes_stable') is not True for s in summaries.values()):
        raise ValueError('Both complete unchanged-parent diagnostics required.')
    identities=[]
    for summary in summaries.values():
        rows=summary['records']
        if len(rows)!=20 or len(summary['gradients'])!=8:raise ValueError('Incomplete identity or gradient coverage.')
        for r in rows:
            if any(v['exact'] is not True for v in r['modes'].values()) or r['modes']['on']['action_digest']!=r['modes']['off']['action_digest']:
                raise ValueError('Proposed parent lost deployed zero-residual identity.')
        identities.append(rows)
    if identities[0]!=identities[1]:raise ValueError('Parent production identities differ.')
    return dict(complete=True,actual_inputs_and_actions_equal=True,
                results={a:dict(gradients=s['gradients'],parameters_unchanged=True) for a,s in summaries.items()},
                scope='Production identity and gradient feasibility, no optimizer updates or CF score.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','recovery-root','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--seconds',type=int,default=900)
    args=ap.parse_args()
    if not 1<=args.seconds<=900:ap.error('Maximum duration is900seconds.')
    source=args.source_root.resolve();recovery=args.recovery_root.resolve();root=args.output.resolve()
    if root.exists():raise ValueError('Output must be new.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Commit code first.')
    driver=read(recovery/'driver.json');audit=read(recovery/'recovery_audit.json')
    if not driver['complete'] or not driver['terminal'] or any(j['exit_code']!=0 for j in driver['jobs'].values()):
        raise ValueError('Recovery has not completed successfully.')
    if any(audit.get(k) is not True for k in ('complete','actual_weights_unchanged','retained_cells_unchanged','no_training_performed','final_matrix_complete')):
        raise ValueError('Recovery integrity audit failed.')
    old_pid=read(recovery.with_name(recovery.name+'-launch.json'))['pid']
    if live(old_pid,recovery) or any(live(j['pid'],recovery) for j in driver['jobs'].values()):
        raise ValueError('Recovery processes still live.')
    if shutil.disk_usage(source).free<3*1024**3:raise ValueError('Need3GiB disk reserve.')
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    uuids={u.strip() for line in gpu_rows.splitlines() for i,u in [line.split(',')] if int(i.strip()) in (0,1)}
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if len(uuids)!=2 or any(line.split(',')[0].strip() in uuids for line in active.splitlines()):
        raise ValueError('Diagnostic GPUs0and1 are not free.')
    plan=read(source/'protocol.json')
    models={a:Path(plan['semantic_parents'][k]) for a,k in [('eraf_only','ordinary'),('eraf_fg','fg')]}
    bindings={str(p):sha(p) for p in [Path(plan['manifest']),Path(plan['correct_teacher']),Path(plan['strongest_checkpoint']),*models.values()]}
    if bindings[plan['manifest']]!=plan['manifest_sha256']:raise ValueError('Manifest changed.')
    from experiments.robotwin.eraf_action_protocol import validate_joint_identity_audit
    for arm,kind in [('eraf_only','ordinary'),('eraf_fg','fg')]:
        identity=read(Path(plan['identity_root'])/kind/'summary.json')
        validate_joint_identity_audit(identity,checkpoint_sha256=bindings[str(models[arm])],manifest_sha256=bindings[plan['manifest']])
    from scripts.probe_robotwin_semantic_oracle_actions import selected_rows
    selected=selected_rows(read(plan['manifest'])['states'])
    root.mkdir(parents=True,exist_ok=False);processes={};cutoff=time.monotonic()+args.seconds
    state=dict(complete=False,terminal=False,jobs={})
    def write(name,value):
        p=root/(name+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(root/name)
    write('protocol.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_root=str(source),recovery_root=str(recovery),seconds=args.seconds,gpus=[0,1],
        input_sha256=bindings,selected_ids=[r['id'] for r in selected],started_at=datetime.now().astimezone().isoformat(),
        optimizer_updates=0,platform_shutdown=None,model_storage='Server only; JSON and code may be archived locally.'))
    write('driver.json',state)
    env=os.environ|dict(PYTHONPATH=str(REPO/'src')+':'+str(REPO),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
    def stop(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for gpu,arm in enumerate(ARMS):
            cmd=[sys.executable,'-u',str(REPO/'scripts/probe_robotwin_deployed_action_objective.py'),
                '--manifest',plan['manifest'],'--source-bank',plan['source_bank'],
                '--correct-teacher',plan['correct_teacher'],'--cf-teacher',plan['strongest_checkpoint'],
                '--fg','off' if arm=='eraf_only' else 'full',
                '--checkpoint',str(models[arm]),'--output',str(root/arm)]
            with (root/(arm+'.log')).open('x') as log:
                p=subprocess.Popen(cmd,cwd=REPO,env=env|dict(CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[arm]=p;state['jobs'][arm]=dict(pid=p.pid,command=cmd,gpu=gpu,exit_code=None);write('driver.json',state)
        while any(p.poll() is None for p in processes.values()):
            if time.monotonic()>=cutoff:raise TimeoutError('Diagnostic budget reached.')
            for arm,p in processes.items():
                rc=p.poll()
                if rc is not None and rc!=0:raise RuntimeError(arm+' probe failed:'+str(rc))
            time.sleep(1)
        if any(p.returncode!=0 for p in processes.values()):raise RuntimeError('A probe failed.')
        if any(sha(p)!=digest for p,digest in bindings.items()):raise ValueError('An input changed.')
        write('comparison.json',compare({a:read(root/a/'summary.json') for a in ARMS}))
        state['complete']=True
    except BaseException as error:
        state['error']=repr(error);raise
    finally:
        for p in processes.values():
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for arm,p in processes.items():
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            state['jobs'][arm]['exit_code']=p.returncode
        state.update(terminal=True,finished_at=datetime.now().astimezone().isoformat());write('driver.json',state)


if __name__=='__main__':main()
