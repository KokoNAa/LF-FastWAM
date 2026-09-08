#!/usr/bin/env python3
"""Bounded fixed-state action queries on idle GPUs while the control finishes."""
from __future__ import annotations
import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from scripts.run_robotwin_formal_five40 import read, records, sha, write, RUNS
from scripts.probe_robotwin_balanced_postjoint_actions import compare_actions


def process(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text().split(') ',1)[1].split()
        command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        return dict(live=stat[0]!='Z' and bool(command),start_time=stat[19],command=command)
    except FileNotFoundError:return dict(live=False)


def eligible(state, control_step):
    if state['status']!='training' or state['complete']:
        raise ValueError('Primary trial must still be training.')
    if state['jobs']['train_no_eraf'].get('exit_code') is not None or not 1<=control_step<=160:
        raise ValueError('Need at least40 pending control steps; diagnostics must yield GPUs first.')
    if any(state['jobs']['train_'+a].get('exit_code')!=0 for a in ('eraf_only','eraf_fg')):
        raise ValueError('Both final ERAF training jobs must have exited successfully.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seconds',type=int,default=240)
    args=ap.parse_args();source=args.source_root.resolve();root=args.output.resolve()
    if not 1<=args.seconds<=240 or root.exists() or not root.is_relative_to(RUNS):
        ap.error('Use a new data-disk output and at most240 seconds.')
    plan=read(source/'protocol.json');state=read(source/'status.json');receipt=read(source.with_name(source.name+'-launch.json'))
    if plan['format']!='robotwin_five_task_expert_trial_v1' or plan['steps']!=200 or plan['smoke_only']:
        raise ValueError('Expected the fixed five-task expert200 experiment.')
    def primary_live():
        p=process(receipt['pid'])
        return p['live'] and p['start_time']==receipt['start_time'] and str(source) in p['command']
    content=(source/'no_eraf/joint/rank0.jsonl').read_text();lines=content.splitlines()
    import json
    rows=[json.loads(x) for x in (lines if content.endswith('\n') else lines[:-1]) if x.strip()]
    eligible(state,rows[-1]['step'])
    if not primary_live() or not process(state['jobs']['train_no_eraf']['pid'])['live']:
        raise ValueError('Primary controller/control worker is not live.')
    if any(process(state['jobs']['train_'+a]['pid'])['live'] for a in ('eraf_only','eraf_fg')):
        raise ValueError('An ERAF training process is still live.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Commit code first.')
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in devices.splitlines()}
    if any(g not in memory or memory[g]>1000 for g in (1,2,3)):
        raise ValueError('Diagnostic GPUs1,2,3 must be idle.')
    models=dict(parent_R=dict(checkpoint=plan['strongest_checkpoint'],eraf='off',gpu=1),
                eraf_only=dict(checkpoint=str(source/'eraf_only/joint/step_000200.pt'),eraf='on',gpu=2),
                eraf_fg=dict(checkpoint=str(source/'eraf_fg/joint/step_000200.pt'),eraf='on',gpu=3))
    bindings={str(source/'protocol.json'):sha(source/'protocol.json'),plan['manifest']:sha(plan['manifest'])}
    if bindings[plan['manifest']]!=plan['input_sha256'][plan['manifest']]:raise ValueError('Training manifest changed.')
    from scripts.audit_robotwin_eraf_fg_action_stage import audit
    audits={a:audit(source/a/'joint') for a in ('eraf_only','eraf_fg')}
    for spec in models.values():
        spec['sha256']=sha(spec['checkpoint']);bindings[spec['checkpoint']]=spec['sha256']
    if bindings[models['parent_R']['checkpoint']]!=plan['input_sha256'][models['parent_R']['checkpoint']]:
        raise ValueError('Common parent changed.')
    root.mkdir(parents=True,exist_ok=False)
    write(root/'protocol.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source=str(source),primary_receipt=receipt,source_control_step=rows[-1]['step'],models=models,input_sha256=bindings,
        seconds=args.seconds,tasks=['blocks_ranking_rgb','place_a2b_left'],training_audits=audits,
        optimizer_updates=0,goal_achieved=False,
        scope='Fixed18 action queries/model:6train and12replay-holdout. The parent_R is an initialization reference, not the final no-eraf control. Reset-state diagnostics only; no CF scoring, new test scenes, model selection or updates.'))
    driver=dict(complete=False,terminal=False,jobs={});processes={};cutoff=time.monotonic()+args.seconds
    env=os.environ|dict(PYTHONPATH=str(REPO/'src')+':'+str(REPO),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
    def stop(sig,frame):raise InterruptedError(str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for name,spec in models.items():
            cmd=[sys.executable,'-u',str(REPO/'scripts/probe_robotwin_expanded_fg_actions.py'),
                '--manifest',plan['manifest'],'--source-bank',plan['source_bank'],'--checkpoint',spec['checkpoint'],
                '--output',str(root/name),'--eraf',spec['eraf'],'--tasks','blocks_ranking_rgb','place_a2b_left']
            with (root/(name+'.log')).open('x') as log:
                p=subprocess.Popen(cmd,cwd=REPO,env=env|{'CUDA_VISIBLE_DEVICES':str(spec['gpu'])},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[name]=p;driver['jobs'][name]=dict(pid=p.pid,command=cmd,gpu=spec['gpu'],exit_code=None)
            write(root/'driver.json',driver)
        while any(p.poll() is None for p in processes.values()):
            if time.monotonic()>=cutoff:raise TimeoutError('240second diagnostic limit reached.')
            if not primary_live() or read(source/'status.json')['status']!='training':
                raise RuntimeError('Yield GPUs to the primary evaluation.')
            if any(p.poll() not in (None,0) for p in processes.values()):raise RuntimeError('A diagnostic worker failed.')
            time.sleep(1)
        if any(p.returncode!=0 for p in processes.values()):raise RuntimeError('A diagnostic worker failed.')
        if any(sha(path)!=digest for path,digest in bindings.items()):raise ValueError('An input changed.')
        reports={a:read(root/a/'summary.json') for a in models}
        comparisons={a:compare_actions(reports['parent_R'],reports[a]) for a in ('eraf_only','eraf_fg')}
        comparisons['full_vs_eraf']=compare_actions(reports['eraf_only'],reports['eraf_fg'])
        write(root/'comparison.json',dict(complete=True,comparisons=comparisons,goal_achieved=False,
            scope='Same-observation action MSE only; no CF-success or final no-eraf comparison.'))
        driver['complete']=True
    except BaseException as error:driver['error']=repr(error);raise
    finally:
        for p in processes.values():
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for name,p in processes.items():
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            driver['jobs'][name]['exit_code']=p.returncode
        driver.update(terminal=True,finished_at=datetime.now().astimezone().isoformat());write(root/'driver.json',driver)


if __name__=='__main__':main()
