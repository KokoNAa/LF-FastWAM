#!/usr/bin/env python3
"""Bounded read-only gradient checks while the two control arms continue training."""
from __future__ import annotations
import argparse
from datetime import datetime
import json,os,signal,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from scripts.probe_robotwin_deployed_postjoint_actions import eligible,live


def compare(reports):
    if set(reports)!={'eraf_only','eraf_fg'}:raise ValueError('Both arms required.')
    for r in reports.values():
        if (not r.get('complete') or not r.get('parameters_unchanged') or not r.get('input_hashes_stable')
                or r.get('optimizer_updates')!=0 or len(r['records'])!=24):raise ValueError('Incomplete read-only gradient reports.')
    a=reports['eraf_only']['records'];b=reports['eraf_fg']['records']
    replacements=0
    for x,y in zip(a,b,strict=True):
        if (x['step'],x['index'],x['seed'],x['pair_id'])!=(y['step'],y['index'],y['seed'],y['pair_id']):
            raise ValueError('Gradient queries differ in schedule or task.')
        if y['group']=='fg':
            if x['group']!='ordinary_cf_control':raise ValueError('Unmatched FG control.')
            replacements+=1
        elif any(x[k]!=y[k] for k in ('id','group','row_sha256')):
            raise ValueError('Actual common gradient rows differ.')
    if replacements!=6:raise ValueError('Expected six FG replacement positions.')
    return dict(complete=True,matched_common_queries=18,matched_fg_control_positions=6,
        alignment={k:r['alignment'] for k,r in reports.items()},
        scope='Local Euclidean gradient alignment at two different final checkpoints, not actual Adam updates, causal CF attribution or checkpoint selection.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('source-root','output'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--seconds',type=int,default=360);args=ap.parse_args()
    if not 1<=args.seconds<=360:ap.error('Maximum 360 seconds.')
    source=args.source_root.resolve();root=args.output.resolve();read=lambda p:json.loads(Path(p).read_text())
    if root.exists():raise ValueError('Output already exists.')
    pid=read(source.with_name(source.name+'-launch.json'))['pid'];driver=read(source/'driver.json')
    if not live(pid,source):raise ValueError('Primary is not live.')
    steps={a:readline(source/a/'joint/rank0.jsonl')['step'] for a in ('no_eraf','fg_only')}
    eligible(driver,steps)
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Commit first.')
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    uuids={u.strip() for line in gpu_rows.splitlines() for i,u in [line.split(',')] if int(i.strip()) in (0,2)}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if len(uuids)!=2 or any(line.split(',')[0].strip() in uuids for line in apps.splitlines()):raise ValueError('GPUs not free.')
    root.mkdir(parents=True);processes={};cutoff=time.monotonic()+args.seconds
    state=dict(complete=False,terminal=False,jobs={})
    def write(name,value):
        p=root/(name+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(root/name)
    write('protocol.json',dict(source_root=str(source),source_pid=pid,source_steps=steps,seconds=args.seconds,gpus=[0,2],
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        started_at=datetime.now().astimezone().isoformat(),optimizer_updates=0,platform_shutdown=None))
    write('driver.json',state)
    env=os.environ|dict(PYTHONPATH=str(REPO/'src')+':'+str(REPO),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
    def stop(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for arm,gpu in [('eraf_only',0),('eraf_fg',2)]:
            cmd=[sys.executable,'-u',str(REPO/'scripts/probe_robotwin_deployed_gradients.py'),
                 '--stage',str(source/arm/'joint'),'--output',str(root/arm)]
            with (root/(arm+'.log')).open('x') as log:
                p=subprocess.Popen(cmd,cwd=REPO,env=env|dict(CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[arm]=p;state['jobs'][arm]=dict(pid=p.pid,gpu=gpu,command=cmd,exit_code=None);write('driver.json',state)
        while any(p.poll() is None for p in processes.values()):
            if time.monotonic()>=cutoff:raise TimeoutError('Gradient probe budget reached.')
            if not live(pid,source) or read(source/'driver.json')['stage']!='joint_training':raise RuntimeError('Yield to primary.')
            if any(p.poll() not in (None,0) for p in processes.values()):raise RuntimeError('Gradient worker failed.')
            time.sleep(1)
        if any(p.returncode!=0 for p in processes.values()):raise RuntimeError('Gradient worker failed.')
        write('comparison.json',compare({a:read(root/a/'summary.json') for a in processes}));state['complete']=True
    except BaseException as error:state['error']=repr(error);raise
    finally:
        for p in processes.values():
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for a,p in processes.items():
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            state['jobs'][a]['exit_code']=p.returncode
        state.update(terminal=True,finished_at=datetime.now().astimezone().isoformat());write('driver.json',state)


def readline(path):
    rows=[]
    for line in path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    return rows[-1] if rows else {'step':0}

if __name__=='__main__':main()
