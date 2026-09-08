#!/usr/bin/env python3
"""Run six bounded train-only pilots after the complete four-arm CF experiment."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time
from datetime import datetime
from pathlib import Path
from statistics import mean
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from scripts.probe_robotwin_action_learnability import TASKS,REGIMES,STEPS,EVAL_SEEDS

def live(pid,root):
    p=Path('/proc')/str(pid)
    try:return (p/'stat').read_text().split(') ')[1][0]!='Z' and str(root).encode() in (p/'cmdline').read_bytes()
    except FileNotFoundError:return False

def compare(reports):
    expected={t+'__'+r for t in TASKS for r in REGIMES}
    if set(reports)!=expected:raise ValueError('All six pilots required.')
    result={}
    for task in TASKS:
        baseline=None;payloads=None;bindings=None
        for regime in REGIMES:
            key=task+'__'+regime;r=reports[key];p=r['protocol']
            if (not r.get('complete') or r['optimizer_steps']!=STEPS or not r['input_hashes_stable']
                    or not r['optimizer_master_binding'] or r['model_files_written'] or r['unexpected_changes']
                    or not r['changed_parameters'] or p['task']!=task or p['regime']!=regime):raise ValueError('Incomplete or invalid pilot.')
            records=r['records'];ids={x['id'] for x in p['selected_rows']}
            triples={(x['step'],x['id'],x['noise_seed']) for x in records}
            required={(s,i,n) for s in (0,16,STEPS) for i in ids for n in EVAL_SEEDS}
            if len(ids)!=9 or len(records)!=54 or triples!=required:raise ValueError('Incomplete evaluation grid.')
            current=sorted((x for x in records if x['step']==0),key=lambda x:(x['id'],x['noise_seed']))
            if baseline is None:baseline=current;payloads=p['actual_payload_hashes'];bindings=p['input_sha256']
            elif current!=baseline or p['actual_payload_hashes']!=payloads or p['input_sha256']!=bindings:raise ValueError('Initial predictions or actual inputs differ across regimes.')
            metrics={}
            for split in ('train','replay_holdout'):
                for kind in ('ordinary','fg'):
                    values={s:mean(x['mse_first24'] for x in records if x['step']==s and x['split']==split and x['kind']==kind) for s in (0,16,STEPS)}
                    metrics[split+'__'+kind]={'mse':values,'final_relative_change':values[STEPS]/values[0]-1 if values[0] else None}
            result[key]=metrics
    return dict(complete=True,pilots=result,steps_per_pilot=STEPS,goal_achievement_claim=False,checkpoint_selection=False,
        scope='Same task inputs and parent baseline across regimes. Mean over two fixed evaluation noises; pilot groups have different observation counts and no retention. Lower expert action MSE is not CF success or independent generalization.')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('source-root','output'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--seconds',type=int,default=1200);args=ap.parse_args()
    if not 1<=args.seconds<=1200:ap.error('At most1200 seconds; stops own processes only.')
    source=args.source_root.resolve();root=args.output.resolve();read=lambda p:json.loads(Path(p).read_text())
    if root.exists():raise ValueError('Output already exists.')
    d=read(source/'driver.json');v=read(source/'terminal_verification.json')
    if not (d['complete'] and d['terminal'] and v['complete'] and v['episodes']==180):raise ValueError('Source CF must be complete and verified.')
    if live(v['controller_pid'],source) or any(j['exit_code']!=0 or live(j['pid'],source) for j in d['jobs'].values()):raise ValueError('Source workers still live or failed.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Commit code first.')
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    if file_sha256(source/'eraf_fg/joint/step_000200.pt')!=v['models']['eraf_fg']['sha256']:raise ValueError('Actual source weights changed.')
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if {int(x.split(',')[0]) for x in inventory}!=set(range(6)):raise ValueError('Need six GPUs.')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if apps.strip():raise ValueError('GPUs already occupied.')
    import shutil
    if shutil.disk_usage(source).free<1024**3:raise ValueError('Need1GiB spare; no model files written by pilot.')
    root.mkdir(parents=True);processes={};cutoff=time.monotonic()+args.seconds
    state=dict(complete=False,terminal=False,stage='optimizing',jobs={})
    def write(n,v):
        p=root/(n+'.tmp');p.write_text(json.dumps(v,indent=2)+'\n');p.replace(root/n)
    write('protocol.json',dict(source_root=str(source),source_verification_sha256=file_sha256(source/'terminal_verification.json'),
        source_checkpoint_sha256=v['models']['eraf_fg']['sha256'],seconds=args.seconds,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        started_at=datetime.now().astimezone().isoformat(),steps_per_pilot=STEPS,model_files_written=False,platform_shutdown=None))
    write('driver.json',state)
    env=os.environ|dict(PYTHONPATH=str(REPO/'src')+':'+str(REPO),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
    def stop(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for gpu,(task,regime) in enumerate((t,r) for t in TASKS for r in REGIMES):
            key=task+'__'+regime
            cmd=[sys.executable,'-u',str(REPO/'scripts/probe_robotwin_action_learnability.py'),'--stage',str(source/'eraf_fg/joint'),
                '--output',str(root/key),'--task',task,'--regime',regime]
            with (root/(key+'.log')).open('x') as log:
                p=subprocess.Popen(cmd,cwd=REPO,env=env|dict(CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[key]=p;state['jobs'][key]=dict(pid=p.pid,gpu=gpu,command=cmd,exit_code=None);write('driver.json',state)
        while any(p.poll() is None for p in processes.values()):
            for key,p in processes.items():state['jobs'][key]['exit_code']=p.poll()
            write('driver.json',state)
            if time.monotonic()>=cutoff:raise TimeoutError('Pilot budget reached.')
            if any(p.poll() not in (None,0) for p in processes.values()):raise RuntimeError('Pilot worker failed; preserve partial logs.')
            time.sleep(3)
        if any(p.returncode!=0 for p in processes.values()):raise RuntimeError('Pilot failed.')
        write('comparison.json',compare({a:read(root/a/'summary.json') for a in processes}));state.update(complete=True,stage='complete')
    except BaseException as error:state.update(error=repr(error),stage='failed');raise
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

if __name__=='__main__':main()
