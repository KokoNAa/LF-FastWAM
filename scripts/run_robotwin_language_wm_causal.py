#!/usr/bin/env python3
"""Schedule independent model/seed shards after verified GPU smoke controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from scripts.probe_robotwin_language_wm_causal import read,sha,write


def verify_state(folder,plan_hash,generated=False):
    proof=read(folder/'complete.json')
    if not proof.get('complete') or proof['plan_sha256']!=plan_hash:
        raise ValueError('State not verified: '+str(folder))
    if any(sha(folder/p)!=h for p,h in proof['output_sha256'].items()):
        raise ValueError('State artifact changed')
    if generated and read(folder/'generated/generations.json')['clips']!=27:
        raise ValueError('Generated conditions incomplete')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpus',type=int,nargs='+',default=[0,1,2])
    args=ap.parse_args();r=args.output
    lock=(r/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if len(set(args.gpus))!=len(args.gpus):raise ValueError('Duplicate GPU')
    plan=read(r/'plan.json');ph=sha(r/'plan.json')
    for model in plan['models']:
        verify_state(r/'smoke'/model/'seed42'/plan['states'][0]['id'],ph)
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used','--format=csv,noheader'],text=True)
    for line in inventory.strip().splitlines():
        parts=[s.strip() for s in line.split(',')]
        if int(parts[0]) in args.gpus and int(parts[3].split()[0])>1024:
            raise ValueError('Selected GPU is occupied: '+line)
    old=read(r/'status.json') if (r/'status.json').exists() else None
    status=dict(format='robotwin_language_wm_causal_controller_v1',complete=False,compute_complete=False,
                stage='running',plan_sha256=ph,controller_pid=os.getpid(),
                controller_start=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ',1)[1].split()[19],
                controller_sha256=sha(Path(__file__)),gpu_inventory=inventory,jobs={},started_at=time.time())
    if old:write(r/f'controller_history_{time.time_ns()}.json',old)
    queue=[(m,s) for s in plan['noise_seeds'] for m in ['released','no_eraf']]
    running={}
    def save():
        status['updated_at']=time.time();status['active_jobs']=[j['label'] for j in running.values()]
        write(r/'status.json',status)
    def stop(signum,frame):raise InterruptedError(f'Controller received signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save()
    try:
        while queue or running:
            for gpu,j in list(running.items()):
                code=j['process'].poll()
                if code is None:continue
                j['log'].close();status['jobs'][j['label']].update(exit_code=code,finished_at=time.time())
                del running[gpu]
                if code!=0:raise RuntimeError('Shard failed: '+j['label'])
                for row in plan['states']:
                    verify_state(r/'results'/j['model']/f"seed{j['seed']}"/row['id'],ph,
                                 generated=j['seed']==42 and row['id'] in plan['generation_states'])
                status['jobs'][j['label']]['verified_states']=len(plan['states']);save()
            for gpu in args.gpus:
                if gpu in running or not queue:continue
                model,seed=queue.pop(0);label=f'{model}-seed{seed}'
                cmd=['/opt/conda/bin/python','scripts/probe_robotwin_language_wm_causal.py','worker',
                     '--output',str(r),'--model',model,'--seed',str(seed)]
                env=os.environ|dict(PATH='/opt/conda/bin:'+os.environ['PATH'],PYTHONPATH=str(REPO/'src')+':'+str(REPO),
                    CUDA_VISIBLE_DEVICES=str(gpu),DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                log=(r/f'{label}-{time.time_ns()}.log').open('x')
                p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[gpu]=dict(label=label,model=model,seed=seed,process=p,log=log)
                status['jobs'][label]=dict(model=model,seed=seed,gpu=gpu,pid=p.pid,command=cmd,log=str(log.name),
                    start_time=Path(f'/proc/{p.pid}/stat').read_text().rsplit(') ',1)[1].split()[19],launched_at=time.time())
                save();print(json.dumps(dict(started=label,gpu=gpu,pid=p.pid)),flush=True)
            time.sleep(5)
        status.update(compute_complete=True,stage='awaiting_report_and_generated_video_review')
    except BaseException as exc:
        status['error']=repr(exc);status['stage']='stopped_on_error'
        for j in running.values():
            if j['process'].poll() is None:os.killpg(j['process'].pid,signal.SIGTERM)
        for j in running.values():
            try:code=j['process'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(j['process'].pid,signal.SIGKILL);code=j['process'].wait(timeout=10)
            status['jobs'][j['label']].update(exit_code=code,finished_at=time.time());j['log'].close()
        running.clear();raise
    finally:save()


if __name__=='__main__':main()
