#!/usr/bin/env python3
"""Bound a six-GPU cup baseline run, preserving partial results on failure."""
from pathlib import Path
import argparse
import datetime
import json
import os
import signal
import subprocess
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--models', type=Path, required=True)
    ap.add_argument('--deadline', required=True, help='ISO datetime including UTC offset')
    ap.add_argument('--robotwin-root', type=Path, required=True)
    args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1]
    root=args.output.resolve(); root.mkdir(exist_ok=False, parents=True)
    models=json.loads(args.models.read_text())
    if len(models)!=3: raise ValueError('Expected exactly three baseline models')
    deadline=datetime.datetime.fromisoformat(args.deadline).timestamp()
    processes={}; logs={}
    state={'format':'robotwin_cup_baseline_campaign_v1','complete':False,'status':'running',
           'started_at':datetime.datetime.now().isoformat(),'deadline':args.deadline,
           'optimizer_updates':0,'hash_scans':False,'jobs':{},
           'code':subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=repo,text=True).strip()}
    def save():
        for name,p in processes.items():state['jobs'][name]['exit_code']=p.poll()
        (root/'driver.json').write_text(json.dumps(state,indent=2)+'\n')
    def launch(name,gpu,command):
        seconds=int(deadline-time.time())
        if seconds<60:raise TimeoutError('Work deadline reached')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',
                 PYTHONPATH=str(repo/'src')+':'+str(repo),
                 VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
                 DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
        logs[name]=(root/(name+'.log')).open('w')
        cmd=['timeout','--signal=TERM','--kill-after=15',str(seconds),os.sys.executable]+command
        p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=logs[name],stderr=subprocess.STDOUT,start_new_session=True)
        processes[name]=p;state['jobs'][name]={'pid':p.pid,'gpu':gpu,'command':cmd}
        save();print('[start]',name,p.pid,flush=True)
    def wait(names):
        while any(processes[n].poll() is None for n in names):
            save()
            if time.time()>=deadline:raise TimeoutError('Work deadline reached')
            if any(processes[n].poll() not in (None,0) for n in names):
                raise RuntimeError('A job failed; preserving all partial results')
            time.sleep(3)
        if any(processes[n].returncode!=0 for n in names):raise RuntimeError('A job failed')
    try:
        catalog=root/'catalog_dev/catalog.json'
        base=['scripts/eval_robotwin_cup_cf.py']
        launch('catalog',0,base+['catalog','--output',str(catalog.parent),
            '--robotwin-root',str(args.robotwin_root.resolve()),'--episodes','6',
            '--max-attempts','24','--start-seed','93000000','--split','dev'])
        wait(['catalog'])
        c=json.loads(catalog.read_text())
        if not c['complete']:raise ValueError('Catalog incomplete')
        print('[catalog-complete]',len(c['episodes']),'cf_expert',
              sum(r['cf_expert_success'] for r in c['episodes']),flush=True)
        names=[]
        for index,(name,model) in enumerate(models.items()):
            for shard in range(2):
                job=f'{name}_{shard}';names.append(job)
                launch(job,index*2+shard,base+['worker','--output',str(root/job),
                    '--robotwin-root',str(args.robotwin_root.resolve()),'--catalog',str(catalog),
                    '--manifest',str(args.manifest.resolve()),'--checkpoint',model['checkpoint'],
                    '--policy-kind',model['policy_kind'],'--episode-start',str(3*shard),
                    '--episode-count','3'])
        wait(names)
        summaries={}
        for name in models:
            cmd=[os.sys.executable]+base+['summarize','--output',str(root/(name+'_summary.json')),
                '--catalog',str(catalog),'--workers',str(root/f'{name}_0/worker.json'),
                str(root/f'{name}_1/worker.json')]
            subprocess.run(cmd,cwd=repo,check=True,stdout=subprocess.DEVNULL,timeout=30)
            r=json.loads((root/(name+'_summary.json')).read_text())
            summaries[name]={k:r[k] for k in ('complete','scenes','source_successes','cf_successes','cf_source_goal_only')}
        state.update(complete=True,status='complete',summaries=summaries)
    except Exception as error:
        state.update(status='failed',error=repr(error))
        for p in processes.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes.values():
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise
    finally:
        state['finished_at']=datetime.datetime.now().isoformat();save()
        for log in logs.values():log.close()
        print(json.dumps(state),flush=True)


if __name__=='__main__':main()
