#!/usr/bin/env python3
"""One-GPU serial execution with durable process records and verified resumes."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from scripts.run_robotwin_world_language_collection import write,TASKS
from scripts.probe_robotwin_world_language import read,sha,MANIFEST,CHECKPOINTS


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'study.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ|dict(PATH='/opt/conda/bin:'+os.environ['PATH'],PYTHONPATH=str(REPO/'src')+':'+str(REPO),
        CUDA_VISIBLE_DEVICES='0',DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Commit all study code before launching')
    status=read(root/'status.json') if (root/'status.json').exists() else dict(jobs={},complete=False,code_commit=commit)
    assert status['code_commit']==commit,'Do not mutate the running study code'
    status['controller_pid']=os.getpid()

    def run(name,script,arguments):
        old=status['jobs'].get(name)
        if old:
            if old.get('exit_code')==0:
                print('[already exited zero] '+name,flush=True);return
            raise ValueError('Inspect previous incomplete job before resuming '+name)
        if shutil.disk_usage(root).free < 8*1024**3:raise ValueError('Less than 8 GiB data-disk reserve')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise ValueError('Another GPU process is present before '+name)
        command=[sys.executable,str(REPO/'scripts'/script),*map(str,arguments)]
        with (root/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            start=Path(f'/proc/{child.pid}/stat').read_text().rsplit(') ',1)[1].split()[19]
            status['jobs'][name]=dict(pid=child.pid,start_time=start,command=command,status='running')
            status['stage']=name;write(root/'status.json',status)
            code=child.wait()
        status['jobs'][name].update(exit_code=code,status='exited');write(root/'status.json',status)
        if code:raise RuntimeError(f'{name} exited {code}; no subsequent job was started')

    try:
        run('collection','run_robotwin_world_language_collection.py',['--output',root/'collection'])
        assert read(root/'collection/status.json')['complete']
        run('prepare','probe_robotwin_world_language.py',['prepare','--output',root/'probes','--collection',root/'collection'])
        plan=read(root/'probes/plan.json')
        assert len(plan['scenes'])==50
        for experiment in ['features_cache','video']:
            for model in ['released','no_eraf']:
                common=['worker','--output',root/'probes','--model',model,'--experiment',experiment]
                run(f'{model}-{experiment}-smoke','probe_robotwin_world_language.py',[*common,'--limit','1'])
                run(f'{model}-{experiment}','probe_robotwin_world_language.py',common)
                proofs=[]
                for row in plan['states']:
                    folder=root/'probes'/model/experiment/row['id'];proof=read(folder/'complete.json')
                    assert proof['complete'] and proof['plan_sha256']==sha(root/'probes/plan.json')
                    assert proof['output_sha256']
                    assert all(sha(folder/p)==digest for p,digest in proof['output_sha256'].items())
                    proofs.append(proof)
                assert len(proofs)==len(plan['states'])
        for model in ['released','no_eraf']:
            for seed in [42,43,44]:
                for video in ['source','target']:
                    for action in ['source','target']:
                        name=f'{model}-closed-v_{video}-a_{action}-seed{seed}'
                        run(name,'eval_robotwin_world_language.py',[
                            '--world-video-language',video,'--policy-seed',seed,'worker',
                            '--output',root/'closed_loop'/name,'--manifest',MANIFEST,
                            '--checkpoint',CHECKPOINTS[model],'--catalog-root',root/'probes/catalog',
                            '--tasks',*TASKS,'--conditions','correct' if action=='source' else 'counterfactual',
                            '--task-config','demo_randomized','--episodes','10','--gpu','0','--eraf','off',
                            '--policy-kind','legacy' if model=='released' else 'repair',
                            '--instruction-type','canonical','--interventions',REPO/'configs/eval/robotwin_cis_ten_tasks.json',
                            '--manipulation-metrics'])
        status['stage']='awaiting_independent_audit_and_video_semantic_scoring'
        status['compute_complete']=True
        # Scientific completion also requires semantic video inspection and report.
        status['complete']=False
    except BaseException as exc:
        status['error']=repr(exc)
        raise
    finally:
        status['updated_at']=time.time();write(root/'status.json',status)


if __name__=='__main__':main()
