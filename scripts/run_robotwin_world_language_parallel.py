#!/usr/bin/env python3
"""Resume independent closed-loop arms on dedicated GPUs with frozen inference."""
from __future__ import annotations
import argparse
import fcntl
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
TASKS=['blocks_ranking_rgb','stack_blocks_two','place_a2b_left','place_a2b_right','place_burger_fries']
def read(p):return json.loads(p.read_text())
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,data):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,p)
def identity(pid):
    try:
        x=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split();return dict(state=x[0],start=x[19])
    except FileNotFoundError:return None
def require_dead(pid,start):
    x=identity(pid);require(x is None or x['start']!=start or x['state']=='Z',f'Owned process still live: {pid}')
def arms():
    return [(m,s,v,a) for m in ['released','no_eraf'] for s in [42,43,44] for v in ['source','target'] for a in ['source','target']]
def name(cell):
    m,s,v,a=cell;return f'{m}-closed-v_{v}-a_{a}-seed{s}'
def command(root,frozen,plan,manifest,cell,gpu):
    m,s,v,a=cell
    return [sys.executable,str(frozen/'scripts/eval_robotwin_world_language.py'),'--world-video-language',v,'--policy-seed',str(s),'worker','--output',str(root/'closed_loop'/name(cell)),'--manifest',manifest,'--checkpoint',str(plan['checkpoints'][m]),'--catalog-root',str(root/'probes/catalog'),'--tasks',*TASKS,'--conditions','correct' if a=='source' else 'counterfactual','--task-config','demo_randomized','--episodes','10','--gpu',str(gpu),'--eraf','off','--policy-kind','legacy' if m=='released' else 'repair','--instruction-type','canonical','--interventions',str(frozen/'configs/eval/robotwin_cis_ten_tasks.json'),'--manipulation-metrics']
def check_inputs(rows,reference,cell):
    _,seed,video,action=cell;seen=set()
    for r in rows:
        k=(r['source_task'],r['scene_seed']);require(k in reference and k not in seen,'Unknown or duplicate first-input scene');seen.add(k)
        require(r['initial_observation_sha256']==reference[k]['initial_observation_sha256'],'Initial RGB/proprio changed on resumed GPU')
        require(r['noise_seed']==seed and r['video_language']==video,'Changed intervention seed/language')
        for side,goal in [('video',video),('action',action)]:
            field='source_instruction' if goal=='source' else 'counterfactual_instruction'
            require(r[side+'_instruction']==reference[k][field],'Changed instruction text')
def journal_rows(p):
    if not p.exists():return []
    # A concurrently written final line is read only after its newline is present.
    return [json.loads(x) for x in p.read_bytes().split(b'\n')[:-1] if x]
def preflight(root,frozen,status):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=frozen,text=True).strip()
    require(head==status['code_commit'],'Frozen inference commit changed')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=frozen,text=True).strip(),'Frozen inference dirty')
    plan=read(root/'probes/plan.json');require(plan['training_performed'] is False,'Training not authorized')
    groups={
        'runtime_sources':read(root/'runtime_sources_snapshot_resume1.json')['files'],
        'actual_termination_sources':read(root/'termination_source_review.json')['source_sha256'],
        'shared_assets':read(root/'shared_inference_assets.json')['files'],
        'checkpoints':{plan['checkpoints'][m]:plan['checkpoint_sha256'][m] for m in plan['models']},
        'planned_inputs':plan['input_sha256'],
    }
    verified={}
    for label,entries in groups.items():
        for path,value in entries.items():
            expected=value['sha256'] if isinstance(value,dict) else value
            if path not in verified:verified[path]=sha(Path(path))
            require(verified[path]==expected,'Changed runtime/input file: '+path)
        print(json.dumps(dict(preflight_group=label,files=len(entries))),flush=True)
    return plan,dict(complete=True,scope='Resume input/source verification only',frozen_commit=head,plan_sha256=sha(root/'probes/plan.json'),files=verified,verified_at=time.time())
def recover(root,status):
    pause=read(root/'user_pause_4gpu.json');require(status.get('paused') is True,'Expected explicit user pause')
    for job in status['jobs'].values():require_dead(job['pid'],job['start_time'])
    original=read(root/'launch_resume1.json');require_dead(original['pid'],original['start_time'])
    stage=pause['interrupted_stage'];job=status['jobs'][stage]
    require(job['status']=='interrupted_by_user','Unexpected incomplete job')
    folder=root/'closed_loop'/stage;require(not (folder/'world_language_complete.json').exists(),'Partial group unexpectedly complete')
    recovery=root/'recovery/user-pause-three-gpu';recovery.mkdir(parents=True,exist_ok=False)
    write(recovery/'status_before.json',status)
    before={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
    count=sum(len(p.read_text().splitlines()) for p in folder.rglob('episodes.jsonl'))
    require(count==pause['completed_episode_records']-pause['terminal_audited_episodes'],'Partial record count changed after pause')
    shutil.move(str(folder),str(recovery/stage));shutil.move(str(root/(stage+'.log')),str(recovery/(stage+'.log')))
    require(all(sha(recovery/stage/p)==digest for p,digest in before.items()),'Recovery move changed file bytes')
    receipt=dict(complete=True,scope='Preserve interrupted group before a full rerun; not scientific completion',interrupted_job=job,arm=stage,partial_episodes=count,files=before,status_before_sha256=sha(recovery/'status_before.json'),created_at=time.time())
    write(recovery/'receipt.json',receipt);status['jobs'].pop(stage)
    status['recovery_receipts']=status.get('recovery_receipts',[])+[str(recovery/'receipt.json')]
    return recovery/'receipt.json'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--inference-root',type=Path,required=True);ap.add_argument('--gpus',type=int,nargs='+',required=True);args=ap.parse_args()
    root=args.output.resolve();frozen=args.inference_root.resolve()
    require(len(args.gpus)==len(set(args.gpus)) and all(x>=0 for x in args.gpus),'Duplicate/invalid GPU ids')
    lock=(root/'study.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    launch_path=root/'launch_multigpu.json';require(not launch_path.exists(),'Inspect existing multi-GPU launch before retry')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(),'Commit scheduler before launch')
    scheduler_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU compute apps present before resume')
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    inventory={int(x.split(',')[0]):[y.strip() for y in x.split(',')] for x in gpu_rows}
    require(all(x in inventory for x in args.gpus),'Requested GPU unavailable')
    status=read(root/'status.json');plan,proof=preflight(root,frozen,status);write(root/'resume_multigpu_input_audit.json',proof)
    # Reuse only terminal audited arms. The sole interrupted group is archived whole.
    subprocess.run([sys.executable,str(REPO/'scripts/audit_robotwin_world_language_closed.py'),'--root',str(root),'--allow-partial'],check=True)
    audit=read(root/'closed_data_audit_partial.json');require(audit['validated_episodes']==300,'Unexpected completed prefix; inspect before recovery')
    template=status['jobs']['released-closed-v_source-a_source-seed42']['command'];manifest=template[template.index('--manifest')+1]
    require(command(root,frozen,plan,manifest,('released',42,'source','source'),0)==template,'Scheduler changed frozen worker arguments')
    reference_path=root/'closed_loop/released-closed-v_source-a_source-seed42/world_language_inputs.jsonl'
    reference={(r['source_task'],r['scene_seed']):r for r in journal_rows(reference_path)};require(len(reference)==50,'Missing baseline input references')
    recovery=recover(root,status)
    pending=[c for c in arms() if name(c) not in status['jobs']]
    priorities=['released-closed-v_target-a_source-seed43','no_eraf-closed-v_source-a_source-seed42','released-closed-v_target-a_target-seed43']
    pending.sort(key=lambda c:priorities.index(name(c)) if name(c) in priorities else len(priorities)+arms().index(c))
    launch=dict(pid=os.getpid(),start_time=identity(os.getpid())['start'],code_commit=status['code_commit'],scheduler_commit=scheduler_commit,scheduler_path=str(Path(__file__).resolve()),scheduler_sha256=sha(Path(__file__)),gpu_inventory=inventory,gpus=args.gpus,created_at=time.time(),previous_controller=read(root/'launch_resume1.json'),input_audit_sha256=sha(root/'resume_multigpu_input_audit.json'),recovery_receipt_sha256=sha(recovery),recovery_receipt=str(recovery),queue=[name(c) for c in pending])
    write(launch_path,launch)
    status.update(controller_pid=os.getpid(),controller_launch='launch_multigpu.json',scheduler_commit=scheduler_commit,paused=False,stage='parallel_closed_loop',active_jobs=[],compute_complete=False,complete=False)
    status.pop('error',None);status['updated_at']=time.time();write(root/'status.json',status)
    running={}
    def stop_signal(signum,frame):raise InterruptedError(f'Controller signal {signum}')
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    def save():
        status['active_jobs']=[x['name'] for x in running.values()];status['updated_at']=time.time();write(root/'status.json',status)
    try:
        while pending or running:
            for gpu,item in list(running.items()):
                rows=journal_rows(root/'closed_loop'/item['name']/'world_language_inputs.jsonl');check_inputs(rows,reference,item['cell'])
                code=item['process'].poll()
                if code is None:continue
                item['log'].close();job=status['jobs'][item['name']];job.update(status='exited',exit_code=code,finished_at=time.time(),verified_first_inputs=len(rows));del running[gpu];save()
                require(code==0,'Worker failed: '+item['name'])
                require(len(rows)==50 and (root/'closed_loop'/item['name']/'world_language_complete.json').exists(),'Incomplete successful worker')
                print(json.dumps(dict(completed=item['name'],gpu=gpu,exit_code=code)),flush=True)
            for gpu in args.gpus:
                if gpu in running or not pending:continue
                require(shutil.disk_usage(root).free>=8*1024**3,'Insufficient data-disk reserve')
                cell=pending.pop(0);label=name(cell);require(not (root/'closed_loop'/label).exists(),'Refuse preexisting output: '+label)
                cmd=command(root,frozen,plan,manifest,cell,gpu)
                env=os.environ|dict(PATH='/opt/conda/bin:'+os.environ['PATH'],PYTHONPATH=str(frozen/'src')+':'+str(frozen),CUDA_VISIBLE_DEVICES=str(gpu),DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                log=(root/(label+'.log')).open('x');child=subprocess.Popen(cmd,cwd=frozen,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[gpu]=dict(name=label,cell=cell,process=child,log=log)
                ident=identity(child.pid);require(ident is not None,'Worker disappeared during launch')
                status['jobs'][label]=dict(pid=child.pid,start_time=ident['start'],command=cmd,status='running',gpu=gpu,gpu_uuid=inventory[gpu][1],scheduler_commit=scheduler_commit,launched_at=time.time())
                save();print(json.dumps(dict(started=label,gpu=gpu,pid=child.pid)),flush=True)
            if running:time.sleep(5)
        status.update(stage='awaiting_independent_audit_and_video_semantic_scoring',compute_complete=True)
    except BaseException as exc:
        status['error']=repr(exc);status['stage']='parallel_stopped_on_error'
        for item in running.values():
            if item['process'].poll() is None:os.killpg(item['process'].pid,signal.SIGTERM)
        for item in running.values():
            try:code=item['process'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(item['process'].pid,signal.SIGKILL);code=item['process'].wait(timeout=10)
            status['jobs'][item['name']].update(status='exited',exit_code=code,finished_at=time.time());item['log'].close()
        running.clear();raise
    finally:save()

if __name__=='__main__':main()
