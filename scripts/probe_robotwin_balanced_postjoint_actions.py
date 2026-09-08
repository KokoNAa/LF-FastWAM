#!/usr/bin/env python3
"""Read-only balanced-target action fit after verified completed CF evaluation."""
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


def validate_completed(driver,receipt):
    if not (driver.get('complete') and driver.get('terminal') and driver.get('stage')=='complete'):
        raise ValueError('Source trial must have completed.')
    if not driver.get('jobs') or any(j.get('exit_code')!=0 for j in driver['jobs'].values()):
        raise ValueError('Source trial jobs must all exit zero.')
    if not all(receipt.get(k) is True for k in ('complete','controller_gone','all_jobs_exit_zero_and_gone')):
        raise ValueError('Missing verified terminal receipt.')
    if receipt.get('episodes')!=180 or receipt.get('comparison_methods')!=29:
        raise ValueError('Incomplete source CF evaluation.')


def compare_actions(previous,current):
    if not previous['complete'] or not current['complete']:
        raise ValueError('Incomplete action diagnostic.')
    for report in (previous,current):
        protocol=report['protocol']
        if (protocol.get('inference_steps')!=10 or protocol.get('noise_seed')!=42
                or protocol.get('optimizer_updates')!=0 or not report.get('input_hashes_stable')
                or not all(r.get('repeat_identical') for r in report['records'])):
            raise ValueError('Unverified inference protocol or changed diagnostic inputs.')
    if previous['protocol']['manifest_sha256']!=current['protocol']['manifest_sha256']:
        raise ValueError('Diagnostic manifests differ.')
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
    ap.add_argument('--prior-diagnostic',type=Path,required=True)
    ap.add_argument('--strongest-summary',type=Path,required=True)
    ap.add_argument('--seconds',type=int,default=240)
    args=ap.parse_args()
    if not 1<=args.seconds<=240:ap.error('At most240seconds of diagnostic work.')
    source=args.source_root.resolve();root=args.output.resolve();read=lambda p:json.loads(Path(p).read_text())
    plan=read(source/'protocol.json');driver=read(source/'driver.json')
    receipt=read(source/'terminal_verification.json')
    pid=read(source.with_name(source.name+'-launch.json'))['pid']
    validate_completed(driver,receipt)
    from scripts.run_robotwin_deployed_action_probe import live as process_live
    if process_live(pid,source) or any(process_live(j['pid'],source) for j in driver['jobs'].values()):
        raise ValueError('Completed trial processes are still live.')
    if root.exists():raise ValueError('Output must be new.')
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    uuids={u.strip() for line in gpu_rows.splitlines() for i,u in [line.split(',')] if int(i.strip()) in (0,2)}
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if len(uuids)!=2 or any(line.split(',')[0].strip() in uuids for line in active.splitlines()):
        raise ValueError('Diagnostic GPUs0and2 are not free.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Commit first.')
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    models={m:source/a/'joint/step_000200.pt' for m,a in [('ordinary','eraf_only'),('fg','eraf_fg')]}
    prior=args.prior_diagnostic.resolve();strongest=args.strongest_summary.resolve()
    references=[prior/m/'summary.json' for m in ('ordinary','fg')]+[strongest]
    for path in references:
        if not read(path).get('complete'):raise ValueError('Incomplete reference action diagnostic.')
    for model in models.values():
        if read(model.parent/'plan.json').get('action_objective')!='balanced_target_rollout_v1':
            raise ValueError('Expected a completed balanced-target model.')
    bindings={str(p):file_sha256(p) for p in [source/'protocol.json',source/'terminal_verification.json',Path(plan['manifest']),*models.values(),*references]}
    for mode,arm in [('ordinary','eraf_only'),('fg','eraf_fg')]:
        if bindings[str(models[mode])]!=receipt['models'][arm]['sha256']:
            raise ValueError('Actual model changed since CF evaluation.')
        if not read(models[mode].parent/'freeze_audit.json')['complete']:
            raise ValueError('Missing completed training audit.')
    root.mkdir(parents=True,exist_ok=False);processes={};cutoff=time.monotonic()+args.seconds
    state=dict(complete=False,terminal=False,jobs={})
    def write(name,value):
        p=root/(name+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(root/name)
    write('protocol.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_root=str(source),source_pid=pid,source_terminal_verified=True,seconds=args.seconds,gpus=[0,2],
        input_sha256=bindings,prior_diagnostic=str(prior),strongest_summary=str(strongest),started_at=datetime.now().astimezone().isoformat(),
        scope='Fixed18train/holdout action queries per finished action model; no model updates, CF scoring or checkpoint selection. Run only after the source trial is terminal and all its processes have exited.'))
    write('driver.json',state)
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
            for mode,p in processes.items():
                rc=p.poll()
                if rc is not None and rc!=0:raise RuntimeError(mode+' probe failed:'+str(rc))
            time.sleep(1)
        if any(p.returncode!=0 for p in processes.values()):raise RuntimeError('An action probe failed.')
        if any(file_sha256(p)!=digest for p,digest in bindings.items()):raise ValueError('An input changed.')
        compared={}
        for mode,arm in [('ordinary','eraf_only'),('fg','eraf_fg')]:
            previous=read(prior/mode/'summary.json');current=read(root/mode/'summary.json')
            compared[arm]=dict(previous_objective=compare_actions(previous,current),
                strongest_control=compare_actions(read(strongest),current))
        write('comparison.json',dict(complete=True,comparisons=compared,
            prior_diagnostic=str(prior),strongest_summary=str(strongest),input_sha256=bindings,
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
