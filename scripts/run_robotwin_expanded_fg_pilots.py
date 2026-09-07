#!/usr/bin/env python3
"""Run six bounded, server-only FG collection pilots with explicit ownership."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]


def pilot_jobs():
    from experiments.robotwin.expanded_fg import TASKS
    return [dict(name=task, task=task, split='train', start_seed=86000000+1000*i, gpu=i)
            for i, task in enumerate(TASKS)] + [dict(name='cup_holdout', task='place_empty_cup',
                split='replay_holdout', start_seed=87000000, gpu=5)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'checkpoint', 'output', 'robotwin-root'):
        ap.add_argument('--'+key, required=True, type=Path)
    for key in ('checkpoint-sha256', 'manifest-sha256', 'deadline'):
        ap.add_argument('--'+key, required=True)
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or not 0 < cutoff.timestamp()-time.time() <= 3600:
        ap.error('Require a future absolute pilot cutoff within one hour.')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Commit pilot code before starting.')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('GPUs are owned by another job.')
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    from experiments.robotwin.expanded_fg import validate_scope
    from scripts.collect_robotwin_eraf_fg import historical_scene_keys
    if file_sha256(args.checkpoint) != args.checkpoint_sha256 or file_sha256(args.manifest) != args.manifest_sha256:
        raise ValueError('Pilot checkpoint or input manifest identity changed.')
    manifest = json.loads(args.manifest.read_text())
    excluded = historical_scene_keys(manifest['states'])
    jobs = pilot_jobs()
    for job in jobs:
        validate_scope(job['task'], job['split'], job['start_seed'], 3)
        if any((job['task'], 'demo_clean', seed) in excluded for seed in range(job['start_seed'], job['start_seed']+3)):
            raise ValueError('Pilot scene was already used in the input manifest.')
    root = args.output.resolve()
    if shutil.disk_usage(root.parent).free < 4.5*1024**3:
        raise RuntimeError('Require 3GiB reserve plus1.5GiB pilot allocation.')
    root.mkdir(exist_ok=False)
    state = dict(complete=False, terminal=False, stage='starting', jobs={})
    processes = {}
    def write(name, data):
        temp=root/(name+'.tmp');temp.write_text(json.dumps(data, indent=2)+'\n');temp.replace(root/name)
    protocol = dict(format='robotwin_expanded_fg_pilots_v1',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        manifest=str(args.manifest.resolve()), manifest_sha256=args.manifest_sha256,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=args.checkpoint_sha256,
        jobs=jobs, scenes_per_job=1, attempts_per_job=3, candidates_per_scene=6,
        candidate_order='late_first', policy_kind='repair', eraf_mode='on', memory_mode='carry',
        deadline=args.deadline, disk_reserve_GiB=3, platform_shutdown=None,
        scope='Collection feasibility only; no model training, no policy evaluation score, no goal achievement.',
        model_storage='server_only')
    write('protocol.json', protocol);write('driver.json', state)
    def stop(signum, frame):raise InterruptedError('Pilot interrupted: '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env=os.environ | dict(PATH='/opt/conda/bin:'+os.environ['PATH'], PYTHONPATH=str(REPO/'src')+':'+str(REPO),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
    try:
        for job in jobs:
            name=job['name']
            cmd=[sys.executable, str(REPO/'scripts/collect_robotwin_eraf_fg.py'),
                '--manifest',str(args.manifest),'--checkpoint',str(args.checkpoint),
                '--robotwin-root',str(args.robotwin_root),'--output',str(root/name),
                '--task',job['task'],'--gpu',str(job['gpu']),'--protocol','expanded_v2',
                '--split',job['split'],'--start-seed',str(job['start_seed']),
                '--scenes','1','--holdout-scenes','0','--max-attempts','3','--candidates','6',
                '--policy-kind','repair','--eraf','on','--deadline',args.deadline,'--reserve-gib','3']
            with (root/(name+'.log')).open('x') as log:
                process=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[name]=process
            state['jobs'][name]=dict(pid=process.pid, command=cmd, exit_code=None)
            write('driver.json',state)
        state['stage']='collecting'
        while True:
            active=[]
            for name, process in processes.items():
                state['jobs'][name]['exit_code']=process.poll()
                if process.returncode is None:active.append(name)
            write('driver.json',state)
            if not active:break
            if time.time() >= cutoff.timestamp():raise TimeoutError('Pilot cutoff; stop only owned process groups.')
            if shutil.disk_usage(root).free < 3*1024**3:raise RuntimeError('Pilot disk reserve reached.')
            time.sleep(5)
        reports={}
        for name in processes:
            p=root/name/'manifest.json'
            reports[name]=json.loads(p.read_text()) if p.exists() else {'complete':False,'records':[]}
        from experiments.robotwin.eraf_fg_contract import validate_correction
        for report in reports.values():
            for row in report['records']:
                validate_correction(row)
                for path, sha in [(row['frame_path'],row['frame_sha256']),
                                  (str(Path(row['frame_path']).with_name('controls.pkl')),row['controls_sha256'])]:
                    if file_sha256(path)!=sha:raise ValueError('Pilot recorded file identity changed.')
        if file_sha256(args.checkpoint)!=args.checkpoint_sha256:raise ValueError('Pilot modified its source checkpoint.')
        state.update(terminal=True, complete=all(r.get('complete') for r in reports.values())
                     and all(p.returncode == 0 for p in processes.values()),stage='finished',
                     accepted={name:len(r['records']) for name,r in reports.items()})
        write('summary.json',dict(state,checkpoint_unchanged=True,performance_gain_claim=False))
    except BaseException as error:
        state.update(stage='stopped',error=repr(error))
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
        for name, process in processes.items():
            try:process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            state['jobs'][name]['exit_code']=process.returncode
        state.update(terminal=True,finished_at=datetime.now().astimezone().isoformat())
        write('driver.json',state)


if __name__=='__main__':main()
