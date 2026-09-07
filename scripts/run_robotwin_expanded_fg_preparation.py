#!/usr/bin/env python3
"""After formal FG collection, cache complete tails and replay every FG mask."""
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

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]


def source_process(pid, root, *, proc_root=Path('/proc')):
    path=proc_root/str(pid)
    try:
        fields=(path/'stat').read_text().rsplit(') ',1)[1].split()
        argv=(path/'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:return None
    if fields[0]=='Z':return None
    if (str(root).encode() not in argv
            or not any(a.endswith(b'/run_robotwin_expanded_fg_pilots.py') for a in argv)):
        raise ValueError('Source PID is owned by a different command.')
    return fields[19]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--source-pid',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--deadline',required=True)
    args=ap.parse_args()
    cutoff=datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or cutoff.timestamp()<=time.time():ap.error('Require a future absolute cutoff.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Commit preparation code before execution.')
    source=args.source_root.resolve();root=args.output.resolve()
    launch=json.loads(source.with_name(source.name+'-launch.json').read_text())
    if launch['pid']!=args.source_pid:raise ValueError('Source launch receipt and PID differ.')
    start=source_process(args.source_pid,source)
    old=json.loads((source/'protocol.json').read_text())
    if old.get('mode')!='cup_pill_expansion' or old['requested_scenes']!=60:
        raise ValueError('Require the declared sixty-scene cup/pill collection.')
    root.mkdir(parents=True,exist_ok=False)
    state=dict(complete=False,terminal=False,stage='waiting_for_collection',jobs={})
    processes={}
    def write(name,data):
        p=root/(name+'.tmp');p.write_text(json.dumps(data,indent=2)+'\n');p.replace(root/name)
    protocol=dict(format='robotwin_expanded_fg_preparation_v1',source_root=str(source),source_pid=args.source_pid,
        source_process_start_time=start,source_launch=launch,manifest=old['manifest'],checkpoint=old['checkpoint'],
        manifest_sha256=old['manifest_sha256'],checkpoint_sha256=old['checkpoint_sha256'],deadline=args.deadline,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        disk_reserve_GiB=3,platform_shutdown=None,optimizer_updates=0,
        scope='Preserve all parent rows; append60verified FG corrections with all real goal-tail windows. Verify normalized targets. Reproduce RGB/physical states and masks for all old and new FG scenes, not an unsupported old-index extension.',
        artifact_storage='server_only; Mac metadata only')
    write('protocol.json',protocol);write('driver.json',state)
    env=os.environ|dict(PATH='/opt/conda/bin:'+os.environ['PATH'],PYTHONPATH=str(REPO/'src')+':'+str(REPO),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
    def budget():
        if time.time()>=cutoff.timestamp():raise TimeoutError('Preparation cutoff; server remains on.')
        if shutil.disk_usage(root).free<3*1024**3:raise RuntimeError('Preparation disk reserve reached.')
    def stop(signum,frame):raise InterruptedError('Signal'+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run_group(stage, commands):
        state['stage']=stage
        active={}
        for name,cmd in commands.items():
            budget()
            with (root/(name+'.log')).open('x') as log:
                p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=p;processes[name]=p
            state['jobs'][name]=dict(pid=p.pid,command=cmd,exit_code=None)
            write('driver.json',state)
        while active:
            budget()
            for name,p in list(active.items()):
                rc=p.poll()
                if rc is not None:
                    state['jobs'][name]['exit_code']=rc;del active[name]
                    write('driver.json',state)
                    if rc:raise RuntimeError(name+' failed with exit'+str(rc))
            if active:time.sleep(5)
    try:
        while True:
            budget();current=source_process(args.source_pid,source)
            if current is None:break
            if start is None or start!=current:raise ValueError('Source PID start time changed.')
            time.sleep(10)
        completed=json.loads((source/'driver.json').read_text())
        if (not completed.get('complete') or not completed.get('terminal')
                or any(j.get('exit_code')!=0 for j in completed['jobs'].values())):
            raise ValueError('Collection ended without the complete declared dataset.')
        from experiments.robotwin.eraf_fg_bridge import file_sha256
        if file_sha256(old['manifest'])!=old['manifest_sha256'] or file_sha256(old['checkpoint'])!=old['checkpoint_sha256']:
            raise ValueError('Collection inputs changed before preparation.')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise ValueError('Another job owns the GPUs after collection.')
        collection=source/'collections.json'
        records=json.loads(collection.read_text())
        if not records.get('complete') or len(records['records'])!=60:raise ValueError('Incomplete formal correction collection.')
        protocol['collection_sha256']=file_sha256(collection);write('protocol.json',protocol)
        cache=root/'cache'
        common=['--manifest',old['manifest'],'--checkpoint',old['checkpoint'],'--output',str(cache),
                '--collections',str(collection),'--shards','6']
        run_group('caching',{f'cache{i}':[sys.executable,str(REPO/'scripts/prepare_robotwin_eraf_fg_replay.py'),
            'worker',*common,'--shard',str(i),'--gpu',str(i)] for i in range(6)})
        run_group('merging',{'merge':[sys.executable,str(REPO/'scripts/prepare_robotwin_eraf_fg_replay.py'),'merge',*common]})
        manifest=cache/'manifest.json'
        run_group('auditing_cache',{'audit_cache':[sys.executable,str(REPO/'scripts/audit_robotwin_expanded_fg_cache.py'),
            '--parent',old['manifest'],'--manifest',str(manifest),'--collection',str(collection),
            '--output',str(root/'cache_audit.json')]})
        masks=root/'masks';masks.mkdir()
        run_group('replaying_all_masks',{f'masks{i}':[sys.executable,str(REPO/'scripts/recapture_robotwin_fg_masks.py'),
            '--manifest',str(manifest),'--output',str(masks/f'shard{i}'),
            '--robotwin-root','/root/gpufree-data/LF-FastWAM/third_party/RoboTwin','--gpu',str(i),
            '--num-shards','6','--shard-index',str(i),'--limit-per-cell','9999'] for i in range(6)})
        from experiments.robotwin.expanded_fg_preparation import merge_mask_shards
        mask_audit=merge_mask_shards(masks,manifest)
        if file_sha256(collection)!=protocol['collection_sha256'] or file_sha256(old['checkpoint'])!=old['checkpoint_sha256']:
            raise ValueError('Preparation changed source collection or checkpoint.')
        state.update(complete=True,terminal=True,stage='complete',manifest=str(manifest),manifest_sha256=file_sha256(manifest),
                     mask_audit=mask_audit,cache_audit=json.loads((root/'cache_audit.json').read_text()))
        write('summary.json',state)
    except BaseException as error:
        state.update(stage='stopped',error=repr(error));raise
    finally:
        for p in processes.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for name,p in processes.items():
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            state['jobs'][name]['exit_code']=p.returncode
        state.update(terminal=True,finished_at=datetime.now().astimezone().isoformat());write('driver.json',state)


if __name__=='__main__':main()
