#!/usr/bin/env python3
"""Final-only verification of owned process exits and immutable runtime inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def expected_jobs():
    jobs=['collection','prepare']
    for e in ['features_cache','video']:
        for m in ['released','no_eraf']:jobs.extend([f'{m}-{e}-smoke',f'{m}-{e}'])
    jobs.extend(f'{m}-closed-v_{v}-a_{a}-seed{s}' for m in ['released','no_eraf'] for s in [42,43,44] for v in ['source','target'] for a in ['source','target'])
    return jobs

def process_identity(pid,proc_root=Path('/proc')):
    p=proc_root/str(pid)/'stat'
    if not p.exists():return None
    fields=p.read_text().rsplit(') ',1)[1].split()
    return dict(start=fields[19],state=fields[0])

def validate_processes(status,launch,proc_root=Path('/proc')):
    require(status.get('compute_complete') is True,'Controller has not completed all compute')
    require(not status.get('error'),'Controller recorded an error')
    require(set(status['jobs'])==set(expected_jobs()),'Missing or unexpected study jobs')
    require(status['controller_pid']==launch['pid'] and status['code_commit']==launch['code_commit'],'Controller identity mismatch')
    checked=[]
    for name,job in status['jobs'].items():
        require(job.get('status')=='exited' and job.get('exit_code')==0,'Job has no successful process exit: '+name)
        actual=process_identity(job['pid'],proc_root)
        require(actual is None or actual['start']!=job['start_time'] or actual['state']=='Z','Owned job still live: '+name)
        checked.append(dict(name=name,pid=job['pid'],start_time=job['start_time'],exit_code=job['exit_code'],current_pid_identity=actual))
    actual=process_identity(launch['pid'],proc_root)
    require(actual is None or actual['start']!=launch['start_time'] or actual['state']=='Z','Owned controller still live')
    return dict(jobs=checked,controller=dict(**launch,current_pid_identity=actual))

def audit(root):
    status_path=root/'status.json';launch_path=root/'launch_resume1.json';plan_path=root/'probes/plan.json'
    status=read(status_path);launch=read(launch_path);plan=read(plan_path)
    processes=validate_processes(status,launch)
    require(plan['training_performed'] is False,'Unexpected training in inference-only study')
    frozen=Path(status['jobs']['prepare']['command'][1]).parents[1]
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=frozen,text=True).strip()
    require(head==launch['code_commit'],'Frozen inference commit changed')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=frozen,text=True).strip(),'Frozen inference checkout dirty')
    snapshot_path=root/'runtime_sources_snapshot_resume1.json';assets_path=root/'shared_inference_assets.json';review_path=root/'termination_source_review.json'
    snapshot=read(snapshot_path);assets=read(assets_path);review=read(review_path)
    require(snapshot['code_commit']==head,'Source snapshot commit mismatch')
    require(assets['complete'] and assets['plan_sha256']==sha(plan_path),'Shared inference assets bound to a different plan')
    require(review['complete'] and review['runtime_snapshot_sha256']==sha(snapshot_path),'Termination review snapshot mismatch')
    records={};groups={}
    def verify(label,entries):
        count=0;size=0
        for name,value in entries.items():
            p=Path(name);metadata=value if isinstance(value,dict) else dict(sha256=value)
            before=p.stat();expected=metadata['sha256']
            if 'size' in metadata:require(before.st_size==metadata['size'],'Changed size: '+name)
            if 'resolved_path' in metadata:require(str(p.resolve())==metadata['resolved_path'],'Changed symlink target: '+name)
            if name not in records:
                digest=sha(p);after=p.stat()
                require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'File changed while hashing: '+name)
                records[name]=dict(sha256=digest,size=after.st_size,mtime_ns=after.st_mtime_ns,resolved_path=str(p.resolve()))
            require(records[name]['sha256']==expected,'Changed bytes: '+name)
            count+=1;size+=before.st_size
        groups[label]=dict(files=count,bytes=size)
        print(json.dumps(dict(verified_group=label,**groups[label])),flush=True)
    verify('runtime_snapshot',snapshot['files'])
    # The actual versioned eval_policy path is absent from the old snapshot map.
    # Its reviewed hash is separately bound to the frozen git object.
    actual_eval=frozen/'third_party/RoboTwin/script/eval_policy.py'
    require(str(actual_eval) in review['source_sha256'],'Actual evaluation loader missing from source review')
    blob=subprocess.check_output(['git','show',head+':third_party/RoboTwin/script/eval_policy.py'],cwd=frozen)
    require(hashlib.sha256(blob).hexdigest()==review['source_sha256'][str(actual_eval)],'Actual evaluator is not the frozen versioned source')
    verify('termination_review_actual_paths',review['source_sha256'])
    verify('shared_inference_assets',assets['files'])
    verify('checkpoints',{str(plan['checkpoints'][m]):plan['checkpoint_sha256'][m] for m in plan['models']})
    verify('planned_inputs',plan['input_sha256'])
    for name,job in status['jobs'].items():
        if '-closed-' in name:
            marker=root/'closed_loop'/name/'world_language_complete.json'
            require(marker.exists() and read(marker)['complete'],'Missing closed worker marker: '+name)
    # Guard against another controller mutation during this final-only audit.
    require(read(status_path)==status,'Controller status changed during final audit')
    result=dict(format='robotwin_world_language_final_runtime_audit_v1',complete=True,verified_at=time.time(),frozen_worktree=str(frozen),frozen_commit=head,processes=processes,verified_groups=groups,files=records,evidence_sha256={str(p):sha(p) for p in [status_path,launch_path,plan_path,snapshot_path,assets_path,review_path]},scope='Final owned process exits and immutable source/checkpoint/input/assets only. Requires separate completed data, semantic review and scientific-report audits; this is not whole-study completion.')
    path=root/'final_runtime_audit.json';path.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    r=audit(args.root);print(json.dumps(dict(complete=r['complete'],groups=r['verified_groups'])))
