#!/usr/bin/env python3
"""Continue the prespecified action trial after a training-manifest diagnostic mismatch."""
from __future__ import annotations
import argparse
from copy import deepcopy
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
from scripts.run_robotwin_expanded_fg_trial import ARMS,RUNS,run_action_stages


def validate_source(root,*,proc_root=Path('/proc')):
    root=Path(root);read=lambda p:json.loads(p.read_text())
    driver=read(root/'driver.json');launch=read(root.with_name(root.name+'-launch.json'))
    expected={'ordinary','fg','semantic_audit','identity_ordinary','identity_fg','terminal_ordinary','terminal_fg'}
    if (driver.get('complete') or not driver.get('terminal') or driver.get('stage')!='stopped'
            or driver.get('error')!="ValueError('Actual observations, instructions, manifest, or labels differ.')"
            or set(driver['jobs'])!=expected or any(j['exit_code']!=0 for j in driver['jobs'].values())):
        raise ValueError('Only the completed pre-action diagnostic mismatch is eligible.')
    for pid in [launch['pid'],*[j['pid'] for j in driver['jobs'].values()]]:
        try:state=(proc_root/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()[0]
        except FileNotFoundError:continue
        if state!='Z':raise ValueError('A recorded source process is still live; do not duplicate work.')
    if any((root/a/'joint').exists() for a in ARMS):
        raise ValueError('Source already contains action training; do not duplicate optimizer steps.')
    plan=read(root/'protocol.json')
    if plan['format']!='robotwin_expanded_fg_matched_trial_v1':raise ValueError('Unexpected source protocol.')
    return plan


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--deadline',required=True)
    args=ap.parse_args();source=args.source_root.resolve();root=args.output.resolve()
    cutoff=datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or cutoff.timestamp()<=time.time():ap.error('A future absolute cutoff is required.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Commit code first.')
    old=validate_source(source);plan=deepcopy(old)
    plan.update(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        continuation_source=str(source),source_code_commit=old['code_commit'],identity_root=str(source/'identity'),
        deadline=args.deadline,continuation_reason='Training bank changed but the exact held-out diagnostic queries did not. Preserve completed semantic training and probes; execute the originally specified four action arms and CF matrices in a new directory.')
    root.mkdir(parents=True,exist_ok=False)
    state=dict(complete=False,terminal=False,stage='auditing_source',jobs={});processes={}
    def write(name,value):
        p=root/(name+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(root/name)
    write('protocol.json',plan);write('driver.json',state)
    env=os.environ|dict(PATH='/opt/conda/bin:'+os.environ['PATH'],PYTHONPATH=str(REPO/'src')+':'+str(REPO),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
    def budget():
        if time.time()>=cutoff.timestamp():raise TimeoutError('Own trial cutoff; server stays on.')
        if shutil.disk_usage(root).free<3*1024**3:raise RuntimeError('Three GiB reserve reached.')
    def stop(signum,frame):raise InterruptedError('Signal'+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(name,cmd,gpus):
        budget()
        with (root/(name+'.log')).open('x') as log:
            p=subprocess.Popen(cmd,cwd=REPO,env=env|{'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus))},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[name]=p;state['jobs'][name]=dict(pid=p.pid,command=cmd,gpus=gpus,exit_code=None)
        write('driver.json',state);return p
    def group(stage,commands):
        state['stage']=stage;active={n:launch(n,c,g) for n,(c,g) in commands.items()}
        while active:
            budget()
            for name,p in list(active.items()):
                rc=p.poll()
                if rc is not None:
                    state['jobs'][name]['exit_code']=rc;del active[name];write('driver.json',state)
                    if rc:raise RuntimeError(name+' failed: '+str(rc))
            if active:time.sleep(5)
    try:
        budget()
        from experiments.robotwin.eraf_fg_bridge import file_sha256
        from experiments.robotwin.eraf_action_protocol import validate_joint_identity_audit
        from experiments.robotwin.fg_role_masks import VerifiedCorrectionMasks
        from scripts.audit_robotwin_cross_goal_training import audit as semantic_audit
        from scripts.compare_robotwin_cross_goal_semantics import compare
        read=lambda p:json.loads(Path(p).read_text())
        if (file_sha256(plan['manifest'])!=plan['manifest_sha256']
                or file_sha256(plan['strongest_checkpoint'])!=plan['source_policy_sha256']
                or file_sha256(plan['prior_comparison_config'])!=plan['prior_comparison_sha256']):
            raise ValueError('Previously bound input changed.')
        masks=VerifiedCorrectionMasks(plan['masks'],plan['manifest'])
        masks.validate_coverage(read(plan['manifest'])['states'])
        if masks.index_sha256!=plan['mask_index_sha256'] or len(masks.records)!=168:
            raise ValueError('Prepared role masks changed.')
        if shutil.disk_usage(root).free<3*1024**3+plan['reserved_output_bytes']:
            raise RuntimeError('Conservative original output reservation does not fit.')
        refreshed=semantic_audit(source/'semantic')
        recorded=read(source/'semantic/training_audit.json')
        if refreshed!=recorded:raise ValueError('Actual semantic training no longer matches its recorded audit.')
        write('semantic_reaudit.json',refreshed)
        hashes={str(source/name):file_sha256(source/name) for name in ('protocol.json','driver.json','semantic/training_audit.json')}
        for mode,arm in [('ordinary','eraf_only'),('fg','eraf_fg')]:
            checkpoint=Path(plan['arms'][arm]['parent'])
            if checkpoint.resolve()!=(source/'semantic'/mode/'step_001500.pt').resolve():raise ValueError('Semantic parent path changed.')
            path=source/'identity'/mode/'summary.json';identity=read(path)
            validate_joint_identity_audit(identity,checkpoint_sha256=file_sha256(checkpoint),manifest_sha256=plan['manifest_sha256'])
            hashes[str(path)]=file_sha256(path)
        old_probe=RUNS/'robotwin_cross_goal_semantics/20260908-paired500-terminal-audit'
        reports={m+'_1000':read(old_probe/(m+'.json')) for m in ('ordinary','fg')}
        reports.update({m+'_1500':read(source/'terminal'/(m+'.json')) for m in ('ordinary','fg')})
        for mode in ('ordinary','fg'):
            report=reports[mode+'_1500']
            if report['checkpoint_sha256']!=refreshed['arms'][mode]['checkpoint_sha256'] or report['manifest_sha256']!=plan['manifest_sha256']:
                raise ValueError('Terminal diagnostic is not bound to the audited semantic checkpoint.')
            hashes[str(source/'terminal'/(mode+'.json'))]=file_sha256(source/'terminal'/(mode+'.json'))
        write('terminal_comparison.json',compare(reports,allow_distinct_manifests=True))
        write('continuation_audit.json',dict(complete=True,source_artifact_sha256=hashes,
            source_processes_terminal=True,no_action_steps_repeated=True,semantic_training_reaudited=True,
            actual_diagnostic_inputs_and_labels_match=True,identity_checks_bound_to_actual_models=True))
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise ValueError('GPUs are owned by another process.')
        run_action_stages(plan,root,read(plan['prior_comparison_config']),group=group,write=write,launch=launch,budget=budget,processes=processes,state=state)
    except BaseException as error:
        state.update(stage='stopped',error=repr(error));raise
    finally:
        for p in processes.values():
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for name,p in processes.items():
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            state['jobs'][name]['exit_code']=p.returncode
        state.update(terminal=True,finished_at=datetime.now().astimezone().isoformat());write('driver.json',state)


if __name__=='__main__':main()
