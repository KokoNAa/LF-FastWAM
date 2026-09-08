#!/usr/bin/env python3
"""Prespecified complete-trajectory800 candidate, audited training and45CF episodes."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
read=lambda p:json.loads(Path(p).read_text())
STEPS=800


def training_command(previous,root):
    cmd=list(previous)
    cmd[cmd.index('--nproc_per_node')+1]='6'
    cmd[6]=str(REPO/'scripts/train_robotwin_eraf_fg_action.py')
    for flag,value in [('--steps',str(STEPS)),('--save-every',str(STEPS)),('--action-objective','trajectory_target_rollout_v1'),('--output',str(root/'eraf_fg/joint'))]:
        cmd[cmd.index(flag)+1]=value
    return cmd


def coverage(rows,ids):
    from experiments.robotwin.trajectory_target import pools
    from experiments.robotwin.initial_anchor import source_task
    counted=Counter(ids);result={}
    for kind,pool in zip(('cf','ordinary','fg'),pools(rows)):
        for task in sorted({source_task(r) for r in pool}):
            candidates=[r for r in pool if source_task(r)==task]
            values=[counted[r['id']] for r in candidates]
            result[task+'|'+kind]=dict(rows=len(candidates),unique_sampled=sum(v>0 for v in values),exposures=sum(values),minimum_exposure=min(values),maximum_exposure=max(values))
    return result


def require_coverage(result):
    if any(v['minimum_exposure']<2 for k,v in result.items() if k.endswith(('|ordinary','|fg'))):
        raise ValueError('Every ordinary and FG training row must be exposed at least twice.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('source-root','output'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--deadline',required=True);ap.add_argument('--preflight-only',action='store_true')
    a=ap.parse_args();source=a.source_root.resolve();root=a.output.resolve();cutoff=datetime.fromisoformat(a.deadline)
    if cutoff.tzinfo is None or cutoff.timestamp()<=time.time():raise ValueError('Future deadline required.')
    if root.exists():raise ValueError('New output required.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Commit code first.')
    from scripts.run_robotwin_balanced_target_trial import completed_source
    from scripts.run_robotwin_deployed_action_probe import live
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    from experiments.robotwin.eraf_action_protocol import validate_joint_identity_audit
    from experiments.robotwin.trajectory_target import mixture_stream
    from scripts.probe_robotwin_balanced_postjoint_actions import compare_actions
    from scripts.compare_robotwin_ten_task_methods import build_report
    old=completed_source(source);terminal=read(source/'terminal_verification.json')
    if not terminal['complete'] or terminal['episodes']!=180:raise ValueError('Completed balanced trial required.')
    diagnostic=source.with_name('20260908-balanced-target-fit-check')
    d=read(diagnostic/'driver.json');receipt=read(diagnostic/'terminal_verification.json')
    if not d['complete'] or not d['terminal'] or not receipt['complete'] or receipt['actual_records']!=36:raise ValueError('Completed diagnostic required.')
    if live(read(diagnostic.with_name(diagnostic.name+'-launch.json'))['pid'],diagnostic) or any(live(j['pid'],diagnostic) or j['exit_code']!=0 for j in d['jobs'].values()):raise ValueError('Diagnostic workers not terminal.')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPUs busy.')
    if len(subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines())!=6:raise ValueError('Six GPUs required.')
    parent=Path(old['arms']['eraf_fg']['parent']);identity=Path(old['identity_root'])/'fg/summary.json'
    paths=[parent,Path(old['strongest_checkpoint']),Path(old['manifest']),identity,source/'comparison_config.json',source/'comparison.json',source/'trajectory_exposure_audit.json',source/'terminal_verification.json',diagnostic/'terminal_verification.json',diagnostic/'comparison.json',diagnostic/'fg/summary.json']
    bindings={str(p):file_sha256(p) for p in paths}
    if bindings[old['manifest']]!=old['manifest_sha256'] or bindings[old['strongest_checkpoint']]!=old['source_policy_sha256']:raise ValueError('Immutable source changed.')
    validate_joint_identity_audit(read(identity),checkpoint_sha256=bindings[str(parent)],manifest_sha256=old['manifest_sha256'])
    rows=read(old['manifest'])['states'];stream=mixture_stream(rows,42);planned=[r['id'] for _ in range(STEPS) for r in next(stream)]
    planned_coverage=coverage(rows,planned);require_coverage(planned_coverage)
    measured=(source/'eraf_fg/joint/step_000200.pt').stat().st_size+(source/'eraf_fg/joint/optimizer_last.pt').stat().st_size
    reserve=int(measured*1.05)+128*1024**2;floor=512*1024**2
    if shutil.disk_usage(root.parent).free<reserve+floor:raise ValueError('Measured model+optimizer, logs and512MiB spare do not fit.')
    cmd=training_command(read(source/'driver.json')['jobs']['eraf_fg']['command'],root)
    plan=dict(format='robotwin_trajectory_target_candidate_v1',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),source_root=str(source),source_diagnostic=str(diagnostic),input_sha256=bindings,manifest=old['manifest'],source_bank=old['source_bank'],groups=old['groups'],parent=str(parent),strongest_checkpoint=old['strongest_checkpoint'],steps=STEPS,world_size=6,global_batch=12,seed=42,planned_coverage=planned_coverage,training_command=cmd,reserved_output_bytes=reserve,disk_floor_bytes=floor,deadline=a.deadline,platform_shutdown=None,model_storage='server_only',checkpoint_selection='Prespecified final800, no intermediate selection.',recipe='Fresh R/S identity initialization. Same target-only ten-step final24 loss and LR3e-6/3e-5, CF weight4. Replace4initial slots by4complete ordinary trajectory slots.4FG slots use complete per-task shuffled cycles.800steps ensure every ordinary and FG row at least2exposures.',scope='Candidate screening against all29historical methods; new matched ablations and independent test still required for module-superiority claim.',independent_test=False,goal_achievement_claim=False)
    if a.preflight_only:print(json.dumps(plan,indent=2));return
    root.mkdir(parents=True);state=dict(complete=False,terminal=False,stage='admitted',jobs={});processes={}
    def write(name,value):
        p=root/(name+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(root/name)
    write('protocol.json',plan);write('driver.json',state)
    env=os.environ|dict(PATH='/opt/conda/bin:'+os.environ['PATH'],PYTHONPATH=str(REPO/'src')+':'+str(REPO),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
    def budget():
        if time.time()>=cutoff.timestamp():raise TimeoutError('Stop owned jobs only; platform stays on.')
        if shutil.disk_usage(root).free<floor:raise RuntimeError('Disk spare reached.')
    def stop(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(name,command,gpus):
        budget()
        with (root/(name+'.log')).open('x') as f:p=subprocess.Popen(command,cwd=REPO,env=env|{'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus))},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        processes[name]=p;state['jobs'][name]=dict(pid=p.pid,command=command,gpus=gpus,exit_code=None);write('driver.json',state)
    def wait(names):
        active=set(names)
        while active:
            budget()
            for name in list(active):
                rc=processes[name].poll()
                if rc is not None:
                    state['jobs'][name]['exit_code']=rc;write('driver.json',state);active.remove(name)
                    if rc:raise RuntimeError(name+' failed '+str(rc))
            if active:time.sleep(2)
    def stage(name,command,gpus):
        state['stage']=name;launch(name,command,gpus);wait([name])
    try:
        stage('training',cmd,list(range(6)))
        joint=root/'eraf_fg/joint';model=joint/'step_000800.pt'
        stage('training_audit',[sys.executable,str(REPO/'scripts/audit_robotwin_eraf_fg_action_stage.py'),'--output',str(joint)],[])
        actual=[]
        for f in joint.glob('rank*.jsonl'):
            for line in f.read_text().splitlines():actual.extend(r['id'] for r in json.loads(line)['examples'])
        actual_coverage=coverage(rows,actual);require_coverage(actual_coverage)
        if Counter(actual)!=Counter(planned):raise ValueError('Actual exposure differs from plan.')
        write('coverage_audit.json',dict(complete=True,actual_examples=len(actual),groups=actual_coverage,actual_matches_planned=True))
        digest=file_sha256(model);write('final_checkpoint_hashes.json',dict(complete=True,models={'eraf_fg':dict(path=str(model),sha256=digest,bytes=model.stat().st_size)}))
        stage('action_fit',[sys.executable,str(REPO/'scripts/probe_robotwin_expanded_fg_actions.py'),'--manifest',old['manifest'],'--source-bank',old['source_bank'],'--checkpoint',str(model),'--output',str(root/'action_fit'),'--eraf','on'],[0])
        write('action_fit_comparison.json',compare_actions(read(diagnostic/'fg/summary.json'),read(root/'action_fit/summary.json')))
        state['stage']='evaluating';pending=[(g,t) for g,s in old['groups'].items() for t in s['tasks']]
        while pending:
            active=[]
            for gpu in range(min(6,len(pending))):
                g,t=pending.pop(0);s=old['groups'][g];name='eval_'+g+'_'+t
                command=[sys.executable,str(REPO/'scripts/eval_robotwin_eraf_fg.py'),'worker','--output',str(root/('eval_'+g)/'eraf_fg/dev'),'--checkpoint',str(model),'--manifest',old['manifest'],'--catalog-root',s['catalog'],'--episodes',str(s['episodes']),'--tasks',t,'--policy-kind','repair','--eraf','on','--conditions','counterfactual','--gpu',str(gpu),'--skip-file-hashes','--interventions',str(REPO/'configs/eval/robotwin_cis_ten_tasks.json')]
                launch(name,command,[gpu]);active.append(name)
            wait(active)
        for g,s in old['groups'].items():
            stage('summarize_'+g,[sys.executable,str(REPO/'scripts/eval_robotwin_eraf_fg.py'),'summarize','--output',str(root/('eval_'+g)/'eraf_fg/dev'),'--checkpoint',str(model),'--catalog-root',s['catalog'],'--episodes',str(s['episodes']),'--tasks',*s['tasks'],'--conditions','counterfactual','--skip-file-hashes'],[])
        config=read(source/'comparison_config.json');config['methods']['pre_trajectory_eraf_fg']=config['methods'].pop('eraf_fg');config['methods']['eraf_fg']={g:str(root/('eval_'+g)/'eraf_fg/dev') for g in old['groups']};config['target']='eraf_fg'
        write('comparison_config.json',config);write('comparison.json',build_report(config))
        if file_sha256(model)!=digest or any(file_sha256(p)!=h for p,h in bindings.items()):raise ValueError('An input or evaluated model changed.')
        write('post_evaluation_hash_audit.json',dict(complete=True,actual_model_sha256=digest,all_inputs_unchanged=True))
        state.update(complete=True,stage='complete')
    except BaseException as error:state.update(error=repr(error),stage='stopped');raise
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
