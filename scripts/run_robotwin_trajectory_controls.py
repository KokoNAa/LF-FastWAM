#!/usr/bin/env python3
"""Complete the three matched800 controls, preserving the evaluated full candidate."""
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
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from experiments.robotwin.trajectory_controls import (
    BASE_COMMIT, SPECS, command, common_command, audit_arm, audit_pairing, comparison_config)
from scripts.run_robotwin_balanced_target_trial import completed_source
from scripts.run_robotwin_trajectory_target_trial import coverage, require_coverage
from experiments.robotwin.trajectory_target import mixture_stream
read = lambda p: json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--deadline', required=True)
    ap.add_argument('--preflight-only', action='store_true')
    args = ap.parse_args()
    source, root = args.source_root.resolve(), args.output.resolve()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or cutoff.timestamp() <= time.time():
        raise ValueError('Future absolute deadline required.')
    if root.exists():
        raise ValueError('New output directory required.')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Commit code first.')
    from experiments.robotwin.eraf_fg_bridge import file_sha256 as sha
    from experiments.robotwin.eraf_action_protocol import validate_joint_identity_audit
    from scripts.compare_robotwin_ten_task_methods import build_report
    old = completed_source(source)
    receipt = read(source / 'terminal_verification.json')
    if not receipt['complete'] or receipt['episodes'] != 45 or receipt['comparison_methods'] != 30:
        raise ValueError('Verified complete trajectory800 candidate required.')
    if old['code_commit'] != BASE_COMMIT or (old['steps'],old['world_size'],old['global_batch']) != (800,6,12):
        raise ValueError('Wrong candidate recipe.')
    source_config = read(source / 'comparison_config.json')
    source_report = read(source / 'comparison.json')
    if build_report(source_config) != source_report or sha(source / 'comparison.json') != receipt['comparison_sha256']:
        raise ValueError('Source episodes do not reproduce the archived report.')
    bt_root = Path(old['source_root'])
    bt = completed_source(bt_root)
    bt_driver = read(bt_root / 'driver.json')
    source_driver = read(source / 'driver.json')
    source_model = source / 'eraf_fg/joint/step_000800.pt'
    if sha(source_model) != receipt['models']['eraf_fg']['sha256']:
        raise ValueError('Previously evaluated candidate changed.')
    # Only new orchestration/audit/tests/docs may differ; training and inference stay byte-identical.
    allowed = {'experiments/robotwin/trajectory_controls.py', 'scripts/run_robotwin_trajectory_controls.py',
               'tests/test_robotwin_trajectory_controls.py', 'docs/TRAJECTORY_CONTROLS_800.md'}
    changed = set(subprocess.check_output(['git','diff','--name-only',BASE_COMMIT,'HEAD'],cwd=REPO,text=True).splitlines())
    if not changed or not changed <= allowed:
        raise ValueError('Training or evaluation code changed versus the full candidate.')
    source_repo = Path(source_driver['jobs']['training']['command'][6]).parents[1]
    tracked = subprocess.check_output(['git','ls-tree','-r','--name-only',BASE_COMMIT],cwd=REPO,text=True).splitlines()
    code_hashes = {}
    for rel in tracked:
        if rel.startswith(('src/','experiments/','scripts/','configs/')) and Path(rel).suffix in ('.py','.json','.yaml','.yml'):
            if sha(REPO/rel) != sha(source_repo/rel):
                raise ValueError('Actual runtime file differs: '+rel)
            code_hashes[str(REPO/rel)] = sha(REPO/rel)
    bindings = dict(old['input_sha256'])
    for path in [source/'protocol.json',source/'comparison.json',source/'comparison_config.json',
                 source/'terminal_verification.json',source/'final_checkpoint_hashes.json',source_model,
                 bt_root/'protocol.json',bt_root/'driver.json']:
        bindings[str(path)] = sha(path)
    for path,digest in old['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Candidate source dependency changed: '+path)
    ordinary_parent = Path(bt['arms']['eraf_only']['parent'])
    identity = Path(bt['identity_root']) / 'ordinary/summary.json'
    for path in [ordinary_parent,identity]:
        bindings[str(path)] = sha(path)
    validate_joint_identity_audit(read(identity),checkpoint_sha256=sha(ordinary_parent),manifest_sha256=bt['manifest_sha256'])
    if any(bt['arms'][a]['parent'] != bt['strongest_checkpoint'] for a in ('no_eraf','fg_only')):
        raise ValueError('ERAF-off parents must be the strongest shared action initialization.')
    commands = {a: command(bt_driver['jobs'][a]['command'], root, a, REPO) for a in SPECS}
    if any(common_command(c) != common_command(source_driver['jobs']['training']['command']) for c in commands.values()):
        raise ValueError('Control optimizer/inference/retention recipe differs from the full candidate.')
    rows = read(old['manifest'])['states']
    planned = {}
    for arm,spec in SPECS.items():
        stream = mixture_stream(rows,42,spec['fg'])
        ids = [r['id'] for _ in range(800) for r in next(stream)]
        planned[arm] = coverage(rows,ids)
        if spec['fg'] == 'full':
            require_coverage(planned[arm])
        elif any(v['minimum_exposure'] < 2 for k,v in planned[arm].items() if k.endswith('|ordinary')):
            raise ValueError('Control ordinary coverage is incomplete.')
    gpu_rows = subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines()
    if len(gpu_rows) != 6 or subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('Six unoccupied GPUs required.')
    measured = sum((bt_root/a/'joint'/f).stat().st_size for a in SPECS for f in ('step_000200.pt','optimizer_last.pt'))
    reserve, floor = int(measured*1.05)+256*1024**2, 1024**3
    if shutil.disk_usage(root.parent).free < reserve+floor:
        raise ValueError('Final models, optimizers, logs and one-GiB spare do not fit.')
    plan = dict(format='robotwin_trajectory_controls800_v1',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                source_root=str(source),source_balanced_root=str(bt_root),source_model=str(source_model),
                source_model_sha256=sha(source_model),manifest=old['manifest'],source_bank=old['source_bank'],
                groups=old['groups'],arms=SPECS,steps=800,world_size=6,global_batch=12,seed=42,
                training_commands=commands,planned_coverage=planned,input_sha256=bindings,runtime_sha256=code_hashes,
                reserved_output_bytes=reserve,disk_floor_bytes=floor,deadline=args.deadline,platform_shutdown=None,
                output_storage='server_data_disk',model_storage='server_only',training_order=list(SPECS),
                recipe='Reuse unchanged complete-trajectory target-only10-step final24 training. Each new arm receives800steps on6GPUs, global12, same seed/slots/LRs/old-CFteacher. FG-off replaces each of4FGslots by same-task ordinary targets. Reuse audited ordinary S1500 parent for ERAF-only and R for off arms. No semantic refresh or full-candidate retraining.',
                checkpoint_selection='Prespecified final800 only; retain all checkpoints/optimizers and all historical comparisons.',
                evaluation='135newCFepisodes plus the unchanged45full-candidate episodes; same task catalogs, instructions, seeds and scoring; videosoff as full candidate.',
                independent_test=False,goal_achievement_claim=False)
    if args.preflight_only:
        print(json.dumps(plan,indent=2));return
    root.mkdir(parents=True)
    state = dict(complete=False,terminal=False,stage='admitted',jobs={})
    processes = {}
    def write(name,value):
        p=root/(name+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(root/name)
    write('protocol.json',plan);write('driver.json',state)
    env=os.environ|dict(PATH='/opt/conda/bin:'+os.environ['PATH'],PYTHONPATH=str(REPO/'src')+':'+str(REPO),
                        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
                        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
    def budget():
        if time.time() >= cutoff.timestamp():raise TimeoutError('Stop owned jobs only; platform stays on.')
        if shutil.disk_usage(root).free < floor:raise RuntimeError('One-GiB disk spare reached.')
    def stop(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(name,cmd,gpus):
        budget()
        with (root/(name+'.log')).open('x') as log:
            p=subprocess.Popen(cmd,cwd=REPO,env=env|{'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus))},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[name]=p;state['jobs'][name]=dict(pid=p.pid,command=cmd,gpus=gpus,exit_code=None);write('driver.json',state)
    def wait(name,arm=None):
        prefix=False
        while True:
            budget();p=processes[name];rc=p.poll()
            if arm and not prefix:
                joint=root/arm/'joint'
                if all((joint/f'rank{i}.jsonl').exists() and (joint/f'rank{i}.jsonl').stat().st_size for i in range(6)):
                    write(arm+'_first_step_audit.json',audit_arm(joint,rows,SPECS[arm]['fg'],1));prefix=True
            if rc is not None:
                state['jobs'][name]['exit_code']=rc;write('driver.json',state)
                if rc:raise RuntimeError(name+' failed '+str(rc))
                return
            time.sleep(2)
    def stage(name,cmd,gpus,arm=None):
        state['stage']=name;launch(name,cmd,gpus);wait(name,arm)
    try:
        ledger={'eraf_fg':dict(path=str(source_model),sha256=sha(source_model),reused=True)}
        for arm in SPECS:
            stage('train_'+arm,commands[arm],list(range(6)),arm)
            joint=root/arm/'joint';model=joint/'step_000800.pt'
            stage('audit_'+arm,[sys.executable,str(REPO/'scripts/audit_robotwin_eraf_fg_action_stage.py'),'--output',str(joint)],[])
            actual=audit_arm(joint,rows,SPECS[arm]['fg'])
            if actual['coverage'] != planned[arm]:raise ValueError('Actual coverage differs from planned.')
            write(arm+'_coverage_audit.json',actual)
            ledger[arm]=dict(path=str(model),sha256=sha(model),bytes=model.stat().st_size,reused=False)
            write('checkpoint_hashes_progress.json',ledger)
        joints={a:root/a/'joint' for a in SPECS};joints['eraf_fg']=source/'eraf_fg/joint'
        write('paired_action_audit.json',audit_pairing(joints,rows))
        write('final_checkpoint_hashes.json',dict(complete=True,models=ledger))
        state['stage']='evaluating';pending=[(g,t,a) for g,s in old['groups'].items() for t in s['tasks'] for a in SPECS];active={}
        while pending or active:
            budget()
            for name,gpu in list(active.items()):
                rc=processes[name].poll()
                if rc is not None:
                    state['jobs'][name]['exit_code']=rc;del active[name];write('driver.json',state)
                    if rc:raise RuntimeError(name+' failed '+str(rc))
            for gpu in sorted(set(range(6))-set(active.values())):
                if not pending:break
                g,t,arm=pending.pop(0);spec=old['groups'][g];name='eval_'+g+'_'+arm+'_'+t
                cmd=[sys.executable,'-u',str(REPO/'scripts/eval_robotwin_eraf_fg.py'),'worker','--output',str(root/('eval_'+g)/arm/'dev'),
                     '--checkpoint',ledger[arm]['path'],'--manifest',old['manifest'],'--catalog-root',spec['catalog'],'--episodes',str(spec['episodes']),
                     '--tasks',t,'--policy-kind','repair','--eraf',SPECS[arm]['eraf'],'--conditions','counterfactual','--gpu',str(gpu),
                     '--skip-file-hashes','--interventions',str(REPO/'configs/eval/robotwin_cis_ten_tasks.json')]
                launch(name,cmd,[gpu]);active[name]=gpu
            if active:time.sleep(2)
        for g,spec in old['groups'].items():
            for arm in SPECS:
                stage('summarize_'+g+'_'+arm,[sys.executable,str(REPO/'scripts/eval_robotwin_eraf_fg.py'),'summarize',
                      '--output',str(root/('eval_'+g)/arm/'dev'),'--checkpoint',ledger[arm]['path'],'--catalog-root',spec['catalog'],
                      '--episodes',str(spec['episodes']),'--tasks',*spec['tasks'],'--conditions','counterfactual','--skip-file-hashes'],[])
        config=comparison_config(source_config,root,old['groups']);report=build_report(config)
        write('comparison_config.json',config);write('comparison.json',report)
        if any(sha(v['path']) != v['sha256'] for v in ledger.values()) or any(sha(p) != h for p,h in (bindings|code_hashes).items()):
            raise ValueError('A model, input or runtime file changed.')
        write('post_evaluation_hash_audit.json',dict(complete=True,all_four_models_unchanged=True,all_inputs_and_runtime_unchanged=True))
        state.update(complete=True,stage='complete')
    except BaseException as error:
        state.update(error=repr(error),stage='stopped');raise
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


if __name__ == '__main__':main()
