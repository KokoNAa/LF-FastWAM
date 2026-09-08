#!/usr/bin/env python3
"""Matched semantic refresh, four joint arms and unchanged ten-task CF evaluation."""
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
RUNS=Path('/root/gpufree-data/LF-FastWAM/runs')
ARMS={'no_eraf':dict(eraf='off',fg='off',gpus=[4]),
      'fg_only':dict(eraf='off',fg='full',gpus=[5]),
      'eraf_only':dict(eraf='on',fg='off',gpus=[0,1]),
      'eraf_fg':dict(eraf='on',fg='full',gpus=[2,3])}
TARGETS=('place_a2b_left','blocks_ranking_rgb','place_empty_cup','move_pillbottle_pad')


def source_process(pid,root,*,proc_root=Path('/proc')):
    p=proc_root/str(pid)
    try:
        fields=(p/'stat').read_text().rsplit(') ',1)[1].split()
        argv=(p/'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:return None
    if fields[0]=='Z':return None
    if (str(root).encode() not in argv
            or not any(a.endswith(b'/run_robotwin_expanded_fg_preparation.py') for a in argv)):
        raise ValueError('Preparation PID belongs to another command.')
    return fields[19]


def semantic_command(plan,root,mode):
    from experiments.robotwin.pgc_data import ROBOTWIN_TEN_TASK_NAMES
    return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node','3',
        str(REPO/'scripts/train_robotwin_eraf_fg_grounding.py'),'--manifest',plan['manifest'],
        '--source-bank',plan['source_bank'],'--checkpoint',plan['semantic_parents'][mode],
        '--output',str(root/'semantic'/mode),'--label-cache',plan['label_cache'],
        '--steps','500','--save-every','500','--step-offset','1000','--global-batch','12',
        '--learning-rate','0.0001','--position-weight','2','--anchor-weight','2',
        '--fg-role-masks',plan['masks'],'--geometry-replay',mode,'--geometry-replay-slots','3',
        '--task-balanced','--qualify-each-language','--qualification-tasks',*ROBOTWIN_TEN_TASK_NAMES,
        '--skip-file-hashes','--paired-cross-goals']


def action_command(plan,root,arm):
    from scripts.run_robotwin_ten_task_action_expansion import train_command
    cmd=train_command(plan,root,arm)
    cmd[cmd.index('--correction-weight')+1]='1'
    cmd += ['--target-tasks',*TARGETS]
    if plan.get('initial_expert_tasks'):
        cmd += ['--initial-expert-tasks',*plan['initial_expert_tasks'],
                '--correction-task-weights',json.dumps(plan['correction_task_weights'],sort_keys=True)]
    if plan.get('action_objective'):
        cmd += ['--action-objective',plan['action_objective']]
    if plan['arms'][arm]['eraf']=='on':
        mode='fg' if arm=='eraf_fg' else 'ordinary'
        cmd += ['--zero-context-joint','--identity-audit',str(Path(plan.get('identity_root',root/'identity'))/mode/'summary.json')]
    return cmd


def comparison_config(previous,root,groups,history_prefix='pre_expanded_fg_'):
    methods=dict(previous['methods'])
    for arm in ARMS:
        alias=history_prefix+arm
        if alias in methods:raise ValueError('Historical comparison alias already exists.')
        methods[alias]=methods[arm]
        methods[arm]={g:str(root/('eval_'+g)/arm/'dev') for g in groups}
    return dict(target='eraf_fg',methods=methods)


def run_action_stages(plan,root,previous,*,group,write,launch,budget,processes,state):
    from scripts.compare_robotwin_ten_task_methods import build_report
    group('joint_training',{a:(action_command(plan,root,a),s['gpus']) for a,s in plan['arms'].items()})
    group('auditing_joint',{a+'_audit':([sys.executable,str(REPO/'scripts/audit_robotwin_eraf_fg_action_stage.py'),
        '--output',str(root/a/'joint')],[]) for a in ARMS})
    from scripts.audit_robotwin_expanded_fg_trial import audit_action_pairing
    write('paired_action_audit.json',audit_action_pairing(root))
    if plan.get('record_final_hashes'):
        from experiments.robotwin.eraf_fg_bridge import file_sha256
        ledger={}
        for arm in ARMS:
            path=root/arm/'joint/step_000200.pt';stat=path.stat()
            audit=json.loads((path.parent/'freeze_audit.json').read_text())
            identity=dict(path=str(path.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
            if not audit['complete'] or audit['checkpoint_identity']!=identity:
                raise ValueError('Actual final weight differs from the completed training audit.')
            ledger[arm]=identity|dict(sha256=file_sha256(path),actual_audit_identity_matches=True)
        write('final_checkpoint_hashes.json',dict(complete=True,models=ledger,
            scope='Actual audited final200 weights, hashed before CF evaluation.'))
    state['stage']='evaluating';pending=[(g,t,a) for g,s in plan['groups'].items() for t in s['tasks'] for a in ARMS];active={}
    while pending or active:
        budget()
        for name,gpu in list(active.items()):
            rc=processes[name].poll()
            if rc is not None:
                state['jobs'][name]['exit_code']=rc;del active[name];write('driver.json',state)
                if rc:raise RuntimeError(name+' failed: '+str(rc))
        for gpu in sorted(set(range(6))-set(active.values())):
            if not pending:break
            g,t,a=pending.pop(0);s=plan['groups'][g];name=f'eval_{g}_{a}_{t}'
            cmd=[sys.executable,'-u',str(REPO/'scripts/eval_robotwin_eraf_fg.py'),'worker',
                '--output',str(root/('eval_'+g)/a/'dev'),'--checkpoint',str(root/a/'joint/step_000200.pt'),
                '--manifest',plan['manifest'],'--catalog-root',s['catalog'],'--episodes',str(s['episodes']),
                '--tasks',t,'--policy-kind','repair','--eraf',plan['arms'][a]['eraf'],'--conditions','counterfactual',
                '--gpu',str(gpu),'--videos','--skip-file-hashes','--interventions',str(REPO/'configs/eval/robotwin_cis_ten_tasks.json')]
            launch(name,cmd,[gpu]);active[name]=gpu
        if active:time.sleep(5)
    commands={}
    for g,s in plan['groups'].items():
        for a in ARMS:
            commands['summarize_'+g+'_'+a]=([sys.executable,str(REPO/'scripts/eval_robotwin_eraf_fg.py'),'summarize',
                '--output',str(root/('eval_'+g)/a/'dev'),'--checkpoint',str(root/a/'joint/step_000200.pt'),
                '--catalog-root',s['catalog'],'--episodes',str(s['episodes']),'--tasks',*s['tasks'],
                '--conditions','counterfactual','--skip-file-hashes'],[])
    group('summarizing',commands)
    if plan.get('record_final_hashes'):
        for value in ledger.values():
            if file_sha256(value['path'])!=value['sha256']:
                raise ValueError('Actual checkpoint changed during CF evaluation.')
    config=comparison_config(previous,root,plan['groups'],plan.get('comparison_history_prefix','pre_expanded_fg_'))
    write('comparison_config.json',config);write('comparison.json',build_report(config))
    state.update(complete=True,terminal=True,stage='complete')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preparation-root',type=Path,required=True)
    ap.add_argument('--preparation-pid',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--deadline',required=True)
    args=ap.parse_args()
    cutoff=datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or cutoff.timestamp()<=time.time():ap.error('Require a future absolute cutoff.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Commit the complete trial before starting.')
    source=args.preparation_root.resolve();root=args.output.resolve()
    receipt=json.loads(source.with_name(source.name+'-launch.json').read_text())
    if receipt['pid']!=args.preparation_pid:raise ValueError('Preparation receipt and PID disagree.')
    start=source_process(args.preparation_pid,source)
    root.mkdir(parents=True,exist_ok=False)
    state=dict(complete=False,terminal=False,stage='waiting_for_preparation',jobs={})
    processes={}
    def write(name,value):
        p=root/(name+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(root/name)
    strongest=RUNS/'robotwin_cf_priority/20260907-1543-warmoff200-cf4/ordinary_cf/joint/step_000200.pt'
    semantic_root=RUNS/'robotwin_cross_goal_semantics/20260908-paired500'
    previous_path=RUNS/'robotwin_memory_guard_trial/20260908-paired1000-fixed200/comparison_config.json'
    previous=json.loads(previous_path.read_text())
    historical=json.loads((RUNS/'robotwin_cross_goal_joint_trial/20260908-paired1000-zero200/protocol.json').read_text())
    plan=dict(format='robotwin_expanded_fg_matched_trial_v1',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        preparation_root=str(source),preparation_pid=args.preparation_pid,preparation_start_time=start,
        manifest=str(source/'cache/manifest.json'),masks=str(source/'masks/complete.json'),
        source_bank=historical['source_bank'],correct_teacher=historical['correct_teacher'],strongest_checkpoint=str(strongest),
        label_cache=str(RUNS/'robotwin_fg_role_masks/20260907-2255-logfix100/label_cache/ordinary'),
        semantic_parents={m:str(semantic_root/m/'step_001000.pt') for m in ('ordinary','fg')},
        semantic_parent_audit=str(semantic_root/'training_audit.json'),
        semantic_steps=500,semantic_step_offset=1000,semantic_learning_rate=1e-4,
        semantic_recipe='6own-goal shared slots+3opposite-goal views of3shared observations+3matched FG/ordinary partial slots, global12, three ranks per arm, seed42; only declared semantics train.',
        arms={a:dict(s,parent=str(strongest) if s['eraf']=='off' else str(root/'semantic'/('fg' if a=='eraf_fg' else 'ordinary')/'step_001500.pt')) for a,s in ARMS.items()},
        groups=historical['groups'],joint_steps=200,joint_policy_lr=3e-6,joint_interface_lr=3e-5,
        global_batch=12,seed=42,correction_weight=1.,correct_count=2,cf_count=4,correct_weight=1,cf_weight=4,
        action_recipe='Identical common policy initialization and2Correct+4CF+3expert+3FG-or-matched-ordinary rows. Action LoRA and declared ERAF interfaces train; noFG arms never load corrective pixels/actions. No seen language augmentation.',
        target_tasks=list(TARGETS),checkpoint_selection='Predeclared final cumulative1500semantic and freshjoint200; no selection from partial CF matrices.',
        recipe_change='Add60actual cup/pill FG corrections, matched500semantic refresh, and increase correction/ordinary-replacement coefficient from0.1to1 equally across allfour arms. This is a method comparison, not a data-only ablation.',
        prior_comparison_config=str(previous_path),total_eval_episodes=180,
        primary_metric='Exact rational equal-task ten-task CF macro; retain every prior control and supplementary strict burger slots.',
        correct_evaluated=False,independent_test=False,goal_achievement_claim=False,
        deadline=args.deadline,disk_reserve_GiB=3,platform_shutdown=None,model_storage='server_only')
    write('protocol.json',plan);write('driver.json',state)
    env=os.environ|dict(PATH='/opt/conda/bin:'+os.environ['PATH'],PYTHONPATH=str(REPO/'src')+':'+str(REPO),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
    def budget():
        if time.time()>=cutoff.timestamp():raise TimeoutError('Trial cutoff; server remains on.')
        if shutil.disk_usage(root).free<3*1024**3:raise RuntimeError('Trial disk reserve reached.')
    def stop(signum,frame):raise InterruptedError('Signal'+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(name,cmd,gpus):
        budget()
        with (root/(name+'.log')).open('x') as log:
            p=subprocess.Popen(cmd,cwd=REPO,env=env|{'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus))},
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
        while True:
            budget();current=source_process(args.preparation_pid,source)
            if current is None:break
            if start is None or current!=start:raise ValueError('Preparation process identity changed.')
            time.sleep(10)
        done=json.loads((source/'driver.json').read_text())
        if not done.get('complete') or not done.get('terminal') or any(j.get('exit_code')!=0 for j in done['jobs'].values()):
            raise ValueError('Preparation stopped without complete cache and full mask evidence.')
        from experiments.robotwin.eraf_fg_bridge import file_sha256
        from experiments.robotwin.fg_role_masks import VerifiedCorrectionMasks
        from scripts.compare_robotwin_ten_task_methods import build_report
        manifest=json.loads(Path(plan['manifest']).read_text())
        reader=VerifiedCorrectionMasks(plan['masks'],plan['manifest']);reader.validate_coverage(manifest['states'])
        cache_audit=json.loads((source/'cache_audit.json').read_text())
        if (not cache_audit['complete'] or not cache_audit['parent_rows_preserved_exactly']
                or cache_audit['added_scenes']!=60 or not cache_audit['cached_target_audit']['complete']
                or cache_audit['input_sha256'][plan['manifest']]!=file_sha256(plan['manifest'])
                or done['manifest_sha256']!=file_sha256(plan['manifest']) or len(reader.records)!=168):
            raise ValueError('Prepared-bank proof does not match the actual input.')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise ValueError('Another process owns the GPUs.')
        # Reserve actual prior same-architecture final weights/optimizers plus two semantics and media.
        prior=RUNS/'robotwin_ten_task_action_expansion/20260908-0005-joint200'
        def size(path):
            if path.is_file():return path.stat().st_size
            compact=json.loads(path.with_suffix('.compacted.json').read_text())
            if not compact['exact_tensors_and_metadata_verified_after_readback']:raise ValueError('Unverified size reference.')
            return compact['original_bytes']
        allocation=sum(size(prior/a/'joint'/n) for a in ARMS for n in ('step_000200.pt','optimizer_last.pt'))
        allocation+=sum(Path(p).stat().st_size for p in plan['semantic_parents'].values())+400*1024**2
        if shutil.disk_usage(root).free<3*1024**3+allocation:raise RuntimeError('Reserve allfour final models/optimizers, two semantics and media before training.')
        plan.update(manifest_sha256=file_sha256(plan['manifest']),mask_index_sha256=reader.index_sha256,
            preparation_audit_sha256=file_sha256(source/'cache_audit.json'),reserved_output_bytes=allocation,
            source_policy_sha256=file_sha256(strongest),prior_comparison_sha256=file_sha256(previous_path))
        write('protocol.json',plan);write('prior_comparison.json',build_report(previous))
        import torch
        torch.set_num_threads(2)
        from experiments.robotwin.eraf_action_protocol import is_zero_context_parent,validate_joint_identity_audit
        base=torch.load(strongest,map_location='cpu',weights_only=False)
        parent_audit=json.loads(Path(plan['semantic_parent_audit']).read_text())
        if not parent_audit.get('complete') or not parent_audit['actual_samples_and_instruction_branches_verified']:
            raise ValueError('Semantic parents lack their actual training audit.')
        parent_proofs={}
        for mode,path in plan['semantic_parents'].items():
            payload=torch.load(path,map_location='cpu',weights_only=False)
            digest=file_sha256(path)
            if (digest!=parent_audit['arms'][mode]['checkpoint_sha256'] or payload['optimizer_steps']!=1000
                    or not is_zero_context_parent(payload) or payload['mot_trainable'].keys()!=base['mot_trainable'].keys()
                    or any(not torch.equal(v,base['mot_trainable'][k]) for k,v in payload['mot_trainable'].items())):
                raise ValueError('Actual semantic parent is not the audited common strongest policy.')
            parent_proofs[mode]=dict(checkpoint_sha256=digest,policy_tensors_identical=True,residual_output_zero=True)
            del payload
        del base
        write('initial_policy_identity.json',dict(complete=True,strongest_checkpoint_sha256=plan['source_policy_sha256'],arms=parent_proofs))
        group('semantic_refresh',{m:(semantic_command(plan,root,m),g) for m,g in [('ordinary',[0,1,2]),('fg',[3,4,5])]})
        group('auditing_semantics',{'semantic_audit':([sys.executable,str(REPO/'scripts/audit_robotwin_cross_goal_training.py'),
            '--output',str(root/'semantic')],[])})
        probes={}
        for mode,gpu in [('ordinary',0),('fg',3)]:
            ckpt=str(root/'semantic'/mode/'step_001500.pt')
            common=['--manifest',plan['manifest'],'--source-bank',plan['source_bank'],'--checkpoint',ckpt]
            probes['identity_'+mode]=([sys.executable,str(REPO/'scripts/probe_robotwin_eraf_context_tokens.py'),
                *common,'--output',str(root/'identity'/mode),'--require-residual-identity','--include-expanded-tasks'],[gpu])
            probes['terminal_'+mode]=([sys.executable,str(REPO/'scripts/probe_robotwin_cross_goal_semantics.py'),
                *common,'--output',str(root/'terminal'/ (mode+'.json'))],[gpu+1])
        group('semantic_and_deployed_identity_probes',probes)
        from scripts.compare_robotwin_cross_goal_semantics import compare as semantic_compare
        old_probe=RUNS/'robotwin_cross_goal_semantics/20260908-paired500-terminal-audit'
        reports={m+'_1000':json.loads((old_probe/(m+'.json')).read_text()) for m in ('ordinary','fg')}
        reports.update({m+'_1500':json.loads((root/'terminal'/(m+'.json')).read_text()) for m in ('ordinary','fg')})
        write('terminal_comparison.json',semantic_compare(reports,allow_distinct_manifests=True))
        for mode in ('ordinary','fg'):
            audit=json.loads((root/'identity'/mode/'summary.json').read_text())
            validate_joint_identity_audit(audit,checkpoint_sha256=file_sha256(root/'semantic'/mode/'step_001500.pt'),
                                         manifest_sha256=plan['manifest_sha256'])
        run_action_stages(plan,root,previous,group=group,write=write,launch=launch,budget=budget,processes=processes,state=state)
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
