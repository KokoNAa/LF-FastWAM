#!/usr/bin/env python3
"""Train ERAF branches on the current completed no-ERAF control and evaluate all three."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
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
from experiments.robotwin.five_task_action import TASKS, FG_TASKS, OBJECTIVE, mixture_stream, pools
from experiments.robotwin.initial_anchor import source_task
from scripts.run_robotwin_formal_five40 import read, write, sha, records

RUNS = Path('/root/gpufree-data/LF-FastWAM/runs')
SOURCE = RUNS / 'robotwin_expanded_fg/20260908-balanced-target200'
FORMAL = RUNS / 'robotwin_formal_five40/20260908-balanced-target200'
CURRENT = RUNS / 'robotwin_five_task_repair/20260908-five-task-expert200'


def training_command(plan, root, arm):
    spec = plan['arms'][arm]
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           '--nproc_per_node', str(len(spec['gpus'])), str(REPO / 'scripts/train_robotwin_eraf_fg_action.py'),
           '--manifest', plan['manifest'], '--source-bank', plan['source_bank'],
           '--checkpoint', spec['parent'], '--output', str(root / arm / 'joint'),
           '--correct-teacher', plan['strongest_checkpoint'], '--cf-teacher', plan['strongest_checkpoint'],
           '--stage', 'joint', '--eraf', spec['eraf'], '--fg', spec['fg'],
           '--action-objective', OBJECTIVE, '--policy-scope', 'action', '--interface-scope', 'all',
           '--steps', str(plan['steps']), '--save-every', str(plan['steps']),
           '--learning-rate', '0.000003', '--interface-learning-rate', '0.00003',
           '--correct-count', '0', '--cf-count', '4', '--correct-weight', '1', '--cf-weight', '1',
           '--correction-weight', '1', '--task-balanced', '--disable-seen-language-augmentation',
           '--skip-file-hashes', '--seed', '42', '--target-tasks', *FG_TASKS,
           '--cf-retention-tasks', *TASKS]
    if spec['eraf'] == 'off':
        cmd += ['--warm-policy']
    else:
        cmd += ['--zero-context-joint', '--identity-audit', spec['identity_audit']]
    return cmd


def coverage(rows, steps):
    pools(rows)  # validates scene split, required experts, and actual temporal coverage
    full, off = mixture_stream(rows, 42, 'full'), mixture_stream(rows, 42, 'off')
    counts, thirds = Counter(), Counter()
    for _ in range(steps):
        for a, b in zip(next(full), next(off), strict=True):
            kind = 'teacher' if a.get('cf_retention') else 'fg_or_ordinary' if a.get('fg_correction') else 'expert'
            counts[kind, source_task(a)] += 1
            if kind != 'teacher': thirds[kind, source_task(a), a['replay_trajectory_third']] += 1
            if a.get('fg_correction'):
                if not b.get('ordinary_cf_control') or any(a[k] != b[k] for k in ('pair_id','task_config','replay_trajectory_third')):
                    raise ValueError('Corrective controls do not match task/domain/temporal stratum.')
            elif a != b:
                raise ValueError('Common expert/teacher examples differ across arms.')
    return dict(expected_examples=12*steps, counts={'|'.join(k):v for k,v in counts.items()},
                temporal_thirds={'|'.join(map(str,k)):v for k,v in thirds.items()},
                matched_common_samples=True, matched_corrective_task_domain_thirds=True)


def admission(args):
    from experiments.robotwin.eraf_action_protocol import validate_joint_identity_audit
    root = args.output.resolve()
    if root.exists() or not root.is_relative_to(RUNS):
        raise ValueError('Use a new output on the server data disk.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Commit code before execution.')
    if not read(FORMAL/'completion_audit.json')['complete'] or read(FORMAL/'status.json')['episodes'] != 600:
        raise ValueError('The prior formal600 audit must be preserved and complete.')
    source = read(SOURCE/'protocol.json')
    if sha(source['manifest']) != source['manifest_sha256']:
        raise ValueError('Training bank changed.')
    rows = read(source['manifest'])['states']
    arms = {a: dict(eraf=source['arms'][a]['eraf'], fg=source['arms'][a]['fg'], parent=source['arms'][a]['parent'])
            for a in ('no_eraf','eraf_only','eraf_fg')}
    for a, gpus in zip(arms, ([0],[1,2],[3,4]), strict=True): arms[a]['gpus'] = gpus
    bindings = {source['manifest']:sha(source['manifest']), str(SOURCE/'protocol.json'):sha(SOURCE/'protocol.json')}
    for arm, spec in arms.items():
        bindings[spec['parent']] = sha(spec['parent'])
        if spec['eraf'] == 'on':
            mode = 'fg' if arm == 'eraf_fg' else 'ordinary'
            path = str(Path(source['identity_root'])/mode/'summary.json')
            validate_joint_identity_audit(read(path), checkpoint_sha256=bindings[spec['parent']], manifest_sha256=source['manifest_sha256'])
            spec['identity_audit'] = path; bindings[path] = sha(path)
    if bindings[arms['no_eraf']['parent']] != source['source_policy_sha256']:
        raise ValueError('Common warm action parent changed.')
    import torch
    common = torch.load(arms['no_eraf']['parent'], map_location='cpu', weights_only=False)
    for arm in ('eraf_only', 'eraf_fg'):
        other = torch.load(arms[arm]['parent'], map_location='cpu', weights_only=False)
        if common['mot_trainable'].keys() != other['mot_trainable'].keys() or any(
                not torch.equal(value, other['mot_trainable'][key]) for key, value in common['mot_trainable'].items()):
            raise ValueError('Action/video adapter initialization differs across arms: '+arm)
        del other
    del common
    manifest = read(source['manifest'])
    for key in ('original_train_config','stats_path','base_checkpoint'):
        bindings[manifest[key]] = sha(manifest[key])
    if bindings[manifest['base_checkpoint']] != manifest['base_checkpoint_sha256']:
        raise ValueError('Released base changed.')
    exclusions, seeds = [], set()
    for directory, dirs, names in os.walk(RUNS):
        dirs[:] = [d for d in dirs if d not in ('weights','optimizers','.git','wandb')]
        for name in names:
            if name not in ('manifest.json','episodes.jsonl'): continue
            path = Path(directory)/name; data = path.read_bytes()
            if not data.strip(): continue
            value = [json.loads(x) for x in data.splitlines() if x.strip()] if name.endswith('jsonl') else json.loads(data)
            rr = value.get('states') if isinstance(value,dict) else value
            if not isinstance(rr,list) or not rr or not all(type(r.get('scene_seed')) is int for r in rr): continue
            seeds.update(r['scene_seed'] for r in rr)
            exclusions.append(dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),records=len(rr)))
    starts = {t:args.dev_seed+1000*i for i,t in enumerate(TASKS)}
    if any(start <= s < start+300 for start in starts.values() for s in seeds):
        raise ValueError('Proposed development namespace already used. Do not resample automatically.')
    from experiments.robotwin.catalog_protocol import validate_namespace
    for start in starts.values(): validate_namespace('dev',start,300)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('A GPU compute process is already running.')
    devices = subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True)
    if not set(range(5)) <= {int(x) for x in devices.splitlines()}: raise ValueError('Five GPUs required.')
    if shutil.disk_usage(RUNS).free < 12*1024**3: raise ValueError('Need 12GiB free data-disk space.')
    config = REPO/'configs/eval/robotwin_cis_ten_tasks.json'; bindings[str(config)] = sha(config)
    plan = dict(format='robotwin_five_task_expert_trial_v1',complete=False,status='admitted',jobs={},
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        output=str(root),source_trial=str(SOURCE),prior_formal=str(FORMAL),manifest=source['manifest'],
        source_bank=source['source_bank'],strongest_checkpoint=source['strongest_checkpoint'],
        arms=arms,tasks=list(TASKS),fg_tasks=list(FG_TASKS),steps=args.steps,seed=42,
        action_objective=OBJECTIVE,recipe='4 unit-weight old-policy anchors + 4 five-task target expert examples + 4 two-task FG or matched ordinary target examples; temporal thirds balanced.',
        semantic_training_this_trial=False,shared_action_initialization_verified_by_source_identity=True,
        seed_starts=starts,dev_episodes=args.dev_episodes,deadline=args.deadline,input_sha256=bindings,
        exclusions=exclusions,excluded_unique_seeds=len(seeds),expected_schedule=coverage(rows,args.steps),
        checkpoint_selection='Predeclared final step only; no partial-CF selection.',
        primary_metric='Equal-task macro CF on five tasks; all arms and failures retained.',
        independent_test=False,goal_achieved=False,correct_evaluated=False,platform_shutdown=None,
        next_stage='Only a development improvement can nominate frozen models for a new 40-scene-per-task independent test.',
        model_storage='server_only',smoke_only=args.steps==2)
    from experiments.robotwin.current_noeraf import current_baseline, branch_parent
    current = args.current_no_eraf_trial.resolve()
    ledger, audit, current_protocol = (read(current/n) for n in
                                      ('final_models.json','completion_audit.json','protocol.json'))
    baseline, baseline_sha = current_baseline(current, ledger, audit)
    if sha(baseline) != baseline_sha:
        raise ValueError('Current final no-ERAF weights changed.')
    if current_protocol['manifest'] != plan['manifest']:
        raise ValueError('Current baseline and branch training banks differ.')
    policy = torch.load(baseline, map_location='cpu', weights_only=False)
    for arm, fg in (('eraf_only','off'),('eraf_fg','full')):
        donor_path = current_protocol['arms'][arm]['parent']
        donor_sha = sha(donor_path)
        if donor_sha != current_protocol['input_sha256'][donor_path]:
            raise ValueError('Current source semantic donor changed.')
        donor = torch.load(donor_path, map_location='cpu', weights_only=False)
        # Preflight validates actual tensor compatibility without creating model files.
        candidate = branch_parent(policy, donor, policy_path=baseline, policy_sha256=baseline_sha,
            semantic_path=donor_path, semantic_sha256=donor_sha, fg=fg)
        del donor, candidate
        bindings[donor_path] = donor_sha
        arms[arm].update(parent=str(root/'initialization'/(arm+'.pt')),
            identity_audit=str(root/'identity'/arm/'summary.json'), semantic_donor=donor_path)
    del policy
    arms['no_eraf'].update(parent=baseline, final_checkpoint=baseline, final_sha256=baseline_sha,
        frozen_reference=True, additional_optimizer_steps=0, gpus=[])
    for n in ('final_models.json','completion_audit.json','protocol.json'):
        bindings[str(current/n)] = sha(current/n)
    bindings[baseline] = baseline_sha
    plan.update(format='robotwin_current_no_eraf_branch_trial_v1',
        current_no_eraf_trial=str(current), baseline_checkpoint=baseline, baseline_sha256=baseline_sha,
        strongest_checkpoint=baseline, train_arms=['eraf_only','eraf_fg'],
        additional_optimizer_steps={'no_eraf':0,'eraf_only':args.steps,'eraf_fg':args.steps},
        actual_parent_adapter_tensors_equal=False, shared_action_initialization_verified_by_source_identity=False,
        initialization='All action/video adapters from current final no-eraf; separate existing semantic guards are initialization only; no semantic donor policy adapters.',
        comparison_scope='Incremental ERAF/FG additions to a fixed completed baseline; extra training is not matched to the baseline.',
        training_identity_audit_required=True)
    return plan, seeds


def summarize(root, plan):
    from fractions import Fraction
    from scripts.report_robotwin_formal_five40 import summarize_cell
    cells, signatures = [], {}
    for arm, spec in plan['arms'].items():
        for task in TASKS:
            folder = root/'evaluation'/arm/task/'demo_clean/counterfactual'
            complete = read(folder/'complete.json'); eps = records(folder/'episodes.jsonl')
            canonical_path=root/'catalog'/task/'demo_clean/correct/episodes.jsonl'; canonical=records(canonical_path)
            initial=read(folder/'initial_states.json')
            if not complete['complete'] or len(eps)!=plan['dev_episodes'] or len(canonical)!=len(eps) or len(initial)!=len(eps): raise ValueError('Incomplete development cell.')
            if complete['checkpoint_sha256']!=spec['final_sha256'] or complete['canonical_sha256']!=sha(canonical_path): raise ValueError('Model/catalog binding changed.')
            if complete['eraf']!=spec['eraf'] or complete['memory_mode']!='carry' or not complete['manipulation_metrics']: raise ValueError('Inference protocol changed.')
            if complete['deployment']!=dict(action_horizon=32,replan_steps=24,inference_steps=10): raise ValueError('Deployment differs.')
            for e,c,h in zip(eps,canonical,initial,strict=True):
                if any(e[k]!=c[k] for k in ('episode_index','scene_seed','source_instruction','counterfactual_instruction')): raise ValueError('Scene or instruction changed.')
                if e['condition']!='counterfactual' or e['instruction_goal']!='counterfactual' or e['selected_goal']!='counterfactual': raise ValueError('Wrong evaluated goal.')
                if h!=dict(scene_seed=c['scene_seed'],sha256=c['initial_physical_state_sha256']): raise ValueError('Physical state changed.')
            if task in signatures and initial!=signatures[task]: raise ValueError('Unpaired physical scenes.')
            signatures[task]=initial;cells.append(dict(arm=arm,task=task,**summarize_cell(eps)))
    macro={a:sum(Fraction(c['cf'],plan['dev_episodes']) for c in cells if c['arm']==a)/5 for a in plan['arms']}
    return dict(complete=True,episodes=3*5*plan['dev_episodes'],cells=cells,macro_cf={a:float(v) for a,v in macro.items()},
                desired_order_on_dev=macro['eraf_fg']>macro['eraf_only']>macro['no_eraf'],
                independent_test=False,goal_achieved=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--deadline',required=True)
    ap.add_argument('--current-no-eraf-trial',type=Path,default=CURRENT,
                    help='Completed trial whose audited final no-eraf is the fixed action parent.')
    ap.add_argument('--steps',type=int,choices=[2,200],default=200)
    ap.add_argument('--dev-episodes',type=int,default=12);ap.add_argument('--dev-seed',type=int,default=91385000)
    ap.add_argument('--preflight-only',action='store_true')
    args=ap.parse_args();cutoff=datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or cutoff.timestamp()<=time.time() or args.dev_episodes<1: ap.error('Future absolute deadline and positive episode count required.')
    plan,seeds=admission(args)
    if args.preflight_only: print(json.dumps(plan,indent=2));return
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    write(root/'protocol.json',plan);write(root/'exclusion_seeds.json',[dict(scene_seed=s) for s in sorted(seeds)])
    processes={};env=os.environ|dict(PATH='/opt/conda/bin:'+os.environ['PATH'],PYTHONPATH=str(REPO/'src')+':'+str(REPO),
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def save(): write(root/'status.json',plan)
    def budget():
        if time.time()>=cutoff.timestamp(): raise TimeoutError('Own work deadline reached; platform stays on.')
        if shutil.disk_usage(root).free<3*1024**3: raise RuntimeError('Data-disk reserve reached.')
    def launch(name,cmd,gpus):
        budget()
        with (root/(name+'.log')).open('x') as log:
            p=subprocess.Popen(cmd,cwd=REPO,env=env|{'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus))},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[name]=p;plan['jobs'][name]=dict(pid=p.pid,gpus=gpus,command=cmd,
            started_at=datetime.now().astimezone().isoformat());save()
        print('[start]',name,p.pid,flush=True)
    def poll(name):
        rc=processes[name].poll()
        if rc is None:return False
        plan['jobs'][name]['exit_code']=rc;save()
        if rc:raise RuntimeError(name+' failed: '+str(rc))
        return True
    def group(jobs):
        for name,cmd,gpus in jobs:launch(name,cmd,gpus)
        active={name for name,_,_ in jobs}
        while active:
            budget();active={n for n in active if not poll(n)}
            if active:time.sleep(5)
    def queue(jobs):
        pending=list(jobs);active={}
        while pending or active:
            budget()
            for gpu,name in list(active.items()):
                if poll(name):del active[gpu]
            for gpu in range(5):
                if gpu in active or not pending:continue
                name,cmd=pending.pop(0);launch(name,cmd+['--gpu',str(gpu)],[gpu]);active[gpu]=name
            if active:time.sleep(5)
    def stop(sig,frame):raise InterruptedError('Signal '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        plan['status']='initializing_on_current_no_eraf';save()
        group([('prepare_current_parents',[sys.executable,str(REPO/'scripts/prepare_robotwin_current_noeraf.py'),
            '--source-trial',plan['current_no_eraf_trial'],'--output',str(root/'initialization')],[])])
        prepared=read(root/'initialization/preparation.json')
        if not prepared['complete'] or prepared['baseline_sha256']!=plan['baseline_sha256']:
            raise ValueError('Branch preparation did not bind the current baseline.')
        for arm in plan['train_arms']:
            spec=plan['arms'][arm];proof=prepared['arms'][arm]
            if proof['path']!=spec['parent'] or not proof['policy_adapters_equal'] or proof['baseline_sha256']!=plan['baseline_sha256']:
                raise ValueError('An ERAF branch has the wrong action parent.')
            if sha(spec['parent'])!=proof['sha256']:
                raise ValueError('Prepared parent checkpoint changed.')
            plan['input_sha256'][spec['parent']]=proof['sha256']
        group([('identity_'+a,[sys.executable,str(REPO/'scripts/probe_robotwin_eraf_context_tokens.py'),
            '--manifest',plan['manifest'],'--source-bank',plan['source_bank'],'--checkpoint',plan['arms'][a]['parent'],
            '--output',str(root/'identity'/a),'--require-residual-identity','--include-expanded-tasks'],
            [plan['arms'][a]['gpus'][0]]) for a in plan['train_arms']])
        from experiments.robotwin.eraf_action_protocol import validate_joint_identity_audit
        for a in plan['train_arms']:
            spec=plan['arms'][a]
            validate_joint_identity_audit(read(spec['identity_audit']),checkpoint_sha256=sha(spec['parent']),
                manifest_sha256=sha(plan['manifest']))
            plan['input_sha256'][spec['identity_audit']]=sha(spec['identity_audit'])
        plan.update(actual_parent_adapter_tensors_equal=True,
            shared_action_initialization_verified_by_source_identity=True)
        plan['status']='training';save()
        group([('train_'+a,training_command(plan,root,a),plan['arms'][a]['gpus']) for a in plan['train_arms']])
        plan['status']='auditing_training';save()
        group([('audit_'+a,[sys.executable,str(REPO/'scripts/audit_robotwin_eraf_fg_action_stage.py'),'--output',str(root/a/'joint')],[]) for a in plan['train_arms']])
        for arm in plan['train_arms']:
            spec=plan['arms'][arm]
            spec['final_checkpoint']=str(root/arm/'joint'/f"step_{args.steps:06d}.pt");spec['final_sha256']=sha(spec['final_checkpoint'])
        write(root/'final_models.json',plan['arms']);save()
        if not plan['smoke_only']:
            plan['status']='screening_dev_scenes';save()
            queue([('catalog_'+t,[sys.executable,str(REPO/'scripts/catalog_robotwin_formal_five.py'),'--task',t,'--output',str(root/'catalog'),
                '--start-seed',str(plan['seed_starts'][t]),'--episodes',str(args.dev_episodes),'--max-attempts','300','--split','dev',
                '--exclude-records',str(root/'exclusion_seeds.json'),'--deadline',args.deadline]) for t in TASKS])
            write(root/'catalog_frozen.json',{t:sha(root/'catalog'/t/'demo_clean/correct/episodes.jsonl') for t in TASKS})
            plan['status']='evaluating_dev';save()
            queue([('eval_'+a+'_'+t,[sys.executable,str(REPO/'scripts/eval_robotwin_eraf_fg.py'),'worker','--output',str(root/'evaluation'/a),
                '--manifest',plan['manifest'],'--checkpoint',s['final_checkpoint'],'--catalog-root',str(root/'catalog'),
                '--interventions',str(REPO/'configs/eval/robotwin_cis_ten_tasks.json'),'--tasks',t,'--conditions','counterfactual',
                '--episodes',str(args.dev_episodes),'--eraf',s['eraf'],'--policy-kind','repair','--memory-mode','carry','--manipulation-metrics','--videos'])
                for t in TASKS for a,s in plan['arms'].items()])
            result=summarize(root,plan);write(root/'comparison.json',result)
        for path,digest in plan['input_sha256'].items():
            if sha(path)!=digest:raise ValueError('Actual input changed: '+path)
        for spec in plan['arms'].values():
            if sha(spec['final_checkpoint'])!=spec['final_sha256']:raise ValueError('Final model changed.')
        plan.update(complete=True,status='complete',finished_at=datetime.now().astimezone().isoformat());save()
        write(root/'terminal_verification.json',dict(complete=True,additional_optimizer_steps=plan['additional_optimizer_steps'],
            episodes=0 if plan['smoke_only'] else 3*5*args.dev_episodes,all_jobs_exit_zero=True,inputs_unchanged=True,
            independent_test=False,goal_achieved=False))
    except BaseException as error:
        plan.update(status='stopped',error=repr(error));raise
    finally:
        for p in processes.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes.values():
            if p.poll() is None:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
        for name,p in processes.items():plan['jobs'][name]['exit_code']=p.poll()
        save()


if __name__=='__main__':main()
