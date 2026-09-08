#!/usr/bin/env python3
"""Freeze matched200 models, screen fresh scenes, then run 600 CF episodes."""
from __future__ import annotations
import argparse
import csv
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
sys.path[:0] = [str(REPO), str(REPO/'src')]
RUNS = Path('/root/gpufree-data/LF-FastWAM/runs')
BT = RUNS/'robotwin_expanded_fg/20260908-balanced-target200'
TASKS = ['blocks_ranking_rgb', 'stack_blocks_two', 'place_a2b_left', 'place_a2b_right', 'place_burger_fries']
MODEL_HASHES = dict(no_eraf='6f5b5528582b52d4cde2b486dcfcdc50ac816cd36632fbde1956ffbbc698c5aa',
                   eraf_only='fc7a793939978e6d21d371724c0edee5798dfe01f529193d773fc2be6a830cde',
                   eraf_fg='1e65c7fdcf946698172fc45cb5f9812d51c93c9e5aa854c9fc37eb907269ecb6')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def records(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def freeze(root, deadline, gpus):
    from experiments.robotwin.manipulation_metrics import PROTOCOL
    if root.exists() or not root.is_relative_to(RUNS):
        raise ValueError('A fresh server data-disk output is required')
    if shutil.disk_usage(RUNS).free < 10*1024**3:
        raise RuntimeError('Less than 10GiB free for videos and journals')
    devices = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    free = {int(line.split(',')[0]): int(line.split(',')[1]) for line in devices.splitlines()}
    if len(set(gpus)) != len(gpus) or any(gpu not in free or free[gpu] > 1000 for gpu in gpus):
        raise RuntimeError('Selected GPUs are missing or already occupied')
    source = read(BT/'protocol.json')
    models = {}
    bindings = {source['manifest']: sha(source['manifest']), str(BT/'protocol.json'): sha(BT/'protocol.json')}
    if bindings[source['manifest']] != source['manifest_sha256']:
        raise ValueError('Training manifest changed')
    for arm, expected in MODEL_HASHES.items():
        path = BT/arm/'joint/step_000200.pt'
        if sha(path) != expected:
            raise ValueError('Selected model hash mismatch: '+arm)
        models[arm] = dict(checkpoint=str(path), sha256=expected, eraf='off' if arm == 'no_eraf' else 'on')
        bindings[str(path)] = expected
    manifest = read(source['manifest'])
    for field in ('original_train_config', 'stats_path', 'base_checkpoint'):
        bindings[manifest[field]] = sha(manifest[field])
    if bindings[manifest['base_checkpoint']] != manifest['base_checkpoint_sha256']:
        raise ValueError('Released base checkpoint changed')
    # Scan archived scene manifests/catalogs, never weights or training streams.
    # Explicitly record which files were inspected instead of claiming arbitrary
    # external or original released-model training data has been audited.
    exclusions, all_seeds = [], set()
    for directory, dirs, files in os.walk(RUNS):
        dirs[:] = [d for d in dirs if d not in ('weights', 'optimizers', '.git', 'wandb')]
        for name in files:
            if name not in ('manifest.json', 'episodes.jsonl'):
                continue
            path = Path(directory)/name
            content = path.read_bytes()
            if not content.strip():
                continue
            value = [json.loads(x) for x in content.splitlines() if x.strip()] if name.endswith('jsonl') else json.loads(content)
            rows = value.get('states') if isinstance(value, dict) else value
            if not isinstance(rows, list) or not rows or not all(type(x.get('scene_seed')) is int for x in rows):
                continue
            seeds = {x['scene_seed'] for x in rows}
            if any(91750000 <= seed < 91755000 for seed in seeds):
                raise ValueError('Proposed fresh namespace was already accessed: '+str(path))
            all_seeds.update(seeds)
            exclusions.append(dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(), records=len(rows)))
    if source['manifest'] not in {x['path'] for x in exclusions}:
        raise ValueError('Current training records were not excluded')
    for group in source['groups'].values():
        for task in group['tasks']:
            path = str(Path(group['catalog'])/task/'demo_clean/correct/episodes.jsonl')
            if path not in {x['path'] for x in exclusions}:
                raise ValueError('Current development catalog was not excluded: '+path)
    config = REPO/'configs/eval/robotwin_cis_ten_tasks.json'
    bindings[str(config)] = sha(config)
    return dict(format='robotwin_formal_five40_v1', complete=False, status='frozen_before_test',
                frozen_at=datetime.now().astimezone().isoformat(),
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                output=str(root), deadline=deadline, gpus=gpus, models=models, tasks=TASKS,
                episodes_per_task_per_model=40, expected_episodes=600, conditions=['counterfactual'],
                manifest=source['manifest'], input_sha256=bindings, exclusions=exclusions,
                excluded_unique_seeds=len(all_seeds), seed_starts={t:91750000+1000*i for i,t in enumerate(TASKS)},
                metrics=PROTOCOL, primary_metric='equal-task five-task macro CF; 40 fixed scenes per task',
                checkpoint_selection='User-selected balanced-target200 matched three arms; no test selection',
                task_selection='User excludes the five all-zero DEV tasks before this fresh evaluation',
                scope='New physical scenes in these five task families; no claim of unseen tasks or assets',
                training_performed=False, correct_evaluated=False, platform_shutdown=None,
                model_storage='server_only', jobs={})


def summarize(root, plan):
    rows, object_rows, cells = [], [], []
    signatures = {}
    for arm, model in plan['models'].items():
        for task in TASKS:
            base = root/'evaluation'/arm/task/'demo_clean/counterfactual'
            complete = read(base/'complete.json')
            canonical_path = root/'catalog'/task/'demo_clean/correct/episodes.jsonl'
            canonical = records(canonical_path)
            episodes = records(base/'episodes.jsonl')
            hashes = read(base/'initial_states.json')
            if len(episodes) != 40 or len(canonical) != 40 or len(hashes) != 40 or not complete['complete']:
                raise ValueError('Formal cell incomplete')
            if complete['checkpoint_sha256'] != model['sha256'] or complete['canonical_sha256'] != sha(canonical_path):
                raise ValueError('Cell bindings changed')
            if complete['eraf'] != model['eraf'] or not complete['manipulation_metrics']:
                raise ValueError('Cell deployment mode changed')
            selected, picked, placed, all_picked, strict = 0, 0, 0, 0, 0
            signature = []
            for i, (episode, canon, initial) in enumerate(zip(episodes, canonical, hashes, strict=True)):
                for field in ('scene_seed', 'source_instruction', 'counterfactual_instruction', 'episode_index'):
                    if episode[field] != canon[field]:
                        raise ValueError('Canonical scene/instruction mismatch')
                if initial != dict(scene_seed=canon['scene_seed'], sha256=canon['initial_physical_state_sha256']):
                    raise ValueError('Physical initial state mismatch')
                m = episode['manipulation_metrics']
                if m['physics_samples'] <= 0:
                    raise ValueError('No physical metric samples')
                success = bool(episode['counterfactual_goal_ever_success'])
                lift = bool(m['any_correct_object_lifted'])
                after = bool(m['correct_placement_after_lift'])
                slots = m['final_strict_slot_assignment']
                strict_final = all(slots.values()) if slots is not None else None
                row = dict(arm=arm,task=task,episode_index=i,scene_seed=canon['scene_seed'],
                           cf_success=success,cf_final_success=episode['counterfactual_goal_final_success'],
                           correctly_lifted_any=lift,correctly_lifted_all=m['all_instruction_objects_lifted'],
                           correct_placement_after_lift=after,full_goal_after_lift=m['full_goal_after_any_lift'],
                           any_object_placed_after_lift=m['any_object_placed_after_lift'],
                           strict_slots_final=strict_final,physics_samples=m['physics_samples'])
                rows.append(row)
                for obj, values in m['objects'].items():
                    object_rows.append(dict(arm=arm,task=task,episode_index=i,scene_seed=canon['scene_seed'],actor=obj,
                                           **{k:v for k,v in values.items() if k != 'lift_contact_links'}))
                if after and not (success and lift):
                    raise ValueError('Inconsistent lift/goal temporal attribution')
                selected += success; picked += lift; placed += after
                all_picked += m['all_instruction_objects_lifted']; strict += bool(strict_final)
                signature.append(tuple(episode[k] for k in ('scene_seed','source_instruction','counterfactual_instruction')))
            if task in signatures and signatures[task] != signature:
                raise ValueError('Unpaired model evaluation')
            signatures[task] = signature
            cells.append(dict(arm=arm,task=task,episodes=40,cf_successes=selected,cf_rate=selected/40,
                              correct_lift_episodes=picked,correct_lift_rate=picked/40,
                              all_objects_lifted_episodes=all_picked,
                              correct_placement_after_lift_episodes=placed,
                              placement_given_lift_rate=placed/picked if picked else None,
                              strict_slots_final_successes=strict if task == 'place_burger_fries' else None))
    for filename, data in [('episodes.csv',rows),('objects.csv',object_rows),('cells.csv',cells)]:
        with (root/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    paired = []
    for a,b in [('no_eraf','eraf_only'),('no_eraf','eraf_fg'),('eraf_only','eraf_fg')]:
        for task in TASKS:
            aa=[r for r in rows if r['arm']==a and r['task']==task]
            bb=[r for r in rows if r['arm']==b and r['task']==task]
            paired.append(dict(reference=a,candidate=b,task=task,
                               gains=sum(not x['cf_success'] and y['cf_success'] for x,y in zip(aa,bb)),
                               losses=sum(x['cf_success'] and not y['cf_success'] for x,y in zip(aa,bb))))
    result=dict(complete=True,episodes=len(rows),cells=cells,paired_comparisons=paired,
                macro_cf={arm:sum(c['cf_rate'] for c in cells if c['arm']==arm)/5 for arm in plan['models']},
                metric_note='Placement denominator is episodes with any verified correct-object lift. Placement requires release, all goal relations, table height (or stack height), stack base at declared center, burger nearest slot and functional-point height. Original CF and CF-after-lift are also retained. No added settling.')
    write(root/'comparison.json',result)
    return result


def main(*, freeze_fn=None, parser_setup=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--deadline',required=True)
    ap.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3,4])
    ap.add_argument('--preflight-only',action='store_true')
    if parser_setup is not None: parser_setup(ap)
    args=ap.parse_args();root=args.output.resolve()
    cutoff=datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or cutoff.timestamp() <= time.time():
        raise ValueError('Need a future absolute deadline')
    plan=freeze(root,args.deadline,args.gpus) if freeze_fn is None else freeze_fn(args)
    if args.preflight_only:
        print(json.dumps(plan,indent=2));return
    root.mkdir(parents=True,exist_ok=False)
    write(root/'frozen_protocol.json',plan)
    write(root/'exclusion_seeds.json',[dict(scene_seed=s) for s in sorted({row['scene_seed'] for receipt in plan['exclusions']
          for row in (records(receipt['path']) if receipt['path'].endswith('.jsonl') else read(receipt['path'])['states'])})])
    processes={}
    env=dict(os.environ,PATH='/opt/conda/bin:'+os.environ['PATH'],PYTHONPATH=str(REPO/'src')+':'+str(REPO),
             DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
             VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def save(): write(root/'status.json',plan)
    def budget():
        if time.time() >= cutoff.timestamp(): raise TimeoutError('Evaluation cutoff reached; stop owned jobs only')
        if shutil.disk_usage(root).free < 3*1024**3: raise RuntimeError('Evaluation disk floor reached')
    def run_queue(jobs):
        queue=list(jobs);running={}
        while queue or running:
            budget()
            for gpu,name in list(running.items()):
                rc=processes[name].poll()
                if rc is not None:
                    plan['jobs'][name].update(exit_code=rc,finished_at=datetime.now().astimezone().isoformat());save()
                    if rc: raise RuntimeError(name+' failed; inspect its log')
                    del running[gpu]
            for gpu in args.gpus:
                if gpu in running or not queue: continue
                name,command=queue.pop(0)
                cmd=[sys.executable,'-u',str(REPO/command[0]),*command[1:],'--gpu',str(gpu)]
                with (root/(name+'.log')).open('x') as log:
                    p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes[name]=p;running[gpu]=name
                plan['jobs'][name]=dict(pid=p.pid,gpu=gpu,command=cmd,started_at=datetime.now().astimezone().isoformat());save()
            if running: time.sleep(5)
    def stop(signum,frame): raise InterruptedError('User/process signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        plan['status']='checking_physical_metric_apis';save()
        (root/'smoke').mkdir()
        run_queue([('smoke_'+t,['scripts/smoke_robotwin_manipulation_metrics.py','--task',t,'--output',str(root/'smoke'/f'{t}.json'),
                              '--seed',str(91690000+100*i)]) for i,t in enumerate(TASKS)])
        if not all(read(root/'smoke'/f'{t}.json')['complete'] for t in TASKS): raise ValueError('Smoke incomplete')
        plan['status']='screening_fresh_scenes';save()
        run_queue([('catalog_'+t,['scripts/catalog_robotwin_formal_five.py','--task',t,'--output',str(root/'catalog'),
                                '--start-seed',str(plan['seed_starts'][t]),'--episodes','40','--max-attempts','400',
                                '--split','test','--exclude-records',str(root/'exclusion_seeds.json'),'--deadline',args.deadline]) for t in TASKS])
        catalog_hashes={t:sha(root/'catalog'/t/'demo_clean/correct/episodes.jsonl') for t in TASKS}
        write(root/'catalog_frozen.json',dict(complete=True,episodes=200,sha256=catalog_hashes,
                                             frozen_at=datetime.now().astimezone().isoformat()))
        plan['status']='evaluating_fixed_models';save()
        jobs=[]
        for task in TASKS:
            for arm,model in plan['models'].items():
                jobs.append(('eval_'+arm+'_'+task,['scripts/eval_robotwin_eraf_fg.py','worker','--output',str(root/'evaluation'/arm),
                    '--manifest',plan['manifest'],'--checkpoint',model['checkpoint'],'--catalog-root',str(root/'catalog'),
                    '--interventions',str(REPO/'configs/eval/robotwin_cis_ten_tasks.json'),'--tasks',task,
                    '--conditions','counterfactual','--episodes','40','--eraf',model['eraf'],'--policy-kind','repair',
                    '--memory-mode','carry','--manipulation-metrics','--videos']))
        run_queue(jobs)
        for path,digest in plan['input_sha256'].items():
            if sha(path)!=digest: raise ValueError('Input changed during formal evaluation: '+path)
        for task,digest in catalog_hashes.items():
            if sha(root/'catalog'/task/'demo_clean/correct/episodes.jsonl')!=digest: raise ValueError('Catalog changed')
        result=summarize(root,plan)
        plan.update(complete=True,status='complete',finished_at=datetime.now().astimezone().isoformat(),episodes=result['episodes']);save()
        write(root/'terminal_verification.json',dict(complete=True,episodes=600,inputs_unchanged=True,catalogs_unchanged=True,
            all_workers_exited_zero=True,verified_at=datetime.now().astimezone().isoformat()))
    except BaseException as error:
        plan.update(status='stopped',error=repr(error));raise
    finally:
        for p in processes.values():
            if p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
        for p in processes.values():
            if p.poll() is None:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
        save()


if __name__=='__main__':main()
