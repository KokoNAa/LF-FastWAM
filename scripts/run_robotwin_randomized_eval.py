#!/usr/bin/env python3
"""Evaluate the frozen latest three models on a fresh randomized CF catalog."""
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
from scripts.run_robotwin_formal_five40 import read, write, sha, records
from scripts.report_robotwin_five_task_comparison import report, markdown, TASKS, ARMS

DOMAIN = 'demo_randomized'
EXPECTED = dict(no_eraf='e9ebee0cc0b1c532548b0d71444a259c617e6c833005dbee0b0e653a10157540',
    eraf_only='26aea72064d7c17c39992ce2ece74ae63a7496d13f0931bcee3e2ba39e98fdcd',
    eraf_fg='47a6198755ea58bf33ea0e41a5e5b193bb534c2a6aa3b3449588baf19a182fd9')
RUNS = Path('/root/gpufree-data/LF-FastWAM/runs')
SIM = RUNS.parent / 'third_party/RoboTwin'


def verify_domain(actual, expected):
    if actual != expected or not all(actual.get(k) is True for k in
            ('random_background', 'cluttered_table', 'random_light')):
        raise ValueError('The frozen randomized environment is not active.')


def freeze(source, root, gpus):
    import yaml
    from experiments.robotwin.catalog_protocol import validate_namespace
    if root.exists() or root.parent != RUNS / 'robotwin_five_task_repair':
        raise ValueError('A fresh server data-disk directory is required.')
    proof, prior = read(source/'completion_audit.json'), read(source/'protocol.json')
    if not proof['complete'] or not read(source/'status.json')['complete']:
        raise ValueError('The selected source trial must be fully complete.')
    models = read(source/'final_models.json')
    if {a:m['final_sha256'] for a,m in models.items()} != EXPECTED:
        raise ValueError('Use exactly the latest preselected three models.')
    source_plan = read(Path(prior['source_trial'])/'protocol.json')
    manifest = source_plan['manifest']
    config = SIM/'task_config'/f'{DOMAIN}.yml'
    randomization = yaml.safe_load(config.read_text())['domain_randomization']
    verify_domain(randomization, randomization)
    paths = [source/n for n in ('completion_audit.json','protocol.json','final_models.json')]
    paths += [Path(manifest), config, SIM/'task_config/demo_clean.yml',
              REPO/'configs/eval/robotwin_cis_ten_tasks.json', Path(prior['source_trial'])/'protocol.json']
    bindings = {str(p):sha(p) for p in paths}
    for arm, model in models.items():
        if sha(model['final_checkpoint']) != EXPECTED[arm]: raise ValueError('Checkpoint changed: '+arm)
        bindings[model['final_checkpoint']] = EXPECTED[arm]
    git = lambda *cmd: subprocess.check_output(['git',*cmd],cwd=REPO,text=True).strip()
    if git('status','--porcelain'): raise ValueError('Commit evaluation code first.')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPUs are occupied.')
    available = {int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines()}
    if len(gpus) != len(set(gpus)) or not set(gpus) <= available: raise ValueError('Invalid GPUs.')
    if shutil.disk_usage(RUNS).free < 10*1024**3: raise ValueError('Need 10 GiB free for outputs.')
    starts = {task:91305000+1000*i for i,task in enumerate(TASKS)}
    for start in starts.values(): validate_namespace('dev',start,400)
    excluded, receipts = set(), []
    for directory, dirs, names in os.walk(RUNS):
        dirs[:] = [d for d in dirs if d not in ('weights','optimizers','.git','wandb')]
        for name in names:
            if name not in ('manifest.json','episodes.jsonl'): continue
            p = Path(directory)/name
            value = records(p) if name.endswith('jsonl') else read(p)
            rows = value.get('states') if isinstance(value,dict) else value
            if not isinstance(rows,list) or not rows or not all(isinstance(r,dict) and type(r.get('scene_seed')) is int for r in rows): continue
            seeds = {r['scene_seed'] for r in rows}
            if any(start <= s < start+400 for start in starts.values() for s in seeds):
                raise ValueError('Declared namespace already used; do not automatically resample: '+str(p))
            excluded.update(seeds)
            receipts.append(dict(path=str(p),sha256=sha(p),records=len(rows)))
    return dict(format='robotwin_randomized_five_task_eval_v1', complete=False, status='frozen',
        code_commit=git('rev-parse','HEAD'), source_trial=str(source), arms=models, manifest=manifest,
        tasks=list(TASKS), task_config=DOMAIN, domain_randomization=randomization, gpus=gpus,
        dev_episodes=12, expected_episodes=180, conditions=['counterfactual'], seed_starts=starts,
        input_sha256=bindings, exclusions=receipts, excluded_unique_seeds=len(excluded),
        training_comparison=prior['training_comparison'], training_performed=False,
        independent_test=False, correct_evaluated=False, goal_achieved=False, jobs={},
        platform_shutdown=None, deadline=None, checkpoint_selection='Latest fixed three models; no outcome-based selection.',
        scope='Fresh randomized DEV scenes; all five tasks, 12 per model. Clean/random comparison is not scene-paired.',
        frozen_at=datetime.now().astimezone().isoformat()), excluded


def verify_outputs(root, plan):
    from experiments.robotwin.manipulation_metrics import PROTOCOL, actor_attributes
    pairs = {p['pair_id']:p for p in read(REPO/'configs/eval/robotwin_cis_ten_tasks.json')['pairs']}
    videos, observations = {}, {}
    config_hash = plan['input_sha256'][str(SIM/'task_config'/f'{DOMAIN}.yml')]
    for task in TASKS:
        catalog = root/'catalog'/task/DOMAIN/'correct'
        canonical = records(catalog/'episodes.jsonl')
        cat_proof = read(catalog/'catalog_complete.json')
        verify_domain(cat_proof['domain_randomization'],plan['domain_randomization'])
        if not cat_proof['complete'] or cat_proof['task_config'] != DOMAIN or len(canonical) != 12:
            raise ValueError('Incomplete randomized catalog.')
        reference_obs = None
        for arm in ARMS:
            folder = root/'evaluation'/arm/task/DOMAIN/'counterfactual'
            rows, cell = records(folder/'episodes.jsonl'), read(folder/'complete.json')
            verify_domain(cell['domain_randomization'],plan['domain_randomization'])
            if cell['task_config_sha256'] != config_hash: raise ValueError('Environment file changed.')
            obs_rows = [read(p) for p in sorted((root/'initial_observations'/arm/task).glob('*.json'))]
            observed = {x['metadata']['scene_seed']:x for x in obs_rows}
            if len(rows) != 12 or len(observed) != 12: raise ValueError('Missing episodes or initial observations.')
            signatures = []
            for e,c in zip(rows,canonical,strict=True):
                if e['task_config'] != DOMAIN or c['task_config'] != DOMAIN: raise ValueError('Wrong scene domain.')
                o = observed[e['scene_seed']]
                if any(o['metadata'][k] != e[k] for k in ('source_task','task_config','scene_seed','episode_index',
                        'source_instruction','counterfactual_instruction','policy_instruction','condition')):
                    raise ValueError('Observation audit does not match the evaluated episode.')
                if o['metadata']['checkpoint'] != plan['arms'][arm]['final_checkpoint']:
                    raise ValueError('Observation audit checkpoint differs.')
                signatures.append(dict(scene_seed=e['scene_seed'],observation_sha256=o['observation_sha256'],arrays=o['arrays']))
                m = e['manipulation_metrics']
                if m['protocol'] != PROTOCOL or m['physics_samples'] < 1 or set(m['objects']) != set(actor_attributes(pairs[e['pair_id']]['counterfactual_goal'])):
                    raise ValueError('Incomplete physical measurements.')
                for obj in m['objects'].values():
                    if obj['correctly_lifted'] and not (obj['first_lift_tick'] and len(obj['lift_contact_links']) >= 2):
                        raise ValueError('Missing lift evidence.')
                    if obj['placed_after_lift'] and not (obj['correctly_lifted'] and obj['first_placement_tick'] > obj['first_lift_tick']):
                        raise ValueError('Placement precedes lift.')
                v = Path(e['video_path'])
                if not v.exists() or v.parent != folder or str(v) in videos: raise ValueError('Missing/duplicate video.')
                probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
                    'format=duration:stream=codec_type,nb_frames','-of','json',str(v)],text=True))
                if float(probe['format']['duration']) <= 0 or not any(s['codec_type']=='video' and int(s['nb_frames'])>0 for s in probe['streams']):
                    raise ValueError('Invalid video.')
                videos[str(v)] = dict(sha256=sha(v),probe=probe)
            if reference_obs is not None and signatures != reference_obs: raise ValueError('Models saw different initial RGB/state.')
            reference_obs = signatures
        observations[task] = reference_obs
    return dict(complete=True, videos=videos, video_count=len(videos), matched_initial_observations=observations,
        initial_rgb_and_qpos_identical=True, task_config=DOMAIN,
        note='Physics-record and video integrity checks; not manual frame labeling.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-trial',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpus',type=int,nargs='+',default=[0,1,2])
    ap.add_argument('--preflight-only',action='store_true')
    args = ap.parse_args()
    root = args.output.resolve()
    plan, excluded = freeze(args.source_trial.resolve(),root,args.gpus)
    if args.preflight_only:
        print(json.dumps(plan,indent=2)); return
    root.mkdir(parents=True,exist_ok=False)
    write(root/'protocol.json',plan)
    write(root/'final_models.json',plan['arms'])
    write(root/'exclusion_seeds.json',[dict(scene_seed=s) for s in sorted(excluded)])
    processes = {}
    env = os.environ | dict(PATH='/opt/conda/bin:'+os.environ['PATH'],PYTHONPATH=str(REPO/'src')+':'+str(REPO),
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def save(): write(root/'status.json',plan)
    def queue(jobs):
        jobs, active = list(jobs), {}
        while jobs or active:
            if shutil.disk_usage(root).free < 3*1024**3: raise RuntimeError('Data-disk reserve reached.')
            for gpu,name in list(active.items()):
                rc = processes[name].poll()
                if rc is not None:
                    plan['jobs'][name].update(exit_code=rc,finished_at=datetime.now().astimezone().isoformat());save()
                    if rc: raise RuntimeError(name+' failed; inspect its log.')
                    del active[gpu]
            for gpu in args.gpus:
                if gpu in active or not jobs: continue
                name,command,audit_dir = jobs.pop(0)
                cmd = [sys.executable,'-u',str(REPO/command[0]),*command[1:],'--gpu',str(gpu)]
                job_env = env | {'CUDA_VISIBLE_DEVICES':str(gpu)}
                job_env.pop('FASTWAM_ROBOTWIN_INITIAL_OBSERVATION_AUDIT',None)
                if audit_dir: job_env['FASTWAM_ROBOTWIN_INITIAL_OBSERVATION_AUDIT'] = str(audit_dir)
                with (root/(name+'.log')).open('x') as log:
                    p = subprocess.Popen(cmd,cwd=REPO,env=job_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes[name],active[gpu] = p,name
                start = (Path('/proc')/str(p.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19]
                plan['jobs'][name] = dict(pid=p.pid,start_time=start,gpu=gpu,command=cmd,
                    started_at=datetime.now().astimezone().isoformat());save()
            if active: time.sleep(3)
    def stop(signum,frame): raise InterruptedError('Signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        plan['status']='screening_randomized_scenes';save()
        queue([('catalog_'+t,['scripts/catalog_robotwin_formal_five.py','--task',t,'--output',str(root/'catalog'),
            '--start-seed',str(plan['seed_starts'][t]),'--episodes','12','--max-attempts','400','--split','dev',
            '--exclude-records',str(root/'exclusion_seeds.json'),'--task-config',DOMAIN,'--no-deadline'],None) for t in TASKS])
        catalogs = {t:sha(root/'catalog'/t/DOMAIN/'correct/episodes.jsonl') for t in TASKS}
        write(root/'catalog_frozen.json',catalogs)
        plan['status']='evaluating_randomized_180';save()
        queue([('eval_'+a+'_'+t,['scripts/eval_robotwin_eraf_fg.py','worker','--output',str(root/'evaluation'/a),
            '--manifest',plan['manifest'],'--checkpoint',plan['arms'][a]['final_checkpoint'],'--catalog-root',str(root/'catalog'),
            '--interventions',str(REPO/'configs/eval/robotwin_cis_ten_tasks.json'),'--tasks',t,'--conditions','counterfactual',
            '--episodes','12','--eraf',plan['arms'][a]['eraf'],'--policy-kind','repair','--memory-mode','carry',
            '--task-config',DOMAIN,'--manipulation-metrics','--videos'],root/'initial_observations'/a/t) for t in TASKS for a in ARMS])
        plan['status']='verifying_complete_matrix';save()
        for path,digest in plan['input_sha256'].items():
            if sha(path) != digest: raise ValueError('Bound input changed: '+path)
        for t,digest in catalogs.items():
            if sha(root/'catalog'/t/DOMAIN/'correct/episodes.jsonl') != digest: raise ValueError('Catalog changed.')
        evidence = verify_outputs(root,plan)
        plan.update(complete=True,status='complete',finished_at=datetime.now().astimezone().isoformat());save()
        write(root/'terminal_verification.json',dict(complete=True,episodes=180,all_jobs_exit_zero=True,
            inputs_unchanged=True,catalogs_unchanged=True,initial_rgb_and_qpos_identical=True))
        result = report(root)
        (root/'paired_report').mkdir()
        write(root/'paired_report/PAIRED_COMPARISON.json',result)
        (root/'paired_report/PAIRED_COMPARISON.md').write_text(markdown(result))
        evidence.update(code_commit=plan['code_commit'],checkpoint_sha256=EXPECTED,
            paired_report_sha256=sha(root/'paired_report/PAIRED_COMPARISON.json'),
            verified_at=datetime.now().astimezone().isoformat(),independent_test=False,training_performed=False)
        write(root/'completion_audit.json',evidence)
        print(json.dumps(dict(complete=True,macro_cf=result['macro_cf'],episodes=180)),flush=True)
    except BaseException as error:
        plan.update(complete=False,status='stopped',error=repr(error));save();raise
    finally:
        for p in processes.values():
            if p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
        for p in processes.values():
            if p.poll() is None:
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()


if __name__ == '__main__': main()
