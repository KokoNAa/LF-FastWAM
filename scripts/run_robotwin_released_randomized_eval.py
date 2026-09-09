#!/usr/bin/env python3
"""Evaluate released FastWAM on the first ten scenes of the frozen random catalog."""
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
from scripts.report_robotwin_formal_five40 import TASK_NAMES, summarize_cell, frac

DOMAIN = 'demo_randomized'
N = 10
RUNS = Path('/root/gpufree-data/LF-FastWAM/runs/robotwin_five_task_repair')
SIM = RUNS.parent.parent / 'third_party/RoboTwin'
CHECKPOINT = Path('/root/gpufree-data/fastwam/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt')
ORIGINAL_CODE = '774a88a9708453404ef2a6fff4c5415eeede7077'


def first_ten(rows):
    if len(rows) != 12 or len({r['scene_seed'] for r in rows}) != 12:
        raise ValueError('Expected the original twelve unique scenes.')
    if any(r['episode_index'] != i or r['task_config'] != DOMAIN for i, r in enumerate(rows)):
        raise ValueError('Catalog order/domain changed.')
    return rows[:N]


def freeze(source, root, gpus):
    import yaml
    if root.exists() or root.parent != RUNS:
        raise ValueError('Use a fresh output directory on the server data disk.')
    prior = read(source / 'protocol.json')
    proof = read(source / 'completion_audit.json')
    if not proof['complete'] or not read(source / 'status.json')['complete']:
        raise ValueError('Source evaluation must be fully complete.')
    if prior['code_commit'] != ORIGINAL_CODE or prior['task_config'] != DOMAIN:
        raise ValueError('Unexpected reference evaluation.')
    git = lambda *cmd: subprocess.check_output(['git', *cmd], cwd=REPO, text=True).strip()
    if git('status', '--porcelain'):
        raise ValueError('Commit the evaluation code before launch.')
    changed = git('diff', '--name-only', ORIGINAL_CODE, 'HEAD').splitlines()
    allowed = {'scripts/run_robotwin_released_randomized_eval.py', 'tests/test_robotwin_released_randomized_eval.py'}
    if not set(changed) <= allowed:
        raise ValueError('Reference inference/evaluation implementation changed.')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('GPUs are occupied.')
    available = {int(i) for i in subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True).splitlines()}
    if not gpus or len(gpus) != len(set(gpus)) or not set(gpus) <= available:
        raise ValueError('Invalid GPU selection.')
    if shutil.disk_usage(RUNS).free < 5 * 1024**3:
        raise ValueError('Need five GiB free on the data disk.')
    config = SIM / 'task_config' / (DOMAIN + '.yml')
    intervention = REPO / 'configs/eval/robotwin_cis_ten_tasks.json'
    manifest = Path(prior['manifest'])
    for p in (config, intervention, manifest):
        if sha(p) != prior['input_sha256'][str(p)]:
            raise ValueError('Reference input changed: ' + str(p))
    randomization = yaml.safe_load(config.read_text())['domain_randomization']
    if randomization != prior['domain_randomization']:
        raise ValueError('Randomization settings changed.')
    inputs = [source / n for n in ('protocol.json', 'completion_audit.json', 'status.json')]
    inputs += [config, intervention, manifest, CHECKPOINT]
    manifest_data = read(manifest)
    inputs += [Path(manifest_data[k]) for k in ('stats_path', 'original_train_config')]
    selections = {}
    for task in TASK_NAMES:
        catalog = source / 'catalog' / task / DOMAIN / 'correct'
        p = catalog / 'episodes.jsonl'
        if sha(p) != read(source / 'catalog_frozen.json')[task]:
            raise ValueError('Reference catalog changed.')
        selections[task] = first_ten(records(p))
        inputs += [p, catalog / 'catalog_complete.json']
        inputs += sorted((source / 'initial_observations/no_eraf' / task).glob('*.json'))
    plan = dict(format='robotwin_released_randomized_ten_v1', complete=False, status='frozen',
        source_trial=str(source), checkpoint=str(CHECKPOINT), manifest=str(manifest),
        task_config=DOMAIN, tasks=list(TASK_NAMES), episodes_per_task=N, expected_episodes=N*5,
        eraf='off', policy_kind='legacy', gpus=gpus, training_performed=False, correct_evaluated=False,
        scene_selection='First ten entries in each original twelve-scene catalog, preserving order; no outcome selection.',
        independent_test=False, domain_randomization=randomization, code_commit=git('rev-parse', 'HEAD'),
        input_sha256={str(p):sha(p) for p in inputs}, jobs={}, platform_shutdown=None,
        frozen_at=datetime.now().astimezone().isoformat())
    return plan, selections


def verify(root, plan):
    from experiments.robotwin.manipulation_metrics import PROTOCOL, actor_attributes
    pairs = {p['pair_id']:p for p in read(REPO / 'configs/eval/robotwin_cis_ten_tasks.json')['pairs']}
    source = Path(plan['source_trial'])
    cells, pooled, videos = [], [], {}
    for task in TASK_NAMES:
        canonical_path = root / 'catalog' / task / DOMAIN / 'correct/episodes.jsonl'
        canonical = records(canonical_path)
        if canonical != first_ten(records(source / 'catalog' / task / DOMAIN / 'correct/episodes.jsonl')):
            raise ValueError('Selected catalog changed.')
        folder = root / 'evaluation/fastwam_release' / task / DOMAIN / 'counterfactual'
        episodes, complete = records(folder / 'episodes.jsonl'), read(folder / 'complete.json')
        if len(episodes) != N or not complete['complete']:
            raise ValueError('Incomplete task: ' + task)
        for key in ('eraf', 'policy_kind', 'task_config'):
            if complete[key] != plan[key]: raise ValueError('Wrong runtime: ' + key)
        if (complete['checkpoint_sha256'] != plan['input_sha256'][plan['checkpoint']]
            or complete['canonical_sha256'] != sha(canonical_path)
            or complete['task_config_sha256'] != plan['input_sha256'][str(SIM / 'task_config' / (DOMAIN + '.yml'))]
            or complete['domain_randomization'] != plan['domain_randomization']
            or complete['memory_mode'] != 'carry' or not complete['manipulation_metrics']
            or complete['deployment'] != dict(action_horizon=32, replan_steps=24, inference_steps=10)):
            raise ValueError('Checkpoint/catalog/inference binding changed.')
        signatures = [dict(scene_seed=c['scene_seed'],sha256=c['initial_physical_state_sha256']) for c in canonical]
        if read(folder / 'initial_states.json') != signatures:
            raise ValueError('Initial physical states do not match.')
        observations = [read(p) for p in sorted((root / 'initial_observations' / task).glob('*.json'))]
        if len(observations) != N or len({o['metadata']['scene_seed'] for o in observations}) != N:
            raise ValueError('Missing initial observation evidence.')
        observations = {o['metadata']['scene_seed']:o for o in observations}
        reference = {o['metadata']['scene_seed']:o for o in
            [read(p) for p in sorted((source / 'initial_observations/no_eraf' / task).glob('*.json'))]}
        for e,c in zip(episodes, canonical, strict=True):
            if any(e[k] != c[k] for k in ('scene_seed','episode_index','source_instruction','counterfactual_instruction')):
                raise ValueError('Changed scene/instruction pairing.')
            if (e['source_task'] != task or e['task_config'] != DOMAIN
                or any(e[k] != 'counterfactual' for k in ('condition','selected_goal','instruction_goal'))
                or type(e['counterfactual_goal_ever_success']) is not bool):
                raise ValueError('Wrong task/condition/outcome.')
            obs, old = observations[e['scene_seed']], reference[e['scene_seed']]
            if obs['metadata']['checkpoint'] != plan['checkpoint']:
                raise ValueError('Observation came from the wrong checkpoint.')
            for key in ('source_task','task_config','scene_seed','episode_index','source_instruction','counterfactual_instruction','policy_instruction','condition'):
                if obs['metadata'][key] != e[key]: raise ValueError('Observation metadata mismatch.')
            if any(obs[k] != old[k] for k in ('observation_sha256','arrays')):
                raise ValueError('Initial RGB/qpos differs from previous models.')
            metrics = e['manipulation_metrics']
            if (metrics['protocol'] != PROTOCOL or metrics['physics_samples'] < 1
                or set(metrics['objects']) != set(actor_attributes(pairs[e['pair_id']]['counterfactual_goal']))):
                raise ValueError('Missing physical measurements.')
            for obj in metrics['objects'].values():
                if obj['correctly_lifted'] and not (obj['first_lift_tick'] and len(obj['lift_contact_links']) >= 2):
                    raise ValueError('Missing lift contact evidence.')
                if obj['placed_after_lift'] and not (obj['correctly_lifted'] and obj['first_placement_tick'] > obj['first_lift_tick']):
                    raise ValueError('Placement precedes lift.')
            v = Path(e['video_path'])
            if v.parent != folder or str(v) in videos:
                raise ValueError('Wrong or duplicate video.')
            probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
                'format=duration:stream=codec_type,nb_frames','-of','json',str(v)],text=True))
            if float(probe['format']['duration']) <= 0 or not any(s['codec_type']=='video' and int(s['nb_frames'])>0 for s in probe['streams']):
                raise ValueError('Invalid video.')
            videos[str(v)] = dict(sha256=sha(v),probe=probe)
        cells.append(dict(task=task, **summarize_cell(episodes)))
        pooled.extend(episodes)
    for p,digest in plan['input_sha256'].items():
        if sha(p) != digest: raise ValueError('Frozen input changed: ' + p)
    if len(videos) != N*5 or any(j.get('exit_code') != 0 for j in plan['jobs'].values()) or len(plan['jobs']) != 5:
        raise ValueError('Incomplete worker matrix.')
    return dict(format='robotwin_released_randomized_ten_report_v1',complete=True,task_config=DOMAIN,
        checkpoint=plan['checkpoint'],checkpoint_sha256=plan['input_sha256'][plan['checkpoint']],
        episodes_per_task=N,episodes=N*5,cells=cells,pooled=summarize_cell(pooled),
        training_performed=False,correct_evaluated=False,independent_test=False,
        original_catalog=str(source),scene_selection=plan['scene_selection'],
        initial_rgb_and_qpos_identical_to_previous_models=True,videos=videos,
        verified_at=datetime.now().astimezone().isoformat())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-trial',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpus',type=int,nargs='+',default=[0,1,2])
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--verify-only',action='store_true')
    args = ap.parse_args()
    root = args.output.resolve()
    if args.verify_only:
        result = verify(root, read(root/'status.json'))
        print(json.dumps({k:v for k,v in result.items() if k != 'videos'})); return
    plan, selections = freeze(args.source_trial.resolve(),root,args.gpus)
    if args.preflight_only:
        print(json.dumps(plan,indent=2)); return
    root.mkdir(parents=True,exist_ok=False)
    for task,rows in selections.items():
        folder = root / 'catalog' / task / DOMAIN / 'correct'
        folder.mkdir(parents=True)
        (folder/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(root/'protocol.json',plan)
    processes, pending, active = {}, list(TASK_NAMES), {}
    env = os.environ | dict(PATH='/opt/conda/bin:'+os.environ['PATH'],PYTHONPATH=str(REPO/'src')+':'+str(REPO),
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def save(): write(root/'status.json',plan)
    def stop(signum,frame): raise InterruptedError('Signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        plan['status']='evaluating_released_randomized_50';save()
        while pending or active:
            if shutil.disk_usage(root).free < 2*1024**3: raise RuntimeError('Data disk reserve reached.')
            for gpu,task in list(active.items()):
                rc = processes[task].poll()
                if rc is not None:
                    plan['jobs'][task].update(exit_code=rc,finished_at=datetime.now().astimezone().isoformat());save()
                    if rc: raise RuntimeError(task+' failed; inspect its log.')
                    del active[gpu]
            for gpu in args.gpus:
                if gpu in active or not pending: continue
                task = pending.pop(0)
                cmd = [sys.executable,'-u',str(REPO/'scripts/eval_robotwin_eraf_fg.py'),'worker',
                    '--output',str(root/'evaluation/fastwam_release'),'--manifest',plan['manifest'],
                    '--checkpoint',plan['checkpoint'],'--catalog-root',str(root/'catalog'),
                    '--interventions',str(REPO/'configs/eval/robotwin_cis_ten_tasks.json'),'--tasks',task,
                    '--conditions','counterfactual','--episodes',str(N),'--eraf','off','--policy-kind','legacy',
                    '--memory-mode','carry','--task-config',DOMAIN,'--manipulation-metrics','--videos','--gpu',str(gpu)]
                job_env = env | dict(CUDA_VISIBLE_DEVICES=str(gpu),
                    FASTWAM_ROBOTWIN_INITIAL_OBSERVATION_AUDIT=str(root/'initial_observations'/task))
                with (root/(task+'.log')).open('x') as log:
                    p = subprocess.Popen(cmd,cwd=REPO,env=job_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes[task],active[gpu] = p,task
                start = (Path('/proc')/str(p.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19]
                plan['jobs'][task] = dict(pid=p.pid,start_time=start,gpu=gpu,command=cmd,started_at=datetime.now().astimezone().isoformat());save()
            if active: time.sleep(3)
        plan['status']='verifying_all_50';save()
        result = verify(root,plan)
        write(root/'RELEASED_RANDOMIZED_REPORT.json',result)
        text = ['# Released FastWAM demo_randomized 五任务实测','',
            '每任务10回合，共50个CF回合。复用上一轮每任务前10个场景；未训练，未测Correct。','',
            '| 任务 | CF成功 | 正确夹起 | 夹起后任务目标 | 严格脱手放置 |', '|---|---:|---:|---:|---:|']
        for c in result['cells']:
            text.append('| '+TASK_NAMES[c['task']]+' | '+ ' | '.join([frac(c['cf'],N),frac(c['lift'],N),frac(c['cf_after_lift'],c['lift']),frac(c['strict_release_placement'],c['lift'])])+' |')
        c=result['pooled']
        text += ['| 五任务合计 | '+' | '.join([frac(c['cf'],50),frac(c['lift'],50),frac(c['cf_after_lift'],c['lift']),frac(c['strict_release_placement'],c['lift'])])+' |','',
            '正确夹起指至少一个指令物体被验证夹起。严格脱手未确认不能单独判定任务放置失败。',
            '原始次数和成功率未作缩放；这些是10回合实测，不是40回合假设折算。','']
        (root/'RELEASED_RANDOMIZED_REPORT.md').write_text('\n'.join(text))
        plan.update(complete=True,status='complete',finished_at=datetime.now().astimezone().isoformat());save()
        print(json.dumps(dict(complete=True,episodes=50,cf=c['cf'],cf_rate=c['cf_rate'])),flush=True)
    except BaseException as error:
        plan.update(complete=False,status='stopped',error=repr(error));save();raise
    finally:
        for p in processes.values():
            if p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
        for p in processes.values():
            if p.poll() is None:
                try: p.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL);p.wait()


if __name__ == '__main__': main()
