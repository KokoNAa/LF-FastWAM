#!/usr/bin/env python3
"""One-shot dependent validation of a running RoboTwin training experiment.

GPU0 waits for its original arm, then evaluates Base, old step1000, and the new
original arm. GPU1 waits for its treatment arm. No jobs are restarted, no GPU
is rented/stopped, and no training process is interrupted. Every rollout and
normal-policy initial input must match across models before a comparison is
marked complete. Completion here means measurement, not scientific success.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]


def process_is_training(pid, train_root):
    try:
        value = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return False
    return 'run_robotwin_decision_control.py' in value and str(train_root) in value


def initial_comparison(audits):
    """Compare exact observations AND language for every model/condition/seed."""
    indexed = {}
    for name, records in audits.items():
        index = {}
        for record in records:
            meta = record['metadata']
            expected = meta['source_instruction'] if meta['condition'] == 'correct' else meta['counterfactual_instruction']
            if meta['policy_instruction'] != expected or meta['source_instruction'] == meta['counterfactual_instruction']:
                raise ValueError('Policy language does not implement the intended intervention.')
            key = tuple(meta[k] for k in ('pair_id', 'task_config', 'condition', 'scene_seed'))
            if key in index:
                raise ValueError('Duplicate initial observation audit.')
            index[key] = (record['observation_sha256'], meta['source_instruction'],
                          meta['counterfactual_instruction'], meta['policy_instruction'])
        indexed[name] = index
    if not indexed:
        raise ValueError('No observation audits.')
    names = list(indexed)
    reference = indexed[names[0]]
    if not reference:
        raise ValueError('Empty observation audits.')
    for name in names[1:]:
        if indexed[name] != reference:
            raise ValueError(f'Initial scene or instruction differs for {name}.')
    for key, value in reference.items():
        pair, domain, condition, seed = key
        other = reference[(pair, domain, 'counterfactual' if condition == 'correct' else 'correct', seed)]
        if value[:3] != other[:3]:
            raise ValueError('Correct and CF did not start at the same complete observation/instruction pair.')
    return {'models': names, 'episodes_per_model': len(reference),
            'exact_initial_observations_equal': True, 'instruction_pairs_equal': True}


def wait_for_arm(plan, arm, gpu):
    from scripts.probe_robotwin_no_eraf import read_json, sha256
    train_root = Path(plan['train_output'])
    complete = train_root / arm / 'complete.json'
    announced = None
    while True:
        if complete.exists():
            record = read_json(complete)
            if record.get('complete') is not True or record['steps'] != 1000:
                raise ValueError('Unexpected training completion record.')
            apps = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            no_apps = not apps or apps == 'No running processes found'
            used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
            if no_apps and used < 256:
                if sha256(record['checkpoint']) != record['checkpoint_sha256']:
                    raise ValueError('Completed training checkpoint changed.')
                return
            status = 'waiting_for_gpu_release'
        elif not process_is_training(plan['training_pid'], train_root):
            raise RuntimeError('Training controller is no longer live and this arm has no completion record. Not restarting it.')
        else:
            status = 'waiting_for_training'
        if status != announced:
            print(f'[{status}] {arm} gpu={gpu}', flush=True)
            announced = status
        time.sleep(30)


def worker(args):
    from scripts.probe_robotwin_no_eraf import read_json, write_json, sha256
    from experiments.robotwin.run_robotwin_cis_manager import _resolve_ckpt_tag
    from scripts.summarize_robotwin_cis import load_job_output
    root = Path(args.output).resolve()
    plan = read_json(root / 'plan.json')
    if sha256(plan['replay_manifest']) != plan['replay_manifest_sha256'] or sha256(plan['stats_path']) != plan['stats_sha256']:
        raise ValueError('Replay manifest or normalization statistics changed.')
    gpu = args.gpu
    arm = 'original' if gpu == 0 else 'decision_replay'
    wait_for_arm(plan, arm, gpu)
    names = ['base', 'old_step1000', 'original'] if gpu == 0 else ['decision_replay']
    for name in names:
        checkpoint = Path(plan['checkpoints'][name])
        if name in plan['fixed_checkpoint_sha256'] and sha256(checkpoint) != plan['fixed_checkpoint_sha256'][name]:
            raise ValueError('A fixed reference checkpoint changed.')
        if sha256(plan['cis_manifest']) != plan['cis_manifest_sha256']:
            raise ValueError('CIS goal definitions changed.')
        model_root = root / name
        model_root.mkdir(exist_ok=False)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO / 'src') + os.pathsep + str(REPO),
            PYTHON_BIN=sys.executable, RUN_ROOT=str(REPO), FASTWAM_EVAL_MODE='B0',
            ROBOTWIN_TASK_CONFIG='robotwin_uncond_3cam_384_1e-4',
            CIS_TASK_CONFIGS='demo_clean', CIS_CONDITIONS='correct,counterfactual', CIS_TASKS='',
            MAX_TASKS_PER_GPU='1', INSTRUCTION_TYPE='unseen',
            MANIFEST_PATH=plan['cis_manifest'], RUN_TAG=root.name + '_' + name,
            OUTPUT_ROOT=str(model_root / (root.name + '_' + name)),
            FASTWAM_ROBOTWIN_INITIAL_OBSERVATION_AUDIT=str(model_root / 'initial_observations'))
        env['DIFFSYNTH_MODEL_BASE_PATH'] = plan['model_cache']
        for key in list(env):
            if key.startswith('PGC_ROBOTWIN_CLOSED_LOOP_CAPTURE'):
                del env[key]
        record = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint), 'physical_gpu': gpu}
        write_json(model_root / 'plan.json', record)
        cmd = ['bash', 'scripts/eval_robotwin_cis.sh', '1', '3', '10', '42', str(checkpoint), plan['stats_path']]
        print(f'[cis_start] {name} gpu={gpu}', flush=True)
        with (model_root / 'cis.log').open('x') as log:
            subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        cis_root = REPO / 'evaluate_results/robotwin' / _resolve_ckpt_tag(checkpoint) / env['RUN_TAG']
        episodes = []
        for pair in read_json(plan['cis_manifest'])['pairs']:
            for condition in ('correct', 'counterfactual'):
                _, records = load_job_output(cis_root / pair['source_task'] / 'demo_clean' / condition,
                    expected_episodes=3, expected_source_task=pair['source_task'], expected_task_config='demo_clean',
                    expected_condition=condition, expected_checkpoint=checkpoint)
                episodes.extend(records)
        audits = [read_json(p) for p in (model_root / 'initial_observations').glob('*.json')]
        if len(episodes) != 30 or len(audits) != 30:
            raise ValueError('Incomplete CIS episodes or initial-input audit.')
        episode_keys = {(r['pair_id'], r['task_config'], r['condition'], r['scene_seed'], r['policy_instruction']) for r in episodes}
        audit_keys = {(r['metadata']['pair_id'], r['metadata']['task_config'], r['metadata']['condition'], r['metadata']['scene_seed'], r['metadata']['policy_instruction']) for r in audits}
        if episode_keys != audit_keys:
            raise ValueError('Episode outcomes do not match the audited policy inputs.')
        initial_comparison({name: audits})
        write_json(model_root / 'cis_validated.json', {'complete': True, 'run_root': str(cis_root), 'episodes': episodes})
        print(f'[cis_complete] {name}', flush=True)
        # The standard cold-load probe checks every state against normal policy
        # preprocessing. The CIS-only input logger is disabled for this probe.
        env.pop('FASTWAM_ROBOTWIN_INITIAL_OBSERVATION_AUDIT')
        cmd = [sys.executable, '-u', 'scripts/probe_robotwin_decision_replay.py',
            '--manifest', plan['replay_manifest'], '--checkpoint', str(checkpoint),
            '--output', str(model_root / 'offline'), '--gpu', str(gpu),
            *(['--base'] if name == 'base' else ['--step', '1000'])]
        with (model_root / 'offline.log').open('x') as log:
            subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if sha256(checkpoint) != record['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during evaluation.')
        write_json(model_root / 'complete.json', {'complete': True, **record, 'cis_run_root': str(cis_root)})
        print(f'[complete] {name}', flush=True)


def finish(root):
    from scripts.probe_robotwin_no_eraf import read_json, write_json
    plan = read_json(root / 'plan.json')
    training = read_json(Path(plan['train_output']) / 'summary.json')
    if not all(training.get(k) is True for k in ('complete', 'matched_initial_adapters',
            'matched_original_samples_and_rng', 'matched_original_losses_before_first_update')):
        raise ValueError('Full training control did not pass its integrity audit.')
    observations, outcomes = {}, {}
    for name in plan['checkpoints']:
        folder = root / name
        if read_json(folder / 'complete.json').get('complete') is not True:
            raise ValueError('Model evaluation incomplete.')
        if read_json(folder / 'offline/summary.json').get('complete') is not True:
            raise ValueError('Cold-load action probe incomplete.')
        observations[name] = [read_json(p) for p in (folder / 'initial_observations').glob('*.json')]
        episodes = read_json(folder / 'cis_validated.json')['episodes']
        outcomes[name] = {condition: {'episodes': sum(r['condition'] == condition for r in episodes),
            'selected_goal_successes': sum(r['condition'] == condition and r['selected_goal_success'] for r in episodes),
            'source_goal_ever_successes': sum(r['condition'] == condition and r['source_goal_ever_success'] for r in episodes)}
            for condition in ('correct', 'counterfactual')}
    comparison = initial_comparison(observations)
    write_json(root / 'summary.json', {'complete': True, 'training_control_valid': True,
        'initial_inputs': comparison, 'outcomes': outcomes,
        'scope': 'Completed measurements; scientific interpretation and attribution require analysis, not just this completion marker.'})


def main():
    from scripts.probe_robotwin_no_eraf import read_json, write_json, sha256
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['run', 'worker', 'summarize'])
    ap.add_argument('--output', required=True)
    ap.add_argument('--train-output')
    ap.add_argument('--training-pid', type=int)
    ap.add_argument('--replay-manifest')
    ap.add_argument('--gpu', type=int, choices=[0, 1])
    args = ap.parse_args()
    root = Path(args.output).resolve()
    if args.mode == 'worker':
        return worker(args)
    if args.mode == 'summarize':
        return finish(root)
    if not args.train_output or not args.training_pid or not args.replay_manifest:
        ap.error('Need the exact running training job and replay bank.')
    train_root = Path(args.train_output).resolve()
    if not process_is_training(args.training_pid, train_root):
        raise ValueError('Specified training controller is not live; inspect it before changing the pipeline.')
    bank = read_json(args.replay_manifest)
    source = read_json(Path(bank['source_probe']) / 'plan.json')
    checkpoints = {'base': source['base_checkpoint'], 'old_step1000': source['checkpoints']['step1000'],
        **{name: str(train_root / name / 'checkpoints/weights/step_001000.pt') for name in ('original', 'decision_replay')}}
    root.mkdir(parents=True, exist_ok=False)
    plan = {'format': 'robotwin_cf_dependent_validation_v1', 'train_output': str(train_root),
        'training_pid': args.training_pid, 'replay_manifest': str(Path(args.replay_manifest).resolve()),
        'replay_manifest_sha256': sha256(args.replay_manifest), 'checkpoints': checkpoints,
        'stats_path': bank['stats_path'], 'stats_sha256': bank['stats_sha256'],
        'fixed_checkpoint_sha256': {'base': source['checkpoint_sha256']['base'], 'old_step1000': source['checkpoint_sha256']['step1000']},
        'cis_manifest': str(REPO / 'configs/eval/robotwin_cis_v939_four_tasks.json'),
        'cis_manifest_sha256': sha256(REPO / 'configs/eval/robotwin_cis_v939_four_tasks.json'),
        'model_cache': '/root/gpufree-data/fastwam/FastWAM/checkpoints',
        'scope': 'Use each GPU only after its own training arm completes and releases it. Fixed 3 matched Correct/CF episodes x5 tasks; cold-load all100 replay states x3 noise seeds. No training restart.'}
    write_json(root / 'plan.json', plan)
    workers = []
    for gpu in (0, 1):
        log = (root / f'gpu{gpu}.log').open('x')
        child = subprocess.Popen([sys.executable, '-u', __file__, 'worker', '--output', str(root), '--gpu', str(gpu)],
                                 cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        workers.append((child, log))
        print(f'[worker] gpu={gpu} pid={child.pid}', flush=True)
    for child, log in workers:
        rc = child.wait(); log.close()
        print(f'[worker_exit] pid={child.pid} code={rc}', flush=True)
    if any(child.returncode for child, _ in workers):
        raise RuntimeError('A dependent evaluation failed; partial comparisons are not summarized.')
    finish(root)
    print('[complete]', root / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
