#!/usr/bin/env python3
"""Stop our two runs after step250 is saved, then screen 12 CIS rollouts.

This intentionally truncates the original1000-step plans; their completion
markers are not rewritten. Existing step250 files retain the original LR
schedule. Evaluate two tasks x one seed x Correct/CF for three models, using
each GPU as soon as its own training arm releases it. No additional hashes.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]


def run_arm(args, gpu, arm, launcher):
    train = Path(args.train_output).resolve()
    output = Path(args.output).resolve()
    checkpoint = train / arm / 'checkpoints/weights/step_000250.pt'
    draws = train / arm / 'decision_replay/draws_rank0.jsonl'
    print(f'[wait250] {arm} launcher={launcher}', flush=True)
    while True:
        command = Path(f'/proc/{launcher}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        if 'accelerate.commands.launch' not in command or str(train / arm) not in command:
            raise RuntimeError('Training launcher identity changed; inspect before stopping it.')
        # The next microbatch can only be logged after the synchronous step250
        # save has returned. Avoid interrupting a checkpoint write.
        count = sum(1 for _ in draws.open()) if draws.exists() else 0
        if checkpoint.exists() and count > 250 * 16:
            break
        time.sleep(20)
    os.kill(launcher, signal.SIGTERM)
    while int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True).strip()) >= 256:
        time.sleep(10)
    (output / f'{arm}_stopped.json').write_text(json.dumps({'checkpoint': str(checkpoint),
        'kept_optimizer_steps': 250, 'reason': 'User requested smaller experiments and fewer GPU hours.'}, indent=2) + '\n')
    models = [('base', args.base_checkpoint), ('original250', str(checkpoint))] if gpu == 0 else [('replay250', str(checkpoint))]
    from experiments.robotwin.run_robotwin_cis_manager import _resolve_ckpt_tag
    for name, weight in models:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO / 'src') + os.pathsep + str(REPO),
            PYTHON_BIN=sys.executable, RUN_ROOT=str(REPO), FASTWAM_EVAL_MODE='B0',
            ROBOTWIN_TASK_CONFIG='robotwin_uncond_3cam_384_1e-4',
            CIS_TASK_CONFIGS='demo_clean', CIS_CONDITIONS='correct,counterfactual',
            CIS_TASKS='stack_blocks_two,blocks_ranking_rgb', MAX_TASKS_PER_GPU='1', INSTRUCTION_TYPE='unseen',
            MANIFEST_PATH=str(REPO / 'configs/eval/robotwin_cis_v939_four_tasks.json'),
            RUN_TAG=output.name + '_' + name, OUTPUT_ROOT=str(output / (output.name + '_' + name)),
            DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
        for key in list(env):
            if key.startswith('PGC_ROBOTWIN_CLOSED_LOOP_CAPTURE') or key == 'FASTWAM_ROBOTWIN_INITIAL_OBSERVATION_AUDIT':
                env.pop(key)
        print(f'[evaluate] {name} gpu={gpu}', flush=True)
        with (output / f'{name}.log').open('x') as log:
            subprocess.run(['bash', 'scripts/eval_robotwin_cis.sh', '1', '1', '10', '42', weight, args.stats_path],
                cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = REPO / 'evaluate_results/robotwin' / _resolve_ckpt_tag(Path(weight)) / env['RUN_TAG']
        (output / f'{name}_complete.json').write_text(json.dumps({'model': name, 'checkpoint': weight,
            'cis_results': str(result), 'scope': 'Two-task one-seed screening only.'}, indent=2) + '\n')
        print(f'[evaluated] {name} {result}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-output', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--original-launcher', required=True, type=int)
    ap.add_argument('--replay-launcher', required=True, type=int)
    ap.add_argument('--base-checkpoint', required=True)
    ap.add_argument('--stats-path', required=True)
    args = ap.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'plan.json').write_text(json.dumps(vars(args) | {'target_step': 250, 'rollouts': 12,
        'tasks': ['stack_blocks_two', 'blocks_ranking_rgb'], 'reason': 'Lean causal screening per user request.'}, indent=2) + '\n')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_arm, args, 0, 'original', args.original_launcher),
                   pool.submit(run_arm, args, 1, 'decision_replay', args.replay_launcher)]
        for future in futures:
            future.result()
    print('[complete] lean screening; analyze per-task outcomes', flush=True)


if __name__ == '__main__':
    main()
