#!/usr/bin/env python3
"""Train traced common-parent controls and jointly optimized ERAF+FG candidates."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
OLD_TASKS = ['blocks_ranking_rgb', 'stack_blocks_two', 'place_a2b_left',
             'place_a2b_right', 'place_burger_fries']
NEW_TASKS = ['blocks_ranking_size', 'place_empty_cup', 'place_mouse_pad',
             'move_stapler_pad', 'move_pillbottle_pad']
BASE = '/root/gpufree-data/LF-FastWAM/runs'
STUDY = BASE + '/robotwin_target_retention_study_20260907'


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def build_protocol(args, code):
    return dict(format='robotwin_cf_priority_common_warm_policy_v1',
        authorization='User prioritizes ERAF+FG CF performance, permits data/supervision/structure changes with traceability and autonomous server power control; no Correct threshold.',
        primary_metric='Macro CF across the ten declared tasks; report per-task regressions and paired episode outcomes.',
        selection='Original-five development screen first; select each arm by complete CF count, then earlier step on ties. Extend selected arms to five additional tasks. Development selection is not independent test evidence.',
        code_commit=code, deadline=args.deadline, checkpoint=str(args.checkpoint),
        manifest=str(args.manifest), source_bank=str(args.source_bank),
        correct_teacher=str(args.correct_teacher), cf_teacher=str(args.checkpoint),
        old_catalog=str(args.old_catalog), new_catalog=str(args.new_catalog),
        old_tasks=OLD_TASKS, new_tasks=NEW_TASKS, global_batch=12, seed=42,
        joint_steps=args.steps, interface_steps=args.interface_steps,
        checkpoints=[args.steps // 2, args.steps],
        joint_learning_rate=3e-6, interface_joint_learning_rate=3e-5,
        interface_warm_learning_rate=1e-4, correct_weight=1., cf_weight=4., correction_weight=1.,
        mixture={'correct': 2, 'cf': 4, 'pair': 3, 'fg_or_matched_ordinary_cf': 3},
        task_balanced=True, policy_scope='action', seen_language_augmentation=False,
        arms={'ordinary_cf': {'eraf': 'off', 'fg': 'off', 'gpus': [0, 1]},
              'fg_only': {'eraf': 'off', 'fg': 'full', 'gpus': [2, 3]},
              'eraf_fg': {'eraf': 'on', 'fg': 'full', 'gpus': [4, 5]}},
        causal_scope='Matched parent, action steps, data schedule and supervision settings across arms. ERAF arm has separately reported interface-only warmup. FG-off uses ordinary CF at matched positions; no claim of equal total FLOPs or pure horizon ablation.',
        data_scope='Existing five-task bank; additional five tasks are initially transfer evaluations. No evaluation outcomes enter training or scene filtering.',
        input_changes='No image/action normalization, task goal, prompt, horizon, noise or rollout-budget change.',
        independent_test_opened=False, file_hash_scans=False)


def training_command(plan, output, arm, stage, parent, *, warm):
    spec = plan['arms'][arm]
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=2', str(REPO / 'scripts/train_robotwin_eraf_fg_action.py'),
        '--manifest', plan['manifest'], '--source-bank', plan['source_bank'],
        '--checkpoint', str(parent), '--output', str(output / arm / stage),
        '--correct-teacher', plan['correct_teacher'], '--cf-teacher', plan['cf_teacher'],
        '--stage', stage, '--fg', spec['fg'], '--eraf', spec['eraf'],
        '--steps', str(plan['interface_steps'] if stage == 'interface' else plan['joint_steps']),
        '--save-every', str(plan['interface_steps'] if stage == 'interface' else plan['checkpoints'][0]),
        '--learning-rate', str(plan['interface_warm_learning_rate'] if stage == 'interface' else plan['joint_learning_rate']),
        '--correct-weight', str(plan['correct_weight']), '--cf-weight', str(plan['cf_weight']),
        '--correction-weight', str(plan['correction_weight']), '--correct-count', '2', '--cf-count', '4',
        '--policy-scope', 'action', '--cf-retention-tasks', *OLD_TASKS,
        '--seed', '42', '--task-balanced', '--disable-seen-language-augmentation', '--skip-file-hashes']
    if stage == 'joint' and spec['eraf'] == 'on':
        command += ['--interface-learning-rate', str(plan['interface_joint_learning_rate'])]
    if warm:
        command += ['--warm-policy']
    return command


def select_arm(rows):
    """Count complete fixed-CF episodes; use earlier checkpoint only to break ties."""
    if not rows or any(not r['complete'] or r['episodes'] != 30 for r in rows):
        raise ValueError('Selection needs complete thirty-episode original-five CF evaluations.')
    return max(rows, key=lambda r: (r['successes'], -r['step']))


def cf_summary_result(summary, *, name, step):
    cells = summary['cells']
    if (not summary['complete'] or summary['episodes'] != 30 or len(cells) != 5
            or {cell['source_task'] for cell in cells} != set(OLD_TASKS)
            or any(cell['condition'] != 'counterfactual' or cell['total_episodes'] != 6 for cell in cells)):
        raise ValueError('Selection requires all five tasks with six CF episodes each.')
    return dict(name=name, step=step, complete=True, episodes=30,
                successes=sum(cell['selected_goal_successes'] for cell in cells), cells=cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--deadline', required=True)
    ap.add_argument('--execute', action='store_true', help='Without this flag only writes the reviewable protocol.')
    ap.add_argument('--checkpoint', type=Path, default=STUDY + '/all5_off200/joint/step_000200.pt')
    ap.add_argument('--manifest', type=Path, default=STUDY + '/cache_all5/manifest.json')
    ap.add_argument('--source-bank', type=Path, default=BASE + '/robotwin_cf_cause_audit/decision-bank-20260905-220457')
    ap.add_argument('--correct-teacher', type=Path, default=BASE + '/robotwin_cf_dense/20260906-native-retention-v1/repair-dense-native/step_000600.pt')
    ap.add_argument('--old-catalog', type=Path, default=STUDY + '/catalog_dev')
    ap.add_argument('--new-catalog', type=Path, default=BASE + '/robotwin_ten_task_extension_20260907/catalog')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--interface-steps', type=int, default=100)
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or args.steps < 2 or args.steps % 2 or args.interface_steps <= 0:
        ap.error('Timezone-aware cutoff, positive even joint steps and positive interface steps required.')
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    plan = build_protocol(args, code)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'protocol.json', plan)
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False)); return
    if time.time() >= cutoff.timestamp():
        raise TimeoutError('Work cutoff already reached')
    for name in ['checkpoint', 'manifest', 'correct_teacher', 'old_catalog', 'new_catalog', 'source_bank']:
        if not Path(plan[name]).exists(): raise FileNotFoundError(plan[name])
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO, text=True).strip():
        raise RuntimeError('Commit tracked changes before an attributable training run.')
    report = dict(complete=False, started=datetime.now().astimezone().isoformat(), jobs={}, stage='training')
    env = os.environ | {'PATH': '/opt/conda/bin:' + os.environ['PATH'],
        'PYTHONPATH': str(REPO / 'src') + ':' + str(REPO), 'OMP_NUM_THREADS': '2',
        'DIFFSYNTH_MODEL_BASE_PATH': '/root/gpufree-data/fastwam/FastWAM/checkpoints',
        'VK_ICD_FILENAMES': '/etc/vulkan/icd.d/nvidia_icd.json'}
    procs = {}

    def save(): write(output / 'driver.json', report)
    def budget():
        if time.time() >= cutoff.timestamp(): raise TimeoutError('Experimental cutoff reached; preserve partial records')
    def stop(signum, frame): raise InterruptedError(f'Campaign signal {signum}')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)

    def start(name, command, gpus):
        budget()
        with (output / (name + '.log')).open('x') as log:
            p = subprocess.Popen(list(map(str, command)), cwd=REPO,
                env=env | {'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus))},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        procs[name] = p
        report['jobs'][name] = dict(pid=p.pid, gpus=gpus, command=list(map(str, command)),
                                  started=datetime.now().astimezone().isoformat())
        save(); print('[start]', name, p.pid, flush=True)

    def done(name):
        value = procs[name].poll()
        if value is None: return False
        report['jobs'][name]['exit_code'] = value; save()
        if value: raise RuntimeError(f'{name} failed: {value}')
        return True

    def wait(name):
        while not done(name): budget(); time.sleep(5)

    def evaluate(name, models, tasks, catalog, episodes):
        matrix = dict(models=models, tasks=tasks, conditions=['counterfactual'],
                      catalogs={'dev': {'path': str(catalog), 'episodes': episodes}},
                      interventions=str(REPO / 'configs/eval/robotwin_cis_ten_tasks.json'))
        write(output / (name + '_plan.json'), matrix)
        start(name, [sys.executable, '-u', REPO / 'scripts/run_robotwin_fixed_eval_matrix.py',
            '--plan', output / (name + '_plan.json'), '--output', output / name,
            '--gpus', '0', '1', '2', '3', '4', '5', '--deadline', args.deadline], list(range(6)))
        wait(name)

    def model(checkpoint, eraf):
        return dict(checkpoint=str(checkpoint), manifest=plan['manifest'], policy_kind='repair', eraf=eraf)

    try:
        for arm in plan['arms']:
            (output / arm).mkdir()
            stage = 'interface' if arm == 'eraf_fg' else 'joint'
            start(arm + '_' + stage, training_command(plan, output, arm, stage, plan['checkpoint'], warm=True),
                  plan['arms'][arm]['gpus'])
        while not done('eraf_fg_interface'):
            for arm in ('ordinary_cf', 'fg_only'): done(arm + '_joint')
            budget(); time.sleep(5)
        parent = output / 'eraf_fg/interface' / f"step_{plan['interface_steps']:06d}.pt"
        start('eraf_fg_joint', training_command(plan, output, 'eraf_fg', 'joint', parent, warm=False), [4, 5])
        while not all(done(arm + '_joint') for arm in plan['arms']): budget(); time.sleep(5)
        report['stage'] = 'original_five_cf_screen'; save()
        models = {'all5_off200': model(plan['checkpoint'], 'off')}
        for arm, spec in plan['arms'].items():
            for step in plan['checkpoints']:
                models[f'{arm}_{step}'] = model(output / arm / 'joint' / f'step_{step:06d}.pt', spec['eraf'])
        evaluate('eval_original_five', models, OLD_TASKS, plan['old_catalog'], 6)
        selection = {}
        for arm in plan['arms']:
            options = []
            for step in plan['checkpoints']:
                name = f'{arm}_{step}'
                summary = json.loads((output / 'eval_original_five' / name / 'dev/summary.json').read_text())
                options.append(cf_summary_result(summary, name=name, step=step))
            selection[arm] = select_arm(options)
        write(output / 'selection.json', selection)
        final_models = {'all5_off200': models['all5_off200']}
        final_models.update({value['name']: models[value['name']] for value in selection.values()})
        selected_eraf = final_models[selection['eraf_fg']['name']]
        # Same learned checkpoint with ERAF bypassed measures actual module use.
        evaluate('eval_eraf_bypass', {'selected_eraf_fg_bypass': selected_eraf | {'eraf': 'off'}},
                 OLD_TASKS, plan['old_catalog'], 6)
        report['stage'] = 'additional_five_cf'; save()
        evaluate('eval_additional_five', final_models, NEW_TASKS, plan['new_catalog'], 3)
        report.update(complete=True, stage='complete', completed=datetime.now().astimezone().isoformat(),
                      scope='Development screens and selected-arm transfer only; no independent test or Correct claim.')
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        for process in procs.values():
            if process.poll() is None:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        for process in procs.values():
            if process.poll() is None:
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)
        save()


if __name__ == '__main__': main()
