#!/usr/bin/env python3
"""Train matched ten-task action controls and ERAF/FG arms, then evaluate fixed CF scenes."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
RUNS = Path('/root/gpufree-data/LF-FastWAM/runs')


def train_command(plan, root, arm, *, interface=False):
    spec = plan['arms'][arm]
    parent = plan['ordinary_bootstrap'] if interface else spec['parent']
    dest = root / arm / ('interface' if interface else 'joint')
    steps = 100 if interface else 200
    cmd = ['/opt/conda/bin/python', '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node', str(len(spec['gpus'])), str(REPO / 'scripts/train_robotwin_eraf_fg_action.py'),
        '--manifest', plan['old_manifest'] if interface else plan['manifest'], '--source-bank', plan['source_bank'],
        '--checkpoint', parent, '--output', str(dest), '--correct-teacher', plan['correct_teacher'],
        '--cf-teacher', plan['strongest_checkpoint'], '--stage', 'interface' if interface else 'joint',
        '--eraf', spec['eraf'], '--fg', spec['fg'], '--policy-scope', 'action', '--interface-scope', 'all',
        '--steps', str(steps), '--save-every', str(steps), '--learning-rate', '0.0001' if interface else '0.000003',
        '--correct-count', '2', '--cf-count', '4', '--correct-weight', '1', '--cf-weight', '4',
        '--correction-weight', '1' if interface else '0.1', '--task-balanced',
        '--disable-seen-language-augmentation', '--skip-file-hashes', '--cf-retention-tasks',
        'place_a2b_right', 'place_burger_fries', 'stack_blocks_two', 'place_a2b_left', 'blocks_ranking_rgb']
    if not interface:
        cmd += ['--interface-learning-rate', '0.00003']
        if spec['eraf'] == 'off':
            cmd += ['--warm-policy']
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--wait-for', type=Path, required=True)
    ap.add_argument('--deadline', required=True)
    args = ap.parse_args()
    root = args.output.resolve()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        ap.error('Deadline requires timezone.')
    strongest = str(RUNS / 'robotwin_cf_priority/20260907-1543-warmoff200-cf4/ordinary_cf/joint/step_000200.pt')
    manifest = RUNS / 'robotwin_joint_expert_expansion/20260907-formal16/cache_ten_tasks/manifest.json'
    preflight = json.loads(manifest.parent.parent.joinpath('action_expansion_preflight.json').read_text())
    assert preflight['complete'] and preflight['parent_rows_preserved_exactly'] and preflight['development_seeds_excluded']
    assert preflight['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert preflight['all_non_fg_sample_slots_identical_across_arms'] and not preflight['new_data_are_fg']
    root.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        path = root / name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)

    plan = dict(format='robotwin_ten_task_action_expansion_v1',
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        manifest=str(manifest), manifest_sha256=preflight['manifest_sha256'], preflight=preflight,
        old_manifest=str(RUNS / 'robotwin_target_retention_study_20260907/cache_all5/manifest.json'),
        source_bank=str(RUNS / 'robotwin_cf_cause_audit/decision-bank-20260905-220457'),
        strongest_checkpoint=strongest,
        correct_teacher=str(RUNS / 'robotwin_cf_dense/20260906-native-retention-v1/repair-dense-native/step_000600.pt'),
        ordinary_bootstrap=str(RUNS / 'robotwin_context_residual/20260907-2300-identity/ordinary_bootstrap.pt'),
        arms={
            'no_eraf': dict(eraf='off', fg='off', gpus=[2], parent=strongest),
            'fg_only': dict(eraf='off', fg='full', gpus=[3], parent=strongest),
            'eraf_only': dict(eraf='on', fg='off', gpus=[0, 1], parent=str(root / 'eraf_only/interface/step_000100.pt')),
            'eraf_fg': dict(eraf='on', fg='full', gpus=[4, 5],
                parent=str(RUNS / 'robotwin_context_residual/20260907-2310-action100/candidate/fg_semantic_fg/step_000100.pt'))},
        groups={
            'original_five': dict(tasks=['blocks_ranking_rgb', 'stack_blocks_two', 'place_a2b_left', 'place_a2b_right', 'place_burger_fries'],
                episodes=6, catalog=str(RUNS / 'robotwin_target_retention_study_20260907/catalog_dev')),
            'additional_five': dict(tasks=['blocks_ranking_size', 'place_empty_cup', 'place_mouse_pad', 'move_stapler_pad', 'move_pillbottle_pad'],
                episodes=3, catalog=str(RUNS / 'robotwin_ten_task_extension_20260907/catalog'))},
        joint_steps=200, joint_policy_lr=3e-6, joint_interface_lr=3e-5, correction_weight=0.1,
        global_batch=12, seed=42, mixture=dict(correct=2, cf=4, expert=3, fg_or_matched_ordinary=3),
        full_arm_interface='Reuse declared completed FG semantic100 plus residual interface100, no additional selection.',
        eraf_only_interface='Ordinary semantic100 residual bootstrap, then fresh FG-off interface100 on original five-task bank. No failed-state images or FG targets consumed.',
        causal_scope='All arms start joint training with identical strongest action policy tensors and receive identical new expert data, retention and common paired samples. FG changes partial semantic and corrective action supervision; ERAF adds its interfaces. Equal action optimizer steps, not equal total compute.',
        new_data_scope='Eighty expert-paired scenes, twelve train and four holdout per added task. Expert trajectories are not Full Goal corrections. New data used only in joint training.',
        recipe_change='Unfreeze action LoRA, lower correction coefficient to0.1, and add shared ten-task expert replay. Four-arm method comparison; not an isolated attribution of all three changes.',
        checkpoint_selection='Predeclared final joint200 only. Fresh optimizer per stage; no smoke updates inherited.',
        total_eval_episodes=180, independent_test=False, correct_evaluated=False,
        primary_metric='Equal-task macro CF over ten unchanged development tasks; compare all new arms and strongest archived35%.',
        memory_mode='carry', deadline=args.deadline, platform_shutdown=None,
        power_authorization='User requires server remain on until the full goal is achieved.',
        local_checkpoint_storage=False)
    write('protocol.json', plan)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Commit code before execution.')
    report = dict(complete=False, stage='waiting_for_diagnostic', jobs={})
    processes = {}
    env = os.environ | dict(PATH='/opt/conda/bin:' + os.environ['PATH'], PYTHONPATH=str(REPO / 'src') + ':' + str(REPO),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')

    def budget():
        if time.time() >= deadline.timestamp():
            raise TimeoutError('Experiment work cutoff; server remains on.')
        if shutil.disk_usage(root).free < 5 * 1024**3:
            raise RuntimeError('Five GiB disk reserve reached.')

    def launch(name, cmd, gpus):
        with (root / (name + '.log')).open('x') as log:
            proc = subprocess.Popen(cmd, cwd=REPO, env=env | {'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus))},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = proc
        report['jobs'][name] = dict(pid=proc.pid, gpus=gpus, command=cmd)
        write('driver.json', report)
        return proc

    def audit(arm, stage):
        dest = root / arm / stage
        subprocess.run(['/opt/conda/bin/python', str(REPO / 'scripts/audit_robotwin_eraf_fg_action_stage.py'),
            '--output', str(dest)], cwd=REPO, env=env, check=True, timeout=120, stdout=subprocess.DEVNULL)
        result = json.loads((dest / 'freeze_audit.json').read_text())
        assert result['complete'] and result['all_sampled_state_ids_match'] and not result['unexpected_changes']
        if stage == 'interface':
            assert result['changed_lora_tensors'] == 0

    def stop(signum, _frame):
        raise InterruptedError(f'Signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        write('driver.json', report)
        previous = args.wait_for.resolve()
        pid = json.loads(previous.with_name(previous.name + '-launch.json').read_text())['pid']
        while True:
            budget()
            prior = json.loads((previous / 'driver.json').read_text())
            proc = Path('/proc') / str(pid)
            try:
                live = (proc / 'stat').read_text().split(') ')[1][0] != 'Z' and b'run_robotwin_memory_diagnostic.py' in (proc / 'cmdline').read_bytes()
            except FileNotFoundError:
                live = False
            if prior.get('error'):
                raise RuntimeError('Diagnostic failed: ' + prior['error'])
            if prior['complete'] and not live:
                break
            if not live:
                raise RuntimeError('Diagnostic producer absent without completion.')
            time.sleep(5)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        # Check action-policy identity on the server before any training update.
        import sys
        sys.path[:0] = [str(REPO), str(REPO / 'src')]
        import torch
        from experiments.robotwin.eraf_action_protocol import validate_action_parent
        base = torch.load(strongest, map_location='cpu', weights_only=False)
        for name, path in [('eraf_fg', plan['arms']['eraf_fg']['parent']), ('eraf_only', plan['ordinary_bootstrap'])]:
            parent = torch.load(path, map_location='cpu', weights_only=False)
            assert parent['mot_trainable'].keys() == base['mot_trainable'].keys()
            assert all(torch.equal(v, base['mot_trainable'][k]) for k, v in parent['mot_trainable'].items())
            assert parent['context_injection_mode'] == 'context_residual_v1'
            validate_action_parent(parent, stage='joint' if name == 'eraf_fg' else 'interface', eraf='on', fg=plan['arms'][name]['fg'])
            del parent
        for name in ['no_eraf', 'fg_only']:
            validate_action_parent(base, stage='joint', eraf='off', fg=plan['arms'][name]['fg'], warm_policy=True)
        del base
        write('initial_policy_identity.json', dict(complete=True, all_action_policy_tensors_identical=True,
            strongest_checkpoint=strongest, new_training_updates_at_check=0))
        report['stage'] = 'training'
        active = {}
        for arm, spec in plan['arms'].items():
            stage = 'interface' if arm == 'eraf_only' else 'joint'
            key = arm + '_' + stage
            active[key] = (arm, stage, launch(key, train_command(plan, root, arm, interface=stage == 'interface'), spec['gpus']))
        while active:
            budget()
            for name, (arm, stage, proc) in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                report['jobs'][name]['exit_code'] = rc
                write('driver.json', report)
                del active[name]
                if rc:
                    raise RuntimeError(f'{name} failed: {rc}')
                audit(arm, stage)
                if stage == 'interface':
                    key = arm + '_joint'
                    active[key] = (arm, 'joint', launch(key, train_command(plan, root, arm), plan['arms'][arm]['gpus']))
            if active:
                time.sleep(5)
        report['stage'] = 'evaluating'
        pending = [(group, task, arm) for group, spec in plan['groups'].items() for task in spec['tasks'] for arm in plan['arms']]
        active = {}
        while pending or active:
            budget()
            for name, gpu in list(active.items()):
                rc = processes[name].poll()
                if rc is not None:
                    report['jobs'][name]['exit_code'] = rc
                    del active[name]
                    write('driver.json', report)
                    if rc:
                        raise RuntimeError(f'{name} failed: {rc}')
            for gpu in sorted(set(range(6)) - set(active.values())):
                if not pending:
                    break
                group, task, arm = pending.pop(0)
                spec = plan['groups'][group]
                key = f'eval_{group}_{arm}_{task}'
                cmd = ['/opt/conda/bin/python', '-u', str(REPO / 'scripts/eval_robotwin_eraf_fg.py'), 'worker',
                    '--output', str(root / ('eval_' + group) / arm / 'dev'), '--checkpoint', str(root / arm / 'joint/step_000200.pt'),
                    '--manifest', plan['manifest'], '--catalog-root', spec['catalog'], '--episodes', str(spec['episodes']),
                    '--tasks', task, '--policy-kind', 'repair', '--eraf', plan['arms'][arm]['eraf'], '--conditions', 'counterfactual',
                    '--gpu', str(gpu), '--videos', '--skip-file-hashes', '--interventions', str(REPO / 'configs/eval/robotwin_cis_ten_tasks.json')]
                launch(key, cmd, [gpu])
                active[key] = gpu
            if active:
                time.sleep(5)
        scores = {arm: {} for arm in plan['arms']}
        for group, spec in plan['groups'].items():
            for arm in plan['arms']:
                dest = root / ('eval_' + group) / arm / 'dev'
                subprocess.run(['/opt/conda/bin/python', str(REPO / 'scripts/eval_robotwin_eraf_fg.py'), 'summarize',
                    '--output', str(dest), '--checkpoint', str(root / arm / 'joint/step_000200.pt'), '--catalog-root', spec['catalog'],
                    '--episodes', str(spec['episodes']), '--tasks', *spec['tasks'], '--conditions', 'counterfactual', '--skip-file-hashes'],
                    cwd=REPO, env=env, check=True, timeout=60, stdout=subprocess.DEVNULL)
                summary = json.loads((dest / 'summary.json').read_text())
                assert summary['complete'] and summary['episodes'] == 5 * spec['episodes']
                scores[arm][group] = {cell['source_task']: cell['selected_goal_successes'] for cell in summary['cells']}
                for task in spec['tasks']:
                    suffix = Path(task) / 'demo_clean/counterfactual/initial_states.json'
                    control = RUNS / 'robotwin_cf_priority/20260907-1543-warmoff200-cf4' / ('eval_' + group) / 'ordinary_cf_200/dev' / suffix
                    assert json.loads((dest / suffix).read_text()) == json.loads(control.read_text())
        for arm in scores:
            scores[arm]['ten_task_macro_cf'] = sum(sum(scores[arm][g].values()) / spec['episodes'] for g, spec in plan['groups'].items()) / 10
        write('comparison.json', dict(complete=True, models=scores, episodes=180, matched_initial_states=True,
            strongest_archived_macro_cf=0.35, independent_test=False, correct_evaluated=False))
        report.update(complete=True, stage='complete')
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        for proc in processes.values():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for name, proc in processes.items():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            report['jobs'][name]['exit_code'] = proc.returncode
        report['finished'] = datetime.now().astimezone().isoformat()
        write('driver.json', report)


if __name__ == '__main__':
    main()
