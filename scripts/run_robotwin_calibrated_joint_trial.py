#!/usr/bin/env python3
"""Matched joint200 from calibrated, verified identity bootstraps, followed by90CF."""
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
from scripts.run_robotwin_ten_task_action_expansion import train_command
from scripts.compare_robotwin_ten_task_methods import build_report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--deadline', required=True)
    ap.add_argument('--identity-root', type=Path,
                    help='Use a separately audited residual semantic preparation.')
    ap.add_argument('--prior-comparison-config', type=Path,
                    help='Preserve all prior controls when extending a completed comparison.')
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None:
        ap.error('Require an absolute timezone-aware cutoff.')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Commit code before execution.')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('GPUs are occupied.')
    runs = Path('/root/gpufree-data/LF-FastWAM/runs')
    identity = args.identity_root or runs / 'robotwin_calibrated_context_identity/20260908-semantic500'
    prior = runs / 'robotwin_ten_task_action_expansion/20260908-0005-joint200'
    previous = json.loads((prior / 'protocol.json').read_text())
    bootstraps = json.loads((identity / 'plan.json').read_text())
    semantic_audit = (json.loads(Path(bootstraps['training_audit']).read_text())
                      if bootstraps.get('training_audit') else None)
    if semantic_audit is not None:
        if not semantic_audit['complete'] or not semantic_audit['actual_samples_and_instruction_branches_verified']:
            raise ValueError('Semantic preparation has not passed its paired training audit.')
        comparison = json.loads(Path(bootstraps['terminal_comparison']).read_text())
        if not comparison['complete'] or not comparison['inputs_and_labels_match'] or comparison['matched_queries'] != 80:
            raise ValueError('Complete matched terminal semantic comparison required.')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        temporary = root / (name + '.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(root / name)

    plan = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        manifest=previous['manifest'], manifest_sha256=previous['manifest_sha256'], source_bank=previous['source_bank'],
        correct_teacher=previous['correct_teacher'], strongest_checkpoint=previous['strongest_checkpoint'],
        groups=previous['groups'], arms={
            'eraf_only': dict(eraf='on', fg='off', gpus=[0, 1, 2], parent=bootstraps['models']['ordinary'],
                              identity_audit=str(identity / 'identity/ordinary/summary.json')),
            'eraf_fg': dict(eraf='on', fg='full', gpus=[3, 4, 5], parent=bootstraps['models']['fg'],
                            identity_audit=str(identity / 'identity/fg/summary.json'))},
        identity_root=str(identity), joint_steps=200, joint_policy_lr=3e-6, joint_interface_lr=3e-5,
        global_batch=12, seed=42, correction_weight=.1, correct_count=2, cf_count=4, correct_weight=1, cf_weight=4,
        initialization='Opt-in zero_context_joint: exact30-query residual identity, no interface-warmup updates.',
        semantic_qualification_passed=False,
        scientific_scope='Exploratory CF trial after the recorded matched semantic preparation. Both new arms share policy initialization, action steps, data schedule and learning rates. Historical controls remain in the comparison; semantic diagnostics alone do not establish CF superiority.',
        semantic_training_audit=bootstraps.get('training_audit'),
        terminal_semantic_comparison=bootstraps.get('terminal_comparison'),
        checkpoint_selection='Predeclared final joint200 only, fresh optimizer. No checkpoint chosen from partial CF results.',
        total_eval_episodes=90, correct_evaluated=False, independent_test=False,
        deadline=args.deadline, platform_shutdown=None, model_storage='server_only')
    write('protocol.json', plan)
    report = dict(complete=False, stage='preflight', jobs={})
    processes = {}
    env = os.environ | dict(PATH='/opt/conda/bin:' + os.environ['PATH'], PYTHONPATH=str(REPO / 'src') + ':' + str(REPO),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')

    def budget():
        if time.time() >= cutoff.timestamp():
            raise TimeoutError('Trial cutoff; server remains on.')
        if shutil.disk_usage(root).free < 5 * 1024**3:
            raise RuntimeError('Five GiB disk reserve reached.')

    def launch(name, command, gpus):
        with (root / (name + '.log')).open('x') as log:
            process = subprocess.Popen(command, cwd=REPO, env=env | {'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus))},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = process
        report['jobs'][name] = dict(pid=process.pid, gpus=gpus, command=command)
        write('driver.json', report)
        return process

    def stop(signum, frame):
        raise InterruptedError(f'Signal{signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        budget()
        # Actual prior joint files provide the storage estimate for identical scopes.
        def original_bytes(path):
            if path.exists():
                return path.stat().st_size
            receipt = json.loads(path.with_suffix('.compacted.json').read_text())
            if not receipt['exact_tensors_and_metadata_verified_after_readback']:
                raise ValueError('Unverified compacted size reference.')
            return receipt['original_bytes']
        expected_bytes = sum(original_bytes(prior / arm / 'joint' / name)
                             for arm in plan['arms'] for name in ('step_000200.pt', 'optimizer_last.pt'))
        if shutil.disk_usage(root).free < 5 * 1024**3 + expected_bytes + 160 * 1024**2:
            raise RuntimeError('Insufficient reserve for both final models, optimizers and evaluation media.')
        import torch
        from experiments.robotwin.eraf_fg_bridge import file_sha256
        from experiments.robotwin.eraf_action_protocol import validate_action_parent, validate_joint_identity_audit
        base = torch.load(plan['strongest_checkpoint'], map_location='cpu', weights_only=False)
        proofs = {}
        for arm, spec in plan['arms'].items():
            mode = 'ordinary' if arm == 'eraf_only' else 'fg'
            assert json.loads((identity / (mode + '.exit.json')).read_text())['exit_code'] == 0
            payload = torch.load(spec['parent'], map_location='cpu', weights_only=False)
            if payload['stage'] == 'grounding':
                if semantic_audit is None:
                    raise ValueError('Grounding parents require their completed paired training audit.')
                evidence = semantic_audit['arms'][mode]
                if (evidence['checkpoint_sha256'] != file_sha256(spec['parent'])
                        or not evidence['policy_tensors_identical'] or not evidence['only_declared_semantics_changed']
                        or not evidence['residual_output_still_zero']
                        or comparison['models'][mode + '_paired500']['checkpoint_sha256'] != evidence['checkpoint_sha256']):
                    raise ValueError('Semantic or diagnostic evidence does not bind this actual parent.')
            validate_action_parent(payload, stage='joint', eraf='on', fg=spec['fg'], zero_context_joint=True)
            proof = json.loads(Path(spec['identity_audit']).read_text())
            validate_joint_identity_audit(proof, checkpoint_sha256=file_sha256(spec['parent']),
                                          manifest_sha256=file_sha256(plan['manifest']))
            assert proof['queries'] == 30 and proof['task_domain_count'] == 15
            assert payload['mot_trainable'].keys() == base['mot_trainable'].keys()
            assert all(torch.equal(v, base['mot_trainable'][k]) for k, v in payload['mot_trainable'].items())
            proofs[arm] = dict(checkpoint_sha256=proof['checkpoint_sha256'], queries=30,
                               identity_audit_sha256=file_sha256(spec['identity_audit']))
            del payload
        del base
        write('initial_policy_identity.json', dict(complete=True, strongest_policy_tensors_identical=True,
                                                   all_deployed_identity_queries_exact=True, arms=proofs))
        report['stage'] = 'training'
        active = {}
        for arm, spec in plan['arms'].items():
            command = train_command(plan, root, arm) + ['--zero-context-joint', '--identity-audit', spec['identity_audit']]
            active[arm] = launch(arm + '_joint', command, spec['gpus'])
        while active:
            budget()
            for arm, process in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                report['jobs'][arm + '_joint']['exit_code'] = rc
                del active[arm]
                write('driver.json', report)
                if rc:
                    raise RuntimeError(f'{arm} failed: {rc}')
                subprocess.run(['/opt/conda/bin/python', str(REPO / 'scripts/audit_robotwin_eraf_fg_action_stage.py'),
                    '--output', str(root / arm / 'joint')], cwd=REPO, env=env, check=True, timeout=120, stdout=subprocess.DEVNULL)
                audit = json.loads((root / arm / 'joint/freeze_audit.json').read_text())
                assert audit['complete'] and audit['all_sampled_state_ids_match'] and not audit['unexpected_changes']
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
                name = f'eval_{group}_{arm}_{task}'
                command = ['/opt/conda/bin/python', '-u', str(REPO / 'scripts/eval_robotwin_eraf_fg.py'), 'worker',
                    '--output', str(root / ('eval_' + group) / arm / 'dev'), '--checkpoint', str(root / arm / 'joint/step_000200.pt'),
                    '--manifest', plan['manifest'], '--catalog-root', spec['catalog'], '--episodes', str(spec['episodes']),
                    '--tasks', task, '--policy-kind', 'repair', '--eraf', 'on', '--conditions', 'counterfactual',
                    '--gpu', str(gpu), '--videos', '--skip-file-hashes', '--interventions', str(REPO / 'configs/eval/robotwin_cis_ten_tasks.json')]
                launch(name, command, [gpu])
                active[name] = gpu
            if active:
                time.sleep(5)
        for group, spec in plan['groups'].items():
            for arm in plan['arms']:
                subprocess.run(['/opt/conda/bin/python', str(REPO / 'scripts/eval_robotwin_eraf_fg.py'), 'summarize',
                    '--output', str(root / ('eval_' + group) / arm / 'dev'), '--checkpoint', str(root / arm / 'joint/step_000200.pt'),
                    '--catalog-root', spec['catalog'], '--episodes', str(spec['episodes']), '--tasks', *spec['tasks'],
                    '--conditions', 'counterfactual', '--skip-file-hashes'], cwd=REPO, env=env, check=True,
                    timeout=60, stdout=subprocess.DEVNULL)
        config = (json.loads(args.prior_comparison_config.read_text()) if args.prior_comparison_config else
                  json.loads((runs / 'robotwin_ten_task_semantic_audit/20260908-post-joint200/plan.json').read_text())['comparison'])
        methods = dict(config['methods'])
        for arm in plan['arms']:
            archive_name = ('pre_cross_goal_' if args.prior_comparison_config else 'previous_') + arm
            if archive_name in methods:
                raise ValueError('Historical comparison name already exists: ' + archive_name)
            methods[archive_name] = methods[arm]
            methods[arm] = {group: str(root / ('eval_' + group) / arm / 'dev') for group in plan['groups']}
        config = dict(target='eraf_fg', methods=methods)
        write('comparison_config.json', config)
        write('comparison.json', build_report(config))
        report.update(complete=True, stage='complete')
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for name, process in processes.items():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            report['jobs'][name]['exit_code'] = process.returncode
        report['finished'] = datetime.now().astimezone().isoformat()
        write('driver.json', report)


if __name__ == '__main__':
    main()
