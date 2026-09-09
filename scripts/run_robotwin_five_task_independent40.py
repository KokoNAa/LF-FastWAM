#!/usr/bin/env python3
"""Freeze all three complete DEV candidates before a new independent 600 CF test."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from scripts import run_robotwin_formal_five40 as formal
from scripts.run_robotwin_formal_five40 import read, sha, write, records, RUNS, TASKS
from scripts.report_robotwin_five_task_comparison import report, markdown, ARMS


def serial_evidence(trial, plan, status, terminal):
    """Require the audited current control -> ERAF -> FG chain before nomination."""
    from experiments.robotwin.current_noeraf import current_baseline
    added = dict(no_eraf=0, eraf_only=200, eraf_fg=200)
    cumulative = dict(no_eraf=0, eraf_only=200, eraf_fg=400)
    for document in (plan, status, terminal):
        if (document.get('additional_optimizer_steps') != added
                or document.get('cumulative_optimizer_steps_since_baseline') != cumulative):
            raise ValueError('Serial training budgets must remain 0/200/400 since the current control.')
    if (status.get('train_arms') != ['eraf_only', 'eraf_fg']
            or not status.get('eraf_initialization_matches_current_no_eraf')
            or not status.get('fg_initialization_matches_completed_eraf')):
        raise ValueError('Both serial initialization checks must be complete.')
    source = Path(plan['current_no_eraf_trial'])
    baseline, digest = current_baseline(source, read(source/'final_models.json'), read(source/'completion_audit.json'))
    reference = status['arms']['no_eraf']
    if (plan['baseline_checkpoint'] != baseline or plan['baseline_sha256'] != digest
            or reference['final_checkpoint'] != baseline or reference['final_sha256'] != digest
            or not reference.get('frozen_reference') or reference.get('additional_optimizer_steps') != 0):
        raise ValueError('The fixed control must be the unchanged current final no-ERAF model.')
    paths = [trial/'completion_audit.json', source/'final_models.json', source/'completion_audit.json',
             trial/'eraf_fg/joint/plan.json', trial/'eraf_fg/joint/continuation_initialization.json']
    proof = read(paths[0])
    expected_jobs = {'prepare_current_parents', 'identity_eraf_only', 'train_eraf_only', 'audit_eraf_only',
                     'train_eraf_fg', 'audit_eraf_fg'}
    expected_jobs |= {'catalog_'+t for t in TASKS}
    expected_jobs |= {'eval_'+a+'_'+t for a in ARMS for t in TASKS}
    if (not proof.get('complete') or proof.get('format') != 'robotwin_serial3_completion_audit_v1'
            or proof.get('root') != str(trial) or proof.get('smoke_only')
            or proof.get('code_commit') != plan['code_commit']
            or proof.get('episodes') != terminal['episodes'] or proof.get('video_count') != terminal['episodes']
            or not proof.get('eraf_initialization_matches_current_no_eraf')
            or not proof.get('fg_inherits_completed_eraf') or proof.get('controller', {}).get('live') is not False
            or set(status['jobs']) != expected_jobs or set(proof.get('jobs', {})) != expected_jobs):
        raise ValueError('A complete independent serial DEV audit is required.')
    for name, job in status['jobs'].items():
        audited = proof['jobs'][name]
        if audited['pid'] != job['pid'] or audited['exit_code'] != 0 or audited['process']['live'] is not False:
            raise ValueError('An audited development process is incomplete or changed.')
    if proof['checkpoint_sha256'] != {a:s['final_sha256'] for a,s in status['arms'].items()}:
        raise ValueError('Audited final model hashes differ from the nominated models.')
    inputs = dict(status['input_sha256'])
    if any(inputs.get(p) != h for p,h in plan['input_sha256'].items()) or proof['input_sha256'] != inputs:
        raise ValueError('Serial input bindings changed after initialization or training.')
    expected_episodes = {'evaluation/'+a+'/'+t+'/demo_clean/counterfactual/episodes.jsonl' for a in ARMS for t in TASKS}
    if set(proof['episode_sha256']) != expected_episodes:
        raise ValueError('The serial audit must cover every development cell.')
    for relative, digest in proof['episode_sha256'].items():
        if sha(trial/relative) != digest:
            raise ValueError('Development episodes changed after their independent audit.')
    parent = status['arms']['eraf_only']
    full = status['arms']['eraf_fg']
    training, initial = read(paths[3]), read(paths[4])
    if (full.get('parent') != parent['final_checkpoint'] or full.get('parent_sha256') != parent['final_sha256']
            or training.get('checkpoint') != parent['final_checkpoint']
            or training.get('continuation_parent_sha256') != parent['final_sha256']
            or not training.get('continue_eraf_with_fg') or training.get('zero_context_joint')
            or training.get('warm_policy') or training.get('resume_state') is not None
            or not initial.get('complete') or not initial.get('all_policy_and_eraf_tensors_equal')
            or initial.get('eraf_reinitialized') is not False
            or initial.get('parent_checkpoint') != parent['final_checkpoint']
            or initial.get('parent_sha256') != parent['final_sha256'] or initial.get('parent_optimizer_steps') != 200):
        raise ValueError('FG must inherit all weights of the completed current ERAF stage.')
    for arm in ('eraf_only', 'eraf_fg'):
        path = trial/arm/'joint/freeze_audit.json'
        audit = read(path)
        if (audit != proof['actual_training_audits'][arm] or audit.get('final_step') != 200
                or audit.get('global_batch') != 12 or audit.get('world_size') != len(plan['gpus'])
                or (arm == 'eraf_fg' and not audit.get('fg_continuation_parent_and_initialization_verified'))):
            raise ValueError('The independently checked serial stage audit changed: '+arm)
    return inputs, paths


def candidates(trial):
    """Recompute complete DEV comparisons and bind every selected weight file."""
    trial = Path(trial).resolve()
    plan, status, terminal = (read(trial/name) for name in ('protocol.json', 'status.json', 'terminal_verification.json'))
    serial = plan.get('format') == 'robotwin_current_no_eraf_serial_trial_v1'
    if (plan.get('format') not in ('robotwin_five_task_expert_trial_v1', 'robotwin_current_no_eraf_serial_trial_v1') or plan['steps'] != 200
        or plan['smoke_only'] or not status['complete'] or status['status'] != 'complete'
        or not terminal['complete'] or (not serial and terminal.get('steps_per_arm') != 200)
        or terminal['episodes'] != 15*plan['dev_episodes']
        or set(plan['tasks']) != set(TASKS) or set(status['arms']) != set(ARMS)):
        raise ValueError('A complete three-model final200 DEV trial is required.')
    if any(j.get('exit_code') != 0 for j in status['jobs'].values()):
        raise ValueError('A development job failed or has not exited.')
    inputs, evidence = serial_evidence(trial, plan, status, terminal) if serial else (plan['input_sha256'], [])
    result = report(trial)
    if result['independent_test'] or not result['desired_order_observed']:
        raise ValueError('The full DEV matrix must support the proposed ordering before nomination.')
    if not read(trial/'comparison.json')['desired_order_on_dev']:
        raise ValueError('Stored DEV comparison disagrees with the recomputed result.')
    models, bindings = {}, {}
    for name in ('protocol.json', 'status.json', 'terminal_verification.json', 'comparison.json', 'final_models.json'):
        bindings[str(trial/name)] = sha(trial/name)
    for path in evidence: bindings[str(path)] = sha(path)
    finals = read(trial/'final_models.json')
    for arm in ARMS:
        spec = status['arms'][arm]
        if (spec['eraf'] != ('off' if arm == 'no_eraf' else 'on')
                or (serial and spec.get('fg') != ('full' if arm == 'eraf_fg' else 'off'))):
            raise ValueError('The matched ERAF ablation mode changed: '+arm)
        checkpoint = trial/arm/'joint/step_000200.pt'
        if serial and arm == 'no_eraf': checkpoint = Path(plan['baseline_checkpoint'])
        audit_path = checkpoint.parent/'freeze_audit.json'
        audit = read(audit_path)
        if (not audit['complete'] or not audit['optimizer_checkpoint_and_contract_match'] or audit['unexpected_changes']
            or (audit['changed_guard_tensors'] > 0) != (arm != 'no_eraf')):
            raise ValueError('A frozen-parameter/optimizer audit failed: '+arm)
        expected = spec['final_sha256']
        if (spec['final_checkpoint'] != str(checkpoint) or sha(checkpoint) != expected
            or finals[arm]['final_sha256'] != expected or finals[arm]['final_checkpoint'] != str(checkpoint)):
            raise ValueError('Final candidate checkpoint binding changed: '+arm)
        models[arm] = dict(checkpoint=str(checkpoint), sha256=expected, eraf=spec['eraf'])
        bindings[str(checkpoint)] = expected
        bindings[str(audit_path)] = sha(audit_path)
    for path, digest in inputs.items():
        if sha(path) != digest: raise ValueError('Actual DEV input changed: '+path)
        bindings[path] = digest
    for relative, digest in result['source_sha256'].items(): bindings[str(trial/relative)] = digest
    return plan, models, bindings, result


def exclusion_inventory(starts):
    """Require unused namespaces and retain the full archived-scene inventory."""
    from experiments.robotwin.catalog_protocol import validate_namespace
    for start in starts.values(): validate_namespace('test', start, 400)
    seeds, receipts = set(), []
    for directory, dirs, names in os.walk(RUNS):
        dirs[:] = [d for d in dirs if d not in ('weights', 'optimizers', '.git', 'wandb')]
        for name in names:
            if name not in ('manifest.json', 'episodes.jsonl'): continue
            path = Path(directory)/name
            content = path.read_bytes()
            if not content.strip(): continue
            value = [json.loads(x) for x in content.splitlines() if x.strip()] if name.endswith('jsonl') else json.loads(content)
            rows = value.get('states') if isinstance(value, dict) else value
            if not isinstance(rows, list) or not rows or not all(type(r.get('scene_seed')) is int for r in rows): continue
            found = {r['scene_seed'] for r in rows}
            if any(start <= seed < start+1000 for start in starts.values() for seed in found):
                raise ValueError('The declared test namespace was already accessed; do not automatically resample: '+str(path))
            seeds.update(found)
            receipts.append(dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(), records=len(rows)))
    return seeds, receipts


def freeze(args):
    from experiments.robotwin.manipulation_metrics import PROTOCOL
    root, trial = args.output.resolve(), args.trial.resolve()
    if root.exists() or not root.is_relative_to(RUNS) or not trial.is_relative_to(RUNS):
        raise ValueError('Use a fresh server data-disk output and an existing server DEV trial.')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Commit the independent evaluator before execution.')
    source, models, bindings, dev = candidates(trial)
    unchanged = ['src', 'experiments', 'configs', 'scripts/eval_robotwin_eraf_fg.py',
                 'scripts/catalog_robotwin_formal_five.py', 'scripts/collect_pgc_robotwin_pairs.py',
                 'scripts/train_robotwin_cf_decision_adapter.py', 'scripts/smoke_robotwin_manipulation_metrics.py']
    subprocess.run(['git', 'diff', '--exit-code', source['code_commit'], '--', *unchanged], cwd=REPO, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if shutil.disk_usage(RUNS).free < 10*1024**3: raise ValueError('Need at least 10GiB for complete600 outputs.')
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    devices = {int(line.split(',')[0]):int(line.split(',')[1]) for line in used.splitlines()}
    if len(set(args.gpus)) != len(args.gpus) or any(g not in devices or devices[g] > 1000 for g in args.gpus):
        raise ValueError('Requested GPUs are unavailable or occupied.')
    starts = {t:args.test_seed+1000*i for i,t in enumerate(TASKS)}
    seeds, exclusions = exclusion_inventory(starts)
    required = {source['manifest']} | {str(trial/'catalog'/t/'demo_clean/correct/episodes.jsonl') for t in TASKS}
    required |= {str(Path(source['prior_formal'])/'catalog'/t/'demo_clean/correct/episodes.jsonl') for t in TASKS}
    if not required <= {x['path'] for x in exclusions}:
        raise ValueError('Training, current DEV and prior formal scenes must all be explicitly excluded.')
    config = REPO/'configs/eval/robotwin_cis_ten_tasks.json'
    bindings[str(config)] = sha(config)
    return dict(format='robotwin_five_task_independent40_v1', complete=False, status='frozen_before_test',
        frozen_at=datetime.now().astimezone().isoformat(), code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        output=str(root), deadline=args.deadline, gpus=args.gpus, models=models, tasks=TASKS,
        episodes_per_task_per_model=40, expected_episodes=600, conditions=['counterfactual'],
        manifest=source['manifest'], input_sha256=bindings, exclusions=exclusions, excluded_unique_seeds=len(seeds),
        seed_starts=starts, metrics=PROTOCOL, primary_metric='Equal-task five-task macro CF; all three frozen models and all 40 scenes/task retained.',
        checkpoint_selection=('Fixed current no-ERAF; final ERAF200 then final FG200 continued from ERAF. '
            'All three nominated by the complete paired DEV matrix before test access.'
            if source['format'] == 'robotwin_current_no_eraf_serial_trial_v1' else
            'Final step200 only, all three arms nominated by the complete paired DEV matrix before test access.'),
        training_comparison=dict(source_format=source['format'],
            additional_optimizer_steps=source.get('additional_optimizer_steps', dict.fromkeys(ARMS, 200)),
            cumulative_optimizer_steps_since_baseline=source.get('cumulative_optimizer_steps_since_baseline', dict.fromkeys(ARMS, 200)),
            equal_additional_training_budget=source['format'] != 'robotwin_current_no_eraf_serial_trial_v1'),
        nomination=dict(trial=str(trial), dev_macro_cf=dev['macro_cf'], desired_order_on_dev=True),
        task_selection='Same user-selected five tasks; no post-test task selection.',
        scope='New physical scenes in these five fixed task families; no unseen-task or asset claims.',
        independent_test=True, goal_achieved=False, training_performed=False, correct_evaluated=False,
        platform_shutdown=None, model_storage='server_only', jobs={})


def parser_setup(ap):
    ap.description = __doc__
    ap.set_defaults(gpus=[0, 1, 2])
    ap.add_argument('--trial', type=Path, required=True)
    ap.add_argument('--test-seed', type=int, default=91770000)


if __name__ == '__main__':
    formal.main(freeze_fn=freeze, parser_setup=parser_setup)
