"""Matched controls for the prespecified trajectory-target800 candidate."""
from collections import Counter
import json
import math
from pathlib import Path

from experiments.robotwin.trajectory_target import mixture_stream
from scripts.run_robotwin_trajectory_target_trial import coverage, require_coverage

STEPS = 800
SPECS = {'no_eraf': {'eraf': 'off', 'fg': 'off'},
         'eraf_only': {'eraf': 'on', 'fg': 'off'},
         'fg_only': {'eraf': 'off', 'fg': 'full'}}
BASE_COMMIT = 'f794d7176b3d5f4dcb3603040ee73c5d0309ac4f'


def command(previous, root, arm, repo):
    if arm not in SPECS:
        raise ValueError('Unknown control.')
    cmd = list(previous)
    cmd[6] = str(Path(repo) / 'scripts/train_robotwin_eraf_fg_action.py')
    for flag, value in [('--nproc_per_node', '6'), ('--steps', '800'), ('--save-every', '800'),
                        ('--output', str(Path(root) / arm / 'joint')),
                        ('--action-objective', 'trajectory_target_rollout_v1')]:
        cmd[cmd.index(flag) + 1] = value
    for flag, value in [('--eraf', SPECS[arm]['eraf']), ('--fg', SPECS[arm]['fg'])]:
        if cmd[cmd.index(flag) + 1] != value:
            raise ValueError('Source arm has different ablation switches.')
    return cmd


def common_command(cmd):
    """Exclude only declared ablation switches, lineage paths and output path."""
    cmd = list(cmd)
    cmd[6] = '<unchanged-trainer>'
    for flag in ['--checkpoint', '--output', '--eraf', '--fg', '--identity-audit']:
        if flag in cmd:
            index = cmd.index(flag)
            del cmd[index:index + 2]
    return [x for x in cmd if x not in ['--warm-policy', '--zero-context-joint']]


def audit_arm(joint, rows, fg, steps=STEPS):
    """Recount actual rank journals, including the exact target-only objective."""
    joint = Path(joint)
    plan = json.loads((joint / 'plan.json').read_text())
    required = dict(steps=800, world_size=6, global_batch=12, seed=42, start_optimizer_step=0,
                    fg=fg, correct_count=0, cf_count=4, action_objective='trajectory_target_rollout_v1')
    if any(plan.get(k) != v for k, v in required.items()):
        raise ValueError('Actual training plan differs from the matched recipe.')
    if not 1 <= steps <= STEPS:
        raise ValueError('Invalid audit prefix.')
    journals = []
    for rank in range(6):
        lines = (joint / f'rank{rank}.jsonl').read_text().splitlines()
        if len(lines) < steps or (steps == STEPS and len(lines) != steps):
            raise ValueError('Missing or excess optimizer steps.')
        journals.append([json.loads(x) for x in lines[:steps]])
    stream = mixture_stream(rows, 42, fg)
    ids = []
    kinds = Counter()
    for i in range(steps):
        batch = next(stream)
        if any(j[i]['step'] != i + 1 or not math.isfinite(j[i]['grad_norm']) for j in journals):
            raise ValueError('Invalid step sequence or norm.')
        if len({j[i]['grad_norm'] for j in journals}) != 1:
            raise ValueError('Ranks disagree on gradient norm.')
        for rank, journal in enumerate(journals):
            examples = journal[i]['examples']
            expected = batch[rank::6]
            if [e['id'] for e in examples] != [r['id'] for r in expected]:
                raise ValueError('Actual sample IDs differ from deterministic schedule.')
            for e, r in zip(examples, expected):
                if (r['replay_split'] != 'train' or e['seen_variant'] is not None
                        or e['supervised_languages'] != ['target'] or e['conditional_difference']
                        or e['action_objective'] != 'trajectory_target_rollout_v1'
                        or e['denoising_steps'] != 10 or e['executed_horizon'] != 24
                        or e['gradient_horizon'] != 'all_ten_steps'
                        or not math.isfinite(e['deployed_objective'])):
                    raise ValueError('Actual supervised objective differs.')
                for flag in ['ordinary_target_trajectory', 'ordinary_cf_control', 'initial_expert_anchor']:
                    if e[flag] != bool(r.get(flag)):
                        raise ValueError('Actual training stratum differs.')
                ids.append(e['id'])
                kinds['fg' if r.get('fg_correction') else 'replacement' if r.get('ordinary_cf_control') else 'common'] += 1
    result = coverage(rows, ids)
    if steps == STEPS:
        if fg == 'full':
            require_coverage(result)
        elif any(v['minimum_exposure'] < 2 for k, v in result.items() if k.endswith('|ordinary')):
            raise ValueError('An ordinary control trajectory row was omitted.')
        if fg == 'off' and any(v['exposures'] for k, v in result.items() if k.endswith('|fg')):
            raise ValueError('FG-off arm trained on FG data.')
    return dict(complete=True,optimizer_steps=steps,actual_examples=len(ids),counts=dict(kinds),
                exact_six_rank_schedule=True,all_rows_train=True,all_rank_norms_equal=True,coverage=result)


def audit_pairing(joints, rows):
    settings = dict(SPECS, eraf_fg={'eraf': 'on', 'fg': 'full'})
    if set(joints) != set(settings):
        raise ValueError('All four actual arms are required.')
    audits = {a: audit_arm(p, rows, settings[a]['fg']) for a, p in joints.items()}
    full = mixture_stream(rows, 42, 'full')
    off = mixture_stream(rows, 42, 'off')
    shared = replaced = 0
    for _ in range(STEPS):
        for x, y in zip(next(full), next(off)):
            if x.get('fg_correction'):
                if (not y.get('ordinary_cf_control') or y.get('fg_correction')
                        or (x['pair_id'], x['task_config']) != (y['pair_id'], y['task_config'])):
                    raise ValueError('FG replacement task or slot differs.')
                replaced += 1
            else:
                if x != y:
                    raise ValueError('Common example differs.')
                shared += 1
    if (shared, replaced) != (6400, 3200):
        raise ValueError('Unmatched number of shared/replaced slots.')
    return dict(complete=True,actual_examples_per_arm=9600,actual_examples_all_arms=38400,
                common_examples_per_arm=shared,matched_fg_or_ordinary_slots_per_arm=replaced,
                same_step_slot_noise_seed_formula='42 + (step - 1) * 12 + slot',
                same_world_size=6,arms=audits)


def comparison_config(previous, root, groups):
    methods = dict(previous['methods'])
    for arm in SPECS:
        alias = 'pre_trajectory_' + arm
        if alias in methods:
            raise ValueError('Historical control alias already exists.')
        methods[alias] = methods[arm]
        methods[arm] = {g: str(Path(root) / ('eval_' + g) / arm / 'dev') for g in groups}
    return dict(target='eraf_fg', methods=methods)
