"""Explicit admission and rollout configuration for new-task FG corrections.

This protocol preserves the original full-goal replay requirements. It only
extends task coverage, and records the actual collector model configuration.
"""
from __future__ import annotations

from datetime import datetime
import shutil

from experiments.robotwin.pgc_data import (
    ROBOTWIN_REPLACEMENT_PAIR_SPECS, ROBOTWIN_TEN_TASK_EXTRA_SPECS,
)

FORMAT = 'robotwin_expanded_full_goal_failure_replay_v2'
SPECS = (*ROBOTWIN_REPLACEMENT_PAIR_SPECS, *ROBOTWIN_TEN_TASK_EXTRA_SPECS)
TASKS = tuple(s.source_task for s in SPECS)
SEED_RANGES = {'train': (86000000, 87000000), 'replay_holdout': (87000000, 88000000)}


def validate_scope(task, split, start_seed, attempts):
    if task not in TASKS or split not in SEED_RANGES:
        raise ValueError('Expanded FG requires an explicit new task and data split.')
    lower, upper = SEED_RANGES[split]
    if not (attempts > 0 and lower <= start_seed < start_seed + attempts <= upper):
        raise ValueError('Expanded FG seed range is outside its reserved data split.')


def canonical_instructions(spec):
    if spec not in SPECS:
        raise ValueError('No expanded FG instruction contract for this task.')
    return {'source': spec.source_instruction, 'target': spec.counterfactual_instruction}


def validate_header(row):
    validate_scope(row['source_task'], row['replay_split'], int(row['scene_seed']), 1)
    spec = next(s for s in SPECS if s.source_task == row['source_task'])
    if (row.get('pair_id') != spec.pair_id
            or row.get('source_instruction') != spec.source_instruction
            or row.get('counterfactual_instruction') != spec.counterfactual_instruction):
        raise ValueError('Expanded FG task, pair and canonical instructions disagree.')
    if row.get('policy_kind') not in {'legacy', 'repair'} or row.get('eraf_mode') not in {'on', 'off'}:
        raise ValueError('Expanded FG must declare its actual policy loader and ERAF mode.')
    if row['policy_kind'] == 'legacy' and row['eraf_mode'] != 'off':
        raise ValueError('Legacy collection cannot claim ERAF was enabled.')
    if row.get('memory_mode') != 'carry' or row.get('policy_seed') != 42:
        raise ValueError('Expanded FG collection requires the declared deployment seed and memory.')
    for key, size in [('collector_commit', 40), ('input_manifest_sha256', 64)]:
        value = row.get(key)
        if not isinstance(value, str) or len(value) != size or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing expanded FG provenance: ' + key)


def load_collection_policy(checkpoint, manifest, *, policy_kind, eraf_mode, task, task_config):
    if policy_kind == 'repair':
        from experiments.robotwin.eraf_fg_bridge import load_policy
        policy = load_policy(checkpoint, manifest, seed=42)
        policy.model.policy_guard_enabled = eraf_mode == 'on'
    elif policy_kind == 'legacy' and eraf_mode == 'off':
        from types import SimpleNamespace
        from scripts.train_robotwin_cf_decision_adapter import load_policy
        policy = load_policy(SimpleNamespace(checkpoint=checkpoint, seed=42), manifest)
    else:
        raise ValueError('Unsupported collector policy/ERAF combination.')
    policy.task_name, policy.task_config = task, task_config
    if bool(policy.model.policy_guard_enabled) != (eraf_mode == 'on'):
        raise ValueError('Loaded policy ERAF mode differs from recorded mode.')
    return policy


class CollectionBudgetExceeded(RuntimeError):
    pass


def check_budget(root, deadline, reserve_gib):
    if deadline is not None and datetime.now().astimezone() >= datetime.fromisoformat(deadline):
        raise CollectionBudgetExceeded('Collection deadline reached; retain partial evidence.')
    if shutil.disk_usage(root).free < reserve_gib * 1024**3:
        raise CollectionBudgetExceeded('Collection disk reserve reached; retain partial evidence.')
