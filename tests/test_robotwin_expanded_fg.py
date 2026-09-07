import sys
from types import SimpleNamespace

import pytest

from experiments.robotwin.expanded_fg import (
    FORMAT, SPECS, validate_scope, canonical_instructions, load_collection_policy,
)
from experiments.robotwin.eraf_fg_contract import validate_correction
from scripts.collect_robotwin_eraf_fg import historical_scene_keys


def record(spec):
    return dict(format=FORMAT, source_task=spec.source_task, pair_id=spec.pair_id,
        task_config='demo_clean', scene_seed=86000001, replay_split='train',
        source_instruction=spec.source_instruction, counterfactual_instruction=spec.counterfactual_instruction,
        policy_kind='repair', eraf_mode='on', memory_mode='carry', policy_seed=42,
        collector_commit='a'*40, input_manifest_sha256='b'*64,
        source_goal_ever_success=True, counterfactual_goal_ever_success=False,
        failure_kind='source_directed', capture_origin='source_directed_failure_replan',
        capture_action_index=48, prefix_action_count=48, initial_state_sha256='a'*64,
        capture_state_sha256='b'*64, prefix_action_sha256='c'*64,
        correction_action_sha256='d'*64, rollout_checkpoint_sha256='e'*64,
        verification_policy='full_goal', full_goal_verified=True, counterfactual_goal_final_success=True,
        both_grippers_open_final=True, verified_replay_count=2, captured_state_count=30,
        candidate_count=4, recorded_action_count=83, state_atol=1e-7, replay_state_max_abs=0.)


@pytest.mark.parametrize('spec', SPECS)
def test_expanded_tasks_have_bound_goals_and_are_excluded_from_reuse(spec):
    row = record(spec)
    assert validate_correction(row) == row
    assert canonical_instructions(spec)['target'] == row['counterfactual_instruction']
    legacy = dict(row, format='robotwin_eraf_fg_failure_replay_v1')
    with pytest.raises(ValueError, match='targets'):
        validate_correction(legacy)
    minimal = {k: row[k] for k in ('pair_id', 'task_config', 'scene_seed')}
    assert historical_scene_keys([minimal]) == {(spec.source_task, 'demo_clean', 86000001)}


@pytest.mark.parametrize('change', [
    {'pair_id': 'place_a2b_left_to_right'}, {'counterfactual_instruction': 'Place it on the pad.'},
    {'source_instruction': 'invented'}, {'scene_seed': 91300000}, {'scene_seed': 91700000},
    {'replay_split': 'replay_holdout'}, {'policy_kind': 'legacy'}, {'memory_mode': 'reset'},
    {'collector_commit': ''}, {'input_manifest_sha256': ''}, {'policy_seed': 0},
    {'capture_action_index': 0, 'prefix_action_count': 0}, {'verified_replay_count': 1},
    {'counterfactual_goal_final_success': False}, {'both_grippers_open_final': False},
    {'replay_state_max_abs': 2e-7}, {'recorded_action_count': 11},
    {'counterfactual_goal_ever_success': True},
])
def test_expansion_keeps_full_goal_physical_replay_and_lineage_requirements(change):
    with pytest.raises(ValueError):
        validate_correction(record(SPECS[0]) | change)


def test_expanded_namespaces_cannot_cross_split_or_enter_dev_test():
    validate_scope(SPECS[0].source_task, 'train', 86999999, 1)
    validate_scope(SPECS[0].source_task, 'replay_holdout', 87000000, 1)
    for split, seed, attempts in [('train', 86999999, 2), ('train', 87000000, 1),
                                  ('replay_holdout', 86000000, 1), ('train', 91700000, 1)]:
        with pytest.raises(ValueError):
            validate_scope(SPECS[0].source_task, split, seed, attempts)


def test_pilot_matrix_covers_all_new_tasks_and_disjoint_holdout():
    from scripts.run_robotwin_expanded_fg_pilots import pilot_jobs
    jobs = pilot_jobs()
    assert len(jobs) == 6 and {j['gpu'] for j in jobs} == set(range(6))
    assert {j['task'] for j in jobs if j['split'] == 'train'} == {s.source_task for s in SPECS}
    assert len({(j['task'], seed) for j in jobs for seed in range(j['start_seed'], j['start_seed']+3)}) == 18
    for job in jobs:
        validate_scope(job['task'], job['split'], job['start_seed'], 3)


def test_repair_collector_uses_real_repair_loader_and_records_actual_task(monkeypatch):
    calls = []
    policy = SimpleNamespace(model=SimpleNamespace(policy_guard_enabled=True))
    def factory(checkpoint, manifest, **kwargs):
        calls.append((checkpoint, manifest, kwargs))
        return policy
    monkeypatch.setitem(sys.modules, 'experiments.robotwin.eraf_fg_bridge', SimpleNamespace(load_policy=factory))
    result = load_collection_policy('repair.pt', {'states': []}, policy_kind='repair', eraf_mode='on',
                                    task='place_empty_cup', task_config='demo_randomized')
    assert result is policy and policy.model.policy_guard_enabled
    assert (policy.task_name, policy.task_config) == ('place_empty_cup', 'demo_randomized')
    assert calls == [('repair.pt', {'states': []}, {'seed': 42})]
    load_collection_policy('repair.pt', {}, policy_kind='repair', eraf_mode='off',
                           task='place_mouse_pad', task_config='demo_clean')
    assert not policy.model.policy_guard_enabled
    with pytest.raises(ValueError):
        load_collection_policy('x', {}, policy_kind='legacy', eraf_mode='on', task='x', task_config='x')
