import json

import pytest

from scripts import run_robotwin_five_task_independent40 as independent
from scripts.run_robotwin_formal_five40 import TASKS, sha, read, write, records
from test_robotwin_formal_five40 import matrix


@pytest.fixture
def completed_dev(matrix):
    root, previous = matrix
    source = root/'training-input.json'
    write(source, dict(training=True))
    arms = {}
    for index, arm in enumerate(previous['models']):
        checkpoint = root/arm/'joint/step_000200.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(arm.encode())
        arms[arm] = dict(final_checkpoint=str(checkpoint), final_sha256=sha(checkpoint),
                         eraf='off' if arm == 'no_eraf' else 'on')
        write(checkpoint.parent/'freeze_audit.json', dict(complete=True,
            optimizer_checkpoint_and_contract_match=True, unexpected_changes=[],
            changed_guard_tensors=0 if arm=='no_eraf' else 43))
        for task in TASKS:
            folder = root/'evaluation'/arm/task/'demo_clean/counterfactual'
            complete = read(folder/'complete.json')
            complete.update(checkpoint_sha256=sha(checkpoint), policy_kind='repair', memory_mode='carry',
                            deployment=dict(action_horizon=32, replan_steps=24, inference_steps=10))
            write(folder/'complete.json', complete)
            episodes = records(folder/'episodes.jsonl')
            for i, episode in enumerate(episodes):
                episode.update(counterfactual_goal_ever_success=i<(10+5*index), source_task=task,
                               condition='counterfactual', selected_goal='counterfactual', instruction_goal='counterfactual')
            (folder/'episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    write(root/'protocol.json', dict(format='robotwin_five_task_expert_trial_v1', steps=200,
        smoke_only=False, dev_episodes=40, tasks=TASKS, input_sha256={str(source):sha(source)}))
    write(root/'status.json', dict(complete=True, status='complete', arms=arms, jobs={'eval':{'exit_code':0}}))
    write(root/'terminal_verification.json', dict(complete=True, steps_per_arm=200, episodes=600))
    write(root/'comparison.json', dict(desired_order_on_dev=True))
    write(root/'final_models.json', arms)
    return root


def test_nomination_binds_all_three_actual_final_checkpoints(completed_dev):
    source, models, bindings, result = independent.candidates(completed_dev)
    assert set(models) == {'no_eraf', 'eraf_only', 'eraf_fg'}
    assert result['desired_order_observed'] and not result['independent_test']
    for model in models.values(): assert bindings[model['checkpoint']] == sha(model['checkpoint'])


def test_raw_dev_outcomes_override_a_claimed_positive_summary(completed_dev):
    for task in TASKS:
        path = completed_dev/'evaluation/eraf_fg'/task/'demo_clean/counterfactual/episodes.jsonl'
        episodes = records(path)
        for i, episode in enumerate(episodes): episode['counterfactual_goal_ever_success'] = i < 10
        path.write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    with pytest.raises(ValueError, match='full DEV matrix'): independent.candidates(completed_dev)


@pytest.mark.parametrize('mutation', ['partial', 'failed_job', 'changed_model', 'changed_input', 'bad_audit', 'bad_mode', 'comparison_disagrees'])
def test_candidate_admission_rejects_unverified_results(completed_dev, mutation):
    root = completed_dev
    if mutation in ('partial', 'failed_job', 'bad_mode'):
        status = read(root/'status.json')
        if mutation == 'partial': status['complete'] = False
        elif mutation == 'failed_job': status['jobs']['eval']['exit_code'] = 1
        else: status['arms']['no_eraf']['eraf'] = 'on'
        write(root/'status.json', status)
    elif mutation == 'changed_model': (root/'eraf_fg/joint/step_000200.pt').write_bytes(b'changed')
    elif mutation == 'changed_input': write(root/'training-input.json', dict(training=False))
    elif mutation == 'comparison_disagrees': write(root/'comparison.json', dict(desired_order_on_dev=False))
    else:
        path = root/'eraf_fg/joint/freeze_audit.json'
        audit = read(path); audit['unexpected_changes'] = ['video_adapter']; write(path, audit)
    with pytest.raises(ValueError): independent.candidates(root)


def test_used_test_namespace_is_rejected_without_resampling(tmp_path, monkeypatch):
    monkeypatch.setattr(independent, 'RUNS', tmp_path)
    starts = {task:91770000+1000*i for i, task in enumerate(TASKS)}
    write(tmp_path/'manifest.json', dict(states=[dict(scene_seed=91370000)]))
    seeds, receipts = independent.exclusion_inventory(starts)
    assert seeds == {91370000} and len(receipts) == 1
    write(tmp_path/'manifest.json', dict(states=[dict(scene_seed=91770999)]))
    with pytest.raises(ValueError, match='already accessed'): independent.exclusion_inventory(starts)


def test_namespace_bounds_are_checked_for_every_task(tmp_path, monkeypatch):
    monkeypatch.setattr(independent, 'RUNS', tmp_path)
    starts = {task:91799000+1000*i for i, task in enumerate(TASKS)}
    with pytest.raises(ValueError): independent.exclusion_inventory(starts)
