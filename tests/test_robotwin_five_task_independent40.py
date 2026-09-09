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


@pytest.fixture
def completed_serial_dev(completed_dev):
    root = completed_dev
    plan, status, terminal = (read(root/n) for n in ('protocol.json', 'status.json', 'terminal_verification.json'))
    arms = status['arms']
    source = root.with_name(root.name+'-current-control')
    baseline = source/'no_eraf/joint/step_000200.pt'
    baseline.parent.mkdir(parents=True)
    original = root/'no_eraf/joint/step_000200.pt'
    original.rename(baseline)
    (original.parent/'freeze_audit.json').rename(baseline.parent/'freeze_audit.json')
    for arm, spec in arms.items(): spec['fg'] = 'full' if arm == 'eraf_fg' else 'off'
    arms['no_eraf'].update(final_checkpoint=str(baseline), frozen_reference=True, additional_optimizer_steps=0)
    write(source/'final_models.json', arms)
    write(source/'completion_audit.json', dict(complete=True, episodes=180,
        checkpoint_sha256={'no_eraf':sha(baseline)}))
    added = dict(no_eraf=0, eraf_only=200, eraf_fg=200)
    cumulative = dict(no_eraf=0, eraf_only=200, eraf_fg=400)
    for document in (plan, status, terminal):
        document.update(additional_optimizer_steps=added, cumulative_optimizer_steps_since_baseline=cumulative)
    plan.update(format='robotwin_current_no_eraf_serial_trial_v1', code_commit='source-code', gpus=[0,1,2],
        current_no_eraf_trial=str(source), baseline_checkpoint=str(baseline), baseline_sha256=sha(baseline))
    status.update(train_arms=['eraf_only','eraf_fg'], eraf_initialization_matches_current_no_eraf=True,
                  fg_initialization_matches_completed_eraf=True)
    parent = arms['eraf_only']
    arms['eraf_fg'].update(parent=parent['final_checkpoint'], parent_sha256=parent['final_sha256'])
    initial = dict(complete=True, parent_checkpoint=parent['final_checkpoint'], parent_sha256=parent['final_sha256'],
        parent_optimizer_steps=200, all_policy_and_eraf_tensors_equal=True, eraf_reinitialized=False)
    write(root/'eraf_fg/joint/continuation_initialization.json', initial)
    write(root/'eraf_fg/joint/plan.json', dict(checkpoint=parent['final_checkpoint'],
        continuation_parent_sha256=parent['final_sha256'], continue_eraf_with_fg=True,
        zero_context_joint=False, warm_policy=None, resume_state=None))
    audits = {}
    for arm in ('eraf_only', 'eraf_fg'):
        path = root/arm/'joint/freeze_audit.json'
        audit = read(path); audit.update(final_step=200, world_size=3, global_batch=12)
        if arm == 'eraf_fg': audit['fg_continuation_parent_and_initialization_verified'] = True
        write(path, audit); audits[arm] = audit
    status['input_sha256'] = plan['input_sha256'] | {str(baseline):sha(baseline), parent['final_checkpoint']:parent['final_sha256']}
    names = ['prepare_current_parents', 'identity_eraf_only', 'train_eraf_only', 'audit_eraf_only', 'train_eraf_fg', 'audit_eraf_fg']
    names += ['catalog_'+t for t in TASKS] + ['eval_'+a+'_'+t for a in independent.ARMS for t in TASKS]
    status['jobs'] = {n:dict(pid=i+100, exit_code=0) for i,n in enumerate(names)}
    paths = [str(p.relative_to(root)) for p in (root/'evaluation').rglob('episodes.jsonl')]
    proof = dict(format='robotwin_serial3_completion_audit_v1', complete=True, smoke_only=False,
        root=str(root), code_commit=plan['code_commit'], episodes=600, video_count=600,
        eraf_initialization_matches_current_no_eraf=True, fg_inherits_completed_eraf=True,
        controller=dict(live=False), jobs={n:j|{'process':dict(live=False)} for n,j in status['jobs'].items()},
        checkpoint_sha256={a:s['final_sha256'] for a,s in arms.items()},
        actual_training_audits=audits, input_sha256=status['input_sha256'],
        episode_sha256={p:sha(root/p) for p in paths})
    write(root/'completion_audit.json', proof)
    for name, document in [('protocol.json',plan), ('status.json',status), ('terminal_verification.json',terminal), ('final_models.json',arms)]:
        write(root/name, document)
    return root


def test_serial_nomination_keeps_external_control_and_reports_unequal_budget(completed_serial_dev):
    root = completed_serial_dev
    source, models, bindings, result = independent.candidates(root)
    assert models['no_eraf']['checkpoint'] == source['baseline_checkpoint']
    assert not (root/'no_eraf/joint/step_000200.pt').exists()
    assert bindings[str(root/'completion_audit.json')] == sha(root/'completion_audit.json')
    assert bindings[str(root/'eraf_fg/joint/continuation_initialization.json')]
    assert not result['training_comparison']['equal_additional_training_budget']
    assert '0 / 200 / 400' in independent.markdown(result)
    assert result['desired_order_observed'] and not result['goal_achieved']


@pytest.mark.parametrize('mutation', ['wrong_baseline', 'wrong_budget', 'direct_fg_branch', 'reset_eraf',
    'wrong_initial_parent', 'failed_audit', 'live_worker', 'unaudited_cell', 'changed_episodes',
    'changed_late_input', 'mixed_source_inputs', 'missing_completion', 'wrong_fg_mode'])
def test_serial_nomination_rejects_lineage_and_completion_drift(completed_serial_dev, mutation):
    root = completed_serial_dev
    filename = 'status.json'
    if mutation == 'wrong_budget': filename = 'terminal_verification.json'
    elif mutation in ('direct_fg_branch',): filename = 'eraf_fg/joint/plan.json'
    elif mutation in ('reset_eraf', 'wrong_initial_parent'): filename = 'eraf_fg/joint/continuation_initialization.json'
    elif mutation in ('live_worker', 'unaudited_cell', 'missing_completion'): filename = 'completion_audit.json'
    elif mutation == 'failed_audit': filename = 'eraf_fg/joint/freeze_audit.json'
    document = read(root/filename)
    if mutation == 'wrong_baseline': document['arms']['no_eraf']['final_checkpoint'] = str(root/'old-parent.pt')
    elif mutation == 'wrong_budget': document['cumulative_optimizer_steps_since_baseline']['eraf_fg'] = 200
    elif mutation == 'direct_fg_branch': document['checkpoint'] = read(root/'protocol.json')['baseline_checkpoint']
    elif mutation == 'reset_eraf': document['eraf_reinitialized'] = True
    elif mutation == 'wrong_initial_parent': document['parent_sha256'] = 'old-eraf'
    elif mutation == 'failed_audit': document['fg_continuation_parent_and_initialization_verified'] = False
    elif mutation == 'live_worker': document['jobs']['train_eraf_fg']['process']['live'] = True
    elif mutation == 'unaudited_cell': document['episode_sha256'].pop(next(iter(document['episode_sha256'])))
    elif mutation == 'missing_completion': document['complete'] = False
    elif mutation == 'changed_late_input': (root/'eraf_only/joint/step_000200.pt').write_bytes(b'replaced')
    elif mutation == 'mixed_source_inputs': document['input_sha256'][str(root/'training-input.json')] = 'replaced'
    elif mutation == 'wrong_fg_mode': document['arms']['eraf_fg']['fg'] = 'off'
    else:
        path = root/'evaluation/eraf_fg'/TASKS[0]/'demo_clean/counterfactual/episodes.jsonl'
        episodes = records(path); episodes[0]['counterfactual_goal_ever_success'] = False
        path.write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    write(root/filename, document)
    with pytest.raises(ValueError): independent.candidates(root)
