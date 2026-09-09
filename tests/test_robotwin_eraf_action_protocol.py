import copy
from collections import Counter
from itertools import islice

import pytest
import torch

from experiments.robotwin.eraf_action_protocol import parameter_learning_rates, validate_action_parent
from experiments.robotwin.eraf_fg_bridge import MasterAdamW
from experiments.robotwin.eraf_fg_training import balanced_group_stream, mixture_counts, mixture_stream


def test_fg_continuation_only_accepts_trained_eraf_without_resetting_it():
    from experiments.robotwin.eraf_action_protocol import verify_continuation_weights
    parent=dict(stage='joint', optimizer_steps=200, fg_supervision='off', provenance={'eraf':'on'},
        mot_trainable={'action':torch.tensor([3.])}, policy_guard={'learned_context':torch.tensor([7.])})
    kwargs=dict(stage='joint',eraf='on',fg='full',continue_eraf_with_fg=True)
    validate_action_parent(parent,**kwargs)
    verify_continuation_weights(parent,copy.deepcopy(parent['mot_trainable']),copy.deepcopy(parent['policy_guard']))
    for patch in ({'stage':'grounding'}, {'optimizer_steps':0}, {'fg_supervision':'full'},
                  {'provenance':{'eraf':'off'}}):
        with pytest.raises(ValueError,match='FG continuation'):
            validate_action_parent(parent|patch,**kwargs)
    for patch in ({'fg':'off'}, {'stage':'interface'}, {'eraf':'off'}, {'zero_context_joint':True},
                  {'warm_policy':True}, {'resume':True}):
        with pytest.raises(ValueError,match='FG continuation'):
            validate_action_parent(parent,**(kwargs|patch))
    with pytest.raises(ValueError,match='ERAF weights'):
        verify_continuation_weights(parent,parent['mot_trainable'],{'learned_context':torch.zeros(1)})
    with pytest.raises(ValueError,match='policy weights'):
        verify_continuation_weights(parent,{'action':torch.zeros(1)},parent['policy_guard'])


def test_joint_zero_context_is_explicit_and_cannot_accept_an_updated_or_wrong_arm():
    from experiments.robotwin.context_residual import ZEROED
    parent = dict(stage='bootstrap', optimizer_steps=0, context_injection_mode='context_residual_v1',
                  fg_supervision='full', provenance={'context_residual_initialization': True},
                  policy_guard={key: torch.zeros(2) for key in ZEROED})
    with pytest.raises(ValueError, match='matching interface'):
        validate_action_parent(parent, stage='joint', eraf='on', fg='full')
    validate_action_parent(parent, stage='joint', eraf='on', fg='full', zero_context_joint=True)
    for patch in [{'optimizer_steps': 1}, {'fg_supervision': 'off'}, {'context_injection_mode': 'append_v1'},
                  {'provenance': {}}, {'policy_guard': {key: torch.ones(2) for key in ZEROED}}]:
        with pytest.raises(ValueError, match='untouched residual bootstrap'):
            validate_action_parent(parent | patch, stage='joint', eraf='on', fg='full', zero_context_joint=True)
    for kwargs in [{'resume': True}, {'warm_policy': True}]:
        with pytest.raises(ValueError, match='untouched residual bootstrap'):
            validate_action_parent(parent, stage='joint', eraf='on', fg='full', zero_context_joint=True, **kwargs)


def test_joint_identity_requires_exact_matching_repeated_ten_task_evidence():
    from experiments.robotwin.pgc_data import ROBOTWIN_TEN_TASK_SPECS
    from experiments.robotwin.eraf_action_protocol import validate_joint_identity_audit
    records = [dict(pair_id=s.pair_id, task_config='demo_clean', language=language,
                    comparisons_to_eraf_off={mode: dict(raw_actions_exact=True, normalized_actions_exact=True, max_abs=0.)
                                             for mode in ('full', 'repeat_full')})
               for s in ROBOTWIN_TEN_TASK_SPECS for language in ('source', 'target')]
    audit = dict(complete=True, full_eraf_equals_off_exactly=True, include_expanded_tasks=True,
                 task_count=10, task_domain_count=10, queries=20, records=records,
                 checkpoint_sha256='a'*64, manifest_sha256='b'*64, denoising_steps=10, seed=42)
    validate_joint_identity_audit(audit, checkpoint_sha256='a'*64, manifest_sha256='b'*64)
    for patch in [{'checkpoint_sha256': 'c'*64}, {'manifest_sha256': 'c'*64}, {'complete': False},
                  {'denoising_steps': 1}, {'task_count': 5}, {'records': records[:-1]},
                  {'records': records + records[:1]}]:
        with pytest.raises(ValueError):
            validate_joint_identity_audit(audit | patch, checkpoint_sha256='a'*64, manifest_sha256='b'*64)
    changed = copy.deepcopy(audit)
    changed['records'][0]['comparisons_to_eraf_off']['repeat_full']['raw_actions_exact'] = False
    with pytest.raises(ValueError, match='changed deployed action'):
        validate_joint_identity_audit(changed, checkpoint_sha256='a'*64, manifest_sha256='b'*64)


def test_semantically_trained_residual_keeps_its_steps_and_requires_zero_output():
    from experiments.robotwin.context_residual import ZEROED
    from experiments.robotwin.eraf_action_protocol import is_zero_context_parent
    parent = dict(stage='grounding', optimizer_steps=1000, context_injection_mode='context_residual_v1',
        fg_supervision='full', provenance={'paired_cross_goals': True},
        policy_guard={key: torch.zeros(2) for key in ZEROED})
    validate_action_parent(parent, stage='joint', eraf='on', fg='full', zero_context_joint=True)
    assert parent['stage'] == 'grounding' and parent['optimizer_steps'] == 1000
    for patch in ({'stage': 'joint'}, {'optimizer_steps': 0}, {'provenance': {}},
                  {'policy_guard': {key: torch.ones(2) for key in ZEROED}}):
        assert not is_zero_context_parent(parent | patch)


def test_common_warm_policy_does_not_leak_fg_training_into_off_controls():
    parent = dict(stage='joint', fg_supervision='off', provenance={'eraf': 'off'})
    for stage, eraf, fg in [('joint', 'off', 'off'), ('joint', 'off', 'full'), ('interface', 'on', 'full')]:
        validate_action_parent(parent, stage=stage, eraf=eraf, fg=fg, warm_policy=True)
    with pytest.raises(ValueError, match='warmup'):
        validate_action_parent(parent, stage='joint', eraf='on', fg='full', warm_policy=True)
    with pytest.raises(ValueError, match='common warm policy'):
        validate_action_parent(parent | {'fg_supervision': 'full'}, stage='joint', eraf='off', fg='off', warm_policy=True)
    with pytest.raises(ValueError, match='different operations'):
        validate_action_parent(parent, stage='joint', eraf='off', fg='off', warm_policy=True, resume=True)
    with pytest.raises(ValueError, match='matching interface'):
        validate_action_parent(dict(stage='interface', fg_supervision='full'), stage='joint', eraf='on', fg='off')


def test_interface_can_learn_faster_without_accelerating_policy_and_resume_is_exact():
    names = ['mot.x', 'guard.y', 'mot.z']
    rates = parameter_learning_rates(names, 1e-5, 1e-3)
    assert rates == [1e-5, 1e-3, 1e-5]
    live = [torch.nn.Parameter(torch.ones(1)) for _ in names]
    opt = MasterAdamW(live, lr=1e-5, learning_rates=rates)
    for p in live: p.grad = torch.ones_like(p)
    opt.step()
    assert 50 < float((1 - live[1]).detach() / (1 - live[0]).detach()) < 150
    state = copy.deepcopy(opt.state_dict())
    restored = [torch.nn.Parameter(p.detach().clone()) for p in live]
    other = MasterAdamW(restored, lr=1e-5, learning_rates=rates)
    other.load_state_dict(state); other.set_learning_rates(rates)
    for optimizer, params in [(opt, live), (other, restored)]:
        optimizer.zero_grad()
        for p in params: p.grad = torch.full_like(p, .3)
        optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(live, restored, strict=True))
    with pytest.raises(ValueError, match='positive'):
        parameter_learning_rates(names, 0, 1e-3)


def test_task_balancing_does_not_double_old_tasks_with_two_domains():
    rows = [dict(id=f'{task}_{domain}_{i}', pair_id=task, task_config=domain, scene_seed=i)
            for task, domains in [('old', ['clean', 'randomized']), ('new', ['clean'])]
            for domain in domains for i in range(3)]
    draws = list(islice(balanced_group_stream(rows, 42, task_balanced=True), 100))
    assert Counter(r['pair_id'] for r in draws) == {'old': 50, 'new': 50}
    assert {r['task_config'] for r in draws if r['pair_id'] == 'old'} == {'clean', 'randomized'}


def test_cf_priority_mixture_keeps_matched_fg_positions_and_expert_exposure():
    rows = [dict(id=f'{kind}_{task}_{i}', pair_id=task, task_config='clean', scene_seed=i,
                 replay_split='train', frame_index=0, **{kind: True})
            for kind in ('native_retention', 'cf_retention', 'pair', 'fg_correction')
            for task in ('left', 'rank') for i in range(3)]
    full = mixture_stream(rows, 42, 'full', correct_count=2, cf_count=4, task_balanced=True)
    off = mixture_stream(rows, 42, 'off', correct_count=2, cf_count=4, task_balanced=True)
    for _ in range(10):
        a, b = next(full), next(off)
        assert len(a) == len(b) == 12
        assert sum(bool(r.get('native_retention')) for r in a) == 2
        assert sum(bool(r.get('cf_retention')) for r in a) == 4
        assert sum(bool(r.get('pair')) for r in a) == 3
        assert sum(bool(r.get('ordinary_cf_control')) for r in b) == 3
        assert not any(r.get('fg_correction') for r in b)
        for x, y in zip(a, b, strict=True):
            assert (x['pair_id'], x['task_config']) == (y['pair_id'], y['task_config'])
            if not x.get('fg_correction'): assert x == y
    for counts in [(0, 6), (3, 4), (2.5, 3.5)]:
        with pytest.raises(ValueError): mixture_counts(*counts)
