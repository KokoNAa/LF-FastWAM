import copy

import pytest
import torch

from experiments.robotwin.eraf_fg_bridge import CHECKPOINT_FORMAT, PROTOCOL, eraf_guard_config
from scripts.compose_robotwin_eraf_fg_checkpoint import compose


@pytest.mark.parametrize('case', ['valid', 'semantic_changed', 'different_parent'])
def test_composition_retains_the_policy_and_only_admits_its_trained_interface(case):
    interface_key = 'eraf_action_context_injector.weight'
    semantic_key = 'entity_relation_affordance.predicate_head.weight'
    policy = dict(format=CHECKPOINT_FORMAT, protocol=PROTOCOL, stage='joint', fg_supervision='full',
                  optimizer_steps=200, parent_checkpoint='/common.pt', base_checkpoint='/release.pt',
                  guard_config=eraf_guard_config(),
                  lora_config={'enabled': True, 'experts': ['video', 'action']},
                  geometry=dict(action_dim=14, proprio_dim=14, camera_count=3,
                                camera_layout='robotwin_mosaic', action_horizon=32,
                                replan_steps=24, inference_steps=10),
                  mot_trainable={'action.lora_A': torch.tensor([.4])},
                  policy_guard={interface_key: torch.zeros(2), semantic_key: torch.ones(2)},
                  provenance={'eraf': 'off', 'policy_scope': 'action'})
    donor = copy.deepcopy(policy)
    donor.update(stage='interface', optimizer_steps=10, provenance={'eraf': 'on'})
    donor['mot_trainable']['action.lora_A'].zero_()
    donor['policy_guard'][interface_key].fill_(.2)
    selected = {'guard.' + interface_key}
    if case == 'semantic_changed':
        donor['policy_guard'][semantic_key].zero_()
        # Even a wrong plan cannot authorize changing a semantic head.
        selected.add('guard.' + semantic_key)
    if case == 'different_parent':
        donor['parent_checkpoint'] = '/different.pt'
    before = copy.deepcopy(policy)
    if case != 'valid':
        with pytest.raises(ValueError):
            compose(policy, donor, selected)
        return
    result, changed = compose(policy, donor, selected)
    assert changed == [interface_key]
    assert torch.equal(result['mot_trainable']['action.lora_A'], before['mot_trainable']['action.lora_A'])
    assert torch.equal(result['policy_guard'][semantic_key], before['policy_guard'][semantic_key])
    assert torch.equal(policy['policy_guard'][interface_key], before['policy_guard'][interface_key])
    assert result['provenance']['aggregate_component_optimizer_steps'] == 210
    assert result['provenance']['composition_jointly_optimized'] is False
