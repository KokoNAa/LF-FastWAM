import copy
import pytest
import torch

from scripts.robotwin_checkpoint_delta import make_delta, restore, fingerprint


def test_roundtrip_keeps_bf16_scalar_and_metadata_exact_without_unchanged_video():
    parent = dict(format='checkpoint', stage='joint', optimizer_steps=200,
        mot_trainable={'video': torch.ones(2, dtype=torch.bfloat16), 'action': torch.zeros(2)},
        policy_guard={'gate': torch.tensor(0., dtype=torch.bfloat16)})
    candidate = copy.deepcopy(parent)
    candidate['mot_trainable']['action'] += .01
    candidate['policy_guard']['gate'] += .125
    candidate['optimizer_steps'] = 400
    delta = make_delta(parent, candidate)
    assert set(delta['changed']['mot_trainable']) == {'action'}
    restored = restore(parent, delta)
    assert fingerprint(restored) == fingerprint(candidate)
    assert restored['policy_guard']['gate'].dtype == torch.bfloat16
    assert restored['optimizer_steps'] == 400
    wrong_parent = copy.deepcopy(parent); wrong_parent['mot_trainable']['video'][0] = 2
    with pytest.raises(ValueError, match='parent checkpoint'):
        restore(wrong_parent, delta)
    broken = copy.deepcopy(delta); broken['changed']['mot_trainable']['action'][0] += 1
    with pytest.raises(ValueError, match='source tensors'):
        restore(parent, broken)
    broken = copy.deepcopy(delta); broken['checkpoint_metadata']['optimizer_steps'] = 300
    with pytest.raises(ValueError, match='source tensors'):
        restore(parent, broken)


def test_equal_numeric_values_with_changed_dtype_are_not_dropped():
    parent = dict(mot_trainable={'x': torch.ones(1, dtype=torch.float32)}, policy_guard={})
    candidate = dict(mot_trainable={'x': torch.ones(1, dtype=torch.bfloat16)}, policy_guard={})
    delta = make_delta(parent, candidate)
    assert 'x' in delta['changed']['mot_trainable']
    assert restore(parent, delta)['mot_trainable']['x'].dtype == torch.bfloat16
