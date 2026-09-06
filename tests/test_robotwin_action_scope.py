import torch
import pytest
from test_robotwin_eraf_fg_bridge import case
from experiments.robotwin.eraf_fg_bridge import trainable_parameters, MasterAdamW
from experiments.robotwin.eraf_fg_training import backward_example


def test_action_only_real_update_preserves_video_and_guard(case):
    model, captured, noise, time = case
    selected = trainable_parameters(model, 'joint', eraf=False, policy_scope='action')
    assert selected and all('.action.' in n for n in selected)
    frozen = {n: p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    before = {n: p.detach().clone() for n,p in selected.items()}
    opt = MasterAdamW(selected.values(), lr=1e-3)
    backward_example(model, {'fg_correction': True},
        {'captured': {'target': captured}, 'references': {'target': torch.zeros_like(noise)}},
        noise,time,teachers={},eraf=False)
    opt.step()
    assert any(not torch.equal(p, before[n]) for n,p in selected.items())
    assert all(torch.equal(p, frozen[n]) and p.grad is None
               for n,p in model.named_parameters() if n in frozen)


@pytest.mark.parametrize('row,scale', [({'fg_correction':True}, .1),
                                      ({'ordinary_cf_control':True}, .1), ({}, 1.)])
def test_correction_weight_scales_full_and_matched_control_gradients(case,row,scale):
    model,captured,noise,time=case
    selected=trainable_parameters(model,'interface')
    payload={'captured':{'target':captured},'references':{'target':torch.zeros_like(noise)}}
    backward_example(model,row,payload,noise,time,teachers={},correction_weight=1.)
    original={n:p.grad.clone() for n,p in selected.items() if p.grad is not None}
    for p in selected.values():p.grad=None
    backward_example(model,row,payload,noise,time,teachers={},correction_weight=.1)
    assert original
    for n,expected in original.items():
        assert torch.allclose(selected[n].grad,expected*scale,atol=2e-6,rtol=2e-4),n
