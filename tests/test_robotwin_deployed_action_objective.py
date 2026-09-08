from types import SimpleNamespace
import pytest
import torch
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from experiments.robotwin import deployed_action_objective as objective


@pytest.mark.parametrize('checkpoint',[False,True])
def test_full_unroll_gradient_matches_analytic_euler_product(checkpoint):
    scheduler=WanContinuousFlowMatchScheduler(shift=5.)
    initial=torch.tensor([[[0.7]]],dtype=torch.float64)
    w=torch.tensor(0.3,dtype=torch.float64,requires_grad=True)
    result=objective.denoise(initial,scheduler,lambda x,t:w*x,checkpoint=checkpoint)
    _,deltas=scheduler.build_inference_schedule(10,initial.device,initial.dtype)
    product=(1+w*deltas).prod()
    expected=initial*product
    expected_gradient=(initial*product*(deltas/(1+w*deltas)).sum()).item()
    result.sum().backward()
    torch.testing.assert_close(result,expected)
    assert w.grad.item()==pytest.approx(expected_gradient,abs=1e-12)


def test_training_and_nograd_forward_are_identical():
    initial=torch.randn(1,32,14,dtype=torch.float64)
    scheduler=WanContinuousFlowMatchScheduler()
    w=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    with torch.no_grad(): expected=objective.denoise(initial,scheduler,lambda x,t:x*w+t/1000)
    actual=objective.denoise(initial,scheduler,lambda x,t:x*w+t/1000,checkpoint=True)
    assert torch.equal(actual,expected)


def model_payload():
    model=SimpleNamespace(torch_dtype=torch.float32,weight=torch.tensor(0.2,requires_grad=True))
    captured={'video_inputs':{'x':torch.zeros(1)},'proprio':torch.zeros(1)}
    payload={'captured':{'target':captured},'references':{'target':torch.zeros(1,32,14)}}
    return model,payload


def test_reference_never_enters_sampler_and_unexecuted_tail_is_not_target(monkeypatch):
    model,payload=model_payload();noise=torch.ones(1,32,14)
    observed=[]
    def sample(m,c,n,**kw):
        observed.append(n.clone());return n*m.weight
    monkeypatch.setattr(objective,'sample',sample)
    first=objective.backward_deployed_example(model,{},payload,noise,None,teachers={})
    grad=model.weight.grad.clone();model.weight.grad=None
    payload['references']['target'][:,24:]=10000
    second=objective.backward_deployed_example(model,{},payload,noise,None,teachers={})
    assert first==second and torch.equal(grad,model.weight.grad)
    assert all(torch.equal(n,noise) for n in observed)


def test_teacher_completes_before_student_forward(monkeypatch):
    model,payload=model_payload();events=[];noise=torch.ones(1,32,14)
    def teacher(*args):events.append('teacher');return torch.zeros_like(noise)
    def student(m,c,n,**kw):events.append('student');return n*m.weight
    monkeypatch.setattr(objective,'teacher_action',teacher);monkeypatch.setattr(objective,'sample',student)
    metrics=objective.backward_deployed_example(model,{'cf_retention':True},payload,noise,None,teachers={'cf':object()})
    assert events==['teacher','student']
    assert metrics['retention_target']=='same_noise_ten_step_teacher'
    assert model.weight.grad!=0


def test_conditional_loss_requires_actual_identical_state(monkeypatch):
    model,payload=model_payload()
    payload['references']['source']=torch.zeros(1,32,14)
    payload['captured']['source']={'video_inputs':{'x':torch.ones(1)},'proprio':torch.zeros(1)}
    with pytest.raises(ValueError,match='identical actual'):
        objective.backward_deployed_example(model,{},payload,torch.ones(1,32,14),None,teachers={})


def test_teacher_adapter_swap_restores_after_sampler_failure(monkeypatch):
    from experiments.robotwin.native_teacher import teacher_parameters
    p=torch.nn.Parameter(torch.tensor(2.)); teacher=SimpleNamespace(model=object(),parameters={'w':p},values={'w':torch.tensor(9.)})
    def fail(*args,**kwargs):
        assert p.item()==9.
        raise RuntimeError('sampler failed')
    monkeypatch.setattr(objective,'sample',fail)
    with pytest.raises(RuntimeError,match='sampler failed'):objective.teacher_action(teacher,{},torch.ones(1))
    assert p.item()==2.


def test_all_four_commands_share_explicit_objective():
    from pathlib import Path
    from scripts.run_robotwin_expanded_fg_trial import action_command,ARMS
    plan=dict(manifest='/manifest',source_bank='/source',correct_teacher='/correct',strongest_checkpoint='/best',
              arms={a:dict(s,parent='/parent/'+a) for a,s in ARMS.items()},action_objective='deployed_rollout_v1')
    for arm in ARMS:
        cmd=action_command(plan,Path('/trial'),arm)
        assert cmd[cmd.index('--action-objective')+1]=='deployed_rollout_v1'
        assert cmd[cmd.index('--steps')+1]=='200'
        assert cmd[cmd.index('--save-every')+1]=='200'


def test_paired_audit_rejects_silent_objective_change(tmp_path):
    import json
    from tests.test_robotwin_expanded_fg_trial import action_evidence
    from scripts.audit_robotwin_expanded_fg_trial import audit_action_pairing
    action_evidence(tmp_path)
    path=tmp_path/'protocol.json';p=json.loads(path.read_text());p['action_objective']='deployed_rollout_v1';path.write_text(json.dumps(p))
    with pytest.raises(AssertionError):audit_action_pairing(tmp_path)
