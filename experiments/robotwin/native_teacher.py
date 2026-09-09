"""Preserve original instruction behavior while fitting counterfactual experts."""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def teacher_parameters(parameters, values):
    """Use the shared frozen backbone; restore student storage before autograd."""
    original = {name: p.data for name, p in parameters.items()}
    try:
        for name, p in parameters.items():
            p.data = values[name]
        yield
    finally:
        for name, p in parameters.items():
            p.data = original[name]


class NativeTeacher:
    def __init__(self, model, parameters, checkpoint, *, eraf=False):
        import torch
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = payload['mot_trainable']
        if set(state) != set(parameters):
            raise ValueError('Native teacher must have exactly the same adapter parameter names.')
        self.model = model
        self.eraf = eraf
        self.parameters = dict(parameters)
        if eraf:
            if (payload.get('stage') != 'joint' or payload.get('optimizer_steps', 0) <= 0
                    or payload.get('fg_supervision') != 'off'
                    or payload.get('provenance', {}).get('eraf') != 'on'):
                raise ValueError('Full ERAF teacher requires a completed ERAF-on, FG-off joint checkpoint.')
            guard = model.policy_guard_modules
            # Keep the live objects (including persistent buffers), swapping only storage.
            guard_tensors = guard.state_dict(keep_vars=True)
            if set(guard_tensors) != set(payload['policy_guard']):
                raise ValueError('Teacher must have exactly the same ERAF tensor names.')
            if set(dict(guard.named_buffers())) - set(guard_tensors):
                raise ValueError('ERAF teacher cannot leave unbound nonpersistent buffers.')
            state = dict(state)
            for name, tensor in guard_tensors.items():
                key = 'policy_guard.' + name
                if key in self.parameters:
                    raise ValueError('Teacher tensor name collision: ' + key)
                self.parameters[key] = tensor
                state[key] = payload['policy_guard'][name]
        self.values = {}
        for name, parameter in self.parameters.items():
            if state[name].shape != parameter.shape:
                raise ValueError(f'Teacher adapter shape mismatch: {name}')
            self.values[name] = state[name].to(parameter).detach().clone()

    def predict(self, captured, noisy, timestep):
        if self.eraf:
            raise ValueError('Full ERAF teacher requires the deployed ten-step teacher_action sampler.')
        import torch
        from experiments.robotwin.joint_adapter_repair import build_cache
        with torch.no_grad(), teacher_parameters(self.parameters, self.values):
            return self.model._predict_action_noise_with_cache(
                latents_action=noisy, timestep_action=timestep,
                **build_cache(self.model, captured)).detach()


def retention_backward(model, captured, reference, noise, time, teacher, weight, endpoint_weight):
    """Distill native velocity fields on actual policy states; no CF label here."""
    import torch
    from experiments.robotwin.joint_adapter_repair import predict
    scheduler = model.train_action_scheduler
    endpoint = torch.tensor([scheduler.num_train_timesteps], device=model.device, dtype=model.torch_dtype)
    noisy = scheduler.add_noise(reference, noise, time)
    # Finish every teacher swap before constructing student autograd graphs.
    targets = [teacher.predict(captured, noisy, time), teacher.predict(captured, noise, endpoint)]
    terms = {}
    for name, x, t, target, coefficient in (
        ('flow', noisy, time, targets[0], float(scheduler.training_weight(time).item())),
        ('endpoint', noise, endpoint, targets[1], endpoint_weight),
    ):
        prediction = predict(model, captured, x, t)
        loss = (prediction.float() - target.float()).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite native retention loss.')
        (weight * coefficient * loss).backward()
        terms['retention_' + name + '_mse'] = float(loss.detach())
    return terms
