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
    def __init__(self, model, parameters, checkpoint):
        import torch
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = payload['mot_trainable']
        if set(state) != set(parameters):
            raise ValueError('Native teacher must have exactly the same adapter parameter names.')
        self.model = model
        self.parameters = parameters
        self.values = {}
        for name, parameter in parameters.items():
            if state[name].shape != parameter.shape:
                raise ValueError(f'Teacher adapter shape mismatch: {name}')
            self.values[name] = state[name].to(parameter).detach()

    def predict(self, captured, noisy, timestep):
        import torch
        from experiments.robotwin.joint_adapter_repair import build_cache
        with torch.no_grad(), teacher_parameters(self.parameters, self.values):
            return self.model._predict_action_noise_with_cache(
                latents_action=noisy, timestep_action=timestep,
                **build_cache(self.model, captured)).detach()
