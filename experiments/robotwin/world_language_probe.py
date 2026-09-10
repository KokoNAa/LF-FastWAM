"""Causal video-cache interventions; production inference is the reference."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import numpy as np


@contextmanager
def replace_method(obj, name, value):
    existed, old = name in vars(obj), vars(obj).get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if existed: setattr(obj, name, old)
        else: delattr(obj, name)


def metrics(a, b):
    """Magnitude-normalized differences; both-zero cosine distance is zero."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite or mismatched tensors')
    d = a-b
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cosine = float(np.sum(a*b)/(na*nb)) if na*nb else (1. if na == nb else 0.)
    return dict(rms=float(np.mean(d*d)**.5), max_abs=float(np.max(np.abs(d))),
                relative_l2=float(np.linalg.norm(d)/max((na+nb)/2, 1e-12)),
                cosine_distance=float(1-np.clip(cosine, -1, 1)))


def preference(action, source, target):
    """Expert proximity is diagnostic, never a substitute for goal success."""
    source_error = metrics(action, source)['rms']
    target_error = metrics(action, target)['rms']
    delta = np.asarray(target, dtype=np.float64)-source
    separation = float(np.sum(delta*delta))
    projection = None if separation < 1e-12 else float(np.sum((action-source)*delta)/separation)
    return dict(source_rmse=source_error, target_rmse=target_error,
        source_minus_target_rmse=source_error-target_error,
        target_axis_projection=projection, reference_separation_rms=float(np.mean(delta*delta)**.5))


def numpy_tensor(x):
    return x.detach().float().cpu().numpy()


def hash_tensor(x):
    # Hash the exact bits, including BF16, without lossy float conversion.
    raw = x.detach().contiguous().cpu()
    import torch
    return hashlib.sha256((str(raw.dtype)+str(tuple(raw.shape))).encode()+
                          raw.view(torch.uint8).numpy().tobytes()).hexdigest()


def capture_world(policy, observation, instruction):
    from experiments.robotwin.denoising_probe import capture_action_cache
    from experiments.robotwin.same_state_repair import move_cache
    model = policy.model
    if model.policy_guard_enabled or model.uses_transition_queries:
        raise ValueError('Primary intervention requires ordinary no-ERAF cache inference')
    hidden, video_inputs = {}, {}
    indexes = {id(block): i for i, block in enumerate(model.video_expert.blocks)}
    post = model.mot._apply_post_with_optional_checkpoint
    pre = model.video_expert.pre_dit

    def observe_post(*args, **kwargs):
        result = post(*args, **kwargs)
        index = indexes.get(id(kwargs.get('block')))
        if index is not None:
            if index in hidden: raise ValueError('Video block executed more than once')
            hidden[index] = move_cache(result, 'cpu', clone=True)
        return result

    def observe_pre(*args, **kwargs):
        if args or video_inputs: raise ValueError('Expected one keyword-only Video prefill')
        video_inputs.update(move_cache(kwargs, 'cpu', clone=True))
        return pre(**kwargs)

    policy.reset()
    with replace_method(model.mot, '_apply_post_with_optional_checkpoint', observe_post), \
         replace_method(model.video_expert, 'pre_dit', observe_pre):
        action, cache, calls = capture_action_cache(model,
            lambda: policy._infer_action_chunk(observation, instruction))
    assert calls == policy.num_inference_steps and len(hidden) == len(indexes)
    return dict(action=action, cache=move_cache(cache, 'cpu', clone=True),
                hidden=hidden, video_inputs=video_inputs, predictor_calls=calls)


def compare_world(a, b):
    output = []
    for layer in sorted(a['hidden']):
        for kind in ['hidden', 'k', 'v']:
            x = a['hidden'][layer] if kind == 'hidden' else a['cache']['video_kv_cache'][layer][kind]
            y = b['hidden'][layer] if kind == 'hidden' else b['cache']['video_kv_cache'][layer][kind]
            xx, yy = numpy_tensor(x), numpy_tensor(y)
            # Each spatial token is retained. Aggregation is left to reporting.
            spatial = np.sqrt(np.mean((xx-yy)**2, axis=-1))[0].tolist()
            output.append(dict(layer=layer, kind=kind, **metrics(xx, yy), token_delta_rms=spatial))
    return output


def hybrid_cache(video_capture, action_capture, device):
    from experiments.robotwin.same_state_repair import move_cache
    a, v = action_capture['cache'], video_capture['cache']
    assert a['video_seq_len'] == v['video_seq_len']
    import torch
    assert torch.equal(a['attention_mask'], v['attention_mask'])
    result = dict(a, video_kv_cache=v['video_kv_cache'])
    return move_cache(result, device)


def sample_cache(model, cache, seed, steps=10):
    import torch
    initial = torch.randn((1, 32, model.action_expert.action_dim),
        generator=torch.Generator(device='cpu').manual_seed(seed), dtype=torch.float32)
    latent = initial.to(device=model.device, dtype=model.torch_dtype)
    initial_hash = hash_tensor(latent)
    times, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=steps, device=model.device, dtype=model.torch_dtype, shift_override=None)
    with torch.no_grad():
        for t, delta in zip(times, deltas, strict=True):
            velocity = model._predict_action_noise_with_cache(latents_action=latent,
                timestep_action=t.unsqueeze(0), **cache)
            latent = model.infer_action_scheduler.step(velocity, delta, latent)
    return latent, initial_hash


@contextmanager
def video_language_override(model, prompt):
    """Change only Video text; retain the exact appended proprioception token."""
    import torch
    context, mask = model.encode_prompt(prompt)
    original = model.video_expert.pre_dit

    def changed(*args, **kwargs):
        if args: raise ValueError('Expected keyword-only pre_dit')
        old, old_mask = kwargs['context'], kwargs['context_mask']
        # Production encode_prompt pads to fixed tokenizer length.
        if old.shape[1] != context.shape[1]+1 or model.proprio_dim != 14:
            raise ValueError('Unexpected language/proprioception token boundary')
        new = dict(kwargs, context=torch.cat([context.to(old), old[:, -1:]], dim=1),
                   context_mask=torch.cat([mask.to(old_mask), old_mask[:, -1:]], dim=1))
        return original(**new)

    with replace_method(model.video_expert, 'pre_dit', changed):
        yield


def video_velocity(model, latent, timestep, context, context_mask):
    """Video-only equivalent of _predict_joint_noise under the verified mask."""
    if model.video_expert.action_conditioned:
        raise ValueError('This probe does not silently omit required action conditioning')
    pre = model.video_expert.pre_dit(x=latent, timestep=timestep, context=context,
        context_mask=context_mask, action=None,
        fuse_vae_embedding_in_latents=model.video_expert.fuse_vae_embedding_in_latents)
    _, hidden = model.mot.prefill_video_cache(video_tokens=pre['tokens'],
        video_freqs=pre['freqs'], video_t_mod=pre['t_mod'],
        video_context_payload=dict(context=pre['context'], mask=pre['context_mask']),
        video_attention_mask=model.video_expert.build_video_to_video_mask(
            video_seq_len=pre['tokens'].shape[1], video_tokens_per_frame=pre['meta']['tokens_per_frame'],
            device=latent.device), return_final_hidden=True)
    return model.video_expert.post_dit(hidden, pre)
