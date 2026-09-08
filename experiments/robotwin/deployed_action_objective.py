"""Supervise the actual ten-step action sampler, with complete unrolled gradients.

No expert action enters a denoising input. The ordinary flow objective remains
available unchanged; this is an explicit, separately recorded experiment.
"""
from __future__ import annotations
import math

OBJECTIVE = 'deployed_rollout_v1'
STEPS = 10
EXECUTED = 24


def denoise(initial, scheduler, velocity, *, checkpoint=False):
    """Same schedule, dtype and Euler arithmetic as production, without detach."""
    import torch
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
    action = initial.detach().clone()
    times, deltas = scheduler.build_inference_schedule(
        num_inference_steps=STEPS, device=action.device, dtype=action.dtype, shift_override=None)
    for timestep, delta in zip(times, deltas, strict=True):
        time = timestep.unsqueeze(0).to(device=action.device, dtype=action.dtype)
        prediction = (activation_checkpoint(velocity, action, time, use_reentrant=False)
                      if checkpoint and torch.is_grad_enabled() else velocity(action, time))
        action = scheduler.step(prediction, delta, action)
    return action


def sample(model, captured, noise, *, eraf, checkpoint=True):
    """Build the same visual/semantic context once, then unroll all ten steps."""
    from experiments.robotwin.eraf_fg_bridge import build_cache
    action, queries, _ = build_cache(model, captured, eraf=eraf)
    if eraf:
        def velocity(x, time):
            return model._forward_policy_guard_action_from_cache(
                action_tokens=x, timestep_action=time, context=action['context'],
                full_context_mask=action['context_mask'], state_only_context_mask=action['state_only_context_mask'],
                video_kv_cache=action['video_kv_cache'], video_seq_len=action['video_seq_len'],
                video_tokens_per_frame=action['video_tokens_per_frame'], routed_goal_queries=queries)
    else:
        body = getattr(type(model)._predict_action_noise_with_cache, '__wrapped__', None)
        if body is None: raise ValueError('Expected the production no-grad predictor.')
        def velocity(x, time):
            return body(model, latents_action=x, timestep_action=time, **action)
    return denoise(noise, model.infer_action_scheduler, velocity, checkpoint=checkpoint)


def teacher_action(teacher, captured, noise):
    import torch
    from experiments.robotwin.native_teacher import teacher_parameters
    with torch.no_grad(), teacher_parameters(teacher.parameters, teacher.values):
        return sample(teacher.model, captured, noise, eraf=False, checkpoint=False).detach()


def backward_deployed_example(model, row, payload, noise, time, *, teachers, coefficient=1.,
                              eraf=True, fg='full', correct_weight=2., cf_weight=1., target_weight=2.,
                              conditional_gain=4., correction_weight=1., fg_gradient_route='joint'):
    """Final-action fit plus same-observation instruction difference and retention.

Teacher actions are sampled using the student's starting noise before building
any student graph. The teacher swaps its adapters back before student forward.
Only the first24 executed positions of each real horizon receive supervision.
Full-goal corrective rows still span the entire corrective episode in the bank.
"""
    del time  # Training RNG draws are preserved for exact paired schedules.
    import torch
    from experiments.robotwin.eraf_fg_training import same_observation, validate_fg_gradient_route
    from experiments.robotwin.eraf_fg_bridge import masked_mse
    from experiments.robotwin.same_state_repair import paired_velocity_losses
    if fg not in ('full','off'): raise ValueError('Deployed objective currently requires full/off FG.')
    validate_fg_gradient_route(fg_gradient_route, eraf=eraf, fg=fg)
    if not math.isfinite(correction_weight) or correction_weight<=0: raise ValueError('Invalid correction weight.')
    if row.get('fg_correction') or row.get('ordinary_cf_control'): coefficient *= correction_weight
    captured = payload['captured']
    refs = {k:v.to(model.torch_dtype) for k,v in payload['references'].items()}
    valid = {k:payload.get('valid',{}).get(k,torch.ones(v.shape[:2],device=v.device,dtype=torch.bool))[:,:EXECUTED]
             for k,v in refs.items()}
    retention = 'correct' if row.get('native_retention') else 'cf' if row.get('cf_retention') else None
    paired = set(refs)=={'source','target'}
    gain = conditional_gain if paired and row.get('initial_observations_exactly_equal',True) else 1.
    if gain!=1 and not same_observation(captured): raise ValueError('Conditional difference needs identical actual observations.')
    targets = ({k:teacher_action(teachers[retention],captured[k],noise) for k in refs} if retention else refs)
    predictions = {k:sample(model,captured[k],noise,eraf=eraf) for k in refs}
    def weight(k):
        if retention: return correct_weight if retention=='correct' else cf_weight
        return (target_weight if k=='target' else 1.) / len(refs)
    losses = {k:masked_mse(predictions[k][:,:EXECUTED],targets[k][:,:EXECUTED],valid[k]) for k in refs}
    objective = sum(weight(k)*value for k,value in losses.items())
    metrics = {f'deployed_{k}_mse_first24':float(v.detach()) for k,v in losses.items()}
    if gain!=1:
        if not all(bool(v.all()) for v in valid.values()): raise ValueError('Conditional horizon must be entirely valid.')
        diff=paired_velocity_losses({k:v[:,:EXECUTED] for k,v in predictions.items()},
                                    {k:v[:,:EXECUTED] for k,v in targets.items()})['conditional_mse']
        objective = objective + (gain-1.)*diff
        metrics['deployed_conditional_mse_first24'] = float(diff.detach())
    if not bool(torch.isfinite(objective)): raise ValueError('Nonfinite deployed-action objective.')
    if fg_gradient_route=='eraf_only' and row.get('fg_correction'):
        receivers=tuple(p for p in model.policy_guard_modules.parameters() if p.requires_grad)
        if not receivers: raise ValueError('No trainable ERAF receivers.')
        (coefficient*objective).backward(inputs=receivers)
    else:
        (coefficient*objective).backward()
    metrics.update(action_objective=OBJECTIVE,denoising_steps=STEPS,executed_horizon=EXECUTED,
                   deployed_objective=float(objective.detach()),gradient_horizon='all_ten_steps',
                   retention_target='same_noise_ten_step_teacher' if retention else 'expert_actions')
    return metrics
