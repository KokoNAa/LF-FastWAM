#!/usr/bin/env python3
"""Fixed-input action and objective-gradient diagnosis; never creates an optimizer."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def action_probe(policy, rows, payloads, raw, output, check_budget):
    import h5py
    import numpy as np
    import torch
    from experiments.robotwin.eraf_fg_contract import CAMERAS
    from experiments.robotwin.image_io import decode_legacy_robotwin_rgb
    selected = {}
    for row in rows:
        if row['replay_split'] != 'replay_holdout' or any(row.get(k) for k in
                ('native_retention', 'cf_retention', 'fg_correction')):
            continue
        if any(row.get(k + '_frame_index', row.get('frame_index', 0)) != 0 for k in ('source', 'target')):
            continue
        selected.setdefault((row['pair_id'], row['task_config']), row)
    if len(selected) != 10:
        raise ValueError('Expected the existing ten task/domain initial holdout states.')
    model = policy.model
    norm = policy.processor.normalizer.normalizers['action'][policy.processor.shape_meta['action'][0]['key']]
    records, arrays = [], {}
    enabled = model.policy_guard_enabled
    try:
        model.policy_guard_enabled = False
        for row in selected.values():
            observations, predictions, errors = {}, {}, {}
            payload = payloads[row['id']]
            for language in ('source', 'target'):
                check_budget()
                path, frame = raw.locate(row, language)
                with h5py.File(path, 'r') as handle:
                    observation = {'joint_action': {'vector': handle['joint_action/vector'][frame].astype(np.float32)},
                                   'observation': {c: {'rgb': decode_legacy_robotwin_rgb(handle[f'observation/{c}/rgb'][frame])}
                                                   for c in CAMERAS}}
                observations[language] = observation
                instruction = row['source_instruction' if language == 'source' else 'counterfactual_instruction']
                policy.reset()
                with torch.no_grad():
                    action = policy._infer_action_chunk(observation, instruction)
                normalized = norm.forward(torch.as_tensor(action).float().unsqueeze(0)).detach().cpu()[0]
                reference = payload['references'][language].detach().float().cpu()[0]
                valid = payload.get('valid', {}).get(language, torch.ones((1, 32), dtype=torch.bool)).detach().cpu().bool()[0]
                if normalized.shape != (32, 14) or not bool(torch.isfinite(normalized).all()):
                    raise ValueError('Invalid action output.')
                errors[language] = {}
                for horizon in (12, 24, 32):
                    mask = valid[:horizon]
                    if not bool(mask.any()):
                        raise ValueError('No real reference actions in diagnostic horizon.')
                    errors[language][f'rmse_first{horizon}'] = float((normalized[:horizon][mask] - reference[:horizon][mask]).square().mean().sqrt())
                predictions[language] = normalized
                arrays[f'q{len(records):02d}_{language}'] = normalized.numpy()
            source, target = observations['source'], observations['target']
            equal = np.array_equal(source['joint_action']['vector'], target['joint_action']['vector']) and all(
                np.array_equal(source['observation'][c]['rgb'], target['observation'][c]['rgb']) for c in CAMERAS)
            if not equal:
                raise ValueError('Two instructions do not have identical physical observations.')
            records.append({'id': row['id'], 'pair_id': row['pair_id'], 'task_config': row['task_config'],
                            'scene_seed': row['scene_seed'], 'initial_observations_equal': equal,
                            'source_instruction': row['source_instruction'],
                            'counterfactual_instruction': row['counterfactual_instruction'],
                            'reference_errors': errors,
                            'condition_difference_rmse24': float((predictions['source'][:24] - predictions['target'][:24]).square().mean().sqrt())})
            (output / 'action_probe.json').write_text(json.dumps({'complete': False, 'records': records}, indent=2))
            print(f'[actions] paired_states={len(records)}/10', flush=True)
    finally:
        model.policy_guard_enabled = enabled
        policy.reset()
    result = {'complete': True, 'seed': policy.seed if hasattr(policy, 'seed') else 42,
              'inference_steps': 10, 'records': records,
              'scope': 'Existing initial data holdout only; action error is not closed-loop task success.'}
    (output / 'action_probe.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(output / 'actions.npz', **arrays)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'source-bank', 'checkpoint', 'output', 'correct-teacher', 'cf-teacher'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--batches', type=int, default=12)
    parser.add_argument('--start-batch', type=int, default=1)
    parser.add_argument('--max-seconds', type=int, default=2100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-actions', action='store_true', help='Second disjoint gradient shard; avoid duplicate action probes.')
    args = parser.parse_args()
    if not 1 <= args.batches <= 12 or args.start_batch < 1 or args.start_batch + args.batches - 1 > 12 or args.max_seconds <= 0:
        parser.error('Use1–12 fixed global batches and a positive runtime limit.')
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = {'complete': False, 'settings': vars(args), 'optimizer_updates': 0,
              'checkpoint': str(Path(args.checkpoint).resolve()), 'batches_completed': 0,
              'scope': 'Fixed-input diagnosis, ERAF off, original Correct4/CF2 and full FG; no model selection.'}
    def check_budget():
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError('Diagnostic runtime limit reached; no automatic extension.')
    try:
        import torch
        from experiments.robotwin.eraf_fg_bridge import load_policy, trainable_parameters
        from experiments.robotwin.eraf_fg_data import RawReplay
        from experiments.robotwin.eraf_fg_training import backward_example, mixture_stream, supervision_payload
        from experiments.robotwin.gradient_diagnostic import GradientCollector, bucket
        from experiments.robotwin.compact_replay import ReplayPayloads
        from experiments.robotwin.native_teacher import NativeTeacher
        from experiments.robotwin.same_state_repair import noise_tensor
        from experiments.robotwin.decision_language_replay import build_seen_contexts, replace_language
        torch.manual_seed(args.seed)
        manifest = json.loads(Path(args.manifest).read_text())
        if not manifest.get('complete'):
            raise ValueError('Require complete prepared replay.')
        rows = manifest['states']
        policy = load_policy(args.checkpoint, manifest, seed=args.seed)
        model = policy.model
        payloads, raw = ReplayPayloads(rows, model.device), RawReplay(args.source_bank)
        if not args.skip_actions:
            result['action_probe_complete'] = action_probe(policy, rows, payloads, raw, output, check_budget)['complete']
        selected = trainable_parameters(model, 'joint', eraf=False)
        original = {n: p.detach().cpu().clone() for n, p in selected.items()}
        adapters = {n: p for n, p in model.mot.named_parameters() if n.endswith(('.lora_A', '.lora_B'))}
        teachers = {'correct': NativeTeacher(model, adapters, args.correct_teacher),
                    'cf': NativeTeacher(model, adapters, args.cf_teacher)}
        seen = build_seen_contexts(model, REPO, [r for r in rows if not r.get('native_retention') and not r.get('cf_retention')])
        stream = mixture_stream(rows, args.seed, 'full')
        for _ in range(args.start_batch - 1):
            next(stream)
        with (output / 'gradients.jsonl').open('x', buffering=1) as journal:
            for step in range(args.start_batch, args.start_batch + args.batches):
                check_budget()
                collector = GradientCollector(selected)
                entries = []
                for index, row in enumerate(next(stream)):
                    check_budget()
                    payload = supervision_payload(row, payloads[row['id']])
                    if row.get('frozen_input_protocol') != 'robotwin_eraf_fg_pre_dit_v1':
                        payload = raw.attach_proprio(payload, row, policy)
                    seed = args.seed + (step - 1) * 12 + index
                    variants = [] if row.get('native_retention') or row.get('cf_retention') else seen.get(row.get('language_replay_key', row['pair_id']), [])
                    variant_index = None
                    rng = random.Random(seed)
                    if variants and rng.random() < .5:
                        variant_index = rng.randrange(len(variants))
                        variant = variants[variant_index]
                        payload = dict(payload, captured={k: replace_language(v, *variant[k]) for k, v in payload['captured'].items()})
                    noise = noise_tensor((1, 32, 14), seed, model)
                    u = torch.rand((1,), generator=torch.Generator(device='cpu').manual_seed(seed + 1_000_000))
                    scheduler = model.train_action_scheduler
                    timestep = (scheduler._phi(u, scheduler.shift) * scheduler.num_train_timesteps).to(model.device, model.torch_dtype)
                    group = bucket(row)
                    report = backward_example(model, row, payload, noise, timestep, teachers=teachers,
                        coefficient=1/12, eraf=False, fg='full', correct_weight=4., cf_weight=2.,
                        gradient_observer=collector.observer(group, row['id']))
                    entries.append({'id': row['id'], 'group': group, 'pair_id': row['pair_id'],
                                    'scene_seed': row['scene_seed'], 'seed': seed, 'seen_variant': variant_index,
                                    'timestep': float(timestep), 'metrics': report})
                journal.write(json.dumps({'batch': step, 'examples': entries, **collector.summary()}) + '\n')
                if any(p.grad is not None for p in selected.values()):
                    raise ValueError('Diagnostic unexpectedly accumulated parameter.grad.')
                result['batches_completed'] = step - args.start_batch + 1
                (output / 'summary.json').write_text(json.dumps(result, indent=2))
                del collector
                gc.collect()
                print(f'[gradients] batches={result["batches_completed"]}/{args.batches} global_batch={step} seconds={time.monotonic()-started:.1f}', flush=True)
        unchanged = all(torch.equal(p.detach().cpu(), original[n]) for n, p in selected.items())
        if not unchanged:
            raise ValueError('Diagnostic changed policy parameters.')
        result.update(complete=True, policy_parameters_unchanged=unchanged,
                      trainable_tensors=len(selected), elapsed_seconds=time.monotonic()-started)
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
