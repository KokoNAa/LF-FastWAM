#!/usr/bin/env python3
"""Warm the ERAF action interface and jointly repair the shared policy."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'source-bank', 'checkpoint', 'output', 'correct-teacher', 'cf-teacher'):
        ap.add_argument('--' + key, required=True)
    ap.add_argument('--stage', choices=['interface', 'joint'], required=True)
    ap.add_argument('--fg', choices=['off', 'local', 'full'], default='full')
    ap.add_argument('--eraf', choices=['on', 'off'], default='on')
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--save-every', type=int, default=200)
    ap.add_argument('--learning-rate', type=float, default=1e-5)
    ap.add_argument('--correct-weight', type=float, default=2.)
    ap.add_argument('--cf-weight', type=float, default=1.)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--disable-seen-language-augmentation', action='store_true')
    ap.add_argument('--resume-state', help='Resume saved FP32 masters and optimizer at the supplied checkpoint.')
    args = ap.parse_args()
    import torch
    import torch.distributed as dist
    from experiments.robotwin.eraf_fg_bridge import load_policy, trainable_parameters, MasterAdamW, save_repair_checkpoint, validate_payload, file_sha256
    from experiments.robotwin.eraf_fg_data import RawReplay
    from experiments.robotwin.eraf_fg_training import backward_example, mixture_stream, supervision_payload, FG_OFF_PROTOCOL
    from experiments.robotwin.compact_replay import ReplayPayloads
    from experiments.robotwin.native_teacher import NativeTeacher
    from experiments.robotwin.same_state_repair import noise_tensor
    from scripts.train_robotwin_cf_decision_adapter import average_gradients
    world, rank, local = (int(os.environ.get(k, d)) for k, d in
                           (('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')))
    if 12 % world or min(args.steps, args.save_every) <= 0:
        ap.error('Positive steps and world size dividing global batch12 required.')
    if args.stage == 'interface' and args.eraf == 'off':
        ap.error('ERAF-off has no interface warmup.')
    parent_payload = validate_payload(torch.load(args.checkpoint, map_location='cpu', weights_only=False))
    if args.resume_state:
        if (parent_payload['stage'] != args.stage or parent_payload['fg_supervision'] != args.fg
                or parent_payload['provenance'].get('eraf') != args.eraf):
            raise ValueError('Resumed checkpoint must have the same stage and ablation settings.')
    elif args.stage == 'interface':
        if parent_payload['stage'] != 'grounding':
            raise ValueError('Every interface ablation must start from the same semantic checkpoint.')
    elif args.eraf == 'on':
        if parent_payload['stage'] != 'interface' or parent_payload['fg_supervision'] != args.fg:
            raise ValueError('Joint ERAF training needs its own matching interface arm; full FG must not leak into controls.')
    elif parent_payload['stage'] != 'grounding':
        raise ValueError('ERAF-off joint controls must start from the common unchanged-policy semantic checkpoint.')
    del parent_payload
    torch.cuda.set_device(local)
    torch.manual_seed(args.seed)
    if world > 1:
        dist.init_process_group('nccl', device_id=torch.device(f'cuda:{local}'))

    def barrier():
        if world > 1:
            dist.barrier(device_ids=[local])

    root = Path(args.output).resolve()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    barrier()
    manifest = json.loads(Path(args.manifest).read_text())
    if not manifest.get('complete'):
        raise ValueError('Action training requires complete prepared replay.')
    rows = manifest['states']
    if args.steps > 2:
        cf_tasks = {r['source_task'] for r in rows if r.get('cf_retention')}
        if cf_tasks != {'place_a2b_right', 'place_burger_fries', 'stack_blocks_two'}:
            raise ValueError('CF retention must cover the three previously successful tasks.')
        for task in cf_tasks:
            if len({r['scene_seed'] for r in rows if r.get('cf_retention') and r['source_task'] == task}) < 10:
                raise ValueError('Formal action training requires ten CF-retention scenes per preserved task.')
        if args.fg != 'off':
            for split in ('train', 'replay_holdout'):
                if {r['source_task'] for r in rows if r.get('fg_correction') and r['replay_split'] == split} != {'place_a2b_left', 'blocks_ranking_rgb'}:
                    raise ValueError('FG train and holdout must cover both target tasks.')
                minimum = 24 if split == 'train' else 6
                for task in ('place_a2b_left', 'blocks_ranking_rgb'):
                    count = len({r['scene_seed'] for r in rows if r.get('fg_correction')
                                 and r['source_task'] == task and r['replay_split'] == split})
                    if count < minimum:
                        raise ValueError(f'Formal FG training needs {minimum} {split} scenes for {task}, got {count}.')
    policy = load_policy(args.checkpoint, manifest, device=f'cuda:{local}', seed=args.seed)
    model = policy.model
    selected = trainable_parameters(model, args.stage, eraf=args.eraf == 'on')
    adapters = {n: p for n, p in model.mot.named_parameters() if n.endswith(('.lora_A', '.lora_B'))}
    teachers = {'correct': NativeTeacher(model, adapters, args.correct_teacher),
                'cf': NativeTeacher(model, adapters, args.cf_teacher)}
    optimizer = MasterAdamW(selected.values(), lr=args.learning_rate)
    start = 0
    optimization_contract = {k: getattr(args, k) for k in ('stage', 'fg', 'eraf', 'seed',
        'learning_rate', 'correct_weight', 'cf_weight', 'disable_seen_language_augmentation')}
    optimization_contract['manifest_sha256'] = file_sha256(args.manifest)
    if args.fg == 'off':
        optimization_contract['fg_off_protocol'] = FG_OFF_PROTOCOL
    adjustments = {}
    if args.resume_state:
        state = torch.load(args.resume_state, map_location='cpu', weights_only=False)
        if state['parameter_names'] != list(selected) or state['checkpoint_sha256'] != file_sha256(args.checkpoint):
            raise ValueError('Optimizer does not belong to this exact checkpoint and parameter scope.')
        for key, value in optimization_contract.items():
            if state['optimization_contract'].get(key) != value:
                if key not in ('learning_rate', 'correct_weight', 'cf_weight'):
                    raise ValueError(f'Resume changed immutable training contract: {key}')
                adjustments[key] = {'from': state['optimization_contract'][key], 'to': value}
        optimizer.load_state_dict(state['optimizer'])
        for group in optimizer.optimizer.param_groups:
            group['lr'] = args.learning_rate
        start = int(state['step'])
        if args.steps <= start:
            raise ValueError('Resume total-step target must exceed saved optimizer steps.')
        del state
    payloads = ReplayPayloads(rows, model.device)
    raw = RawReplay(args.source_bank)
    stream = mixture_stream(rows, args.seed, args.fg)
    for _ in range(start):
        next(stream)
    from experiments.robotwin.decision_language_replay import build_seen_contexts, replace_language
    seen_contexts = ({} if args.disable_seen_language_augmentation else build_seen_contexts(model, REPO,
        [r for r in rows if not r.get('native_retention') and not r.get('cf_retention')
         and not (args.fg == 'off' and r.get('fg_correction'))]))
    if rank == 0:
        (root / 'plan.json').write_text(json.dumps(vars(args) | {'world_size': world, 'global_batch': 12,
            'start_optimizer_step': start, 'resume_adjustments': adjustments,
            'optimization_contract': optimization_contract,
            'trainable_parameters': list(selected), 'optimizer_precision': 'FP32 master weights',
            'mixture': {'correct_retention': 4, 'cf_retention': 2,
                        'expert_pairs': 3, 'fg': 0 if args.fg == 'off' else 3,
                        'ordinary_cf_control': 3 if args.fg == 'off' else 0}}, indent=2))
    started = time.monotonic()
    journal = (root / f'rank{rank}.jsonl').open('x', buffering=1)
    for step in range(start + 1, args.steps + 1):
        optimizer.zero_grad()
        batch = next(stream)
        reports = []
        for index, row in enumerate(batch):
            if index % world != rank:
                continue
            payload = supervision_payload(row, payloads[row['id']])
            if row.get('frozen_input_protocol') != 'robotwin_eraf_fg_pre_dit_v1':
                payload = raw.attach_proprio(payload, row, policy)
            seed = args.seed + (step - 1) * 12 + index
            variants = ([] if row.get('native_retention') or row.get('cf_retention') else
                        seen_contexts.get(row.get('language_replay_key', row['pair_id']), []))
            variant_index = None
            rng = random.Random(seed)
            if variants and rng.random() < .5:
                variant_index = rng.randrange(len(variants))
                variant = variants[variant_index]
                payload = dict(payload, captured={k: replace_language(v, *variant[k])
                                                  for k, v in payload['captured'].items()})
            noise = noise_tensor((1, 32, 14), seed, model)
            u = torch.rand((1,), generator=torch.Generator(device='cpu').manual_seed(seed + 1_000_000))
            scheduler = model.train_action_scheduler
            t = (scheduler._phi(u, scheduler.shift) * scheduler.num_train_timesteps).to(model.device, model.torch_dtype)
            report = backward_example(model, row, payload, noise, t, teachers=teachers,
                coefficient=1. / (12 // world), eraf=args.eraf == 'on', fg=args.fg,
                correct_weight=args.correct_weight, cf_weight=args.cf_weight)
            reports.append({'id': row['id'], 'seen_variant': variant_index,
                            'ordinary_cf_control': bool(row.get('ordinary_cf_control')), **report})
        average_gradients(selected.values())
        norm = optimizer.step()
        journal.write(json.dumps({'step': step, 'grad_norm': norm, 'examples': reports,
                                  'elapsed': time.monotonic() - started}) + '\n')
        if rank == 0 and (step == start + 1 or step % 10 == 0):
            print(f'[action] step={step}/{args.steps} seconds_per_step={(time.monotonic()-started)/(step-start):.2f} grad={norm:.4f}', flush=True)
        if step % args.save_every == 0 or step == args.steps:
            barrier()
            if rank == 0:
                checkpoint_path = root / f'step_{step:06d}.pt'
                save_repair_checkpoint(model, checkpoint_path, stage=args.stage,
                    steps=step, parent=args.checkpoint, fg_supervision=args.fg,
                    provenance={'plan': str(root / 'plan.json'), 'eraf': args.eraf})
                torch.save({'step': step, 'checkpoint': str(checkpoint_path),
                    'checkpoint_sha256': file_sha256(checkpoint_path), 'parameter_names': list(selected),
                    'optimization_contract': optimization_contract, 'optimizer': optimizer.state_dict()},
                    root / 'optimizer_last.tmp')
                (root / 'optimizer_last.tmp').replace(root / 'optimizer_last.pt')
                print(f'[checkpoint] step={step}', flush=True)
            barrier()
    journal.close()
    if rank == 0:
        (root / 'complete.json').write_text(json.dumps({'complete': True,
            'optimizer_steps': args.steps, 'local_optimizer_steps': args.steps - start}))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
