#!/usr/bin/env python3
"""Check full retention teacher against actual ERAF deployment, without updates."""
from __future__ import annotations
import argparse
from functools import wraps
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'manifest', 'source-bank', 'output'):
        ap.add_argument('--' + key, required=True)
    args = ap.parse_args()
    import h5py
    import numpy as np
    import torch
    from experiments.robotwin.eraf_fg_bridge import load_policy, trainable_parameters, capture_frozen_inputs, file_sha256
    from experiments.robotwin.eraf_fg_data import RawReplay
    from experiments.robotwin.eraf_fg_contract import CAMERAS
    from experiments.robotwin.image_io import decode_legacy_robotwin_rgb
    from experiments.robotwin.same_state_repair import noise_tensor, move_cache
    from experiments.robotwin.decision_replay import tensor_digest
    from experiments.robotwin.deployed_action_objective import teacher_action
    from experiments.robotwin.native_teacher import NativeTeacher
    from experiments.robotwin.five_task_action import TASKS
    from experiments.robotwin.initial_anchor import source_task

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(args.manifest).read_text())
    rows = []
    for task in TASKS:
        choices = [r for r in manifest['states'] if source_task(r) == task
                   and r['replay_split'] == 'train' and r['task_config'] == 'demo_clean'
                   and int(r.get('target_frame_index', r.get('frame_index', 0))) == 0
                   and not any(r.get(k) for k in ('cf_retention', 'native_retention', 'fg_correction'))]
        if not choices:
            raise ValueError('Missing initial ordinary training scene: ' + task)
        rows.append(min(choices, key=lambda r: (r['scene_seed'], r['id'])))
    hashes = {k: file_sha256(getattr(args, k)) for k in ('checkpoint', 'manifest')}
    protocol = dict(checkpoint=args.checkpoint, input_sha256=hashes, selected_ids=[r['id'] for r in rows],
                    optimizer_updates=0, seed=42, inference_steps=10, cf_teacher_mode='parent_eraf')
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    policy = load_policy(args.checkpoint, manifest, seed=42)
    model = policy.model
    selected = trainable_parameters(model, 'joint', eraf=True, policy_scope='action', interface_scope='all')
    adapters = {n: p for n, p in model.mot.named_parameters() if n.endswith(('.lora_A', '.lora_B'))}
    teacher = NativeTeacher(model, adapters, args.checkpoint, eraf=True)
    state = lambda: {'adapters': {k: p.detach() for k, p in adapters.items()},
                     'guard': model.policy_guard_modules.state_dict()}
    before = tensor_digest(state())
    raw = RawReplay(args.source_bank)
    records = []
    for row in rows:
        path, frame = raw.locate(row, 'target')
        with h5py.File(path, 'r') as h:
            obs = {'joint_action': {'vector': h['joint_action/vector'][frame].astype(np.float32)},
                   'observation': {c: {'rgb': decode_legacy_robotwin_rgb(h[f'observation/{c}/rgb'][frame])}
                                   for c in CAMERAS}}
        policy.reset()
        captured = move_cache(capture_frozen_inputs(policy, obs, row['counterfactual_instruction']), model.device)
        policy.reset()
        model.policy_guard_enabled = True
        actual = []
        original = model.infer_action
        @wraps(original)
        def record(*pos, **kw):
            output = original(*pos, **kw)
            actual.append(output['action'].detach().clone())
            return output
        existed, previous = 'infer_action' in vars(model), vars(model).get('infer_action')
        model.infer_action = record
        try:
            with torch.no_grad():
                policy._infer_action_chunk(obs, row['counterfactual_instruction'])
        finally:
            if existed:
                model.infer_action = previous
            else:
                delattr(model, 'infer_action')
        if len(actual) != 1:
            raise ValueError('Expected one deployed action chunk.')
        noise = noise_tensor((1, 32, 14), 42, model)
        predicted = teacher_action(teacher, captured, noise)
        expected = actual[0].to(predicted).unsqueeze(0)
        if not torch.equal(predicted, expected):
            raise ValueError('Teacher differs from deployed ERAF: ' + str(float((predicted - expected).abs().max())))
        # Deliberately change one live action and guard weight to test the swap.
        keys = [next(k for k in selected if k.startswith(prefix)) for prefix in ('mot.', 'guard.')]
        saved = {k: selected[k].detach().clone() for k in keys}
        try:
            with torch.no_grad():
                for k in keys:
                    selected[k].add_(.125)
            modified = tensor_digest(state())
            if not torch.equal(teacher_action(teacher, captured, noise), expected):
                raise ValueError('Teacher drifted after student changes.')
            if tensor_digest(state()) != modified:
                raise ValueError('Teacher did not restore all student tensors.')
        finally:
            with torch.no_grad():
                for k, value in saved.items():
                    selected[k].copy_(value)
        records.append(dict(id=row['id'], task=source_task(row), production_exact=True,
                            fixed_after_student_change=True, student_restored=True,
                            action_digest=tensor_digest(predicted)))
        print('[identity]', len(records), '/', len(rows), flush=True)
    if tensor_digest(state()) != before or any(p.grad is not None for p in selected.values()):
        raise ValueError('Read-only teacher probe changed weights or gradients.')
    if hashes != {k: file_sha256(getattr(args, k)) for k in hashes}:
        raise ValueError('Probe inputs changed during execution.')
    report = protocol | dict(complete=True, all_five_tasks_exact=True, student_unchanged=True,
                             records=records, scope='Deployment identity only; no training or CF evaluation.')
    (root / 'complete.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
