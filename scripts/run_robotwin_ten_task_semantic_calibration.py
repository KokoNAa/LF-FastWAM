#!/usr/bin/env python3
"""Matched ten-task semantic calibration after the completed joint200 diagnostic."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--deadline', required=True)
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None:
        ap.error('Require an absolute timezone-aware cutoff.')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Commit code before launching.')
    runs = Path('/root/gpufree-data/LF-FastWAM/runs')
    evidence = runs / 'robotwin_ten_task_semantic_audit/20260908-post-joint200'
    diagnostic = json.loads((evidence / 'comparison.json').read_text())
    if not diagnostic['complete'] or diagnostic['matched_queries'] != 539:
        raise ValueError('Complete matched semantic diagnostic required.')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('GPUs are occupied; do not overlap other work.')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        temp = root / (name + '.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(root / name)

    parent = runs / 'robotwin_cf_priority/20260907-1543-warmoff200-cf4/ordinary_cf/joint/step_000200.pt'
    manifest = runs / 'robotwin_joint_expert_expansion/20260907-formal16/cache_ten_tasks/manifest.json'
    masks = runs / 'robotwin_ten_task_grounding/20260908-mask-binding/binding.json'
    source = runs / 'robotwin_cf_cause_audit/decision-bank-20260905-220457'
    labels = runs / 'robotwin_fg_role_masks/20260907-2255-logfix100/label_cache/ordinary'
    from experiments.robotwin.fg_role_masks import VerifiedCorrectionMasks
    from experiments.robotwin.fg_geometry_replay import geometry_mixture, validate_scene_splits
    from experiments.robotwin.pgc_data import ROBOTWIN_TEN_TASK_NAMES
    data = json.loads(manifest.read_text())
    rows = data['states']
    validate_scene_splits(rows)
    VerifiedCorrectionMasks(masks, manifest).validate_coverage(rows)
    protocol = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        parent=str(parent), parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        manifest=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        fg_role_mask_binding=str(masks), binding_sha256=hashlib.sha256(masks.read_bytes()).hexdigest(),
        source_bank=str(source), shared_raw_label_cache=str(labels),
        diagnostic=str(evidence / 'comparison.json'), diagnostic_sha256=hashlib.sha256((evidence / 'comparison.json').read_bytes()).hexdigest(),
        steps=500, save_every=500, learning_rate=1e-4, position_weight=2., anchor_weight=2.,
        seed=42, global_batch=12, common_full_semantic_slots=9, matched_partial_slots=3,
        gpus=dict(ordinary=[0, 1, 2], fg=[3, 4, 5]), qualification_tasks=list(ROBOTWIN_TEN_TASK_NAMES),
        recipe='First ten-task semantic-head training. Both arms share strongest35% action policy and expert core; FG changes only three matched partial supervision slots. Increase position/anchor weights and semantic optimizer budget together; this is a matched FG comparison, not a single-factor learning-rate study.',
        labels='Existing audited labels unchanged, including phase labels. Missing FG phase labels remain invalid.',
        policy_and_action_interfaces_frozen=True, fresh_optimizer=True,
        checkpoint_selection='Predeclared final500 diagnostic, compare parent and report every task/language. An eligible geometry selection alone does not establish CF improvement.',
        validation='Existing expert and FG holdout; then repeat the same539 trajectory semantic queries before any action-interface trial.',
        smoke='Existing trainer already exercised on three ranks; expanded labels exercised by the complete539-query diagnostic and masks by324-frame binding audit. No smoke checkpoint inherited.',
        deadline=args.deadline, platform_shutdown=None, model_storage='server_only', disk_reserve_GiB=5)
    write('protocol.json', protocol)
    state = dict(complete=False, stage='training', jobs={})
    write('driver.json', state)
    processes = {}
    env = os.environ | dict(PATH='/opt/conda/bin:' + os.environ['PATH'], PYTHONPATH=str(REPO / 'src') + ':' + str(REPO),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')

    def budget():
        if time.time() >= cutoff.timestamp():
            raise TimeoutError('Calibration work cutoff; server stays on.')
        if shutil.disk_usage(root).free < 5 * 1024**3:
            raise RuntimeError('Five GiB disk reserve reached.')

    try:
        budget()
        # Reserve both final checkpoints plus a conservative label/log allowance.
        if shutil.disk_usage(root).free < 5 * 1024**3 + 2 * parent.stat().st_size + 160 * 1024**2:
            raise RuntimeError('Insufficient space for both final checkpoints and labels.')
        for mode, gpus in protocol['gpus'].items():
            command = ['/opt/conda/bin/python', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '3',
                str(REPO / 'scripts/train_robotwin_eraf_fg_grounding.py'), '--manifest', str(manifest),
                '--source-bank', str(source), '--checkpoint', str(parent), '--output', str(root / mode),
                '--label-cache', str(labels), '--steps', '500', '--save-every', '500', '--global-batch', '12',
                '--learning-rate', '0.0001', '--position-weight', '2', '--anchor-weight', '2',
                '--fg-role-masks', str(masks), '--geometry-replay', mode, '--geometry-replay-slots', '3',
                '--task-balanced', '--qualify-each-language', '--qualification-tasks', *ROBOTWIN_TEN_TASK_NAMES,
                '--skip-file-hashes']
            with (root / (mode + '.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=REPO, env=env | {'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus))},
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes[mode] = process
            state['jobs'][mode] = dict(pid=process.pid, gpus=gpus, command=command)
            write('driver.json', state)
        while any(p.poll() is None for p in processes.values()):
            budget()
            for mode, process in processes.items():
                rc = process.poll()
                if rc is not None:
                    state['jobs'][mode]['exit_code'] = rc
                    if rc:
                        raise RuntimeError(f'{mode} training failed: {rc}')
            write('driver.json', state)
            time.sleep(5)
        import torch
        base = torch.load(parent, map_location='cpu', weights_only=False)
        audit = {}
        journals = {}
        for mode, process in processes.items():
            if process.returncode:
                raise RuntimeError(f'{mode} training failed: {process.returncode}')
            state['jobs'][mode]['exit_code'] = process.returncode
            folder = root / mode
            if not json.loads((folder / 'complete.json').read_text())['complete']:
                raise ValueError('Incomplete semantic stage.')
            payload = torch.load(folder / 'step_000500.pt', map_location='cpu', weights_only=False)
            assert base['mot_trainable'].keys() == payload['mot_trainable'].keys()
            assert all(torch.equal(v, payload['mot_trainable'][k]) for k, v in base['mot_trainable'].items())
            allowed = set(json.loads((folder / 'plan.json').read_text())['trainable_parameters'])
            changed = [k for k, v in base['policy_guard'].items() if not torch.equal(v, payload['policy_guard'][k])]
            assert changed and all('guard.' + k in allowed for k in changed)
            journals[mode] = {rank: [json.loads(line) for line in (folder / f'rank{rank}.jsonl').read_text().splitlines()]
                              for rank in range(3)}
            assert all(len(log) == 500 for log in journals[mode].values())
            audit[mode] = dict(complete=True, steps=500, policy_tensors_identical=True,
                              only_declared_semantics_changed=True, changed_guard_tensors=changed)
            del payload
        streams = {mode: geometry_mixture(rows, 42, batch_size=12, slots=3, mode=mode,
                                         task_balanced=True, world_size=3) for mode in processes}
        exposure = Counter()
        for step in range(500):
            batches = {mode: next(stream) for mode, stream in streams.items()}
            for mode, batch in batches.items():
                norms = [journals[mode][rank][step]['grad_norm'] for rank in range(3)]
                assert len(set(norms)) == 1 and math.isfinite(norms[0]) and norms[0] > 0
                for rank in range(3):
                    record = journals[mode][rank][step]
                    selected = batch[rank::3]
                    assert record['step'] == step + 1 and math.isfinite(record['mean_loss'])
                    assert record['samples'] == [[r['id'], language] for r, language, partial in selected]
                    assert record['partial_geometry_slots'] == [partial for r, language, partial in selected]
                    assert all(r['replay_split'] == 'train' for r, language, partial in selected)
            for (a, la, pa), (b, lb, pb) in zip(batches['ordinary'], batches['fg'], strict=True):
                assert (la, pa) == (lb, pb)
                if pa:
                    assert (a['pair_id'], a['task_config']) == (b['pair_id'], b['task_config'])
                    assert not a.get('fg_correction') and b['fg_correction']
                else:
                    assert a == b
                    exposure[a['pair_id']] += 1
        assert len(exposure) == 10 and sum(exposure.values()) == 4500
        write('training_audit.json', dict(complete=True, arms=audit, common_expert_samples=dict(exposure),
            samples_per_arm=6000, matched_partial_samples=1500, actual_samples_and_rank_norms_verified=True))
        state.update(complete=True, stage='complete')
    except BaseException as error:
        state['error'] = repr(error)
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        state['finished'] = datetime.now().astimezone().isoformat()
        write('driver.json', state)


if __name__ == '__main__':
    main()
