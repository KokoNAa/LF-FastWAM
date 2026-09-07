#!/usr/bin/env python3
"""Paired cross-goal semantic calibration after matched terminal-state diagnostics."""
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
    evidence = runs / 'robotwin_cross_goal_semantics/20260908-terminal-audit'
    diagnostic = json.loads((evidence / 'comparison.json').read_text())
    if not diagnostic['complete'] or diagnostic['matched_queries'] != 80:
        raise ValueError('Complete matched opposite-goal semantic diagnostic required.')
    smoke = json.loads((runs / 'robotwin_cross_goal_semantics/20260908-paired-smoke2/training_audit.json').read_text())
    if not smoke['complete'] or smoke['local_optimizer_steps'] != 2 or not smoke['actual_samples_and_instruction_branches_verified']:
        raise ValueError('Both actual three-rank paired semantic smoke runs must pass their audit.')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('GPUs are occupied; do not overlap other work.')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        temp = root / (name + '.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(root / name)

    parent = runs / 'robotwin_cf_priority/20260907-1543-warmoff200-cf4/ordinary_cf/joint/step_000200.pt'
    parents = {mode: runs / 'robotwin_calibrated_context_identity/20260908-semantic500' / (mode + '_bootstrap.pt') for mode in ('ordinary', 'fg')}
    for mode, path in parents.items():
        if Path(smoke['arms'][mode]['parent_checkpoint']) != path:
            raise ValueError('Smoke validation used a different semantic parent.')
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
        models={mode: str(path) for mode, path in parents.items()},
        parent_semantic_steps=500, step_offset=500, paired_cross_goals=True,
        smoke_audit=str(runs / 'robotwin_cross_goal_semantics/20260908-paired-smoke2/training_audit.json'),
        manifest=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        fg_role_mask_binding=str(masks), binding_sha256=hashlib.sha256(masks.read_bytes()).hexdigest(),
        source_bank=str(source), shared_raw_label_cache=str(labels),
        diagnostic=str(evidence / 'comparison.json'), diagnostic_sha256=hashlib.sha256((evidence / 'comparison.json').read_bytes()).hexdigest(),
        steps=500, save_every=500, learning_rate=1e-4, position_weight=2., anchor_weight=2.,
        seed=42, global_batch=12, common_full_semantic_slots=9, matched_partial_slots=3,
        gpus=dict(ordinary=[0, 1, 2], fg=[3, 4, 5]), qualification_tasks=list(ROBOTWIN_TEN_TASK_NAMES),
        recipe='Start from the original matched semantic500 zero-context bootstraps, never smoke updates. Train500 fresh semantic optimizer steps. Each batch contains6 own-goal common slots,3 opposite-goal slots on3 of those exact observations, and3 unchanged matched ordinary/FG partial slots. Frozen policy and interfaces remain identical to each parent. The same supervision change is applied to both arms.',
        labels='Use the recorded physical trajectory with the selected instruction-specific geometry and truth. Cross-goal phase/history labels are invalid because that goal was not being executed. No expert action is reused under another goal. Existing own-goal and partial labels remain unchanged.',
        policy_and_action_interfaces_frozen=True, fresh_optimizer=True,
        checkpoint_selection='Predeclared final cumulative1000 semantic checkpoint. The initial500 and final1000 are reported; cross-goal terminal audit follows before any action trial. Neither semantic qualification nor reduced false positives establishes CF superiority.',
        validation='Existing expert and FG holdout; repeat the same80 opposite-goal terminal queries after training. Report true positives and false positives together to reject a trivial always-false predictor.',
        smoke='Two completed and audited steps per arm on three ranks, using the exact paired-goal branch. Smoke weights and optimizer updates are not inherited.',
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
                '--source-bank', str(source), '--checkpoint', str(parents[mode]), '--output', str(root / mode),
                '--label-cache', str(labels), '--steps', '500', '--save-every', '500', '--global-batch', '12',
                '--learning-rate', '0.0001', '--position-weight', '2', '--anchor-weight', '2',
                '--fg-role-masks', str(masks), '--geometry-replay', mode, '--geometry-replay-slots', '3',
                '--task-balanced', '--qualify-each-language', '--qualification-tasks', *ROBOTWIN_TEN_TASK_NAMES,
                '--skip-file-hashes', '--paired-cross-goals', '--step-offset', '500']
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
        for mode, process in processes.items():
            state['jobs'][mode]['exit_code'] = process.returncode
            if process.returncode:
                raise RuntimeError(f'{mode} training failed: {process.returncode}')
        from scripts.audit_robotwin_cross_goal_training import audit
        write('training_audit.json', audit(root))
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
