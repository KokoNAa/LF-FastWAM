#!/usr/bin/env python3
"""Collect fresh paired expert futures for inference-only world-language tests."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
TASKS = ['blocks_ranking_rgb', 'stack_blocks_two', 'place_a2b_left',
         'place_a2b_right', 'place_burger_fries']


def write(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def verify_collection(folder, n):
    import h5py
    import numpy as np
    from experiments.robotwin.no_eraf_probe import require_pair
    roots = list(folder.glob('*/native/meta/pgc_episodes.jsonl'))
    if len(roots) != 1:
        raise ValueError('Missing unique paired collection')
    pair = roots[0].parents[2]
    rows = {k: [json.loads(x) for x in (pair/k/'meta/pgc_episodes.jsonl').read_text().splitlines()]
            for k in ['native', 'counterfactual']}
    if any(len(x) != n for x in rows.values()):
        raise ValueError('Incomplete paired expert collection')
    files = {}
    scenes = []
    for a, b in zip(rows['native'], rows['counterfactual'], strict=True):
        require_pair(a, b)
        assert a['goal_verified'] and b['goal_verified']
        paths = {k: pair/k/r['raw_hdf5'] for k, r in [('native', a), ('counterfactual', b)]}
        with h5py.File(paths['native']) as x, h5py.File(paths['counterfactual']) as y:
            assert np.array_equal(x['joint_action/vector'][0], y['joint_action/vector'][0])
            for camera in ['head_camera', 'left_camera', 'right_camera']:
                assert bytes(x[f'observation/{camera}/rgb'][0]) == bytes(y[f'observation/{camera}/rgb'][0])
        for p in paths.values():
            h = hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
            files[str(p)] = h.hexdigest()
        scenes.append(dict(scene_seed=a['scene_seed'], task=a['source_task'],
                           initial_state_sha256=a['initial_state_sha256']))
    assert len({r['scene_seed'] for r in scenes}) == n
    return dict(complete=True, episodes_per_branch=n, scenes=scenes, raw_sha256=files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--robotwin-root', default='/root/gpufree-data/LF-FastWAM/third_party/RoboTwin')
    ap.add_argument('--start-seed', type=int, default=97000000)
    args = ap.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path[:0] = [str(REPO), str(REPO/'src')]
    env = os.environ | dict(PATH='/opt/conda/bin:'+os.environ['PATH'],
        PYTHONPATH=str(REPO/'src')+':'+str(REPO), CUDA_VISIBLE_DEVICES='0',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    protocol = dict(format='robotwin_world_language_collection_v1', tasks=TASKS,
        episodes_per_task=args.episodes, start_seed=args.start_seed, task_config='demo_randomized',
        eval_mode=True, training_allowed=False,
        selection='Both initial goals false and both expert goals feasible; no policy outcome selection.',
        shared_grasp_prefix_tasks=TASKS[2:], code_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
    if (root/'protocol.json').exists():
        old = json.loads((root/'protocol.json').read_text())
        assert old == protocol, 'Cannot change collection protocol on resume'
    else: write(root/'protocol.json', protocol)
    status = dict(complete=False, controller_pid=os.getpid(), jobs={})
    try:
        for index, task in enumerate(TASKS):
            folder = root/task
            proof = root/(task+'-verified.json')
            if proof.exists():
                assert verify_collection(folder, args.episodes) == json.loads(proof.read_text())
                status['jobs'][task] = dict(status='verified'); continue
            if folder.exists():
                raise RuntimeError('Partial collection requires inspection before retry: '+str(folder))
            cmd = [sys.executable, str(REPO/'scripts/collect_robotwin_world_language_pairs.py'),
                '--output-root', str(folder), '--robotwin-root', args.robotwin_root,
                '--task-config', 'demo_randomized', '--episodes', str(args.episodes),
                '--source-tasks', task, '--start-seed', str(args.start_seed+index*1000000),
                '--collection-profile', 'no_eraf_strict', '--max-seed-attempts', '200']
            if task in TASKS[2:]: cmd.append('--shared-grasp-prefix')
            with (root/(task+'.log')).open('x') as log:
                child = subprocess.Popen(cmd, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                start = Path(f'/proc/{child.pid}/stat').read_text().rsplit(') ',1)[1].split()[19]
                status['jobs'][task] = dict(pid=child.pid, start_time=start, status='running', command=cmd)
                write(root/'status.json', status)
                rc = child.wait()
            status['jobs'][task].update(exit_code=rc, status='exited')
            write(root/'status.json', status)
            if rc: raise RuntimeError(f'Collection failed: {task}: {rc}')
            write(proof, verify_collection(folder, args.episodes))
            status['jobs'][task]['status'] = 'verified'
        status['complete'] = True
    except BaseException as error:
        status['error'] = repr(error)
        raise
    finally:
        status['updated_at'] = time.time()
        write(root/'status.json', status)


if __name__ == '__main__': main()
