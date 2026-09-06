#!/usr/bin/env python3
"""Collect bounded goal branches from already recorded real policy failures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('collection', 'output', 'reuse-root', 'robotwin-root'):
        ap.add_argument('--'+key, type=Path, required=True)
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--shards', type=int, default=4)
    ap.add_argument('--max-seconds', type=int, default=1800)
    args = ap.parse_args()
    if not 0 <= args.shard < args.shards: ap.error('Invalid shard')
    import numpy as np
    from experiments.robotwin.cup_goal_branches import FORMAT, branch_record
    collection = json.loads(args.collection.read_text())
    if collection.get('complete') is not True: raise ValueError('Parent failure collection incomplete')
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    report = {'format': FORMAT, 'complete': False, 'records': [], 'attempts': [],
              'parent_collection': str(args.collection.resolve()), 'hash_scans': False}
    deadline = time.monotonic()+args.max_seconds

    def save():
        (root/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')

    save()
    for parent in collection['records'][args.shard::args.shards]:
        attempt = {'scene_seed': parent['scene_seed'], 'candidates': []}; report['attempts'].append(attempt)
        record_path = Path(parent['frame_path']).parent/'record.json'
        with np.load(parent['rollout_path'], allow_pickle=False) as trace:
            states = dict(zip(trace['capture_steps'].tolist(), trace['states']))
            candidates = [step for step in (120, 144, 96, 168)
                          if step in states and states[step][2]-trace['initial'][2] >= .03]
        for step in candidates:
            if time.monotonic() >= deadline: save(); raise TimeoutError('Bounded branch collection deadline')
            reused = args.reuse_root/str(parent['scene_seed'])/'results.json'
            if step == 120 and reused.exists():
                path = reused
            else:
                folder = root/f'scene_{parent["scene_seed"]}_step{step}'
                command = [sys.executable, '-u', str(REPO/'scripts/probe_robotwin_cup_goal_branch.py'),
                    '--record', str(record_path), '--output', str(folder), '--robotwin-root', str(args.robotwin_root),
                    '--prefix-steps', str(step)]
                with (root/f'{parent["scene_seed"]}_{step}.log').open('x') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                                   timeout=min(600, max(1, deadline-time.monotonic())))
                path = folder/'results.json'
            candidate = {'step': step, 'result_path': str(path)}; attempt['candidates'].append(candidate)
            try:
                row = branch_record(parent, json.loads(path.read_text()), path)
                report['records'].append(row)
                candidate['accepted'] = True
                print('[branch-record]', parent['scene_seed'], step, 'paired', row['paired_goal_branch'], flush=True)
                break
            except ValueError as error:
                candidate['rejection'] = str(error)
            finally: save()
        save()
    report.update(complete=True, accepted_scenes=len(report['records']),
                  paired_scenes=sum(r['paired_goal_branch'] for r in report['records']))
    save()
    print(json.dumps({k: report[k] for k in ('complete', 'accepted_scenes', 'paired_scenes')}), flush=True)


if __name__ == '__main__': main()
