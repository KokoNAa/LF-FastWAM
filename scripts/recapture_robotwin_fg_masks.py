#!/usr/bin/env python3
"""Replay recorded FG controls and save labels only for matched observations."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--robotwin-root', default=str(REPO / 'third_party/RoboTwin'))
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--limit-per-cell', type=int, default=1)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    args = ap.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        ap.error('Require a positive shard count and a valid shard index.')
    args.manifest, args.output = str(Path(args.manifest).resolve()), str(Path(args.output).resolve())
    args.robotwin_root = str(Path(args.robotwin_root).resolve())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import numpy as np
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    from experiments.robotwin.eraf_fg_contract import CAMERAS, validate_correction
    from experiments.robotwin.eraf_fg_collection import replay_prefix, replay_continuation, full_goal
    from experiments.robotwin.fg_mask_replay import SCHEMA, VerifiedMaskFrames, capture_verified_masks, select_replay_scenes
    from experiments.robotwin.pgc_data import array_sha256, pair_spec_from_source_task
    from experiments.robotwin.pgc_task_variants import install_pgc_task_contract
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args, _capture_data_type, _close
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    all_rows = select_replay_scenes(json.loads(Path(args.manifest).read_text())['states'], limit_per_cell=args.limit_per_cell)
    rows = all_rows[args.shard_index::args.num_shards]
    if not rows:
        raise ValueError('Selected mask replay shard is empty.')
    (root / 'plan.json').write_text(json.dumps(vars(args) | {'schema': SCHEMA,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'manifest_sha256': file_sha256(args.manifest), 'source_records': rows,
        'unsharded_scene_count': len(all_rows),
        'optimizer_updates': 0, 'source_captures_modified': False,
        'acceptance': 'All original RGB frames identical; qpos and semantic geometry within1e-7; original prefix and control states verified; full goal remains true.'}, indent=2))
    reports = []
    for row in rows:
        task = None
        report = {'pair_id': row['pair_id'], 'task_config': row['task_config'],
                  'scene_seed': row['scene_seed'], 'replay_split': row['replay_split'], 'complete': False}
        try:
            validate_correction(row)
            source = Path(row['frame_path'])
            controls_path = source.parent / 'controls.pkl'
            if file_sha256(source) != row['frame_sha256'] or file_sha256(controls_path) != row['controls_sha256']:
                raise ValueError('Correction or control archive differs from the frozen manifest.')
            with np.load(source, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            with np.load(source.parent / 'failure_rollout.npz', allow_pickle=False) as archive:
                trace = {k: archive[k] for k in archive.files}
            trace['states'] = dict(zip(map(int, trace['capture_steps']), trace['states']))
            step = row['capture_action_index']
            for name, value in [('initial_state_sha256', trace['initial']),
                                ('capture_state_sha256', trace['states'][step]),
                                ('prefix_action_sha256', trace['actions'][:step]),
                                ('correction_action_sha256', arrays['actions'])]:
                if array_sha256(value) != row[name]:
                    raise ValueError('Recorded replay identity mismatch: ' + name)
            # Trusted, hash-bound control sequences created by our collection.
            with controls_path.open('rb') as handle:
                controls = pickle.load(handle)
            spec = pair_spec_from_source_task(row['source_task'])
            task, task_args = _load_robotwin_args(robotwin_root=Path(args.robotwin_root).resolve(),
                task_name=row['source_task'], task_config=row['task_config'], output_root=root)
            install_pgc_task_contract(task, spec)
            task_args.update(data_type=_capture_data_type(task_args), eval_mode=True,
                             need_plan=True, save_data=False, render_freq=0, save_freq=row['save_freq'])
            task._pgc_active_variant = spec.counterfactual_variant
            task.setup_demo(now_ep_num=0, seed=row['scene_seed'], **deepcopy(task_args))
            error = replay_prefix(task, spec, trace, step)
            frames = VerifiedMaskFrames(arrays, row['pair_id'])
            with capture_verified_masks(task, frames):
                error = max(error, replay_continuation(task, controls))
            if not full_goal(task, spec):
                raise ValueError('Mask replay did not retain full-goal completion.')
            labels = frames.finish()
            name = f"{row['pair_id']}_{row['task_config']}_{row['scene_seed']}"
            target = root / (name + '.npz')
            np.savez_compressed(target, **labels)
            report.update(complete=True, labels=str(target), labels_sha256=file_sha256(target),
                frame_path=str(source), frame_sha256=row['frame_sha256'], controls_sha256=row['controls_sha256'],
                failure_archive_sha256=file_sha256(source.parent / 'failure_rollout.npz'),
                frames=len(arrays['actions']), all_rgb_frames_equal=True, replay_state_max_abs=error,
                subject_visible_clauses=int(labels['subject_mask_valid'].sum()),
                reference_visible_clauses=int(labels['reference_mask_valid'].sum()),
                phase_labels_valid=False, temporal_labels_valid=False)
        except Exception as e:
            report['error'] = repr(e)
            report['error_detail'] = str(e)
            report['traceback'] = traceback.format_exc()
        finally:
            if task is not None:
                try:
                    _close(task)
                except Exception as error:
                    report['close_error'] = repr(error)
                    report['complete'] = False
            reports.append(report)
            with (root / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(report) + '\n')
            print(json.dumps(report), flush=True)
    (root / 'complete.json').write_text(json.dumps({'complete': all(r['complete'] for r in reports),
        'schema': SCHEMA, 'scenes': reports, 'optimizer_updates': 0}, indent=2))
    if not all(r['complete'] for r in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
