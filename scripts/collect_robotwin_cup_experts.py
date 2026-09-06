#!/usr/bin/env python3
"""Record ordinary source/CF experts in identical fresh cup scenes."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--robotwin-root', type=Path, required=True)
    ap.add_argument('--start-seed', type=int, required=True)
    ap.add_argument('--scenes', type=int, required=True)
    ap.add_argument('--split', choices=['train', 'replay_holdout'], required=True)
    args = ap.parse_args()
    lower = 84000000 if args.split == 'train' else 85000000
    if not (1 <= args.scenes <= 30 and lower <= args.start_seed
            and args.start_seed + args.scenes <= lower+1000000):
        ap.error('Invalid bounded ordinary expert scene range')
    import numpy as np
    from experiments.robotwin.cup_expert_data import FORMAT, validate_pair
    from experiments.robotwin.cup_counterfactual import SOURCE_INSTRUCTION, FRONT_INSTRUCTION
    from experiments.robotwin.eraf_fg_collection import physical_state, record_continuation, replay_continuation, full_goal
    from experiments.robotwin.eraf_fg_contract import CAMERAS, verify_replayed_state
    from experiments.robotwin.pgc_data import pair_spec_from_source_task
    from experiments.robotwin.pgc_task_variants import install_pgc_task_contract, play_variant
    from scripts.collect_pgc_robotwin_pairs import _load_robotwin_args, _capture_data_type, _close
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = pair_spec_from_source_task('place_empty_cup')
    task, config = _load_robotwin_args(robotwin_root=args.robotwin_root.resolve(), task_name=spec.source_task,
        task_config='demo_clean', output_root=args.output)
    install_pgc_task_contract(task, spec)
    config.update(data_type=_capture_data_type(config), eval_mode=True, need_plan=True,
                  save_data=False, render_freq=0)
    report = {'format': FORMAT, 'complete': False, 'requested_scenes': args.scenes,
              'records': [], 'attempts': [], 'hash_scans': False}
    opened = False

    def save():
        (args.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')

    def setup(seed, variant):
        nonlocal opened
        opened = True
        task._pgc_active_variant = variant
        task.setup_demo(now_ep_num=0, seed=seed, is_test=False, **deepcopy(config))

    def close():
        nonlocal opened
        if opened:
            try: _close(task)
            finally: opened = False

    def frame():
        obs = task.get_obs()
        return {'qpos': np.array(obs['joint_action']['vector'], copy=True),
                'images': {c: np.array(obs['observation'][c]['rgb'], copy=True) for c in CAMERAS},
                'grounding': deepcopy(task.pgc_eraf_snapshot())}

    save()
    for seed in range(args.start_seed, args.start_seed+args.scenes):
        folder = args.output/f'scene_{seed}'; folder.mkdir()
        attempt = {'scene_seed': seed}; report['attempts'].append(attempt)
        try:
            row = {'format': FORMAT, 'source_task': spec.source_task, 'pair_id': spec.pair_id,
                'task_config': 'demo_clean', 'scene_seed': seed, 'replay_split': args.split,
                'source_instruction': SOURCE_INSTRUCTION, 'counterfactual_instruction': FRONT_INSTRUCTION,
                'reference_kind': 'ordinary_expert_from_initial_scene', 'fg_correction': False,
                'initial_observations_exactly_equal': True, 'image_color_space': 'RGB'}
            initial = initial_frame = None
            for language, variant in (('source', spec.source_variant), ('target', spec.counterfactual_variant)):
                setup(seed, variant)
                state, first = physical_state(task), frame()
                if initial is None:
                    initial, initial_frame = state, first
                else:
                    verify_replayed_state(initial, state)
                    if (not np.array_equal(first['qpos'], initial_frame['qpos']) or
                            any(not np.array_equal(first['images'][c], initial_frame['images'][c]) for c in CAMERAS)):
                        raise ValueError('Ordinary experts changed the initial observation')
                with record_continuation(task) as (controls, frames):
                    frames.append(first)
                    play_variant(task, spec, variant)
                okay = bool(task.plan_success and full_goal(task, spec, selected_goal=language))
                attempt[language+'_success'] = okay
                close()
                if not okay or len(frames) < 32:
                    raise ValueError('Ordinary expert failed the complete '+language+' goal')
                setup(seed, variant)
                error = verify_replayed_state(initial, physical_state(task))
                error = max(error, replay_continuation(task, controls))
                verified = full_goal(task, spec, selected_goal=language)
                close()
                if not verified: raise ValueError('Ordinary expert physical replay failed')
                arrays = {'actions': np.stack([f['qpos'] for f in frames]).astype(np.float32)}
                arrays.update({c: np.stack([f['images'][c] for f in frames]) for c in CAMERAS})
                arrays.update({'grounding/'+k: np.stack([f['grounding'][k] for f in frames])
                               for k in frames[0]['grounding']})
                path, control_path = folder/(language+'.npz'), folder/(language+'_controls.pkl')
                np.savez_compressed(path, **arrays)
                with control_path.open('xb') as f: pickle.dump(controls, f, protocol=5)
                row[language] = {'frame_path': str(path), 'control_path': str(control_path),
                    'frames': len(frames), 'full_goal_success': True, 'replay_success': True,
                    'replay_state_max_abs': error}
            row = validate_pair(row)
            (folder/'record.json').write_text(json.dumps(row, indent=2)+'\n')
            report['records'].append(row)
            print('[pair-verified]', seed, row['source']['frames'], row['target']['frames'], flush=True)
        except Exception as error:
            attempt['error'] = repr(error)
            print('[pair-failed]', seed, repr(error), flush=True)
        finally:
            close(); save()
    report.update(complete=True, accepted_scenes=len(report['records']),
                  collection_target_met=len(report['records']) == args.scenes)
    save()
    print(json.dumps({'complete': True, 'accepted_scenes': len(report['records'])}), flush=True)


if __name__ == '__main__': main()
