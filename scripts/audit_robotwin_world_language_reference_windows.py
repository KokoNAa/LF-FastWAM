#!/usr/bin/env python3
"""Identify identical expert future windows that cannot test goal discrimination."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from scripts.probe_robotwin_world_language import read, sha, CAMERAS
from scripts.run_robotwin_world_language_collection import write
from experiments.robotwin.image_io import decode_legacy_robotwin_rgb


def audit(root):
    plan_path = root / 'probes/plan.json'; plan = read(plan_path)
    rows = []; groups = defaultdict(lambda: dict(states=0, identical_action_windows=0, identical_video_windows=0))
    for state in plan['states']:
        if not state['dual_reference_valid']:
            continue
        with h5py.File(state['raw_paths']['native']) as source, h5py.File(state['raw_paths']['counterfactual']) as target:
            frame = state['frame']; horizon = plan['action_horizon']
            a = source['joint_action/vector'][frame:frame + horizon]
            b = target['joint_action/vector'][frame:frame + horizon]
            if a.shape != (32, 14) or b.shape != a.shape or not np.array_equal(a[0], b[0]):
                raise ValueError('Invalid common-observation action reference')
            same_frames = []
            for index in state['video_indices']:
                cameras = []
                for camera in CAMERAS:
                    x = source[f'observation/{camera}/rgb'][index]
                    y = target[f'observation/{camera}/rgb'][index]
                    # Identical encoded bytes imply identical model pixels; if
                    # encodings differ, compare decoded RGB before classifying.
                    cameras.append(bytes(x) == bytes(y) or np.array_equal(
                        decode_legacy_robotwin_rgb(x), decode_legacy_robotwin_rgb(y)))
                same_frames.append(all(cameras))
            if not same_frames[0]:
                raise ValueError('Dual reference begins at different observations')
            same_action = bool(np.array_equal(a, b)); same_video = all(same_frames)
            rows.append(dict(state_id=state['id'], task=state['task'], phase=state['phase'],
                             scene_seed=state['scene_seed'], frame=frame,
                             identical_action_window=same_action, identical_video_window=same_video,
                             video_indices=state['video_indices'], paired_frame_rgb_equal=same_frames))
            group = groups[state['task'], state['phase']]
            group['states'] += 1; group['identical_action_windows'] += same_action
            group['identical_video_windows'] += same_video
    expected = sum(s['dual_reference_valid'] for s in plan['states'])
    if len(rows) != expected:
        raise ValueError('Missing paired reference windows')
    result = dict(format='robotwin_world_language_reference_windows_v1', complete=True,
                  plan_sha256=sha(plan_path), paired_states=len(rows), rows=rows,
                  summary=[dict(task=t, phase=p, **counts) for (t, p), counts in groups.items()],
                  interpretation='Identical future windows are shared-prefix controls, not goal-discriminating targets. Paired mean correct/wrong video loss cancels for identical source/target futures; the expert action target-axis is undefined for identical action references. Use the distinct shared-decision windows for placement direction tests. Pixel inequality alone does not prove semantic divergence.',
                  limitation='This checks observed contents of plan-bound HDF5 inputs. Final study audit must also rehash those files against the frozen input manifest.')
    write(root / 'expert_reference_window_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    result = audit(parser.parse_args().root)
    print(json.dumps({k: result[k] for k in ['complete', 'paired_states', 'summary']}))
