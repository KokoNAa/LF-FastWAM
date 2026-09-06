"""Paired cup evaluation helpers; simulator metadata never enters the policy."""
from __future__ import annotations
import numpy as np

CAMERAS = ('head_camera', 'left_camera', 'right_camera')
DEPLOYMENT = dict(action_horizon=32, replan_steps=24, inference_steps=10, seed=42,
                  state_dim=14, camera_layout='robotwin_mosaic', eraf=False)


def initial_snapshot(task, observation):
    result = {'qpos': np.array(observation['joint_action']['vector']),
              'cup': np.r_[task.cup.get_pose().p, task.cup.get_pose().q],
              'coaster': np.r_[task.coaster.get_pose().p, task.coaster.get_pose().q]}
    result.update({c: np.array(observation['observation'][c]['rgb']) for c in CAMERAS})
    return result


def require_same_initial(expected, actual):
    if set(expected) != set(actual):
        raise ValueError('Initial observation fields differ.')
    changed = [k for k in expected if not np.array_equal(expected[k], actual[k])]
    if changed:
        raise ValueError(f'Initial observation changed: {changed}')


def summarize_records(catalog, records):
    expected = {(r['scene_seed'], condition) for r in catalog['episodes']
                for condition in ('source', 'target')}
    keys = [(r['scene_seed'], r['condition']) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Missing, duplicate, or unexpected paired episodes.')
    for row in records:
        if not row['initial_observations_equal'] or row['deployment'] != DEPLOYMENT:
            raise ValueError('Input or deployment contract failed.')
        goal = 'source_success' if row['condition'] == 'source' else 'target_success'
        if bool(row['success']) != bool(row[goal]):
            raise ValueError('Reported success differs from selected goal.')
    return {'complete': True, 'scenes': len(catalog['episodes']),
            'episodes': len(records),
            'source_successes': sum(r['success'] for r in records if r['condition']=='source'),
            'cf_successes': sum(r['success'] for r in records if r['condition']=='target'),
            'cf_source_goal_only': sum(r['source_success'] and not r['target_success']
                                      for r in records if r['condition']=='target'),
            'records': records}
