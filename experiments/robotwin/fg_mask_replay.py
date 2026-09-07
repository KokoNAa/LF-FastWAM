"""Attach masks only to exactly reproduced correction observations."""
from contextlib import contextmanager

import numpy as np

from experiments.robotwin.eraf_fg_contract import CAMERAS
from experiments.robotwin.closed_loop_capture import CAMERA_GEOMETRY

SCHEMA = 'robotwin_fg_verified_mask_replay_v1'


class VerifiedMaskFrames:
    """Validate frame order, RGB, qpos and semantic geometry before saving labels."""
    def __init__(self, arrays, pair_id):
        self.arrays, self.pair_id = arrays, pair_id
        self.labels = []

    def append(self, observation, snapshot):
        from scripts.build_pgc_robotwin_entity_relations import _generic_role_arrays, _entity_id
        frame = len(self.labels)
        if frame >= len(self.arrays['actions']):
            raise ValueError('Replay produced more camera frames than the correction capture.')
        qpos = np.asarray(observation['joint_action']['vector'], dtype=np.float32)
        expected = self.arrays['actions'][frame]
        if qpos.shape != expected.shape or not np.isfinite(qpos).all() or not np.allclose(qpos, expected, rtol=0, atol=1e-7):
            raise ValueError(f'Correction qpos mismatch at frame {frame}.')
        handle = {}
        for camera in CAMERAS:
            view = observation['observation'][camera]
            if not np.array_equal(view['rgb'], self.arrays[camera][frame]):
                raise ValueError(f'Correction RGB mismatch at frame {frame}, {camera}.')
            segmentation = np.asarray(view['actor_segmentation_ids'])
            if segmentation.ndim != 2 or segmentation.dtype.kind not in 'ui':
                raise ValueError('Expected a raw integer actor-segmentation image.')
            # RoboTwin get_obs stores integer actor IDs at the exact ERAF
            # head/wrist patch geometry; raw full-resolution IDs are also valid.
            if segmentation.shape not in (CAMERA_GEOMETRY[camera], np.asarray(view['rgb']).shape[:2]):
                raise ValueError(f'Unexpected actor segmentation geometry for {camera}: {segmentation.shape}.')
            handle[f'observation/{camera}/actor_segmentation_ids'] = segmentation[None]
        for name, value in snapshot.items():
            value = np.asarray(value)
            expected = self.arrays['grounding/' + name][frame]
            # Simulator actor IDs can change across reconstruction; semantic
            # indices/positions bind roles, and masks use the current IDs.
            if name != 'entity_actor_ids':
                equal = (value.shape == expected.shape and
                         (np.allclose(value, expected, rtol=0, atol=1e-7) if value.dtype.kind == 'f'
                          else np.array_equal(value, expected)))
                if not equal:
                    raise ValueError(f'Correction semantic state mismatch at frame {frame}, {name}.')
            handle['pgc_entity_state/' + name] = value[None]
        actors = np.asarray(snapshot['entity_actor_ids'])[np.asarray(snapshot['entity_valid']).astype(bool)]
        if (actors <= 0).any() or len(np.unique(actors)) != len(actors):
            raise ValueError('Valid entities need distinct non-background actor IDs.')
        labels = _generic_role_arrays(handle=handle, prefix='target',
            entity_ids=[_entity_id(self.pair_id + f'/actor_{i}') for i in range(4)])
        # The correction starts mid-episode and lacks an audited phase history.
        labels['phase_valid'][:] = False
        labels['phase_ids'][:] = 0
        self.labels.append(labels)

    def finish(self):
        if len(self.labels) != len(self.arrays['actions']) or not self.labels:
            raise ValueError('Replay did not reproduce every recorded correction frame.')
        return {key: np.concatenate([row[key] for row in self.labels]) for key in self.labels[0]}


@contextmanager
def capture_verified_masks(task, frames):
    original = task._take_picture
    def capture():
        observation = task.get_obs()
        frames.append(observation, task.pgc_eraf_snapshot())
    task._take_picture = capture
    try:
        yield
    finally:
        task._take_picture = original


def select_replay_scenes(rows, *, limit_per_cell=1):
    """One deterministic pilot per task/domain/split; never include eval scenes."""
    from collections import defaultdict
    from experiments.robotwin.eraf_fg_contract import validate_correction
    from experiments.robotwin.fg_geometry_replay import validate_scene_splits
    validate_scene_splits(rows)
    groups = defaultdict(dict)
    for row in rows:
        if row.get('fg_correction'):
            validate_correction(row)
            key = row['pair_id'], row['task_config'], row['replay_split']
            groups[key].setdefault(row['frame_path'], row)
    if not groups or limit_per_cell < 1:
        raise ValueError('Select a positive number of verified correction scenes per cell.')
    selected = [row for key in sorted(groups) for row in
                sorted(groups[key].values(), key=lambda r: (r['scene_seed'], r['frame_path']))[:limit_per_cell]]
    identities = [(r['pair_id'], r['task_config'], r['scene_seed']) for r in selected]
    if len(set(identities)) != len(identities):
        raise ValueError('Multiple correction archives selected for one physical scene.')
    return selected
