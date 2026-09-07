from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.robotwin.eraf_fg_contract import CAMERAS
from experiments.robotwin.fg_mask_replay import VerifiedMaskFrames, capture_verified_masks, select_replay_scenes
from test_robotwin_eraf_fg_contract import correction


def fixture():
    positions = np.zeros((4, 3), np.float32)
    positions[:, 2] = .74
    snapshot = dict(entity_positions=positions, entity_actor_ids=np.array([11, 22, 33, 44]),
                    entity_valid=np.ones(4, bool), target_subject_indices=np.zeros(4, np.int64),
                    target_reference_indices=np.ones(4, np.int64), target_predicate_ids=np.ones(4, np.int64),
                    target_goal_positions=positions.copy(), target_predicate_truth=np.zeros(4, np.float32),
                    target_clause_valid=np.array([True, False, False, False]))
    seg = np.full((24, 20), 11, np.uint32)
    seg[12:] = 22
    observation = {'joint_action': {'vector': np.zeros(14, np.float32)},
                   'observation': {c: {'rgb': np.zeros((24, 20, 3), np.uint8),
                                       'actor_segmentation_ids': seg.copy()} for c in CAMERAS}}
    arrays = {'actions': np.zeros((2, 14), np.float32)}
    arrays.update({c: np.stack([observation['observation'][c]['rgb']] * 2) for c in CAMERAS})
    arrays.update({'grounding/' + k: np.stack([v] * 2) for k, v in snapshot.items()})
    # A reconstruction may allocate different numeric IDs to the same entities.
    arrays['grounding/entity_actor_ids'][:] = [1, 2, 3, 4]
    return arrays, observation, snapshot


def test_exact_replay_binds_current_actor_masks_and_invalidates_unobserved_phase():
    arrays, observation, snapshot = fixture()
    frames = VerifiedMaskFrames(arrays, 'left')
    frames.append(observation, snapshot)
    with pytest.raises(ValueError, match='every recorded'):
        frames.finish()
    frames.append(observation, snapshot)
    labels = frames.finish()
    assert labels['subject_masks'].shape == (2, 4, 24, 20)
    assert labels['subject_mask_valid'][:, 0].all() and labels['reference_mask_valid'][:, 0].all()
    assert not labels['phase_valid'].any()
    with pytest.raises(ValueError, match='more camera frames'):
        frames.append(observation, snapshot)


@pytest.mark.parametrize('kind', ['qpos', 'rgb', 'geometry', 'actors'])
def test_misaligned_masks_are_rejected_before_labels_are_saved(kind):
    arrays, observation, snapshot = fixture()
    if kind == 'qpos': observation['joint_action']['vector'][0] = .0001
    if kind == 'rgb': observation['observation'][CAMERAS[1]]['rgb'][0, 0, 0] = 1
    if kind == 'geometry': snapshot['entity_positions'][0, 0] += .001
    if kind == 'actors': snapshot['entity_actor_ids'][0] = 0
    frames = VerifiedMaskFrames(arrays, 'left')
    with pytest.raises(ValueError): frames.append(observation, snapshot)
    assert not frames.labels


def test_capture_callback_restored_when_replay_fails():
    arrays, observation, snapshot = fixture()
    old = lambda: None
    task = SimpleNamespace(_take_picture=old, get_obs=lambda: observation, pgc_eraf_snapshot=lambda: snapshot)
    with pytest.raises(RuntimeError):
        with capture_verified_masks(task, VerifiedMaskFrames(arrays, 'left')):
            task._take_picture()
            raise RuntimeError('control mismatch')
    assert task._take_picture is old


def test_production_patch_grid_actor_ids_match_existing_mask_geometry():
    from experiments.robotwin.closed_loop_capture import CAMERA_GEOMETRY
    arrays, observation, snapshot = fixture()
    for c, shape in CAMERA_GEOMETRY.items():
        segmentation = np.full(shape, 11, np.uint32)
        segmentation[shape[0] // 2:] = 22
        observation['observation'][c]['actor_segmentation_ids'] = segmentation
    frames = VerifiedMaskFrames(arrays, 'left')
    frames.append(observation, snapshot)
    frames.append(observation, snapshot)
    labels = frames.finish()
    assert labels['subject_mask_valid'][:, 0].all()
    assert labels['reference_mask_valid'][:, 0].all()
    observation['observation'][CAMERAS[0]]['actor_segmentation_ids'] = np.zeros((13, 19), np.uint32)
    with pytest.raises(ValueError, match='Unexpected actor segmentation geometry'):
        VerifiedMaskFrames(arrays, 'left').append(observation, snapshot)


def test_pilot_deduplicates_frame_rows_without_crossing_splits():
    rows = []
    for task in ['blocks_ranking_rgb', 'place_a2b_left']:
        for split in ['train', 'replay_holdout']:
            for scene in [2, 1]:
                for frame in [0, 1]:
                    rows.append(correction() | {'fg_correction': True, 'source_task': task, 'pair_id': task,
                        'replay_split': split, 'scene_seed': scene + (10 if split == 'replay_holdout' else 0),
                        'frame_path': f'{task}/{split}/{scene}.npz', 'frame_index': frame})
    selected = select_replay_scenes(rows)
    assert len(selected) == 4 and all(r['scene_seed'] in {1, 11} for r in selected)
    assert {r['replay_split'] for r in selected} == {'train', 'replay_holdout'}
