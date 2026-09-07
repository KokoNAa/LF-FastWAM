import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.robotwin.fg_geometry_replay import (
    CorrectionGeometry, geometry_labels, geometry_mixture, partial_geometry_loss,
)
from test_robotwin_eraf_fg_contract import correction


def capture():
    positions = np.zeros((2, 4, 3), np.float32)
    positions[:, :, 2] = .74
    positions[1, 0, 0] = .32
    return {'actions': np.arange(28, dtype=np.float32).reshape(2, 14),
            'grounding/entity_positions': positions,
            'grounding/entity_valid': np.ones((2, 4), bool),
            'grounding/target_subject_indices': np.zeros((2, 4), np.int64),
            'grounding/target_reference_indices': np.ones((2, 4), np.int64),
            'grounding/target_predicate_ids': np.ones((2, 4), np.int64),
            'grounding/target_goal_positions': positions.copy(),
            'grounding/target_predicate_truth': np.zeros((2, 4), np.float32),
            'grounding/target_clause_valid': np.tile([True, False, False, False], (2, 1))}


def test_geometry_preserves_observed_frame_and_does_not_invent_segmentation_or_phase(tmp_path):
    path = tmp_path / 'correction.npz'
    np.savez(path, **capture())
    row = correction() | {'fg_correction': True, 'frame_path': str(path), 'frame_index': 1,
                          'frame_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    reader = CorrectionGeometry()
    labels = reader.labels(row, 'target')
    assert labels['subject_positions'][0, 0, 0] == pytest.approx(.4)
    assert labels['subject_position_valid'][0, 0]
    assert torch.equal(torch.from_numpy(reader.state(row)), torch.arange(14, 28, dtype=torch.float32))
    for key in ['subject_mask_valid', 'reference_mask_valid', 'subject_view_visible', 'phase_valid']:
        assert not labels[key].any()
    with pytest.raises(ValueError, match='counterfactual'):
        reader.labels(row, 'source')
    with pytest.raises(ValueError, match='frozen manifest'):
        reader.labels(row | {'frame_sha256': '0' * 64}, 'target')
    with pytest.raises(ValueError, match='independent prefix'):
        reader.labels(row | {'verified_replay_count': 1}, 'target')


@pytest.mark.parametrize('change', ['entity', 'nan', 'shape', 'predicate'])
def test_bad_geometry_is_rejected(change):
    arrays = capture()
    if change == 'entity': arrays['grounding/target_subject_indices'][0, 0] = 9
    if change == 'nan': arrays['grounding/target_goal_positions'][0, 0, 0] = np.nan
    if change == 'shape': arrays['grounding/entity_valid'] = np.ones((1, 4), bool)
    if change == 'predicate': arrays['grounding/target_predicate_ids'][0, 0] = 999
    with pytest.raises(ValueError): geometry_labels(arrays, 0)


def test_partial_loss_leaves_visibility_and_phase_outputs_without_gradients():
    labels = geometry_labels(capture(), 0)
    shapes = {k: (1, 4, 3) for k in ['subject_position', 'reference_position', 'grasp_anchor', 'goal_anchor', 'interaction_anchor']}
    shapes.update(active_logits=(1, 4), predicate_logits=(1, 4, 10), predicate_truth_logits=(1, 4),
                  subject_visibility_logits=(1, 4), phase_logits=(1, 4, 3))
    outputs = {k: torch.ones(shape, requires_grad=True) for k, shape in shapes.items()}
    weights = SimpleNamespace(entity=1., relation=1., position=.5, anchor=1.)
    loss, _ = partial_geometry_loss(outputs, labels, weights)
    loss.backward()
    assert outputs['subject_position'].grad[0, 0].abs().sum() > 0
    assert outputs['subject_position'].grad[0, 1:].count_nonzero() == 0
    assert outputs['subject_visibility_logits'].grad is None
    assert outputs['phase_logits'].grad is None
    changed = dict(outputs, subject_visibility_logits=torch.full((1, 4), -1000.), phase_logits=torch.full((1, 4, 3), 1000.))
    assert torch.equal(loss.detach(), partial_geometry_loss(changed, labels, weights)[0].detach())


def test_matched_geometry_arms_share_expert_core_and_task_domain_without_holdout():
    rows = []
    for task in ['left', 'rgb']:
        for kind in ['ordinary', 'fg']:
            for split in ['train', 'replay_holdout']:
                for frame in range(3):
                    rows.append(dict(id=f'{task}_{kind}_{split}_{frame}', pair_id=task, task_config='demo_clean',
                        scene_seed=1 if split == 'train' else 2, replay_split=split, frame_index=frame,
                        fg_correction=kind == 'fg'))
    a, b = [geometry_mixture(rows, 42, batch_size=12, slots=3, mode=m, task_balanced=True) for m in ['ordinary', 'fg']]
    for _ in range(5):
        x, y = next(a), next(b)
        assert sum(p for _, _, p in x) == 3
        for (ra, la, pa), (rb, lb, pb) in zip(x, y):
            assert pa == pb and la == lb
            assert ra['replay_split'] == rb['replay_split'] == 'train'
            if pa:
                assert la == 'target' and not ra['fg_correction'] and rb['fg_correction']
                assert ra['pair_id'] == rb['pair_id'] and ra['task_config'] == rb['task_config']
            else:
                assert ra == rb and not ra['fg_correction']
