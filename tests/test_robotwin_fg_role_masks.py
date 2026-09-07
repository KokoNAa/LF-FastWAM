from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.robotwin.fg_role_masks import VerifiedCorrectionMasks, partial_geometry_role_loss
from experiments.robotwin.fg_geometry_replay import geometry_labels
from experiments.robotwin.fg_mask_replay import SCHEMA
from test_robotwin_eraf_fg_contract import correction
from test_robotwin_fg_geometry_replay import capture


def test_role_validation_logging_uses_saved_metrics_after_role_iteration():
    from scripts.train_robotwin_eraf_fg_grounding import evaluation_log_line
    report={'step':0,'role_accuracy':.8125,'relation_accuracy':1.,
            'geometry':{'selection_score_cm':15.52},
            'fg_geometry_holdout':{'role_accuracy':.4,'rows':[]}}
    assert evaluation_log_line(report)==('[grounding-eval] step=0 roles=0.8125 '
        'relations=1.0000 geometry_cm=15.520')


def index_fixture(tmp_path):
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{}')
    path = tmp_path / 'masks.npz'
    masks = np.zeros((2, 4, 24, 20), bool)
    masks[0, 0, :12] = True; masks[1, 0, 12:] = True
    valid = masks.reshape(2, 4, -1).any(-1)
    np.savez(path, subject_masks=masks, reference_masks=masks,
             subject_mask_valid=valid, reference_mask_valid=valid)
    row = correction() | dict(fg_correction=True, frame_path=str(tmp_path / 'correction.npz'), frame_index=1,
                             replay_split='train', pair_id='blocks_ranking_rgb_to_bgr', frame_sha256='a'*64, controls_sha256='b'*64)
    report = {k: row[k] for k in ('pair_id', 'task_config', 'scene_seed', 'replay_split', 'frame_path', 'frame_sha256', 'controls_sha256')}
    report.update(complete=True, all_rgb_frames_equal=True, phase_labels_valid=False, temporal_labels_valid=False,
                  labels=str(path), labels_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), frames=2)
    index = tmp_path / 'complete.json'; index.write_text(json.dumps({'complete': True, 'schema': SCHEMA, 'scenes': [report]}))
    (tmp_path / 'plan.json').write_text(json.dumps({'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}))
    return index, manifest, row


def test_verified_masks_bind_frame_and_preserve_partial_geometry(tmp_path):
    index, manifest, row = index_fixture(tmp_path)
    reader = VerifiedCorrectionMasks(index, manifest); reader.validate_coverage([row])
    labels = geometry_labels(capture(), 1); before = deepcopy(labels)
    result = reader.attach(labels, row)
    assert result['subject_masks'][0, 0, 12:].all() and not result['subject_masks'][0, 0, :12].any()
    assert not result['phase_valid'].any()
    for k in before:
        assert torch.equal(before[k], labels[k])
        if k not in ('subject_masks', 'reference_masks', 'subject_mask_valid', 'reference_mask_valid'):
            assert torch.equal(result[k], before[k])


@pytest.mark.parametrize('field,value', [('scene_seed', 9999), ('replay_split', 'replay_holdout'), ('frame_sha256', '0'*64), ('controls_sha256', '1'*64)])
def test_masks_reject_wrong_scene_split_or_source_identity(tmp_path, field, value):
    index, manifest, row = index_fixture(tmp_path)
    with pytest.raises(ValueError, match='identity mismatch'):
        VerifiedCorrectionMasks(index, manifest).record(row | {field: value})


def test_masks_reject_corrupt_file_missing_coverage_and_manifest_change(tmp_path):
    index, manifest, row = index_fixture(tmp_path)
    reader = VerifiedCorrectionMasks(index, manifest)
    with pytest.raises(ValueError, match='coverage'):
        reader.validate_coverage([row | {'frame_path': str(tmp_path/'other.npz')}])
    (tmp_path/'masks.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='file changed'):
        reader.attach(geometry_labels(capture(), 1), row)
    manifest.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='exact training manifest'):
        VerifiedCorrectionMasks(index, manifest)


def test_role_loss_trains_visible_attention_without_phase_or_invisible_gradients():
    labels = geometry_labels(capture(), 0)
    for role in ('subject', 'reference'):
        labels[role + '_masks'][0, 0, :12] = True
        labels[role + '_mask_valid'][0, 0] = True
    shapes = {k:(1,4,3) for k in ('subject_position','reference_position','grasp_anchor','goal_anchor','interaction_anchor')}
    shapes.update(active_logits=(1,4), predicate_logits=(1,4,10), predicate_truth_logits=(1,4), phase_logits=(1,4,3))
    outputs = {k:torch.zeros(shape, requires_grad=True) for k,shape in shapes.items()}
    attention_logits = {}
    for role in ('subject', 'reference'):
        outputs[role + '_similarity'] = torch.zeros(1,4,480,requires_grad=True)
        attention_logits[role] = torch.zeros(1,4,480,requires_grad=True)
        outputs[role + '_attention'] = attention_logits[role].softmax(-1)
    weights = SimpleNamespace(entity=1., relation=1., position=.5, anchor=1., mask=1., attention_mask=1.)
    loss, metrics = partial_geometry_role_loss(outputs, labels, weights)
    loss.backward()
    assert torch.isfinite(loss) and metrics['role_attention_mass'] > 0
    for role in ('subject', 'reference'):
        assert attention_logits[role].grad[0,0].abs().sum() > 0
        assert attention_logits[role].grad[0,1:].count_nonzero() == 0
        assert outputs[role + '_similarity'].grad[0,1:].count_nonzero() == 0
    assert outputs['phase_logits'].grad is None
