from copy import deepcopy
import hashlib
import json

import pytest
import torch

from experiments.robotwin.fg_role_masks import build_expanded_mask_binding, VerifiedCorrectionMasks
from test_robotwin_fg_role_masks import index_fixture
from test_robotwin_fg_geometry_replay import capture
from experiments.robotwin.fg_geometry_replay import geometry_labels


def expanded_fixture(tmp_path):
    index, source, row = index_fixture(tmp_path)
    row['id'] = 'correction_frame1'
    source.write_text(json.dumps(dict(complete=True, states=[row])))
    (tmp_path / 'plan.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
    target = tmp_path / 'expanded.json'
    # A genuinely new task/scene may be added, while correction labels stay bound.
    expert = dict(id='new_expert', pair_id='new_task', task_config='demo_clean',
                  scene_seed=84000000, replay_split='train', artifact_role='joint_expert_supervision')
    target.write_text(json.dumps(dict(complete=True, states=[row, expert])))
    return index, source, target, row


def test_expansion_reuses_exact_original_masks_without_modifying_replay(tmp_path):
    index, source, target, row = expanded_fixture(tmp_path)
    before = index.read_bytes(), (tmp_path / 'plan.json').read_bytes(), (tmp_path / 'masks.npz').read_bytes()
    binding = build_expanded_mask_binding(index, source, target)
    path = tmp_path / 'binding.json'
    path.write_text(json.dumps(binding))
    original = VerifiedCorrectionMasks(index, source)
    expanded = VerifiedCorrectionMasks(path, target)
    expanded.validate_coverage(json.loads(target.read_text())['states'])
    labels = geometry_labels(capture(), 1)
    a, b = original.attach(labels, row), expanded.attach(labels, row)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert before == (index.read_bytes(), (tmp_path / 'plan.json').read_bytes(), (tmp_path / 'masks.npz').read_bytes())


@pytest.mark.parametrize('change', ['split', 'instruction', 'frame', 'drop', 'duplicate'])
def test_expansion_rejects_changed_or_missing_corrections(tmp_path, change):
    index, source, target, row = expanded_fixture(tmp_path)
    data = json.loads(target.read_text())
    if change == 'split':
        data['states'][0]['replay_split'] = 'replay_holdout'
    elif change == 'instruction':
        data['states'][0]['counterfactual_instruction'] = 'Changed goal'
    elif change == 'frame':
        data['states'][0]['frame_index'] = 0
    elif change == 'drop':
        data['states'] = data['states'][1:]
    else:
        data['states'].append(deepcopy(row))
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        build_expanded_mask_binding(index, source, target)


def test_binding_rechecks_target_and_original_replay_hashes(tmp_path):
    index, source, target, row = expanded_fixture(tmp_path)
    path = tmp_path / 'binding.json'
    path.write_text(json.dumps(build_expanded_mask_binding(index, source, target)))
    target_bytes = target.read_bytes()
    target.write_bytes(target_bytes + b' ')
    with pytest.raises(ValueError, match='identity changed'):
        VerifiedCorrectionMasks(path, target)
    target.write_bytes(target_bytes)
    index.write_bytes(index.read_bytes() + b' ')
    with pytest.raises(ValueError, match='identity changed'):
        VerifiedCorrectionMasks(path, target)


def test_expansion_rejects_a_changed_frozen_input_contract(tmp_path):
    index, source, target, row = expanded_fixture(tmp_path)
    data = json.loads(target.read_text())
    data['stats_sha256'] = 'a' * 64
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='frozen input contract'):
        build_expanded_mask_binding(index, source, target)
