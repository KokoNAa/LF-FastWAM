import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.robotwin.eraf_fg_contract import CAMERAS
from experiments.robotwin.eraf_fg_data import file_metadata
from experiments.robotwin.compact_replay import ReplayPayloads


def prepared_case(tmp_path, monkeypatch):
    from scripts import prepare_robotwin_eraf_fg_replay as preparation
    from experiments.robotwin import eraf_fg_bridge as bridge
    parent, checkpoint, capture = tmp_path / 'base.json', tmp_path / 'parent.pt', tmp_path / 'capture.npz'
    source, output = tmp_path / 'collection.json', tmp_path / 'cache'
    parent.write_text(json.dumps({'complete': True, 'states': [], 'formal_data_audit': '/old/audit.json'}))
    checkpoint.write_bytes(b'frozen checkpoint')
    np.savez(capture, state=np.arange(28, dtype=np.float32).reshape(2, 14),
        reference_action_raw=np.arange(2 * 32 * 14, dtype=np.float32).reshape(2, 32, 14),
        **{c: np.zeros((2, 3, 4, 3), dtype=np.uint8) for c in CAMERAS})
    rows = [dict(id=f'cf_ranking_{index}', source_task='blocks_ranking_rgb',
        pair_id='blocks_ranking_rgb_to_bgr', task_config='demo_clean', scene_seed=80810000,
        replay_split='train', cf_retention=True, full_cf_episode_success=True,
        retention_condition='counterfactual', retention_format='robotwin_policy_retention_v2',
        capture_path=str(capture), capture_metadata=file_metadata(capture),
        teacher_checkpoint=str(checkpoint), teacher_checkpoint_metadata=file_metadata(checkpoint),
        source_instruction='RGB', counterfactual_instruction='BGR', frame_index=index)
        for index in range(2)]
    source.write_text(json.dumps({'complete': True, 'states': rows}))
    normalizer = SimpleNamespace(forward=lambda value: value)
    policy = SimpleNamespace(reset=lambda: None, model=SimpleNamespace(requires_grad_=lambda value: None),
        processor=SimpleNamespace(normalizer=SimpleNamespace(normalizers={'action': {'joints': normalizer}}),
                                  shape_meta={'action': [{'key': 'joints'}]}))
    monkeypatch.setattr(bridge, 'load_policy', lambda *args: policy)
    languages = []

    def frozen(policy, observation, instruction):
        languages.append(instruction)
        proprio = torch.from_numpy(observation['joint_action']['vector']).unsqueeze(0)
        return {'video_inputs': {'x': torch.zeros(1, 2, 3)},
                'action_inputs': {'context': proprio.unsqueeze(1)},
                'proprio': proprio, 'policy_guard_state': None}

    monkeypatch.setattr(bridge, 'capture_frozen_inputs', frozen)
    monkeypatch.setattr(bridge, 'file_sha256', lambda *args: pytest.fail('New retention scanned a file hash.'))
    common = ['--manifest', str(parent), '--checkpoint', str(checkpoint), '--output', str(output),
              '--collections', str(source), '--shards', '1']
    monkeypatch.setattr(sys, 'argv', ['cache', 'worker', *common])
    preparation.main()
    assert languages == ['BGR', 'BGR']
    return preparation, common, parent, checkpoint, capture, source, output


def test_new_target_retention_round_trips_through_real_cache_and_merge(tmp_path, monkeypatch):
    preparation, common, _, _, capture, _, output = prepared_case(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['cache', 'merge', *common])
    preparation.main()
    bank = json.loads((output / 'manifest.json').read_text())
    assert bank['complete'] and len(bank['states']) == 2
    assert 'formal_data_audit' not in bank and bank['parent_formal_data_audit'] == '/old/audit.json'
    reader = ReplayPayloads(bank['states'], 'cpu')
    with np.load(capture) as raw:
        for index, row in enumerate(bank['states']):
            payload = reader[row['id']]
            assert set(payload['captured']) == set(payload['references']) == {'target'}
            assert payload['valid']['target'].all() and payload['references']['target'].shape == (1, 32, 14)
            torch.testing.assert_close(payload['references']['target'][0],
                                       torch.from_numpy(raw['reference_action_raw'][index]))
            torch.testing.assert_close(payload['captured']['target']['proprio'][0],
                                       torch.from_numpy(raw['state'][index]))


@pytest.mark.parametrize('change', ['checkpoint', 'collection', 'missing_payload'])
def test_merge_rejects_changed_inputs_or_missing_payloads(tmp_path, monkeypatch, change):
    preparation, common, _, checkpoint, _, source, output = prepared_case(tmp_path, monkeypatch)
    if change == 'checkpoint':
        checkpoint.write_bytes(b'different frozen checkpoint')
    elif change == 'collection':
        source.write_text(source.read_text() + '\n')
    else:
        next((output / 'shard0/payloads').glob('*.pt')).unlink()
    monkeypatch.setattr(sys, 'argv', ['cache', 'merge', *common])
    with pytest.raises(ValueError, match='changed|mismatch'):
        preparation.main()
    assert not (output / 'manifest.json').exists()
