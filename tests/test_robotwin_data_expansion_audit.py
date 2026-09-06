import json
import pytest
import torch
from scripts.advance_robotwin_eraf_fg_data_expansion import audit_native_cache, verified_snapshot_done
from scripts.assemble_robotwin_eraf_fg_bank import sha


def test_native_smoke_restores_compact_source_and_rejects_wrong_language(tmp_path):
    rows = [dict(id=f'native_{i}', native_retention=True, full_native_episode_success=True,
        retention_condition='correct', source_task='place_a2b_left', task_config='demo_clean',
        scene_seed=80600000, capture_path='/capture.npz', capture_sha256='a',
        teacher_checkpoint_sha256='b', source_instruction='left', counterfactual_instruction='right',
        frame_index=i) for i in range(2)]
    collection = tmp_path / 'collection.json'
    collection.write_text(json.dumps(dict(complete=True, condition='correct', successful_scenes=1, states=rows)))
    cache = tmp_path / 'cache'
    shard = cache / 'shard0'
    shard.mkdir(parents=True)
    parent = shard / 'parent.pt'
    captured = dict(video_inputs={'x': torch.zeros(1)}, action_inputs={'context': torch.zeros(1, 2, 4)},
                    proprio=torch.zeros(1, 14), policy_guard_state=None)
    references = {'source': torch.zeros(1, 32, 14)}
    valid = {'source': torch.ones(1, 32, dtype=torch.bool)}
    torch.save(dict(captured={'source': captured}, references=references, valid=valid), parent)
    child = shard / 'child.pt'
    body = dict(format='robotwin_eraf_fg_compact_v1', parent_payload=str(parent),
        capture_deltas={'source': {'action_inputs': {'context': {'last_token': torch.ones(1, 1, 4)}}}},
        capture_extras={'source': {'proprio': torch.ones(1, 14), 'policy_guard_state': None}},
        references=references, valid=valid)
    torch.save(body, child)
    prepared = [row | dict(payload=str(path), policy_memory='none', fg_correction=False)
                for row, path in zip(rows, (parent, child))]
    (shard / 'states.jsonl').write_text('\n'.join(map(json.dumps, prepared)))
    (shard / 'complete.json').write_text(json.dumps(dict(complete=True, states=2)))
    assert audit_native_cache(collection, cache)['compact_payloads_restored'] == 2
    body['references'] = {'target': references['source']}
    torch.save(body, child)
    with pytest.raises(ValueError, match='CF language'):
        audit_native_cache(collection, cache)


def test_recovery_snapshot_must_match_independent_source_records(tmp_path, monkeypatch):
    from experiments.robotwin import eraf_fg_contract
    # Detailed FG physical-verification rules have separate contract tests.
    monkeypatch.setattr(eraf_fg_contract, 'validate_correction', dict)
    row = dict(source_task='place_a2b_left', task_config='demo_clean', scene_seed=80400001)
    original = tmp_path / 'record.json'
    original.write_text(json.dumps(row))
    folder = tmp_path / 'recovered'
    folder.mkdir()
    path = folder / 'manifest.json'
    snapshot = dict(complete=True, records=[row], record_paths_sha256={str(original): sha(original)})
    path.write_text(json.dumps(snapshot))
    job = dict(verified_snapshot=True, scenes=1, snapshot_sha256=sha(path))
    assert verified_snapshot_done(tmp_path, 'recovered', job)
    snapshot['records'] = [row | {'scene_seed': 80400002}]
    path.write_text(json.dumps(snapshot))
    job['snapshot_sha256'] = sha(path)
    with pytest.raises(ValueError, match='differs from its verified source'):
        verified_snapshot_done(tmp_path, 'recovered', job)
