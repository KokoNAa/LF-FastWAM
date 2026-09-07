import json
from pathlib import Path

import pytest

from scripts.compare_robotwin_eraf_fg_evaluations import compare, evaluation


def fixture(root, *, successes=(False, True), hashed=False):
    cells = []
    for task, success in zip(['left', 'ranking'], successes):
        folder = root / task / 'demo_clean/counterfactual'
        folder.mkdir(parents=True)
        episode = dict(scene_seed=90000001, episode_index=0, source_instruction='source',
            counterfactual_instruction='target', policy_instruction='target', instruction_goal='counterfactual',
            selected_goal='counterfactual', condition='counterfactual', source_task=task,
            selected_goal_success=success, steps=100)
        metadata = dict(complete=True, checkpoint=str(root / 'model.pt'), checkpoint_sha256='a' * 64 if hashed else None,
            canonical_sha256='b' * 64 if hashed else None, skip_file_hashes=not hashed,
            checkpoint_metadata=dict(path=str(root / 'model.pt'), size=300, mtime_ns=123),
            canonical_metadata=dict(path=f'/catalog/{task}.jsonl', size=100, mtime_ns=456),
            deployment={'action_horizon': 32, 'replan_steps': 24}, eraf='off', policy_kind='repair')
        (folder / 'complete.json').write_text(json.dumps(metadata))
        (folder / 'episodes.jsonl').write_text(json.dumps(episode) + '\n')
        (folder / 'initial_states.json').write_text(json.dumps([{'scene_seed': 90000001, 'sha256': 'c' * 64}]))
        cells.append(dict(source_task=task, condition='counterfactual', episodes=1, selected_goal_successes=int(success)))
    (root / 'summary.json').write_text(json.dumps(dict(complete=True, checkpoint=str(root / 'model.pt'), episodes=2, cells=cells)))


@pytest.mark.parametrize('hashed', [False, True])
def test_valid_paired_reports_need_no_local_model_files(tmp_path, hashed):
    left, right = tmp_path / 'a', tmp_path / 'b'
    fixture(left, hashed=hashed)
    fixture(right, successes=(True, False), hashed=hashed)
    report, pairs = compare(left, right)
    assert report['complete'] and report['matched_episodes'] == 2
    assert report['cells'][0]['gained_scene_seeds'] == [90000001]
    assert report['cells'][1]['lost_scene_seeds'] == [90000001]
    assert not (left / 'model.pt').exists()


def mutate(root, fn):
    path = root / 'ranking/demo_clean/counterfactual/complete.json'
    payload = json.loads(path.read_text())
    fn(payload)
    path.write_text(json.dumps(payload))


def test_null_hashes_do_not_hide_mixed_checkpoint_metadata(tmp_path):
    fixture(tmp_path)
    mutate(tmp_path, lambda r: r['checkpoint_metadata'].update(mtime_ns=999))
    with pytest.raises(ValueError, match='multiple checkpoint identities'):
        evaluation(tmp_path)


def test_null_hashes_do_not_hide_changed_catalog(tmp_path):
    left, right = tmp_path / 'a', tmp_path / 'b'
    fixture(left)
    fixture(right)
    mutate(right, lambda r: r['canonical_metadata'].update(mtime_ns=999))
    with pytest.raises(ValueError, match='catalog'):
        compare(left, right)


@pytest.mark.parametrize('metadata', [None, {}, {'path': '/model.pt', 'size': 0, 'mtime_ns': 1}])
def test_missing_checkpoint_identity_is_rejected(tmp_path, metadata):
    fixture(tmp_path)
    mutate(tmp_path, lambda r: r.update(checkpoint_metadata=metadata))
    with pytest.raises(ValueError, match='binding'):
        evaluation(tmp_path)
