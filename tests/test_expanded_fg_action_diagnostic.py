import copy
import pytest
from scripts.probe_robotwin_expanded_fg_actions import select_rows, TASKS


def fixture_rows():
    rows = []
    for task in TASKS:
        for split, seeds in [('train', [100]), ('replay_holdout', [200, 201])]:
            for kind in ['ordinary', 'fg']:
                for seed in seeds:
                    for frame in ([0] if kind == 'ordinary' else [0, 16, 32]):
                        rows.append(dict(id=f'{task}/{split}/{kind}/{seed}/{frame}', source_task=task,
                                         pair_id='unused', task_config='demo_clean', replay_split=split,
                                         scene_seed=seed, frame_index=frame, fg_correction=kind == 'fg'))
    return rows


def test_selection_is_output_blind_deterministic_and_split_explicit():
    rows = fixture_rows()
    selected = select_rows(rows)
    assert [r['id'] for r in selected] == [r['id'] for r in select_rows(list(reversed(rows)))]
    assert len(selected) == 18
    assert sum(r['replay_split'] == 'train' for r in selected) == 6
    assert {r['frame_index'] for r in selected if r['diagnostic_kind'] == 'fg'} == {0, 16}


def test_rejects_same_scene_in_train_and_holdout():
    rows = fixture_rows()
    extra = copy.deepcopy(rows[0]); extra['replay_split'] = 'replay_holdout'
    with pytest.raises(ValueError, match='crosses'):
        select_rows(rows + [extra])


def test_does_not_silently_substitute_later_state_for_failure_start():
    rows = [r for r in fixture_rows() if not (r['fg_correction'] and r['frame_index'] == 0)]
    with pytest.raises(ValueError, match='Missing failure-start'):
        select_rows(rows)


def test_does_not_substitute_train_scenes_for_missing_holdout():
    rows = [r for r in fixture_rows() if r['scene_seed'] != 201]
    with pytest.raises(ValueError, match='Insufficient scenes'):
        select_rows(rows)
