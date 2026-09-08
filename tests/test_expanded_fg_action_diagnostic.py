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


def test_five_task_focus_uses_the_declared_tasks_and_same_sampling_rules():
    tasks = ('blocks_ranking_rgb', 'place_a2b_left')
    rename = dict(zip(TASKS, tasks))
    rows = fixture_rows()
    for row in rows:
        row['id'] = row['id'].replace(row['source_task'], rename[row['source_task']])
        row['source_task'] = rename[row['source_task']]
    selected = select_rows(rows, tasks)
    assert len(selected) == 18 and {r['diagnostic_task'] for r in selected} == set(tasks)
    assert sum(r['replay_split'] == 'replay_holdout' for r in selected) == 12
    assert [r['id'] for r in selected] == [r['id'] for r in select_rows(rows[::-1], tasks)]


@pytest.mark.parametrize('tasks', [('blocks_ranking_rgb',), ('blocks_ranking_rgb', 'blocks_ranking_rgb'), ('unknown', 'place_a2b_left')])
def test_invalid_task_scope_is_rejected(tasks):
    with pytest.raises(ValueError, match='exactly two'): select_rows(fixture_rows(), tasks)
