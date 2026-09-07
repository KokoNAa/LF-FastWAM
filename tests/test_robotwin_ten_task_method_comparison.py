import pytest

from scripts.compare_robotwin_ten_task_methods import GROUPS, score_cells


def fixture():
    return {task: {'episodes': n, 'successes': 0} for n, tasks in GROUPS.values() for task in tasks}


def test_three_scene_task_and_six_scene_task_receive_equal_macro_weight():
    old = fixture()
    old['place_a2b_left']['successes'] = 6
    new = fixture()
    new['place_empty_cup']['successes'] = 3
    assert score_cells(old) == pytest.approx(0.1)
    assert score_cells(new) == pytest.approx(0.1)
    # The historical strongest control is21/30 on old tasks and0/15 on new.
    strongest = fixture()
    for task, successes in zip(['blocks_ranking_rgb', 'stack_blocks_two', 'place_a2b_left', 'place_a2b_right', 'place_burger_fries'], [3, 5, 5, 2, 6]):
        strongest[task]['successes'] = successes
    assert score_cells(strongest) == pytest.approx(0.35)


def test_partial_or_wrong_budget_matrix_is_not_a_complete_score():
    missing = fixture()
    del missing['place_empty_cup']
    with pytest.raises(ValueError, match='ten declared tasks'):
        score_cells(missing)
    partial = fixture()
    partial['place_empty_cup']['episodes'] = 2
    with pytest.raises(ValueError, match='Incomplete'):
        score_cells(partial)


def test_equal_scores_cannot_gain_a_strict_lead_from_task_completion_order():
    from itertools import permutations
    values = fixture()
    tasks = ['blocks_ranking_rgb', 'stack_blocks_two', 'place_a2b_left', 'place_a2b_right', 'place_burger_fries']
    for task, successes in zip(tasks, [2, 5, 5, 3, 3]):
        values[task]['successes'] = successes
    extra = {task: cell for task, cell in values.items() if task not in tasks}
    scores = {score_cells({**{task: values[task] for task in order}, **extra})
              for order in permutations(tasks)}
    assert scores == {0.3}
