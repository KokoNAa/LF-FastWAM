import copy

import numpy as np
import pytest

from experiments.robotwin.eraf_fg_contract import (
    FORMAT, CF_MINIMUM, acceptance, action_windows, assert_scene_disjoint,
    candidate_replans, failure_kind, validate_correction, verify_replayed_state,
)


def correction():
    return {
        "format": FORMAT, "source_task": "blocks_ranking_rgb", "task_config": "demo_clean",
        "scene_seed": 80000001, "source_goal_ever_success": False,
        "counterfactual_goal_ever_success": False, "failure_kind": "neither_goal",
        "capture_origin": "goal_incomplete_failure_replan", "capture_action_index": 48,
        "prefix_action_count": 48, "initial_state_sha256": "a" * 64,
        "capture_state_sha256": "b" * 64, "prefix_action_sha256": "c" * 64,
        "correction_action_sha256": "d" * 64, "rollout_checkpoint_sha256": "e" * 64,
        "verification_policy": "full_goal", "full_goal_verified": True,
        "counterfactual_goal_final_success": True, "both_grippers_open_final": True,
        "verified_replay_count": 2, "captured_state_count": 30, "candidate_count": 5,
        "recorded_action_count": 83, "state_atol": 1e-7, "replay_state_max_abs": 0.,
        "replay_split": "train",
    }


def test_neither_goal_is_a_distinct_legal_failure_not_a_fabricated_source_success():
    row = correction()
    assert validate_correction(row)["failure_kind"] == "neither_goal"
    row["capture_origin"] = "source_directed_failure_replan"
    with pytest.raises(ValueError, match="provenance"):
        validate_correction(row)
    row["source_goal_ever_success"] = True
    row["failure_kind"] = "source_directed"
    assert validate_correction(row)["failure_kind"] == "source_directed"


@pytest.mark.parametrize("change", [
    {"counterfactual_goal_final_success": False},
    {"both_grippers_open_final": False},
    {"capture_action_index": 0, "prefix_action_count": 0},
    {"prefix_action_count": 47}, {"verified_replay_count": 1},
    {"recorded_action_count": 11}, {"replay_state_max_abs": 1.1e-7},
    {"counterfactual_goal_ever_success": True}, {"replay_split": "test"},
])
def test_partial_goal_missing_release_fresh_scene_or_unverified_replay_rejected(change):
    row = correction() | change
    with pytest.raises(ValueError):
        validate_correction(row)


def test_replay_detects_velocity_component_drift_even_if_poses_match():
    a = np.zeros(50)
    b = a.copy()
    b[-1] = .0001
    with pytest.raises(ValueError, match="drift"):
        verify_replayed_state(a, b)
    assert verify_replayed_state(a, a.copy()) == 0
    with pytest.raises(ValueError):
        verify_replayed_state(a, a[:-1])


def test_exclusion_uses_actual_accepted_scene_ids_not_assumed_contiguous_seeds():
    rows = [correction() | {"source_task": "place_a2b_left", "scene_seed": 4300003}]
    with pytest.raises(ValueError, match="overlap"):
        assert_scene_disjoint(rows, [("place_a2b_left", "demo_clean", 4300003)])
    assert_scene_disjoint(rows, [("place_a2b_left", "demo_randomized", 4300003)])


def test_full_windows_cover_terminal_actions_and_local_cannot_see_later_goal():
    actions = np.arange(83 * 14).reshape(83, 14).astype(np.float32)
    full, local = action_windows(actions), action_windows(actions, supervision="local")
    assert len(local) == 1 and local[0].start == 0
    assert local[0].valid.sum() == 12
    np.testing.assert_array_equal(local[0].action[:12], actions[:12])
    assert not local[0].action[12:].any()
    seen = set()
    for item in full:
        count = int(item.valid.sum())
        np.testing.assert_array_equal(item.action[:count], actions[item.start:item.start + count])
        assert not item.action[~item.valid].any()
        seen.update(range(item.start, item.start + count))
    assert seen == set(range(83))


def test_replan_candidates_span_rollout_exclude_initialization_and_try_late_first():
    result = candidate_replans(range(0, 1536, 24))
    assert len(result) == 20 and 0 not in result
    assert result[0] == 1512 and result[-1] == 24


def complete_summary():
    return {"complete": True, "cells": [
        {"source_task": task, "condition": condition, "episodes": 3,
         "task_config": "demo_clean", "selected_goal_successes": 2 if condition == "correct" else minimum}
        for task, minimum in CF_MINIMUM.items() for condition in ("correct", "counterfactual")
    ]}


def test_goal_requires_both_breakthroughs_native_floor_and_each_old_cf_task():
    good = complete_summary()
    assert acceptance(good)["eligible"]
    assert acceptance(good)["counterfactual"] == 7
    for index in range(len(good["cells"])):
        bad = copy.deepcopy(good)
        bad["cells"][index]["selected_goal_successes"] -= 1
        assert not acceptance(bad)["eligible"]
    bad = copy.deepcopy(good)
    bad["cells"].pop()
    with pytest.raises(ValueError, match="complete"):
        acceptance(bad)
