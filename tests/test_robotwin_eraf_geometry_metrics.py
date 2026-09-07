import pytest
import torch
from experiments.robotwin.eraf_geometry_metrics import geometry_errors, summarize_geometry, geometry_parameter, semantic_qualification


def test_high_average_cannot_admit_a_failed_target_task():
    rows = [dict(pair_id='place_a2b_left_to_right', role_hits=7, role_count=10, relation_hits=10, relation_count=10),
            dict(pair_id='blocks_ranking_rgb_to_bgr', role_hits=10, role_count=10, relation_hits=10, relation_count=10)]
    assert sum(r['role_hits'] for r in rows) / sum(r['role_count'] for r in rows) == .85
    assert not semantic_qualification(rows)['target_tasks_eligible']
    rows[0]['role_hits'] = 8
    assert semantic_qualification(rows)['target_tasks_eligible']
    assert not semantic_qualification(rows[1:])['target_tasks_eligible']


def test_new_task_cf_failure_cannot_be_masked_by_source_or_old_task_scores():
    pairs = ['place_a2b_left_to_right', 'blocks_ranking_rgb_to_bgr', 'cup_front']
    rows = [dict(pair_id=pair, language=language, role_hits=100, role_count=100,
                 relation_hits=100, relation_count=100)
            for pair in pairs for language in ('source', 'target')]
    rows[-1].update(role_hits=7, role_count=10, relation_hits=8, relation_count=10)
    assert semantic_qualification(rows, pairs)['target_tasks_eligible']
    report = semantic_qualification(rows, pairs, each_language=True)
    assert report['failed_task_languages'] == [{'pair_id': 'cup_front', 'language': 'target'}]
    assert not report['target_tasks_eligible']
    assert not semantic_qualification(rows[:-1], pairs, each_language=True)['target_tasks_eligible']
    with pytest.raises(ValueError, match='source/target'):
        semantic_qualification([rows[0] | {'language': None}], pairs, each_language=True)


def test_world_coordinate_scale_and_invalid_clause_exclusion():
    outputs = {name: torch.tensor([[[.1, 0., 0.], [99., 99., 99.]]])
               for name in ('subject_position', 'reference_position', 'goal_anchor')}
    labels = {'clause_valid': torch.tensor([[True, False]])}
    for target, valid in (('subject_positions', 'subject_position_valid'),
                          ('reference_positions', 'reference_position_valid'),
                          ('goal_anchors', 'goal_anchor_valid')):
        labels[target] = torch.zeros(1, 2, 3)
        labels[valid] = torch.ones(1, 2, dtype=torch.bool)
    errors = geometry_errors(outputs, labels)
    assert errors['positions'] == pytest.approx([8., 8.])
    assert errors['goals'] == pytest.approx([8.])


def test_selection_balances_task_language_cells_instead_of_clause_count():
    rows = [{'pair_id': 'a', 'language': 'source', 'geometry_cm': {'positions': [2.] * 10, 'goals': [4.] * 10}},
            {'pair_id': 'b', 'language': 'target', 'geometry_cm': {'positions': [10.], 'goals': [8.]}}]
    report = summarize_geometry(rows)
    assert report['macro_position_mean_cm'] == 6.
    assert report['macro_goal_mean_cm'] == 6.
    assert report['selection_score_cm'] == 6.
    rows[0]['geometry_cm']['goals'] = []
    with pytest.raises(ValueError):
        summarize_geometry(rows)


def test_calibration_excludes_role_attention_and_action_parameters():
    assert geometry_parameter('guard.entity_relation_affordance.entity_grounder.position_head.1.weight')
    assert geometry_parameter('guard.entity_relation_affordance.relation_reasoner.goal_anchor_head.weight')
    for name in ('mot.layer.lora_A', 'guard.goal_graph.weight',
                 'guard.entity_relation_affordance.entity_grounder.visual_projection.weight',
                 'guard.entity_relation_affordance.role_decoder.subject_role',
                 'guard.entity_relation_affordance.query_delta_projection.weight'):
        assert not geometry_parameter(name)


def test_trajectory_audit_covers_later_phases_without_adding_scenes(tmp_path):
    import h5py
    import numpy as np
    from scripts.audit_robotwin_eraf_fg_semantics import trajectory_queries
    path = tmp_path / 'expert.h5'
    with h5py.File(path, 'w') as h:
        positions = np.zeros((12, 1, 3)); positions[4:, 0, 2] = .1
        h['pgc_entity_state/entity_positions'] = positions
        for lang in ('source', 'target'):
            truth = np.zeros((12, 1)); truth[8:, 0] = 1
            h[f'pgc_entity_state/{lang}_predicate_truth'] = truth
            h[f'pgc_entity_state/{lang}_clause_valid'] = np.ones((12, 1), dtype=bool)
            h[f'pgc_entity_state/{lang}_subject_indices'] = np.zeros((12, 1), dtype=int)
    class Raw:
        def locate(self, row, language):
            return path, 0
    rows = [{'id': 'heldout', 'scene_seed': 123, 'replay_split': 'replay_holdout'}]
    queries = trajectory_queries(rows + rows, Raw())
    assert len(queries) == 10  # Duplicated manifest rows do not duplicate audit frames.
    for lang in ('source', 'target'):
        frames = [row[lang + '_frame_index'] for row, language in queries if language == lang]
        assert frames == [0, 2, 6, 10, 11]
    assert {row['scene_seed'] for row, _ in queries} == {123}


def test_reversal_audit_catches_relations_broken_between_phase_midpoints(tmp_path):
    import h5py
    import numpy as np
    from scripts.audit_robotwin_eraf_fg_semantics import trajectory_queries
    path = tmp_path / 'nonmonotone_expert.h5'
    truth = np.zeros((20, 2)); truth[4:] = 1; truth[7:10] = 0
    with h5py.File(path, 'w') as h:
        positions = np.zeros((20, 1, 3)); positions[2:, 0, 2] = .1
        h['pgc_entity_state/entity_positions'] = positions
        for lang in ('source', 'target'):
            h[f'pgc_entity_state/{lang}_predicate_truth'] = truth
            valid = np.ones((20, 2), dtype=bool); valid[:, 1] = False
            h[f'pgc_entity_state/{lang}_clause_valid'] = valid
            h[f'pgc_entity_state/{lang}_subject_indices'] = np.zeros((20, 2), dtype=int)
    class Raw:
        def locate(self, row, language):
            return path, 0
    rows = [{'id': 'heldout', 'scene_seed': 123, 'replay_split': 'replay_holdout'}]
    ordinary = trajectory_queries(rows, Raw())
    expanded = trajectory_queries(rows + rows, Raw(), include_truth_reversals=True)
    for lang in ('source', 'target'):
        before = {r[lang + '_frame_index'] for r, language in ordinary if language == lang}
        after = [r[lang + '_frame_index'] for r, language in expanded if language == lang]
        assert not before.intersection({7, 8, 9})
        assert set(after) == before | {7, 8, 9}
        assert after == sorted(set(after))
    assert {r['scene_seed'] for r, _ in expanded} == {123}
    assert rows == [{'id': 'heldout', 'scene_seed': 123, 'replay_split': 'replay_holdout'}]
    with h5py.File(path, 'r') as h:
        np.testing.assert_array_equal(h['pgc_entity_state/target_predicate_truth'][:], truth)
