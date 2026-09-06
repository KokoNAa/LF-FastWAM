import pytest
import torch
from experiments.robotwin.eraf_geometry_metrics import geometry_errors, summarize_geometry, geometry_parameter


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
