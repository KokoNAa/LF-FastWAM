"""World-coordinate semantic validation, separate from policy success."""
from collections import defaultdict
import math

TARGET_PAIRS = ('place_a2b_left_to_right', 'blocks_ranking_rgb_to_bgr')


def semantic_qualification(rows):
    """The initial campaign requires each target task, not just the average."""
    grouped = defaultdict(lambda: dict(role_hits=0, role_count=0, relation_hits=0, relation_count=0))
    for row in rows:
        for key in grouped[row['pair_id']]:
            grouped[row['pair_id']][key] += row[key]
    cells = {pair: dict(role_accuracy=counts['role_hits'] / max(1, counts['role_count']),
                       relation_accuracy=counts['relation_hits'] / max(1, counts['relation_count']),
                       **counts) for pair, counts in grouped.items()}
    failures = [pair for pair in TARGET_PAIRS if pair not in cells
                or cells[pair]['role_accuracy'] < .8 or cells[pair]['relation_accuracy'] < .9]
    return {'target_tasks_eligible': not failures, 'failed_target_pairs': failures, 'per_pair': cells,
            'rule': 'Each target task must have role accuracy>=.8 and relation accuracy>=.9.'}


def geometry_parameter(name):
    return name.startswith((
        'guard.entity_relation_affordance.entity_grounder.position_head.',
        'guard.entity_relation_affordance.relation_reasoner.',
    ))


def geometry_errors(outputs, labels):
    import torch
    from scripts.build_pgc_robotwin_entity_relations import WORKSPACE_MIN, WORKSPACE_MAX
    scale = torch.as_tensor((WORKSPACE_MAX - WORKSPACE_MIN) / 2,
                            device=outputs['subject_position'].device)
    result = {'positions': [], 'goals': []}
    for prediction, target, valid, group in (
        ('subject_position', 'subject_positions', 'subject_position_valid', 'positions'),
        ('reference_position', 'reference_positions', 'reference_position_valid', 'positions'),
        ('goal_anchor', 'goal_anchors', 'goal_anchor_valid', 'goals'),
    ):
        keep = labels['clause_valid'].bool() & labels[valid].bool()
        errors = ((outputs[prediction].float() - labels[target].float()) * scale).norm(dim=-1) * 100
        result[group].extend(errors[keep].detach().cpu().tolist())
    return result


def summarize_geometry(rows):
    cells = defaultdict(lambda: {'positions': [], 'goals': []})
    for row in rows:
        for kind in ('positions', 'goals'):
            cells[row['pair_id'], row['language']][kind].extend(row['geometry_cm'][kind])
    result = []
    for (pair, language), values in sorted(cells.items()):
        means = {kind + '_mean_cm': sum(v) / len(v) if v else None for kind, v in values.items()}
        if any(v is None or not math.isfinite(v) for v in means.values()):
            raise ValueError('Every semantic holdout cell needs finite position and goal errors.')
        result.append({'pair_id': pair, 'language': language, **means})
    if not result:
        raise ValueError('Empty semantic validation.')
    position = sum(c['positions_mean_cm'] for c in result) / len(result)
    goal = sum(c['goals_mean_cm'] for c in result) / len(result)
    return {'cells': result, 'macro_position_mean_cm': position, 'macro_goal_mean_cm': goal,
            'selection_score_cm': (position + goal) / 2}
