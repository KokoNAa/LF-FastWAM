"""Train-only geometry from verified FG corrections, with a matched expert control.

Correction captures contain simulator geometry but no actor segmentation. Missing
visibility, phase and temporal observations must never become negative labels.
"""
from collections import OrderedDict, defaultdict
from pathlib import Path
import hashlib
import random

import numpy as np

SCHEMA = 'robotwin_fg_partial_geometry_v1'


def validate_scene_splits(rows):
    # Legacy expert rows have pair_id, but do not all have source_task.
    splits = defaultdict(set)
    for row in rows:
        splits[row['pair_id'], row['task_config'], row['scene_seed']].add(row['replay_split'])
    if any(len(values) != 1 for values in splits.values()):
        raise ValueError('Semantic replay mixes train and holdout observations of one scene.')


class CorrectionGeometry:
    def __init__(self):
        self.cache = OrderedDict()

    def arrays(self, row):
        from experiments.robotwin.eraf_fg_contract import validate_correction
        validate_correction(row)
        if not row.get('fg_correction'):
            raise ValueError('Geometry correction reader requires an FG row.')
        path = Path(row['frame_path'])
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, row['frame_sha256'])
        if key not in self.cache:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != row['frame_sha256']:
                raise ValueError('FG geometry archive differs from the frozen manifest.')
            with np.load(path, allow_pickle=False) as archive:
                # Never decode camera images just to read privileged geometry.
                arrays = {k: archive[k] for k in archive.files
                          if k == 'actions' or k.startswith('grounding/')}
            self.cache[key] = arrays
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        arrays = self.cache[key]
        frame = row['frame_index']
        if not isinstance(frame, int) or not 0 <= frame < len(arrays['actions']):
            raise ValueError('Invalid FG geometry frame index.')
        return arrays

    def state(self, row):
        # Collection calls this field actions, but records observed joint qpos.
        state = self.arrays(row)['actions'][row['frame_index']].astype(np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('Invalid captured FG proprioception.')
        return state.copy()

    def labels(self, row, language):
        if language != 'target':
            raise ValueError('FG geometry uses the executed counterfactual instruction only.')
        return geometry_labels(self.arrays(row), row['frame_index'])


def geometry_labels(arrays, frame):
    """Only observed geometry/semantics, never infer visibility or phase."""
    import torch
    from scripts.build_pgc_robotwin_entity_relations import _empty_arrays, _normalize_position
    from fastwam.datasets.pgc_libero import PGC_ENTITY_RELATION_PREDICATES
    n = len(arrays['actions'])
    values = {}
    for name, shape in {
        'entity_positions': (n, 4, 3), 'entity_valid': (n, 4),
        'target_subject_indices': (n, 4), 'target_reference_indices': (n, 4),
        'target_predicate_ids': (n, 4), 'target_goal_positions': (n, 4, 3),
        'target_predicate_truth': (n, 4), 'target_clause_valid': (n, 4),
    }.items():
        value = arrays['grounding/' + name]
        if value.shape != shape:
            raise ValueError(f'Invalid FG geometry shape for {name}: {value.shape}')
        values[name] = value[frame]
    out = _empty_arrays(1)
    for c in np.flatnonzero(values['target_clause_valid']):
        s, r = int(values['target_subject_indices'][c]), int(values['target_reference_indices'][c])
        pred = int(values['target_predicate_ids'][c])
        if not (0 <= s < 4 and 0 <= r < 4 and 0 < pred < len(PGC_ENTITY_RELATION_PREDICATES)):
            raise ValueError('Invalid FG geometry entity or predicate.')
        if not (values['entity_valid'][s] and values['entity_valid'][r]):
            raise ValueError('FG geometry clause refers to an invalid entity.')
        subject, reference = values['entity_positions'][[s, r]]
        goal, truth = values['target_goal_positions'][c], values['target_predicate_truth'][c]
        if not all(np.isfinite(x).all() for x in (subject, reference, goal, truth)) or not 0 <= truth <= 1:
            raise ValueError('Nonfinite FG geometry or invalid truth.')
        out['clause_valid'][0, c] = True
        out['predicate_ids'][0, c] = pred
        out['predicate_truth'][0, c] = truth
        out['predicate_truth_valid'][0, c] = True
        for role, value in [('subject', subject), ('reference', reference)]:
            out[role + '_positions'][0, c] = _normalize_position(value)
            out[role + '_position_valid'][0, c] = True
        for name, value in [('grasp', subject), ('goal', goal), ('interaction', goal)]:
            out[name + '_anchors'][0, c] = _normalize_position(value)
            out[name + '_anchor_valid'][0, c] = True
    if not out['clause_valid'].any():
        raise ValueError('Empty FG geometry clauses.')
    return {k: torch.from_numpy(v) for k, v in out.items()}


def partial_geometry_loss(outputs, labels, weights):
    """Common FG/expert loss: positions, anchors, active clauses, relation/truth.

Do not use the full ERAF loss here: its visibility term treats mask_valid=False
as a negative observation, which is wrong when segmentation was not captured.
"""
    import torch
    import torch.nn.functional as F
    valid = labels['clause_valid'].bool()
    def mean(value, keep):
        return value.masked_select(keep).sum() / keep.sum().clamp_min(1)
    active = F.binary_cross_entropy_with_logits(outputs['active_logits'].float(), valid.float())
    relation = mean(F.cross_entropy(outputs['predicate_logits'].float().transpose(1, 2),
                                   labels['predicate_ids'].long(), reduction='none'), valid)
    truth = mean(F.binary_cross_entropy_with_logits(outputs['predicate_truth_logits'].float(),
                 labels['predicate_truth'].float(), reduction='none'), valid & labels['predicate_truth_valid'].bool())
    def geometry(pred, target, mask):
        return mean(F.smooth_l1_loss(outputs[pred].float(), labels[target].float(), reduction='none').mean(-1),
                    valid & labels[mask].bool())
    position = torch.stack([geometry(r + '_position', r + '_positions', r + '_position_valid')
                            for r in ('subject', 'reference')]).mean()
    anchor = torch.stack([geometry(r + '_anchor', r + '_anchors', r + '_anchor_valid')
                          for r in ('grasp', 'goal', 'interaction')]).mean()
    # Match existing weights for supported terms; omit unsupported terms entirely.
    total = (weights.position * position + weights.anchor * anchor
             + weights.entity * active + weights.relation * (relation + truth))
    return total, {'position': position.detach(), 'anchor': anchor.detach(), 'relation': relation.detach(),
                   'active': active.detach(), 'truth': truth.detach()}


def geometry_mixture(rows, seed, *, batch_size, slots, mode, task_balanced=False, world_size=1):
    """Identical ordinary core and task/domain schedule in the two arms."""
    from experiments.robotwin.eraf_fg_training import balanced_group_stream
    from scripts.train_robotwin_eraf_fg_grounding import balanced_rows
    if mode not in {'ordinary', 'fg'} or not 0 < slots < batch_size:
        raise ValueError('Declare a geometry mode and a nonempty expert core.')
    if world_size < 1 or batch_size % world_size or slots % world_size:
        raise ValueError('Geometry slots and batch size must divide into complete rank groups.')
    train = [r for r in rows if r['replay_split'] == 'train']
    fg = [r for r in train if r.get('fg_correction')]
    groups = {(r['pair_id'], r['task_config']) for r in fg}
    ordinary = defaultdict(list)
    for r in train:
        if not any(r.get(k) for k in ('fg_correction', 'native_retention', 'cf_retention')):
            ordinary[r['pair_id'], r['task_config']].append(r)
    if not groups or any(not ordinary[k] for k in groups):
        raise ValueError('Every FG task/domain needs matched ordinary geometry data.')
    replacements = {k: balanced_group_stream(ordinary[k], seed + 40009 + i * 1009)
                    for i, k in enumerate(sorted(groups))}
    core = balanced_rows(rows, seed, task_balanced=task_balanced)
    schedule = balanced_group_stream(fg, seed + 19001, task_balanced=task_balanced)
    rng = random.Random(seed + 29003)
    while True:
        batch = [(r, lang, False) for r, lang in (next(core) for _ in range(batch_size - slots))]
        for _ in range(slots):
            r = next(schedule)
            if mode == 'ordinary':
                r = next(replacements[r['pair_id'], r['task_config']])
            batch.append((r, 'target', True))
        rng.shuffle(batch)
        if world_size > 1:
            # The full ERAF loss contains collectives; the partial loss does not.
            # All ranks must enter the same loss type at every microbatch, not
            # merely the same number of optimizer steps. Preserve randomized
            # samples inside each type, then shuffle complete rank groups.
            batch.sort(key=lambda item: item[2])
            groups = [batch[i:i + world_size] for i in range(0, batch_size, world_size)]
            rng.shuffle(groups)
            batch = [sample for group in groups for sample in group]
        yield batch
