"""Hash-bound corrective role masks and a matched partial semantic objective."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

MASK_FIELDS = ('subject_masks', 'reference_masks', 'subject_mask_valid', 'reference_mask_valid')
SCHEMA = 'robotwin_fg_partial_geometry_role_masks_v1'
BINDING_SCHEMA = 'robotwin_fg_mask_expansion_binding_v1'


def correction_rows_digest(manifest):
    """Bind every corrective row field while allowing additional expert rows."""
    import hashlib
    if manifest.get('complete') is not True:
        raise ValueError('Mask expansion needs a complete training manifest.')
    rows = [row for row in manifest['states'] if row.get('fg_correction')]
    if not rows or any(not isinstance(row.get('id'), str) or not row['id'] for row in rows):
        raise ValueError('Corrective rows need nonempty IDs.')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate corrective row IDs.')
    encoded = json.dumps(sorted(rows, key=lambda row: row['id']), sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def expanded_correction_digest(source, target):
    fields = ('base_checkpoint', 'base_checkpoint_sha256', 'stats_path',
              'stats_sha256', 'image_color_space', 'original_train_config_sha256')
    if any(source.get(key) != target.get(key) for key in fields):
        raise ValueError('Expanded manifest changes the frozen input contract.')
    digest = correction_rows_digest(source)
    if correction_rows_digest(target) != digest:
        raise ValueError('Expanded manifest changes the verified corrective rows.')
    return digest


def build_expanded_mask_binding(source_index, source_manifest, target_manifest):
    """Reuse identical verified corrections, preserving the original replay receipt."""
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    source_index, source_manifest, target_manifest = map(
        lambda p: Path(p).resolve(), (source_index, source_manifest, target_manifest))
    report = json.loads(source_index.read_text())
    if report.get('format') == BINDING_SCHEMA:
        raise ValueError('Bind directly to the original replay index, not another binding.')
    source = json.loads(source_manifest.read_text())
    target = json.loads(target_manifest.read_text())
    digest = expanded_correction_digest(source, target)
    reader = VerifiedCorrectionMasks(source_index, source_manifest)
    reader.validate_coverage(source['states'])
    reader.validate_coverage(target['states'])
    return dict(format=BINDING_SCHEMA, complete=True,
        source_index=str(source_index), source_index_sha256=file_sha256(source_index),
        source_plan_sha256=file_sha256(source_index.with_name('plan.json')),
        source_manifest=str(source_manifest), source_manifest_sha256=file_sha256(source_manifest),
        target_manifest=str(target_manifest), target_manifest_sha256=file_sha256(target_manifest),
        correction_rows_sha256=digest,
        scope='Original replay and label files remain unchanged. Only the manifest binding is extended; every corrective row field is identical. No new labels or temporal supervision are created.')


class VerifiedCorrectionMasks:
    def __init__(self, index, manifest):
        from experiments.robotwin.eraf_fg_bridge import file_sha256
        from experiments.robotwin.fg_mask_replay import SCHEMA as replay_schema
        self.index = Path(index).resolve()
        report = json.loads(self.index.read_text())
        replay_index = self.index
        bound_manifest = Path(manifest).resolve()
        if report.get('format') == BINDING_SCHEMA:
            binding = report
            replay_index = Path(binding['source_index']).resolve()
            source_manifest = Path(binding['source_manifest']).resolve()
            if (binding.get('complete') is not True
                    or Path(binding['target_manifest']).resolve() != bound_manifest
                    or file_sha256(bound_manifest) != binding['target_manifest_sha256']
                    or file_sha256(source_manifest) != binding['source_manifest_sha256']
                    or file_sha256(replay_index) != binding['source_index_sha256']
                    or file_sha256(replay_index.with_name('plan.json')) != binding['source_plan_sha256']):
                raise ValueError('Corrective mask expansion binding identity changed.')
            source = json.loads(source_manifest.read_text())
            target = json.loads(bound_manifest.read_text())
            if expanded_correction_digest(source, target) != binding['correction_rows_sha256']:
                raise ValueError('Expanded manifest changes the verified corrective rows.')
            report = json.loads(replay_index.read_text())
            if report.get('format') == BINDING_SCHEMA:
                raise ValueError('Nested corrective mask bindings are not supported.')
            bound_manifest = source_manifest
        plan = json.loads(replay_index.with_name('plan.json').read_text())
        if (report.get('complete') is not True or report.get('schema') != replay_schema
                or plan.get('manifest_sha256') != file_sha256(bound_manifest)):
            raise ValueError('Complete mask replay must bind the exact training manifest.')
        self.records, self.cache = {}, OrderedDict()
        for row in report['scenes']:
            if (row.get('complete') is not True or not row.get('all_rgb_frames_equal')
                    or row.get('phase_labels_valid') is not False or row.get('temporal_labels_valid') is not False):
                raise ValueError('Unverified role masks or invented temporal labels.')
            key = str(Path(row['frame_path']).resolve())
            if key in self.records: raise ValueError('Duplicate corrective mask archive.')
            self.records[key] = row
        if not self.records: raise ValueError('No verified masks.')
        self.index_sha256 = file_sha256(self.index)

    def record(self, row):
        if not row.get('fg_correction'): raise ValueError('Role mask reader needs a correction row.')
        key = str(Path(row['frame_path']).resolve())
        if key not in self.records: raise ValueError('Missing corrective role mask coverage.')
        record = self.records[key]
        for name in ('pair_id', 'task_config', 'scene_seed', 'replay_split', 'frame_sha256', 'controls_sha256'):
            if record[name] != row[name]: raise ValueError('Corrective role mask identity mismatch: ' + name)
        return record

    def validate_coverage(self, rows):
        from experiments.robotwin.fg_geometry_replay import validate_scene_splits
        validate_scene_splits(rows)
        corrections = [r for r in rows if r.get('fg_correction')]
        if not corrections: raise ValueError('No corrections in the training manifest.')
        for row in corrections: self.record(row)
        if set(self.records) != {str(Path(r['frame_path']).resolve()) for r in corrections}:
            raise ValueError('Mask replay includes scenes outside the frozen correction bank.')

    def attach(self, labels, row):
        import torch
        from experiments.robotwin.eraf_fg_bridge import file_sha256
        record = self.record(row)
        path = Path(record['labels']); stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, record['labels_sha256'])
        if key not in self.cache:
            if file_sha256(path) != record['labels_sha256']: raise ValueError('Corrective role mask file changed.')
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in MASK_FIELDS}
            for k, v in arrays.items():
                shape = (record['frames'], 4) + ((24, 20) if k.endswith('_masks') else ())
                if v.shape != shape or v.dtype != np.bool_: raise ValueError('Invalid corrective mask shape or dtype.')
            self.cache[key] = arrays
            if len(self.cache) > 8: self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        frame = row['frame_index']
        if type(frame) is not int or not 0 <= frame < record['frames']: raise ValueError('Invalid role mask frame index.')
        result = dict(labels)
        for k, v in self.cache[key].items(): result[k] = torch.from_numpy(v[frame:frame + 1].copy())
        for role in ('subject', 'reference'):
            present = result[role + '_masks'].flatten(2).any(-1)
            valid = result[role + '_mask_valid']
            if not torch.equal(present, valid) or (valid & ~result['clause_valid']).any():
                raise ValueError('Mask visibility or clause validity is inconsistent.')
        return result


def partial_geometry_role_loss(outputs, labels, weights):
    """Use only observed geometry and visible role masks; no phase/history loss.

    Ordinary and corrective partial slots share this exact objective. Like the
    existing partial geometry loss, it contains no distributed collectives.
    """
    import torch
    import torch.nn.functional as F
    from experiments.robotwin.fg_geometry_replay import partial_geometry_loss
    from fastwam.models.wan22.entity_relation_affordance import masks_to_patch_targets
    loss, metrics = partial_geometry_loss(outputs, labels, weights)
    masks, attention_losses = [], []
    for role in ('subject', 'reference'):
        logits = outputs[role + '_similarity'].float()
        target, teacher_valid = masks_to_patch_targets(labels[role + '_masks'], token_count=logits.shape[-1])
        valid = labels['clause_valid'].bool() & labels[role + '_mask_valid'].bool() & teacher_valid
        def mean(value): return value.masked_select(valid).sum() / valid.sum().clamp_min(1)
        prediction = logits.sigmoid()
        bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none').mean(-1)
        dice = 1 - (2 * (prediction * target).sum(-1) + 1) / (prediction.sum(-1) + target.sum(-1) + 1)
        masks.append(mean(bce + dice))
        mass = (outputs[role + '_attention'].float() * (target > 0)).sum(-1)
        attention_losses.append(mean(-mass.clamp_min(1e-8).log()))
    mask = torch.stack(masks).mean(); attention = torch.stack(attention_losses).mean()
    return loss + weights.mask * mask + weights.attention_mask * attention, metrics | {
        'role_mask': mask.detach(), 'role_attention_mass': attention.detach()}
