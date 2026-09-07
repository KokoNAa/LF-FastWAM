"""Preserve existing replay and verify every new full-goal correction window."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from experiments.robotwin.eraf_fg_contract import validate_correction, scene_key


def audit_extension(parent, target, collection, *, check_files=True):
    from scripts.collect_robotwin_eraf_fg import historical_scene_keys
    from scripts.assemble_robotwin_eraf_fg_bank import audit_rows
    if not all(m.get('complete') is True for m in (parent,target,collection)):
        raise ValueError('Require complete source collection and cache manifests.')
    for key in ('base_checkpoint','base_checkpoint_sha256','stats_path','stats_sha256',
                'image_color_space','original_train_config','original_train_config_sha256'):
        if target.get(key) != parent.get(key):raise ValueError('Frozen input contract changed: '+key)
    old = parent['states'];rows = target['states']
    if rows[:len(old)] != old or len(rows) <= len(old):
        raise ValueError('Expanded bank must preserve every old row in order and append new rows.')
    records = [validate_correction(r) for r in collection['records']]
    expected = {scene_key(r):r for r in records}
    if len(records) != len(expected) or len(records) != 60:
        raise ValueError('Require60distinct verified new FG scenes.')
    counts = Counter((r['source_task'],r['replay_split']) for r in records)
    if counts != {(t,s):n for t in ('place_empty_cup','move_pillbottle_pad')
                  for s,n in [('train',24),('replay_holdout',6)]}:
        raise ValueError('Require24train+6holdout for cup and pill.')
    if set(expected) & historical_scene_keys(old):raise ValueError('New corrections overlap previous scenes.')
    added = rows[len(old):];groups=defaultdict(list)
    for row in added:
        if not row.get('fg_correction') or row.get('native_retention') or row.get('cf_retention'):
            raise ValueError('The appended extension must consist of FG corrections only.')
        key=scene_key(row)
        if key not in expected or any(row.get(k)!=v for k,v in expected[key].items()):
            raise ValueError('Cached corrective row differs from its exact source record.')
        if row.get('policy_memory') != 'none' or row.get('frozen_input_protocol') != 'robotwin_eraf_fg_pre_dit_v1':
            raise ValueError('New frozen cache has unsupported policy memory or protocol.')
        if check_files and not Path(row['payload']).is_file():raise ValueError('Missing corrective cache payload.')
        groups[key].append(row)
    if set(groups) != set(expected):raise ValueError('Missing prepared corrective scenes.')
    report = audit_rows(rows,formal=False)
    # audit_rows verifies every stride8 window and exact real-action tail masks.
    return dict(complete=True,parent_rows_preserved_exactly=True,parent_rows=len(old),
        added_rows=len(added),added_scenes=len(groups),added_scene_counts=[
            dict(task=k[0],split=k[1],scenes=v) for k,v in sorted(counts.items())],
        all_source_record_fields_preserved=True,all_goal_tail_windows_present=True,
        all_training_and_holdout_scenes_disjoint=True,bank_audit=report)


def audit_cached_targets(parent, target, collection):
    """Check all new normalized targets against recorded physical corrections."""
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer, load_dataset_stats_from_json
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    from experiments.robotwin.pgc_data import array_sha256
    from experiments.robotwin.eraf_fg_contract import action_windows
    from experiments.robotwin.compact_replay import ReplayPayloads
    cfg=OmegaConf.load(parent['original_train_config'])
    settings=cfg.data.train.processor
    if settings.use_stepwise_action_norm or settings.norm_default_mode!='z-score' or settings.norm_exception_mode is not None:
        raise ValueError('Audit requires the released action z-score normalization contract.')
    if file_sha256(parent['stats_path']) != parent['stats_sha256']:
        raise ValueError('Action normalization statistics changed.')
    stats=load_dataset_stats_from_json(parent['stats_path'])['action']['default']
    norm=SingleFieldLinearNormalizer({k.removeprefix('global_'):v for k,v in stats.items() if k.startswith('global_')},'z-score')
    rows=target['states'][len(parent['states']):]
    cache=ReplayPayloads(rows,'cpu',capacity=2)
    by_scene=defaultdict(dict)
    for row in rows:by_scene[scene_key(row)][row['frame_index']]=row
    inspected=0
    for r in collection['records']:
        path=Path(r['frame_path'])
        if file_sha256(path)!=r['frame_sha256'] or file_sha256(path.with_name('controls.pkl'))!=r['controls_sha256']:
            raise ValueError('Raw correction or physical controls changed.')
        with np.load(path,allow_pickle=False) as z:actions=z['actions']
        if array_sha256(actions)!=r['correction_action_sha256']:raise ValueError('Raw correction action identity changed.')
        for window in action_windows(actions):
            row=by_scene[scene_key(r)][window.start];payload=cache[row['id']]
            if (set(payload['references'])!={'target'} or set(payload['captured'])!={'target'}
                    or set(payload['valid'])!={'target'}):
                raise ValueError('FG cache must contain the executed target instruction only.')
            expected=norm.forward(torch.from_numpy(window.action).unsqueeze(0))
            if not torch.equal(payload['references']['target'],expected):
                raise ValueError('Normalized corrective target mismatch: '+row['id'])
            if not torch.equal(payload['valid']['target'],torch.from_numpy(window.valid).unsqueeze(0)):
                raise ValueError('Padded action received supervision: '+row['id'])
            if payload['captured']['target']['policy_guard_state'] is not None:
                raise ValueError('FG cache carries unverified policy memory.')
            inspected+=1
    return dict(complete=True,verified_payloads=inspected,verified_scenes=len(collection['records']),
        all_normalized_targets_exact=True,all_real_action_masks_exact=True,all_policy_memory_empty=True)


def merge_mask_shards(root, manifest, shards=6):
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    from experiments.robotwin.fg_mask_replay import SCHEMA
    from experiments.robotwin.fg_role_masks import VerifiedCorrectionMasks
    digest=file_sha256(manifest);scenes=[];bindings={}
    for i in range(shards):
        folder=Path(root)/f'shard{i}'
        plan=json.loads((folder/'plan.json').read_text());report=json.loads((folder/'complete.json').read_text())
        if (not report.get('complete') or plan['manifest_sha256']!=digest
                or plan['shard_index']!=i or plan['num_shards']!=shards or report['schema']!=SCHEMA):
            raise ValueError('Missing or mismatched full-bank mask replay shard.')
        for row in report['scenes']:
            if not 0 <= row['replay_state_max_abs'] <= 1e-7:raise ValueError('Physical mask replay drift.')
            if file_sha256(row['labels'])!=row['labels_sha256']:raise ValueError('Mask labels changed.')
        scenes.extend(report['scenes'])
        bindings[str(folder/'plan.json')]=file_sha256(folder/'plan.json')
        bindings[str(folder/'complete.json')]=file_sha256(folder/'complete.json')
    root=Path(root)
    if (root/'plan.json').exists() or (root/'complete.json').exists():
        raise FileExistsError('Mask aggregate already exists.')
    staging=root/'.aggregate';staging.mkdir()
    for name,value in [('plan.json',dict(manifest=str(manifest),manifest_sha256=digest,source_shard_hashes=bindings)),
                       ('complete.json',dict(complete=True,schema=SCHEMA,scenes=scenes,optimizer_updates=0))]:
        with (staging/name).open('x') as f:json.dump(value,f,indent=2)
    reader=VerifiedCorrectionMasks(staging/'complete.json',manifest)
    reader.validate_coverage(json.loads(Path(manifest).read_text())['states'])
    for name in ('plan.json','complete.json'):(staging/name).replace(root/name)
    staging.rmdir()
    return dict(complete=True,scenes=len(scenes),frames=sum(r['frames'] for r in scenes),
                index=str(root/'complete.json'),index_sha256=reader.index_sha256,
                all_old_and_new_corrections_replayed=True)
