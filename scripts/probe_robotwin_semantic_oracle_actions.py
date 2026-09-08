#!/usr/bin/env python3
"""Same-state diagnostic of ground-truth ERAF routing; never a learned-policy score."""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]
TASKS = ('place_a2b_left', 'blocks_ranking_rgb', 'place_empty_cup', 'move_pillbottle_pad')


def selected_rows(rows):
    from experiments.robotwin.pgc_data import ROBOTWIN_TEN_TASK_SPECS
    names = {s.pair_id:s.source_task for s in ROBOTWIN_TEN_TASK_SPECS}
    buckets = defaultdict(lambda: defaultdict(dict)); splits = defaultdict(set)
    for row in rows:
        task = row.get('source_task') or names.get(row['pair_id'])
        if task not in TASKS or row['task_config'] != 'demo_clean': continue
        if any(row.get(k) for k in ('fg_correction', 'cf_retention', 'native_retention')): continue
        split, seed = row['replay_split'], row['scene_seed']
        splits[task,seed].add(split)
        frame = int(row.get('target_frame_index', row.get('frame_index',0)))
        current = buckets[task,split][seed].get(frame)
        if current is None or row['id'] < current['id']:
            buckets[task,split][seed][frame] = row
    if any(len(s)>1 for s in splits.values()): raise ValueError('Scene split leakage.')
    result = []
    for task in TASKS:
        for split, count in (('train',1), ('replay_holdout',2)):
            scenes = buckets[task,split]
            if len(scenes)<count: raise ValueError('Insufficient ordinary expert scenes.')
            for seed in sorted(scenes)[:count]:
                frames = sorted(scenes[seed])
                if frames[0]!=0: raise ValueError('Require initial expert state.')
                choices = [('initial',0)]
                # The existing old-task holdout cache contains initial states only.
                if not (task in TASKS[:2] and split == 'replay_holdout'):
                    if len(frames)<2: raise ValueError('Require later expert state.')
                    choices.append(('middle',frames[len(frames)//2]))
                for kind, frame in choices:
                    result.append(dict(scenes[seed][frame], diagnostic_task=task, diagnostic_kind=kind))
    if len(result)!=20 or len({r['id'] for r in result})!=20: raise ValueError('Expected20 unique states.')
    return result


def validate_oracle(labels):
    """Require real active-clause phases; never fabricate missing FG temporal labels."""
    required = ('clause_valid','predicate_ids','subject_masks','reference_masks','subject_mask_valid',
        'reference_mask_valid','subject_positions','reference_positions','subject_position_valid',
        'reference_position_valid','goal_anchors','goal_anchor_valid','predicate_truth','phase_ids','phase_valid')
    if any(k not in labels for k in required): raise ValueError('Incomplete semantic labels.')
    valid = labels['clause_valid'].bool()
    if not bool(valid.any()): raise ValueError('No active semantic clauses.')
    for key in ('phase_valid','subject_position_valid','reference_position_valid','goal_anchor_valid'):
        if bool((valid & ~labels[key].bool()).any()): raise ValueError('Missing active label: '+key)
    if 'predicate_truth_valid' not in labels or bool((valid & ~labels['predicate_truth_valid'].bool()).any()):
        raise ValueError('Missing active predicate truth.')
    return {k:labels[k] for k in required}


@contextmanager
def oracle_route(model, oracle):
    """Instrument the actual production route and restore both methods on failure."""
    import torch
    module = model.policy_guard_modules['entity_relation_affordance']
    counts = dict(encode=0, route=0)
    targets = [(model, '_encode_policy_guard_goal'), (module, 'route_oracle')]
    saved = [(obj, key, key in vars(obj), vars(obj).get(key)) for obj,key in targets]
    original_encode, original_route = [getattr(obj,key) for obj,key in targets]
    def encode(*args, **kwargs):
        if torch.is_grad_enabled(): raise ValueError('Privileged labels are inference-only.')
        counts['encode'] += 1
        return original_encode(*args, **(kwargs | {'policy_guard_eraf_oracle':oracle}))
    def route(*args, **kwargs):
        if kwargs.get('oracle') is not oracle: raise ValueError('Oracle route identity changed.')
        counts['route'] += 1
        return original_route(*args, **kwargs)
    model._encode_policy_guard_goal = encode
    module.route_oracle = route
    try:
        yield counts
        if counts['encode'] == 0 or counts['route'] != counts['encode']:
            raise ValueError('Oracle did not reach every production goal encoding.')
    finally:
        for obj,key,existed,previous in saved:
            if existed: setattr(obj,key,previous)
            else: delattr(obj,key)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','source-bank','label-cache','checkpoint','output'):
        ap.add_argument('--'+key,required=True)
    args=ap.parse_args()
    import h5py
    import numpy as np
    import torch
    from experiments.robotwin.eraf_fg_bridge import load_policy, file_sha256, capture_frozen_inputs, masked_mse
    from experiments.robotwin.eraf_fg_data import RawReplay
    from experiments.robotwin.eraf_fg_contract import CAMERAS
    from experiments.robotwin.compact_replay import ReplayPayloads
    from experiments.robotwin.image_io import decode_legacy_robotwin_rgb
    from experiments.robotwin.pgc_data import array_sha256
    from experiments.robotwin.same_state_repair import move_cache

    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(Path(args.manifest).read_text()); assert manifest['complete']
    rows=selected_rows(manifest['states'])
    hashes=dict(manifest=file_sha256(args.manifest),checkpoint=file_sha256(args.checkpoint))
    protocol=dict(format='robotwin_semantic_oracle_action_probe_v1',checkpoint=args.checkpoint,
        input_hashes=hashes,selected_rows=rows,optimizer_updates=0,inference_steps=10,noise_seed=42,
        modes=['learned','semantic_oracle','eraf_off','repeat'],
        scope='Privileged same-state expert semantic routing diagnostic, not CF success or deployable policy. Twenty ordinary expert states per model: one train and two holdout scenes per task; initial plus middle except old-task holdout has initial only. Excludes FG because its phase labels are explicitly invalid. No ground-truth actions are passed to inference.',
        limits='Oracle changes several semantic factors together. Phases are existing expert-trajectory annotations, not independently verified simulator temporal truth. Invisible mask roles retain learned attention in the production oracle API. No claim of pure geometry causality, perfect visual features, independent generalization, or closed-loop success.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    policy=load_policy(args.checkpoint,manifest,seed=42);model=policy.model
    model.eval().requires_grad_(False)
    norm=policy.processor.normalizer.normalizers['action'][policy.processor.shape_meta['action'][0]['key']]
    payloads=ReplayPayloads(rows,model.device)
    raw=RawReplay(args.source_bank,label_cache=args.label_cache)
    records=[]
    with (root/'records.jsonl').open('x',buffering=1) as journal:
        for row in rows:
            path,frame=raw.locate(row,'target')
            with h5py.File(path,'r') as h:
                obs={'joint_action':{'vector':h['joint_action/vector'][frame].astype(np.float32)},
                    'observation':{c:{'rgb':decode_legacy_robotwin_rgb(h[f'observation/{c}/rgb'][frame])} for c in CAMERAS}}
            payload=raw.attach_proprio(payloads[row['id']],row,policy)
            ref=payload['references']['target'].to(model.torch_dtype)
            valid=payload.get('valid',{}).get('target',torch.ones(ref.shape[:2],device=ref.device,dtype=torch.bool))
            policy.reset();model.policy_guard_enabled=True
            captured=move_cache(capture_frozen_inputs(policy,obs,row['counterfactual_instruction']),model.device)
            cached=payload['captured']['target']
            if not torch.equal(captured['video_inputs']['x'],cached['video_inputs']['x']):
                raise ValueError('Fresh production image differs from expert-cache image.')
            if not torch.equal(captured['proprio'],cached['proprio']):
                raise ValueError('Fresh production state differs from expert-cache state.')
            labels=move_cache(raw.grounding(row,'target'),model.device)
            oracle=validate_oracle(labels)
            actions={};oracle_calls={}
            for mode in protocol['modes']:
                policy.reset();model.policy_guard_enabled=mode!='eraf_off'
                with torch.no_grad():
                    if mode=='semantic_oracle':
                        with oracle_route(model,oracle) as oracle_calls:
                            value=policy._infer_action_chunk(obs,row['counterfactual_instruction'])
                    else:value=policy._infer_action_chunk(obs,row['counterfactual_instruction'])
                actions[mode]=norm.forward(torch.as_tensor(value).float().unsqueeze(0)).to(ref.device)
            if not torch.equal(actions['learned'],actions['repeat']):raise ValueError('Repeat action differs.')
            if not all(bool(torch.isfinite(x).all()) for x in actions.values()):raise ValueError('Nonfinite action.')
            errors={mode:float(masked_mse(action[:,:24],ref[:,:24],valid[:,:24])) for mode,action in actions.items()}
            record=dict(id=row['id'],task=row['diagnostic_task'],kind=row['diagnostic_kind'],split=row['replay_split'],
                scene_seed=row['scene_seed'],frame=frame,instruction=row['counterfactual_instruction'],
                state_sha256=array_sha256(obs['joint_action']['vector']),
                rgb_sha256={c:array_sha256(obs['observation'][c]['rgb']) for c in CAMERAS},
                reference_sha256=array_sha256(ref.float().cpu().numpy()),valid_sha256=array_sha256(valid.cpu().numpy()),
                oracle_label_sha256={k:array_sha256(v.cpu().numpy()) for k,v in oracle.items()},
                active_clauses=int(labels['clause_valid'].sum()),oracle_calls=oracle_calls,repeat_identical=True,
                action_mse_first24=errors,
                semantic_oracle_action_delta_mse_first24=float(masked_mse(actions['semantic_oracle'][:,:24],actions['learned'][:,:24],valid[:,:24])),
                eraf_off_action_delta_mse_first24=float(masked_mse(actions['eraf_off'][:,:24],actions['learned'][:,:24],valid[:,:24])))
            records.append(record);journal.write(json.dumps(record)+'\n');print(f'[probe] {len(records)}/20 {row["id"]}',flush=True)
    if hashes!=dict(manifest=file_sha256(args.manifest),checkpoint=file_sha256(args.checkpoint)):
        raise ValueError('Model or manifest changed during diagnostic.')
    (root/'summary.json').write_text(json.dumps(dict(complete=True,protocol=protocol,records=records,input_hashes_stable=True),indent=2)+'\n')


if __name__=='__main__':main()
