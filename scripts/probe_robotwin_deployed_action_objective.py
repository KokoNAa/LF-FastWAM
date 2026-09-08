#!/usr/bin/env python3
"""Verify production identity and real full-unroll backward before training."""
from __future__ import annotations
import argparse
from functools import wraps
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','manifest','source-bank','correct-teacher','cf-teacher','output'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--fg',choices=['off','full'],required=True)
    args=ap.parse_args()
    import h5py
    import numpy as np
    import torch
    from experiments.robotwin.eraf_fg_bridge import load_policy,trainable_parameters,capture_frozen_inputs,file_sha256
    from experiments.robotwin.eraf_fg_data import RawReplay
    from experiments.robotwin.eraf_fg_contract import CAMERAS
    from experiments.robotwin.image_io import decode_legacy_robotwin_rgb
    from experiments.robotwin.compact_replay import ReplayPayloads
    from experiments.robotwin.same_state_repair import noise_tensor,move_cache
    from experiments.robotwin.decision_replay import tensor_digest
    from experiments.robotwin.deployed_action_objective import sample,backward_deployed_example
    from experiments.robotwin.native_teacher import NativeTeacher
    from experiments.robotwin.eraf_fg_training import mixture_stream
    from scripts.probe_robotwin_semantic_oracle_actions import selected_rows
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(Path(args.manifest).read_text());rows=selected_rows(manifest['states'])
    input_hashes={k:file_sha256(getattr(args,k)) for k in ('checkpoint','manifest','correct_teacher','cf_teacher')}
    policy=load_policy(args.checkpoint,manifest,seed=42);model=policy.model
    selected=trainable_parameters(model,'joint',eraf=True,policy_scope='action')
    adapters={n:p for n,p in model.mot.named_parameters() if n.endswith(('.lora_A','.lora_B'))}
    teachers={k:NativeTeacher(model,adapters,getattr(args,k+'_teacher')) for k in ('correct','cf')}
    state=lambda:{'adapters':{k:p.detach() for k,p in adapters.items()},'guard':model.policy_guard_modules.state_dict()}
    before=tensor_digest(state())
    raw=RawReplay(args.source_bank);payloads=ReplayPayloads(rows,model.device)
    records=[]
    def write(name,value):(root/name).write_text(json.dumps(value,indent=2)+'\n')
    write('protocol.json',dict(checkpoint=args.checkpoint,input_hashes=input_hashes,selected_ids=[r['id'] for r in rows],
                              optimizer_updates=0,inference_steps=10,seed=42,fg=args.fg,objective='deployed_rollout_v1',
                              scope='Production action identity and gradient feasibility only; no CF score or weight updates.'))
    with (root/'records.jsonl').open('x',buffering=1) as log:
        for row in rows:
            path,frame=raw.locate(row,'target')
            with h5py.File(path,'r') as h:
                obs={'joint_action':{'vector':h['joint_action/vector'][frame].astype(np.float32)},
                     'observation':{c:{'rgb':decode_legacy_robotwin_rgb(h[f'observation/{c}/rgb'][frame])} for c in CAMERAS}}
            policy.reset();captured=move_cache(capture_frozen_inputs(policy,obs,row['counterfactual_instruction']),model.device)
            payload=raw.attach_proprio(payloads[row['id']],row,policy)
            if not torch.equal(captured['video_inputs']['x'],payload['captured']['target']['video_inputs']['x']) or not torch.equal(captured['proprio'],payload['captured']['target']['proprio']):
                raise ValueError('Cached expert observation differs from current production input.')
            results={}
            for enabled in (False,True):
                policy.reset();model.policy_guard_enabled=enabled
                actual=[];original=model.infer_action
                @wraps(original)
                def record(*pos,**kw):
                    output=original(*pos,**kw);actual.append(output['action'].detach().clone());return output
                existed='infer_action' in vars(model);previous=vars(model).get('infer_action')
                model.infer_action=record
                try:
                    with torch.no_grad():policy._infer_action_chunk(obs,row['counterfactual_instruction'])
                finally:
                    if existed:model.infer_action=previous
                    else:delattr(model,'infer_action')
                if len(actual)!=1:raise ValueError('Production action capture failed.')
                with torch.no_grad():
                    predicted=sample(model,captured,noise_tensor((1,32,14),42,model),eraf=enabled,checkpoint=False)
                expected=actual[0].to(predicted).unsqueeze(0)
                if not torch.equal(predicted,expected):
                    raise ValueError('Sampler differs from deployment: '+str(float((predicted-expected).abs().max())))
                results['on' if enabled else 'off']=dict(exact=True,action_digest=tensor_digest(predicted),input_digest=tensor_digest(captured))
            item=dict(id=row['id'],modes=results);records.append(item);log.write(json.dumps(item)+'\n')
            print('[identity]',len(records),'/20',flush=True)
    model.policy_guard_enabled=True
    # The first actual planned batch includes paired experts, corrective data,
    # and both preservation teachers. All are train rows, without result selection.
    batch=next(mixture_stream(manifest['states'],42,args.fg,task_balanced=True,correct_count=2,cf_count=4,
                              initial_expert_tasks=['place_empty_cup','move_pillbottle_pad']))
    chosen={}
    for row in batch:
        kind='correct' if row.get('native_retention') else 'cf' if row.get('cf_retention') else 'fg' if row.get('fg_correction') or row.get('ordinary_cf_control') else 'pair'
        chosen.setdefault(kind,row)
    if set(chosen)!={'correct','cf','fg','pair'}:raise ValueError('Missing a training branch.')
    payloads=ReplayPayloads(list(chosen.values()),model.device);gradients=[]
    for index,(kind,row) in enumerate(chosen.items()):
        from experiments.robotwin.eraf_fg_training import supervision_payload
        if row['replay_split']!='train':raise ValueError('Gradient probe used holdout data.')
        payload=supervision_payload(row,payloads[row['id']])
        if row.get('frozen_input_protocol')!='robotwin_eraf_fg_pre_dit_v1':payload=raw.attach_proprio(payload,row,policy)
        for enabled in (False,True):
            model.policy_guard_enabled=enabled
            model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            report=backward_deployed_example(model,row,payload,noise_tensor((1,32,14),42+index,model),None,
                                             teachers=teachers,eraf=enabled,fg=args.fg,cf_weight=4.,correct_weight=1.)
            norms={prefix:sum(float(p.grad.detach().float().square().sum()) for n,p in selected.items()
                             if n.startswith(prefix) and p.grad is not None)**.5 for prefix in ('mot.','guard.')}
            required=('mot.','guard.') if enabled else ('mot.',)
            if not all(np.isfinite(v) for v in norms.values()) or any(norms[k]==0 for k in required) and report['deployed_objective']!=0:
                raise ValueError('Nonfinite or missing gradients despite nonzero objective.')
            if not enabled and norms['guard.']!=0:raise ValueError('ERAF-off gradient reached ERAF.')
            gradients.append(dict(kind=kind,eraf=enabled,id=row['id'],report=report,norms=norms,seconds=time.monotonic()-start,
                                  peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3))
            write('gradients.json',gradients);print('[backward]',kind,enabled,norms,flush=True)
    model.zero_grad(set_to_none=True)
    if before!=tensor_digest(state()):raise ValueError('Parameters changed during backward-only diagnostic.')
    if input_hashes!={k:file_sha256(getattr(args,k)) for k in input_hashes}:raise ValueError('Input files changed.')
    write('summary.json',dict(complete=True,records=records,gradients=gradients,parameters_unchanged=True,
                              input_hashes_stable=True,input_hashes=input_hashes,optimizer_updates=0))


if __name__=='__main__':main()
