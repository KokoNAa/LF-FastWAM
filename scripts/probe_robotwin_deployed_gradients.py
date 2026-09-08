#!/usr/bin/env python3
"""Measure local gradient alignment on two fixed training batches; never update weights."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]


def gradient_group(row):
    if row.get('native_retention'):return 'correct_retention'
    if row.get('cf_retention'):return 'cf_retention'
    if row.get('initial_expert_anchor'):return 'initial_'+row['source_task']
    if row.get('fg_correction'):return 'fg'
    if row.get('ordinary_cf_control'):return 'ordinary_cf_control'
    return 'ordinary_expert'


def alignment(vectors):
    import torch
    if not vectors or any(not bool(torch.isfinite(v).all()) for v in vectors.values()):
        raise ValueError('Missing or nonfinite gradients.')
    if len({tuple(v.shape) for v in vectors.values()})!=1:raise ValueError('Gradient dimensions differ.')
    norms={k:float(v.double().square().sum().sqrt()) for k,v in vectors.items()}
    pairs={}
    for a,x in vectors.items():
        for b,y in vectors.items():
            if a>=b:continue
            dot=float(torch.dot(x.double(),y.double()))
            den=norms[a]*norms[b]
            pairs[a+'|'+b]=dict(dot=dot,cosine=max(-1.,min(1.,dot/den)) if den else None)
    return dict(norms=norms,pairs=pairs)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();stage=args.stage.resolve();root=args.output.resolve()
    read=lambda p:json.loads(Path(p).read_text())
    plan=read(stage/'plan.json')
    if (not read(stage/'complete.json')['complete'] or plan['steps']!=200
            or plan['action_objective']!='deployed_rollout_v1' or not plan['disable_seen_language_augmentation']):
        raise ValueError('Need completed final200 deployed-action model without language augmentation.')
    import torch
    from experiments.robotwin.eraf_fg_bridge import load_policy,trainable_parameters,file_sha256
    from experiments.robotwin.eraf_fg_data import RawReplay
    from experiments.robotwin.compact_replay import ReplayPayloads
    from experiments.robotwin.eraf_fg_training import mixture_stream,supervision_payload
    from experiments.robotwin.initial_anchor import effective_weight
    from experiments.robotwin.same_state_repair import noise_tensor
    from experiments.robotwin.deployed_action_objective import backward_deployed_example
    from experiments.robotwin.native_teacher import NativeTeacher
    from experiments.robotwin.decision_replay import tensor_digest
    checkpoint=stage/'step_000200.pt'
    bindings={str(p):file_sha256(p) for p in [stage/'plan.json',checkpoint,Path(plan['manifest']),Path(plan['correct_teacher']),Path(plan['cf_teacher'])]}
    manifest=read(plan['manifest'])
    stream=mixture_stream(manifest['states'],plan['seed'],plan['fg'],task_balanced=plan['task_balanced'],
        correct_count=plan['correct_count'],cf_count=plan['cf_count'],initial_expert_tasks=plan['initial_expert_tasks'])
    queries=[(step,index,row) for step in (1,2) for index,row in enumerate(next(stream))]
    if len(queries)!=24 or any(r['replay_split']!='train' for _,_,r in queries):raise ValueError('Training batches required.')
    counts=Counter(gradient_group(r) for _,_,r in queries)
    if counts['initial_place_empty_cup']!=1 or counts['initial_move_pillbottle_pad']!=1:
        raise ValueError('Expected one initial anchor from each new task.')
    root.mkdir(parents=True,exist_ok=False)
    def write(name,value):(root/name).write_text(json.dumps(value,indent=2)+'\n')
    write('protocol.json',dict(input_sha256=bindings,optimizer_updates=0,noise='Original step/index training seed',
        batches=[1,2],coefficient=1/24,counts=dict(counts),checkpoint=str(checkpoint),fg=plan['fg'],
        scope='Weighted mean gradients of two fixed training batches evaluated at the final checkpoint. Euclidean local alignment is not the actual Adam update, proof of causal CF regression, or a CF score.'))
    policy=load_policy(str(checkpoint),manifest,seed=plan['seed']);model=policy.model
    selected=trainable_parameters(model,plan['stage'],eraf=True,policy_scope=plan['policy_scope'],interface_scope=plan['interface_scope'])
    if list(selected)!=plan['trainable_parameters']:raise ValueError('Trainable parameter set differs.')
    adapters={n:p for n,p in model.mot.named_parameters() if n.endswith(('.lora_A','.lora_B'))}
    teachers={k:NativeTeacher(model,adapters,plan[k+'_teacher']) for k in ('correct','cf')}
    state=lambda:{'adapters':{n:p.detach() for n,p in adapters.items()},'guard':model.policy_guard_modules.state_dict()}
    before=tensor_digest(state());raw=RawReplay(plan['source_bank'])
    payloads=ReplayPayloads([r for _,_,r in queries],model.device)
    groups={};records=[];model.policy_guard_enabled=True
    with (root/'records.jsonl').open('x',buffering=1) as journal:
        for step,index,row in queries:
            model.zero_grad(set_to_none=True)
            payload=supervision_payload(row,payloads[row['id']])
            if row.get('frozen_input_protocol')!='robotwin_eraf_fg_pre_dit_v1':payload=raw.attach_proprio(payload,row,policy)
            seed=plan['seed']+(step-1)*12+index
            report=backward_deployed_example(model,row,payload,noise_tensor((1,32,14),seed,model),None,
                teachers=teachers,coefficient=1/24,eraf=True,fg=plan['fg'],correct_weight=plan['correct_weight'],
                cf_weight=plan['cf_weight'],correction_weight=effective_weight(row,plan['correction_weight'],plan['correction_task_weights']),
                fg_gradient_route=plan['fg_gradient_route'])
            group=gradient_group(row);vectors={}
            for prefix in ('mot.','guard.'):
                values=[(p.grad.detach().float().cpu() if p.grad is not None else torch.zeros(p.shape,dtype=torch.float32)).reshape(-1)
                        for n,p in selected.items() if n.startswith(prefix)]
                vectors[prefix]=torch.cat(values)
                if not bool(torch.isfinite(vectors[prefix]).all()):raise ValueError('Nonfinite gradients.')
            if group not in groups:groups[group]=vectors
            else:
                for prefix,v in vectors.items():groups[group][prefix].add_(v)
            item=dict(step=step,index=index,seed=seed,id=row['id'],pair_id=row['pair_id'],group=group,
                      row_sha256=__import__('hashlib').sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),report=report)
            records.append(item);journal.write(json.dumps(item)+'\n');print('[gradient]',len(records),'/24',group,flush=True)
    result={prefix:alignment({k:v[prefix] for k,v in groups.items()}) for prefix in ('mot.','guard.')}
    model.zero_grad(set_to_none=True)
    if tensor_digest(state())!=before or any(file_sha256(p)!=h for p,h in bindings.items()):raise ValueError('Diagnostic changed inputs or parameters.')
    write('summary.json',dict(complete=True,records=records,alignment=result,input_sha256=bindings,
        parameters_unchanged=True,input_hashes_stable=True,optimizer_updates=0,counts=dict(counts)))

if __name__=='__main__':main()
