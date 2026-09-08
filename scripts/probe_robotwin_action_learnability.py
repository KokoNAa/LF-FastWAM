#!/usr/bin/env python3
"""Bounded, in-memory optimization pilot; no checkpoint promotion or model files."""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
TASKS=('place_empty_cup','move_pillbottle_pad')
REGIMES=('initial_only','correction_only','balanced')
STEPS=64
EVAL_SEEDS=(42,1000042)

def training_weights(rows,regime):
    if regime not in REGIMES:raise ValueError('Unknown pilot regime.')
    if len(rows)!=3 or any(r['replay_split']!='train' for r in rows):raise ValueError('Exactly three training rows required.')
    if len({r['diagnostic_task'] for r in rows})!=1 or len({r['id'] for r in rows})!=3:raise ValueError('One task, unique rows required.')
    initial=[r for r in rows if r['diagnostic_kind']=='ordinary'];fg=[r for r in rows if r['diagnostic_kind']=='fg']
    if len(initial)!=1 or len(fg)!=2:raise ValueError('Need one initial and two corrective observations.')
    mass={'initial_only':(1.,0.),'correction_only':(0.,1.),'balanced':(.5,.5)}[regime]
    return {r['id']:mass[0] if r['diagnostic_kind']=='ordinary' else mass[1]/2 for r in rows}

def error_metrics(prediction,reference,valid):
    import torch
    from experiments.robotwin.eraf_fg_bridge import masked_mse
    p=prediction[:,:24].float();r=reference[:,:24].float();v=valid[:,:24]
    if p.shape!=r.shape or p.shape[-1]!=14 or not bool(v.any()):raise ValueError('Invalid actual executed-action horizon.')
    per_dim=(((p-r).square()*v.unsqueeze(-1)).sum(dim=(0,1))/v.sum()).tolist()
    mse=float(masked_mse(p,r,v))
    if not math.isfinite(mse) or not all(math.isfinite(x) for x in per_dim):raise ValueError('Nonfinite action error.')
    return dict(mse_first24=mse,mse_by_action_dimension=per_dim,valid_positions=int(v.sum()))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--task',choices=TASKS,required=True);ap.add_argument('--regime',choices=REGIMES,required=True)
    args=ap.parse_args();root=args.output.resolve();stage=args.stage.resolve();read=lambda p:json.loads(Path(p).read_text())
    plan=read(stage/'plan.json');checkpoint=stage/'step_000200.pt'
    if (not read(stage/'complete.json')['complete'] or plan['eraf']!='on' or plan['fg']!='full'
            or plan['action_objective']!='deployed_rollout_v1'):raise ValueError('Need completed deployed full model.')
    import torch
    from experiments.robotwin.eraf_fg_bridge import load_policy,trainable_parameters,MasterAdamW,file_sha256,masked_mse
    from experiments.robotwin.eraf_action_protocol import parameter_learning_rates
    from experiments.robotwin.compact_replay import ReplayPayloads
    from experiments.robotwin.eraf_fg_data import RawReplay
    from experiments.robotwin.deployed_action_objective import sample
    from experiments.robotwin.same_state_repair import noise_tensor
    from experiments.robotwin.decision_replay import tensor_digest
    from scripts.probe_robotwin_expanded_fg_actions import select_rows
    manifest=read(plan['manifest']);assert manifest['complete']
    rows=[r for r in select_rows(manifest['states']) if r['diagnostic_task']==args.task]
    training=[r for r in rows if r['replay_split']=='train'];weights=training_weights(training,args.regime)
    if len(rows)!=9:raise ValueError('Need three train and six heldout observations.')
    bindings={str(p):file_sha256(p) for p in (stage/'plan.json',checkpoint,Path(plan['manifest']))}
    root.mkdir(parents=True,exist_ok=False)
    def write(n,v):
        path=root/(n+'.tmp');path.write_text(json.dumps(v,indent=2)+'\n');path.replace(root/n)
    protocol=dict(task=args.task,regime=args.regime,input_sha256=bindings,selected_rows=rows,training_weights=weights,
        steps=STEPS,learning_rate=plan['learning_rate'],interface_learning_rate=plan['interface_learning_rate'],
        evaluation_noise_seeds=list(EVAL_SEEDS),training_noise='420000 + optimizer step, same seed across observations and regimes',
        snapshots=[0,16,STEPS],checkpoint_output=False,retention=False,conditional_difference=False,
        scope='Target-only learnability pilot. Training state counts and compute differ by regime; this is not a matched module ablation, policy selection, independent test or CF success. In-memory updates are intentionally discarded after recording results.')
    write('protocol.json',protocol)
    policy=load_policy(str(checkpoint),manifest,seed=42);model=policy.model;model.policy_guard_enabled=True
    selected=trainable_parameters(model,'joint',eraf=True,policy_scope='action',interface_scope='all')
    if list(selected)!=plan['trainable_parameters']:raise ValueError('Parameter scope differs from audited parent.')
    tracked={**{'mot.'+n:p for n,p in model.mot.named_parameters() if n.endswith(('.lora_A','.lora_B'))},
             **{'guard.'+n:p for n,p in model.policy_guard_modules.named_parameters()}}
    before={n:tensor_digest(p.detach()) for n,p in tracked.items()}
    optimizer=MasterAdamW(selected.values(),lr=plan['learning_rate'],learning_rates=parameter_learning_rates(selected,plan['learning_rate'],plan['interface_learning_rate']))
    payloads=ReplayPayloads(rows,model.device);raw=RawReplay(plan['source_bank']);payload_hashes={}
    def payload(row):
        p=payloads[row['id']]
        if row.get('frozen_input_protocol')!='robotwin_eraf_fg_pre_dit_v1':p=raw.attach_proprio(p,row,policy)
        return p
    for row in rows:
        p=payload(row)
        payload_hashes[row['id']]=tensor_digest({'captured':p['captured']['target'],'reference':p['references']['target'],'valid':p.get('valid',{}).get('target')})
        del p
    protocol['actual_payload_hashes']=payload_hashes;protocol['trainable_parameters']=list(selected);write('protocol.json',protocol)
    records=[];updates=[]
    def evaluate(step):
        with torch.no_grad(),(root/'evaluations.jsonl').open('a',buffering=1) as journal:
            for row in rows:
                p=payload(row);ref=p['references']['target'].to(model.torch_dtype)
                valid=p.get('valid',{}).get('target',torch.ones(ref.shape[:2],device=ref.device,dtype=torch.bool))
                for seed in EVAL_SEEDS:
                    pred=sample(model,p['captured']['target'],noise_tensor((1,32,14),seed,model),eraf=True,checkpoint=False)
                    r=dict(step=step,id=row['id'],split=row['replay_split'],kind=row['diagnostic_kind'],scene_seed=row['scene_seed'],noise_seed=seed,
                           **error_metrics(pred,ref,valid))
                    records.append(r);journal.write(json.dumps(r)+'\n')
                del p,pred,ref
        print('[snapshot]',step,len(records),flush=True)
    evaluate(0)
    with (root/'training.jsonl').open('x',buffering=1) as journal:
        for step in range(1,STEPS+1):
            optimizer.zero_grad();losses={}
            for row in training:
                w=weights[row['id']]
                if not w:continue
                p=payload(row);ref=p['references']['target'].to(model.torch_dtype)
                valid=p.get('valid',{}).get('target',torch.ones(ref.shape[:2],device=ref.device,dtype=torch.bool))
                pred=sample(model,p['captured']['target'],noise_tensor((1,32,14),420000+step,model),eraf=True,checkpoint=True)
                loss=masked_mse(pred[:,:24],ref[:,:24],valid[:,:24])
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite pilot objective.')
                (w*loss).backward();losses[row['id']]=float(loss.detach());del p,pred,ref,loss
            norm=optimizer.step();update=dict(step=step,noise_seed=420000+step,losses=losses,weights=weights,grad_norm=norm)
            updates.append(update);journal.write(json.dumps(update)+'\n')
            print('[step]',step,'/64',flush=True)
            if step in (16,STEPS):evaluate(step)
    changed=[n for n,p in tracked.items() if tensor_digest(p.detach())!=before[n]]
    if not changed or set(changed)-set(selected):raise ValueError('No update or unexpected parameter changes.')
    if any(not torch.equal(p.detach().float(),m.detach().to(p.dtype).float()) for p,m in zip(optimizer.live,optimizer.master,strict=True)):
        raise ValueError('Optimizer masters differ from current live values.')
    if any(file_sha256(p)!=h for p,h in bindings.items()):raise ValueError('Source files changed.')
    for row in rows:
        p=payload(row)
        if tensor_digest({'captured':p['captured']['target'],'reference':p['references']['target'],'valid':p.get('valid',{}).get('target')})!=payload_hashes[row['id']]:raise ValueError('Payload changed.')
    if [u['step'] for u in updates]!=list(range(1,STEPS+1)):raise ValueError('Actual update journal incomplete.')
    write('summary.json',dict(complete=True,optimizer_steps=len(updates),training_journal_sha256=file_sha256(root/'training.jsonl'),protocol=protocol,records=records,
        changed_parameters=changed,unexpected_changes=[],input_hashes_stable=True,optimizer_master_binding=True,
        model_files_written=False,goal_achievement_claim=False))

if __name__=='__main__':main()
