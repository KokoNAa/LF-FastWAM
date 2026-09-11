#!/usr/bin/env python3
"""Frozen-state causal language interventions inside the video world model."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]
from scripts.probe_robotwin_world_language import read, sha, load_policy, observation, CAMERAS
from experiments.robotwin.language_wm_causal import (
    select_states, layer_groups, stage_active, causal_velocity, generation_conditions)
from experiments.robotwin.world_language_probe import hash_tensor, metrics, numpy_tensor, preference


def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, p)


def prepare(args):
    old = read(args.source/'probes/plan.json')
    if not read(args.source/'completion.json')['complete']:
        raise ValueError('Source experiment is not complete')
    if (args.output/'plan.json').exists(): raise ValueError('Plan already exists')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip():
        raise ValueError('Commit the experimental code before freezing the plan')
    chosen = select_states(old, 2)
    required = {p for row in chosen for p in row['raw_paths'].values()}
    manifest = read(old['manifest'])
    required |= set(old['checkpoints'].values()) | {old['manifest'], manifest['original_train_config'], manifest['stats_path']}
    checked = {}
    for p in sorted(required):
        h = sha(p)
        if h != old['input_sha256'][p]: raise ValueError('Changed source input: '+p)
        checked[p] = h
    files = [Path(__file__), REPO/'experiments/robotwin/language_wm_causal.py',
             REPO/'docs/robotwin_language_wm_causal_protocol.md',
             REPO/'scripts/probe_robotwin_world_language.py',
             REPO/'experiments/robotwin/world_language_probe.py',
             REPO/'src/fastwam/models/wan22/mot.py',
             REPO/'src/fastwam/models/wan22/wan_video_dit.py',
             REPO/'src/fastwam/models/wan22/fastwam.py',
             REPO/'src/fastwam/models/wan22/schedulers/scheduler_continuous.py']
    plan = {**old, 'format': 'robotwin_language_wm_causal_v1', 'complete': False,
            'source_plan': str(args.source/'probes/plan.json'),
            'source_plan_sha256': sha(args.source/'probes/plan.json'),
            'states': chosen, 'sigmas': [.5, 1.], 'input_sha256': checked,
            'code_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            'source_sha256': {str(p.relative_to(REPO)):sha(p) for p in files},
            'generation_states': [next(r['id'] for r in chosen if r['task']==task) for task in old['tasks']],
            'generation_seeds': [42], 'generation_conditions': generation_conditions(),
            'layer_groups': layer_groups(), 'training_performed': False,
            'action_inference_performed': False,
            'statistical_scope': 'Mechanistic pilot: 2 scenes/task, 3 paired noise repeats. Raw metrics; no count scaling.'}
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output/'plan.json', plan)
    print(json.dumps({'states':len(chosen), 'generation_conditions':len(plan['generation_conditions']),
                      'output':str(args.output)}), flush=True)


def load_state(policy, row):
    import h5py
    import numpy as np
    import torch
    from experiments.robotwin.no_eraf_probe import observation_hash
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    model = policy.model
    obs, refs, frames, roi = {}, {}, {}, []
    for branch, raw in [('source','native'), ('target','counterfactual')]:
        with h5py.File(row['raw_paths'][raw]) as h:
            obs[branch] = observation(h, row['frame'])
            frames[branch] = [policy._build_robotwin_image_tensor(observation(h,f))[0] for f in row['video_indices']]
            for f in row['video_indices']:
                ids = np.asarray(h['pgc_entity_state/entity_actor_ids'][f])[
                    np.asarray(h['pgc_entity_state/entity_valid'][f],dtype=bool)]
                masks = [np.isin(h[f'observation/{c}/actor_segmentation_ids'][f],ids) for c in CAMERAS]
                roi.append(np.concatenate([masks[0],np.concatenate(masks[1:],axis=1)],axis=0))
    def oh(x): return observation_hash(dict(state=x['joint_action']['vector'],
                                          **{c:x['observation'][c]['rgb'] for c in CAMERAS}))
    if oh(obs['source']) != oh(obs['target']): raise ValueError('References do not share an observation')
    image = policy._build_robotwin_image_tensor(obs['source'])
    proprio = policy._normalize_state(obs['source']['joint_action']['vector'])
    first = model._encode_input_image_latents_tensor(image)
    contexts = {}
    for name in ['source','target','source_paraphrase','target_paraphrase','empty']:
        c,m = model.encode_prompt(DEFAULT_PROMPT.format(task=row['instructions'][name]))
        contexts[name] = model._append_proprio_to_context(c,m,proprio)
    for c,m in contexts.values():
        if not torch.equal(c[:,-1],contexts['source'][0][:,-1]):
            raise ValueError('Proprio changed with language')
        if c.shape[1] != 513 or model.proprio_dim != 14:
            raise ValueError('Unverified text/proprio boundary')
    for name, fs in frames.items():
        clean = model._encode_video_latents(torch.stack(fs,dim=1).unsqueeze(0))
        if isinstance(clean,list): clean = clean[0].unsqueeze(0)
        clean[:,:,0:1] = first
        refs[name] = clean
    roi = torch.as_tensor(np.any(roi,axis=0),device=model.device,dtype=torch.bool)
    if tuple(roi.shape) != tuple(first.shape[-2:]) or not roi.any() or roi.all():
        raise ValueError('Invalid operation/background regions')
    if torch.equal(refs['source'][:,:,1:],refs['target'][:,:,1:]):
        raise ValueError('Expert futures do not diverge in this window')
    return dict(image=image,first=first,refs=refs,contexts=contexts,roi=roi,frames=frames,
                observation_sha256=oh(obs['source']),proprio=proprio)


def error_regions(prediction, target, roi):
    d = (prediction[:,:,1:].float()-target[:,:,1:].float())**2
    return {'full_mse':float(d.mean()), 'object_mse':float(d[...,roi].mean()),
            'background_mse':float(d[...,~roi].mean())}


def spatial_residuals(a, b, roi, latent):
    import numpy as np
    import torch.nn.functional as F
    ft, h, w = latent.shape[2], latent.shape[3]//2, latent.shape[4]//2
    mask = F.max_pool2d(roi.float()[None,None],2)[0,0].bool()
    records=[]; maps=[]
    for layer in sorted(a):
        delta = (a[layer].float()-b[layer].float()).square().mean(-1)[0].sqrt().reshape(ft,h,w)
        future = delta[1:]
        records.append(dict(layer=layer, object_rms=float(future[:,mask].square().mean().sqrt()),
                            background_rms=float(future[:,~mask].square().mean().sqrt())))
        maps.append(numpy_tensor(delta))
    return records, np.stack(maps)


def run_fixed(policy, row, state, folder, plan, seed, *, smoke=False):
    import numpy as np
    import torch
    model=policy.model; ctx=state['contexts']; records=[]; layer_records=[]; maps={}; checks=[]
    groups=plan['layer_groups']; refs=state['refs']; first=state['first']
    sigmas=[1.] if smoke else plan['sigmas']
    for reference, clean in refs.items():
        for sigma in sigmas:
            t=torch.tensor([sigma*model.train_video_scheduler.num_train_timesteps],device=model.device,dtype=model.torch_dtype)
            noise=torch.randn(clean.shape,generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32).to(clean)
            noisy=model.train_video_scheduler.add_noise(clean,noise,t); noisy[:,:,0:1]=first
            gt=model.train_video_scheduler.training_target(clean,noise,t)
            noisy_hash=hash_tensor(noisy); pred={}; acts={}
            for name in ['source','target','source_paraphrase','target_paraphrase','empty']:
                pred[name],act=causal_velocity(model,noisy,t,*ctx[name],capture=name in ['source','target'])
                if act: acts[name]=act
            repeat,_=causal_velocity(model,noisy,t,*ctx['source'])
            sham,_=causal_velocity(model,noisy,t,*ctx['source'],mode='patch',layers=groups['all'],donor=acts['source'])
            repeated=metrics(numpy_tensor(repeat),numpy_tensor(pred['source']))
            sham_error=metrics(numpy_tensor(sham),numpy_tensor(pred['source']))
            if repeated['max_abs'] != 0 or sham_error['max_abs'] != 0:
                raise ValueError('Repeat/self-patch control failed')
            if not checks:
                full,_=model._predict_joint_noise(latents_video=noisy,
                    latents_action=torch.zeros((1,32,14),device=model.device,dtype=model.torch_dtype),
                    timestep_video=t,timestep_action=t,context=ctx['source'][0],context_mask=ctx['source'][1],
                    state_only_context_mask=None,fuse_vae_embedding_in_latents=model.video_expert.fuse_vae_embedding_in_latents)
                equivalence=metrics(numpy_tensor(full),numpy_tensor(pred['source']))
                if equivalence['max_abs'] > .02: raise ValueError('Production video equivalence failed')
                # This one equivalence check calls the joint predictor; all primary interventions are video-only.
            else: equivalence=None
            rec,spatial=spatial_residuals(acts['source'],acts['target'],state['roi'],noisy)
            layer_records.append(dict(reference=reference,sigma=sigma,rows=rec))
            maps[f'{reference}_{sigma:g}']=spatial
            for base in ['source','target']:
                donor='target' if base=='source' else 'source'
                for band,layers in groups.items():
                    for mode in ['mask_text','patch']:
                        name=f'{base}_{mode}_{band}'
                        pred[name],_=causal_velocity(model,noisy,t,*ctx[base],mode=mode,layers=layers,
                                                    donor=acts[donor] if mode=='patch' else None)
                        if mode=='patch' and band=='all':
                            error=metrics(numpy_tensor(pred[name]),numpy_tensor(pred[donor]))
                            if error['max_abs'] > 1e-5: raise ValueError('Full residual patch did not recover donor')
            if not torch.equal(pred['source_mask_text_all'],pred['target_mask_text_all']):
                raise ValueError('Language survived all-layer text masking')
            natural=numpy_tensor(pred['target'][:,:,1:])-numpy_tensor(pred['source'][:,:,1:])
            natural_power=float(np.mean(natural.astype(np.float64)**2))
            for name,p in pred.items():
                summary=dict(reference=reference,sigma=sigma,seed=seed,condition=name,noisy_sha256=noisy_hash,
                             errors=error_regions(p,gt,state['roi']))
                if name.startswith('source_') or name.startswith('target_'):
                    base='source' if name.startswith('source_') else 'target'
                    other='target' if base=='source' else 'source'
                    summary['donor_axis']=preference(numpy_tensor(p[:,:,1:]),numpy_tensor(pred[base][:,:,1:]),numpy_tensor(pred[other][:,:,1:]))
                summary['natural_language_delta_mse']=natural_power
                records.append(summary)
            checks.append(dict(reference=reference,sigma=sigma,repeat=repeated,self_patch=sham_error,
                               full_patch_recovered=True,all_mask_language_equal=True,joint_equivalence=equivalence,
                               noise_sha256=hash_tensor(noise),noisy_sha256=noisy_hash))
            del pred,acts,repeat,sham
    np.savez_compressed(folder/'cross_attention_spatial.npz',**maps)
    write(folder/'fixed.json',dict(complete=True,rows=records,layer_residuals=layer_records,checks=checks))


def generate(policy, state, plan, seed, condition):
    import torch
    model=policy.model; first=state['first']; template=state['refs']['source']
    latent=torch.randn(template.shape,generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32).to(template)
    noise_hash=hash_tensor(latent); latent[:,:,0:1]=first
    initial_hash=hash_tensor(latent)
    ts,ds=model.infer_video_scheduler.build_inference_schedule(num_inference_steps=20,device=model.device,dtype=model.torch_dtype,shift_override=None)
    steps=[]; ctx=state['contexts']; group=plan['layer_groups'][condition['layer']]
    for i,(t,delta) in enumerate(zip(ts,ds,strict=True)):
        active=stage_active(condition['stage'],i,20)
        mode=condition['mode'] if active else 'baseline'; donor=None
        before_hash=hash_tensor(latent)
        if mode=='patch':
            _,donor=causal_velocity(model,latent,t.unsqueeze(0),*ctx[condition['donor']],capture=True)
        velocity,_=causal_velocity(model,latent,t.unsqueeze(0),*ctx[condition['base']],mode=mode,layers=group,donor=donor)
        latent=model.infer_video_scheduler.step(velocity,delta,latent)
        latent[:,:,0:1]=first
        if not torch.equal(latent[:,:,0:1],first): raise ValueError('Initial clamp failed')
        steps.append(dict(step=i,timestep=float(t),delta=float(delta),intervention_active=active,
                          recipient_noisy_sha256=before_hash,donor_same_input=mode=='patch'))
        del donor,velocity
    frames=model._decode_latents(latent,tiled=False)
    if len(frames)!=9: raise ValueError('Unexpected generated frame count')
    return latent,frames,dict(initial_noise_sha256=noise_hash,initial_clamped_sha256=initial_hash,
                             steps=steps,final_latent_sha256=hash_tensor(latent))


def run_generation(policy,row,state,folder,plan,seed,*,smoke=False):
    import numpy as np
    from PIL import Image
    conditions=plan['generation_conditions']
    if smoke: conditions=[c for c in conditions if c['name'] in ['source','target','source_mask_layer_all']]
    rows=[]; expected_noise=None
    for branch,frames in state['frames'].items():
        d=folder/f'expert_{branch}';d.mkdir(exist_ok=True)
        for i,t in enumerate(frames):
            arr=((numpy_tensor(t).transpose(1,2,0)+1)*127.5).clip(0,255).astype('uint8')
            Image.fromarray(arr).save(d/f'{i:02d}.png')
    for c in conditions:
        dest=folder/c['name'];dest.mkdir()
        latent,frames,proof=generate(policy,state,plan,seed,c)
        if expected_noise is None: expected_noise=proof['initial_noise_sha256']
        if expected_noise!=proof['initial_noise_sha256']: raise ValueError('Mismatched generation noise')
        for i,frame in enumerate(frames): frame.save(dest/f'{i:02d}.png')
        np.savez_compressed(dest/'latent.npz',latent=numpy_tensor(latent))
        losses={r:error_regions(latent,ref,state['roi']) for r,ref in state['refs'].items()}
        item=dict(condition=c,proof=proof,latent_reference_errors=losses,
                  frames=9,semantics_reviewed=False)
        write(dest/'generation.json',item); rows.append(item)
        print(json.dumps(dict(generated=c['name'],state=row['id'],seed=seed)),flush=True)
        del latent,frames
    write(folder/'generations.json',dict(complete=True,clips=len(rows),rows=rows,
        limitation='Latent error and image differences do not establish task semantics or success.'))


def worker(args):
    import torch
    plan=read(args.output/'plan.json'); plan_hash=sha(args.output/'plan.json')
    for name,h in plan['source_sha256'].items():
        if sha(REPO/name)!=h: raise ValueError('Runtime source changed: '+name)
    if args.seed not in plan['noise_seeds']: raise ValueError('Unplanned seed')
    if shutil.disk_usage(args.output).free < 12*1024**3: raise ValueError('Data disk reserve <12 GiB')
    policy=load_policy(plan,args.model)
    if len(policy.model.video_expert.blocks)!=30: raise ValueError('Unverified video architecture')
    selected=plan['states'][:1] if args.smoke else plan['states']
    branch='smoke' if args.smoke else 'results'
    with torch.inference_mode():
        for row in selected:
            folder=args.output/branch/args.model/f'seed{args.seed}'/row['id']
            completion=folder/'complete.json'
            if completion.exists():
                proof=read(completion)
                if proof['plan_sha256']!=plan_hash or any(sha(folder/p)!=h for p,h in proof['output_sha256'].items()):
                    raise ValueError('Invalid completed-state resume')
                continue
            if folder.exists() and list(folder.iterdir()):
                # Preserve interrupted output; deterministic whole-state replay, never merge partial data.
                archived=folder.with_name(folder.name+f'.interrupted-{time.time_ns()}')
                folder.rename(archived)
            folder.mkdir(parents=True,exist_ok=True)
            for p in row['raw_paths'].values():
                if sha(p)!=plan['input_sha256'][p]: raise ValueError('Changed paired trajectory')
            start=time.time();torch.cuda.reset_peak_memory_stats()
            state=load_state(policy,row)
            write(folder/'inputs.json',dict(state=row,model=args.model,seed=args.seed,
                  plan_sha256=plan_hash,observation_sha256=state['observation_sha256'],
                  contexts={k:dict(context_sha256=hash_tensor(v[0]),mask_sha256=hash_tensor(v[1])) for k,v in state['contexts'].items()}))
            run_fixed(policy,row,state,folder,plan,args.seed,smoke=args.smoke)
            if args.smoke or (row['id'] in plan['generation_states'] and args.seed in plan['generation_seeds']):
                gen=folder/'generated';gen.mkdir()
                run_generation(policy,row,state,gen,plan,args.seed,smoke=args.smoke)
            proof=dict(complete=True,state=row['id'],model=args.model,seed=args.seed,smoke=args.smoke,
                       plan_sha256=plan_hash,elapsed_seconds=time.time()-start,
                       peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                       output_sha256={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()})
            write(completion,proof)
            print(json.dumps(proof|{'output_sha256':'recorded'}),flush=True)
            del state;gc.collect();torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['prepare','worker'])
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--source',type=Path)
    ap.add_argument('--model',choices=['released','no_eraf'])
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.mode=='prepare': prepare(args)
    else: worker(args)


if __name__=='__main__': main()
