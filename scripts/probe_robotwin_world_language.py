#!/usr/bin/env python3
"""Fixed-state video representation, cache mediation and future-video probes."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]
from scripts.run_robotwin_world_language_collection import TASKS, write

RUNS = Path('/root/gpufree-data/LF-FastWAM/runs')
MANIFEST = RUNS/'robotwin_expanded_fg/20260908-cup-pill-prepared/cache/manifest.json'
CHECKPOINTS = {
    'released': Path('/root/gpufree-data/fastwam/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt'),
    'no_eraf': RUNS/'robotwin_five_task_repair/20260908-five-task-expert200/no_eraf/joint/step_000200.pt',
}
EXPECTED_SHA = dict(released='776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63',
                    no_eraf='e9ebee0cc0b1c532548b0d71444a259c617e6c833005dbee0b0e653a10157540')
CAMERAS = ['head_camera', 'left_camera', 'right_camera']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def instructions(row):
    from experiments.robotwin.decision_language_replay import bound_spatial_instruction_pairs
    a, b = row['source_instruction'], row['counterfactual_instruction']
    task = row['source_task']
    if task.startswith('place_a2b_'):
        pairs = bound_spatial_instruction_pairs(REPO, row['pair_id'], row['scene_info']['info'])
        a, b = pairs[0]['source'], pairs[0]['target']
        # Keep exactly the same object names; change only the command construction.
        def paraphrase(s):
            if not s.startswith('Place '): raise ValueError('Unreviewed placement template')
            return 'Please position '+s[len('Place '):]
        pa, pb = paraphrase(a), paraphrase(b)
    elif task == 'blocks_ranking_rgb':
        pa = 'Put the blocks in a row: red on the left, green in the middle, and blue on the right.'
        pb = 'Put the blocks in a row: blue on the left, green in the middle, and red on the right.'
    elif task == 'stack_blocks_two':
        pa = 'Bring the blocks to the center and put the green block on top of the red block.'
        pb = 'Bring the green block to the center and put the red block on top of it.'
    else:
        pa = 'Put the hamburger in the tray’s left slot and the french fries in its right slot.'
        pb = 'Put the hamburger in the tray’s right slot and the french fries in its left slot.'
    return dict(source=a, target=b, source_paraphrase=pa, target_paraphrase=pb, empty='')


def prepare(args):
    import h5py
    import numpy as np
    import yaml
    from experiments.robotwin.no_eraf_probe import require_pair, last_equal_qpos_prefix
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').exists(): raise ValueError('Plan already exists; use it unchanged')
    proof = read(args.collection/'status.json')
    if not proof['complete']: raise ValueError('Wait for verified paired collection completion')
    manifest = read(MANIFEST)
    config = yaml.safe_load(Path(manifest['original_train_config']).read_text())
    assert config['data']['train']['num_frames'] == 33
    assert config['data']['train']['action_video_freq_ratio'] == 4
    assert not config['model']['video_dit_config']['action_conditioned']
    files, states, scenes = {}, [], []
    known = {(r['pair_id'], r.get('task_config'), r.get('scene_seed')) for r in manifest['states']}
    for task in TASKS:
        verification = read(args.collection/(task+'-verified.json'))
        files.update(verification['raw_sha256'])
        pair = next((args.collection/task).glob('*/native/meta/pgc_episodes.jsonl')).parents[2]
        rows = {k: [json.loads(x) for x in (pair/k/'meta/pgc_episodes.jsonl').read_text().splitlines()]
                for k in ['native','counterfactual']}
        for a, b in zip(rows['native'], rows['counterfactual'], strict=True):
            require_pair(a,b)
            assert (a['pair_id'],'demo_randomized',a['scene_seed']) not in known
            paths = {k:str(pair/k/r['raw_hdf5']) for k,r in [('native',a),('counterfactual',b)]}
            scene = dict(task=task, pair_id=a['pair_id'], scene_seed=a['scene_seed'],
                episode_index=a['episode_index'], instructions=instructions(a), raw_paths=paths,
                initial_state_sha256=a['initial_state_sha256'])
            scenes.append(scene)
            with h5py.File(paths['native']) as x, h5py.File(paths['counterfactual']) as y:
                q = {'source':x['joint_action/vector'][:], 'target':y['joint_action/vector'][:]}
                def equal(f):
                    return (np.array_equal(q['source'][f],q['target'][f]) and all(
                        bytes(x[f'observation/{c}/rgb'][f]) == bytes(y[f'observation/{c}/rgb'][f]) for c in CAMERAS))
                assert equal(0)
                selected = [('initial','source',0,True)]
                end = last_equal_qpos_prefix(q['source'],q['target'],33)
                if end is not None and end > 0:
                    while end > 0 and not equal(end): end -= 1
                    if end > 0: selected.append(('shared_decision','source',end,True))
                for branch in ['source','target']:
                    n = len(q[branch])
                    if n < 33: raise ValueError('No full 33-action video window')
                    for phase,f in [('mid',(n-33)//2),('late',n-33)]:
                        if f > 0: selected.append((branch+'_'+phase,branch,f,False))
                for phase,branch,f,dual in selected:
                    ident = f'{task}_{a["scene_seed"]}_{phase}_f{f}'
                    states.append(scene | dict(id=ident,phase=phase,observation_branch=branch,
                        frame=f,dual_reference_valid=dual,video_indices=[f+4*i for i in range(9)]))
    for name,p in CHECKPOINTS.items():
        digest=sha(p)
        assert digest == EXPECTED_SHA[name], f'Unexpected {name} checkpoint'
        files[str(p)]=digest
    for p in [MANIFEST,Path(manifest['original_train_config']),Path(manifest['stats_path'])]:files[str(p)]=sha(p)
    # Also bind all collection metadata, not just image payloads.
    for p in args.collection.rglob('*.jsonl'): files[str(p)]=sha(p)
    plan = dict(format='robotwin_world_language_v1', complete=False, models=list(CHECKPOINTS),
        checkpoints={k:str(v) for k,v in CHECKPOINTS.items()}, checkpoint_sha256=EXPECTED_SHA,
        manifest=str(MANIFEST), collection=str(args.collection), tasks=TASKS, states=states, scenes=scenes,
        task_config='demo_randomized', noise_seeds=[42,43,44], sigmas=[.2,.5,.8,1.],
        video_frames=9, frame_stride=4, action_horizon=32, inference_steps=10,
        video_inference_steps=20, video_action_conditioned=False,
        training_scene_check='Disjoint from known current manifest; released pretraining scene inventory unavailable.',
        input_sha256=files, training_performed=False,
        code_commit=os.popen('git rev-parse HEAD').read().strip())
    write(root/'plan.json',plan)
    print(json.dumps(dict(states=len(states),scenes=len(scenes),tasks=TASKS)),flush=True)


def observation(handle, frame):
    import numpy as np
    from experiments.robotwin.image_io import decode_legacy_robotwin_rgb
    return dict(joint_action=dict(vector=np.asarray(handle['joint_action/vector'][frame],dtype=np.float32)),
        observation={c:dict(rgb=decode_legacy_robotwin_rgb(handle[f'observation/{c}/rgb'][frame])) for c in CAMERAS})


def load_policy(plan, name):
    from scripts.train_robotwin_cf_decision_adapter import load_policy as legacy
    from experiments.robotwin.eraf_fg_bridge import load_policy as repair
    if sha(plan['checkpoints'][name]) != plan['checkpoint_sha256'][name]:
        raise ValueError('Checkpoint changed after plan freeze')
    if name == 'released':
        policy=legacy(SimpleNamespace(checkpoint=plan['checkpoints'][name],seed=42),read(plan['manifest']))
    else:
        policy=repair(plan['checkpoints'][name],read(plan['manifest']),seed=42)
        policy.model.policy_guard_enabled=False
    policy.model.requires_grad_(False)
    policy.model.eval()
    assert not policy.model.video_expert.action_conditioned
    assert not policy.model.uses_transition_queries
    return policy


def run_features(policy, row, folder):
    import h5py
    import numpy as np
    import torch
    from experiments.robotwin.world_language_probe import (
        capture_world, compare_world, hybrid_cache, sample_cache, metrics, preference, numpy_tensor, hash_tensor)
    from experiments.robotwin.no_eraf_probe import observation_hash
    kind={'source':'native','target':'counterfactual'}
    with h5py.File(row['raw_paths'][kind[row['observation_branch']]]) as h:
        obs=observation(h,row['frame'])
    obs_hash=observation_hash(dict(state=obs['joint_action']['vector'],**{c:obs['observation'][c]['rgb'] for c in CAMERAS}))
    normalizer=policy.processor.normalizer.normalizers['action'][policy.processor.shape_meta['action'][0]['key']]
    captures={}
    for label in ['source','target','source_paraphrase','target_paraphrase','empty','repeat_source']:
        policy.seed=42
        cap=capture_world(policy,obs,row['instructions']['source' if label=='repeat_source' else label])
        if label in ['source','target']: captures[label]=cap
        comparisons=[]
        if label != 'source':
            against='target' if label=='target_paraphrase' else 'source'
            comparisons=compare_world(captures[against],cap)
            if label=='repeat_source':
                assert max(r['max_abs'] for r in comparisons)==0, 'Repeated prefill is not deterministic'
                assert metrics(captures['source']['action'],cap['action'])['max_abs']==0
            write(folder/(label+'-features.json'),dict(id=row['id'],against=against,
                observation_sha256=obs_hash, comparisons=comparisons))
        if label not in ['source','target']: del cap
    refs={}
    if row['dual_reference_valid']:
        for branch,k in kind.items():
            with h5py.File(row['raw_paths'][k]) as h:
                raw=np.asarray(h['joint_action/vector'][row['frame']:row['frame']+32],dtype=np.float32)
                refs[branch]=numpy_tensor(normalizer.forward(torch.from_numpy(raw).unsqueeze(0)))[0]
    results=[]; outputs={}
    with torch.no_grad():
        for v in ['source','target']:
            for a in ['source','target']:
                cache=hybrid_cache(captures[v],captures[a],policy.model.device)
                for seed in [42,43,44]:
                    prediction, noise_hash=sample_cache(policy.model,cache,seed,policy.num_inference_steps)
                    normalized=numpy_tensor(prediction)[0]
                    key=f'v_{v}_a_{a}_seed{seed}'
                    outputs[key]=normalized
                    result=dict(id=row['id'],video_language=v,action_language=a,seed=seed,
                        initial_noise_sha256=noise_hash,observation_sha256=obs_hash)
                    if refs:result['preference']=preference(normalized,refs['source'],refs['target'])
                    if v==a and seed==42:
                        deployed=policy._denormalize_action(prediction)[0]
                        replay=metrics(deployed,captures[a]['action'])
                        assert replay['max_abs'] <= 1e-5, ('Production replay mismatch',replay)
                        result['production_replay']=replay
                    results.append(result)
                del cache
    np.savez_compressed(folder/'actions.npz',**outputs,**{'reference_'+k:v for k,v in refs.items()})
    # Independent hook path used by closed-loop experiments must reproduce cache exchange.
    from experiments.robotwin.world_language_probe import video_language_override
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    cross_check={}
    for v,a in [('source','target'),('target','source')]:
        policy.seed=42;policy.reset()
        with video_language_override(policy.model,DEFAULT_PROMPT.format(task=row['instructions'][v])):
            raw=policy._infer_action_chunk(obs,row['instructions'][a])
        norm=numpy_tensor(normalizer.forward(torch.from_numpy(raw).unsqueeze(0)))[0]
        error=metrics(norm,outputs[f'v_{v}_a_{a}_seed42'])
        assert error['max_abs'] <= 1e-5, ('Closed-loop override differs from cache swap',error)
        cross_check[v+'_'+a]=error
    write(folder/'cache_actions.json',dict(id=row['id'],results=results,override_crosscheck=cross_check))
    del captures
    gc.collect()
    return dict(complete=True,id=row['id'],experiment='features_cache',observation_sha256=obs_hash,
                action_samples=12,comparisons=5,checkpoint=policy.model.lora_base_checkpoint if hasattr(policy.model,'lora_base_checkpoint') else None)


def run_video(policy,row,folder):
    import h5py
    import numpy as np
    import torch
    from experiments.robotwin.world_language_probe import video_velocity,hash_tensor,metrics,numpy_tensor
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    model=policy.model
    kind={'source':'native','target':'counterfactual'}
    with h5py.File(row['raw_paths'][kind[row['observation_branch']]]) as h:
        obs=observation(h,row['frame'])
    image=policy._build_robotwin_image_tensor(obs)
    proprio=policy._normalize_state(obs['joint_action']['vector'])
    contexts={}
    for lang in ['source','target']:
        c,m=model.encode_prompt(DEFAULT_PROMPT.format(task=row['instructions'][lang]))
        contexts[lang]=model._append_proprio_to_context(c,m,proprio)
    branches=['source','target'] if row['dual_reference_valid'] else [row['observation_branch']]
    first=model._encode_input_image_latents_tensor(image)
    losses=[]; equivalence=None
    for branch in branches:
        with h5py.File(row['raw_paths'][kind[branch]]) as h:
            frames=[policy._build_robotwin_image_tensor(observation(h,f))[0] for f in row['video_indices']]
            masks=[]
            for f in row['video_indices']:
                ids=np.asarray(h['pgc_entity_state/entity_actor_ids'][f])[
                    np.asarray(h['pgc_entity_state/entity_valid'][f],dtype=bool)]
                camera_masks=[np.isin(h[f'observation/{c}/actor_segmentation_ids'][f],ids) for c in CAMERAS]
                masks.append(np.concatenate([camera_masks[0],np.concatenate(camera_masks[1:],axis=1)],axis=0))
        clip=torch.stack(frames,dim=1).unsqueeze(0)
        clean=model._encode_video_latents(clip)
        if isinstance(clean,list):clean=clean[0].unsqueeze(0)
        clean[:,:,0:1]=first
        # Conservative union over all frames, then repeated over future latent time.
        roi=torch.as_tensor(np.any(masks,axis=0),device=model.device)
        assert tuple(roi.shape)==tuple(clean.shape[-2:])
        roi=roi[None,None,None].expand_as(clean[:,:,1:])
        if not roi.any():raise ValueError('No task-object ROI in paired future')
        for sigma in [.2,.5,.8,1.]:
            t=torch.tensor([sigma*model.train_video_scheduler.num_train_timesteps],device=model.device,dtype=model.torch_dtype)
            for seed in [42,43,44]:
                noise=torch.randn(clean.shape,generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32).to(clean)
                noisy=model.train_video_scheduler.add_noise(clean,noise,t)
                target=model.train_video_scheduler.training_target(clean,noise,t)
                noisy[:,:,0:1]=first
                pred={lang:video_velocity(model,noisy,t,*contexts[lang]) for lang in ['source','target']}
                if equivalence is None:
                    action_noise=torch.zeros((1,32,14),device=model.device,dtype=model.torch_dtype)
                    full,_=model._predict_joint_noise(latents_video=noisy,latents_action=action_noise,
                        timestep_video=t,timestep_action=t,context=contexts['source'][0],
                        context_mask=contexts['source'][1],state_only_context_mask=None,
                        fuse_vae_embedding_in_latents=model.video_expert.fuse_vae_embedding_in_latents)
                    equivalence=metrics(numpy_tensor(full),numpy_tensor(pred['source']))
                    if equivalence['max_abs'] > .02: raise ValueError(('Video-only/joint mismatch',equivalence))
                errors={}
                for lang,p in pred.items():
                    squared=(p[:,:,1:].float()-target[:,:,1:].float())**2
                    errors[lang]=dict(mse=float(squared.mean()),object_roi_mse=float(squared[roi].mean()))
                wrong='target' if branch=='source' else 'source'
                losses.append(dict(reference=branch,sigma=sigma,seed=seed,noisy_sha256=hash_tensor(noisy),
                    errors=errors,wrong_minus_correct_mse=errors[wrong]['mse']-errors[branch]['mse'],
                    wrong_minus_correct_roi_mse=errors[wrong]['object_roi_mse']-errors[branch]['object_roi_mse'],
                    velocity_delta=metrics(numpy_tensor(pred['source'][:,:,1:]),numpy_tensor(pred['target'][:,:,1:]))))
                del pred
        del clip,clean
    write(folder/'video_losses.json',dict(id=row['id'],rows=losses,joint_equivalence=equivalence,
        roi='Union of task-entity actor masks over nine frames, at VAE spatial resolution; initial latent excluded.'))
    # Save all paired generated frames losslessly for independent visual scoring.
    generations=[]
    for seed in [42,43,44]:
        for language in ['source','target']:
            result=model.infer_joint(prompt=DEFAULT_PROMPT.format(task=row['instructions'][language]),
                input_image=image,num_video_frames=9,action_horizon=32,proprio=proprio,
                num_inference_steps=20,seed=seed,rand_device='cpu',test_action_with_infer_action=False)
            dest=folder/f'video_{language}_{seed}';dest.mkdir()
            for i,frame in enumerate(result['video']):frame.save(dest/f'{i:02d}.png')
            generations.append(dict(language=language,seed=seed,frames=len(result['video']),directory=str(dest)))
            del result
    write(folder/'generations.json',dict(id=row['id'],generations=generations,
        temporal_protocol='9 frames every 4 action observations; 32-action window. Auxiliary joint inference, not deployed policy video.'))
    return dict(complete=True,id=row['id'],experiment='video',loss_rows=len(losses),generated_clips=6)


def worker(args):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ.setdefault('DIFFSYNTH_MODEL_BASE_PATH','/root/gpufree-data/fastwam/FastWAM/checkpoints')
    import torch
    plan=read(args.output/'plan.json')
    rows=plan['states']
    if args.limit:rows=rows[:args.limit]
    policy=load_policy(plan,args.model)
    # No gradients and no persistent guard history enter any primary experiment.
    with torch.no_grad():
        for i,row in enumerate(rows):
            for p in row['raw_paths'].values():
                if sha(p)!=plan['input_sha256'][p]:raise ValueError('Raw paired trajectory changed')
            folder=args.output/args.model/args.experiment/row['id']
            proof=folder/'complete.json'
            if proof.exists():
                raise ValueError('Explicitly verify completed outputs before resuming an existing worker')
            folder.mkdir(parents=True,exist_ok=True)
            torch.cuda.reset_peak_memory_stats()
            start=time.time()
            result=(run_features if args.experiment=='features_cache' else run_video)(policy,row,folder)
            result.update(elapsed_seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
            write(proof,result)
            print(json.dumps(dict(model=args.model,experiment=args.experiment,state=i+1,total=len(rows),**result)),flush=True)
            torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['prepare','worker'])
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--collection',type=Path)
    ap.add_argument('--model',choices=list(CHECKPOINTS))
    ap.add_argument('--experiment',choices=['features_cache','video'])
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    if args.mode=='prepare':prepare(args)
    else:worker(args)


if __name__=='__main__':main()
