#!/usr/bin/env python3
"""Prepare outcome-independent video panels and measure full-coverage sensitivity."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import textwrap
import numpy as np
from PIL import Image,ImageDraw

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from scripts.probe_robotwin_world_language import read,sha,CAMERAS
from scripts.run_robotwin_world_language_collection import write
from scripts.report_robotwin_world_language import verified_state,clustered

BRANCH={'source':'native','target':'counterfactual'}


def selected_states(plan):
    """The first two catalog entries; never inspect predictions for selection."""
    scenes={(r['task'],r['scene_seed']) for task in plan['tasks']
            for r in [s for s in plan['scenes'] if s['task']==task][:2]}
    phases={'initial','shared_decision','source_late','target_late'}
    return [r for r in plan['states'] if (r['task'],r['scene_seed']) in scenes and r['phase'] in phases]


def load_prediction(folder,language,seed):
    frames=[np.asarray(Image.open(folder/f'video_{language}_{seed}/{i:02d}.png').convert('RGB')) for i in range(9)]
    result=np.stack(frames)
    if result.shape!=(9,384,320,3):raise ValueError('Unexpected camera mosaic/video shape')
    return result


def raw_mosaic(handle,frame):
    from experiments.robotwin.image_io import decode_legacy_robotwin_rgb
    sizes=[(320,256),(160,128),(160,128)]
    images=[np.asarray(Image.fromarray(decode_legacy_robotwin_rgb(handle[f'observation/{c}/rgb'][frame])).resize(
        size,Image.Resampling.BILINEAR)) for c,size in zip(CAMERAS,sizes,strict=True)]
    return np.concatenate([images[0],np.concatenate(images[1:],axis=1)],axis=0)


def roi_masks(row):
    import h5py
    branches=['source','target'] if row['dual_reference_valid'] else [row['observation_branch']]
    masks=[]
    for branch in branches:
        with h5py.File(row['raw_paths'][BRANCH[branch]]) as h:
            for frame in row['video_indices']:
                ids=h['pgc_entity_state/entity_actor_ids'][frame][h['pgc_entity_state/entity_valid'][frame]]
                small=[np.isin(h[f'observation/{c}/actor_segmentation_ids'][frame],ids) for c in CAMERAS]
                mosaic=np.concatenate([small[0],np.concatenate(small[1:],axis=1)],axis=0)
                masks.append(np.asarray(Image.fromarray(mosaic).resize((320,384),Image.Resampling.NEAREST)))
    mask=np.any(masks,axis=0)
    if not mask.any():raise ValueError('Task entity mask is empty')
    return mask


def image_delta(a,b,mask):
    # Exclude the fixed/autoencoded initial frame; RGB values normalized to 0..1.
    diff=(a[1:].astype(np.float64)-b[1:].astype(np.float64))/255
    return dict(full_rmse=float(np.sqrt(np.mean(diff**2))),
                task_roi_rmse=float(np.sqrt(np.mean(diff[:,mask]**2))))


def panel(row,probe,seed,path):
    import h5py
    # Four generated rows, then one or two valid expert references.
    streams=[]
    for model in ['released','no_eraf']:
        folder=probe/model/'video'/row['id']
        for language in ['source','target']:
            streams.append((model+' / '+language,load_prediction(folder,language,seed)))
    branches=['source','target'] if row['dual_reference_valid'] else [row['observation_branch']]
    for branch in branches:
        with h5py.File(row['raw_paths'][BRANCH[branch]]) as h:
            streams.append(('expert / '+branch,np.stack([raw_mosaic(h,f) for f in row['video_indices']])))
    positions=[0,2,4,6,8]
    head_height=256
    width=320*len(positions);top=112;row_height=head_height+30
    canvas=Image.new('RGB',(width,top+len(streams)*row_height),'white');draw=ImageDraw.Draw(canvas)
    lines=[row['id']+f' | noise {seed} | head camera | action offsets 0,8,16,24,32',
           'SOURCE: '+row['instructions']['source'],'TARGET: '+row['instructions']['target'],
           'Same observation for both references: '+str(row['dual_reference_valid'])]
    y=4
    for line in lines:
        for wrapped in textwrap.wrap(line,200):draw.text((8,y),wrapped,fill='black');y+=16
    for i,(label,frames) in enumerate(streams):
        yy=top+i*row_height
        draw.text((8,yy+5),label,fill='black')
        for j,f in enumerate(positions):
            canvas.paste(Image.fromarray(frames[f,:head_height]),(j*320,yy+25))
    canvas.save(path,quality=93)
    return dict(path=str(path),sha256=sha(path),rows=[x[0] for x in streams],frame_positions=positions)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--mode',choices=['freeze','measure','panels'],required=True);args=ap.parse_args()
    probe=args.root/'probes';plan=read(probe/'plan.json');plan_hash=sha(probe/'plan.json')
    out=args.root/'video_review';out.mkdir(exist_ok=True)
    selection=dict(format='robotwin_world_language_video_review_v1',plan_sha256=plan_hash,
        selection='First two scenes/task in catalog order; initial/shared_decision/source_late/target_late; all noise seeds and models.',
        states=[r['id'] for r in selected_states(plan)],models=plan['models'],noise_seeds=plan['noise_seeds'],
        protocol_sha256=sha(REPO/'docs/robotwin_world_language_video_review.md'))
    frozen=out/'selection.json'
    if frozen.exists():assert read(frozen)==selection
    else:write(frozen,selection)
    if args.mode=='freeze':print(json.dumps(selection));return
    lookup={r['id']:r for r in plan['states']}
    if args.mode=='panels':
        manifest=[]
        for ident in selection['states']:
            for model in plan['models']:
                if not verified_state(probe/model/'video'/ident,plan_hash):raise ValueError('Selected video not completed: '+ident)
            for seed in plan['noise_seeds']:
                path=out/f'{ident}_seed{seed}.jpg'
                record=panel(lookup[ident],probe,seed,path)
                manifest.append(dict(state_id=ident,seed=seed,**record))
        write(out/'panels.json',dict(complete=True,panels=manifest,selection_sha256=sha(frozen)))
        # Template stays separate; never overwrite actual review annotations.
        template=out/'annotation_template.json'
        if not template.exists():
            annotations=[]
            for ident in selection['states']:
                for model in plan['models']:
                    for seed in plan['noise_seeds']:
                        for language in ['source','target']:
                            annotations.append(dict(state_id=ident,model=model,seed=seed,language=language,
                                visible_change=None,relation_established=None,clear_contradiction=None,
                                generation_quality=None,paired_semantic_change=None,evidence=None))
            write(template,dict(complete=False,selection_sha256=sha(frozen),annotations=annotations))
        print(json.dumps(dict(panels=len(manifest),output=str(out))));return
    values=defaultdict(list);records=[]
    for model in plan['models']:
        for row in plan['states']:
            folder=probe/model/'video'/row['id']
            if not verified_state(folder,plan_hash):raise ValueError('All-state video measurement cannot summarize partial coverage')
            mask=roi_masks(row)
            clips={(lang,seed):load_prediction(folder,lang,seed) for lang in ['source','target'] for seed in plan['noise_seeds']}
            for seed in plan['noise_seeds']:
                effect=image_delta(clips['source',seed],clips['target',seed],mask)
                records.append(dict(id=row['id'],model=model,kind='language',seed=seed,**effect))
                for name,value in effect.items():values[(model,row['task'],row['phase'],'language_'+name)].append((row['scene_seed'],value))
            for lang in ['source','target']:
                seeds=plan['noise_seeds']
                for i,a in enumerate(seeds):
                    for b in seeds[i+1:]:
                        effect=image_delta(clips[lang,a],clips[lang,b],mask)
                        records.append(dict(id=row['id'],model=model,kind='noise',language=lang,seed_a=a,seed_b=b,**effect))
                        for name,value in effect.items():values[(model,row['task'],row['phase'],'noise_'+name)].append((row['scene_seed'],value))
    stats=[dict(model=m,task=t,phase=p,metric=k,**clustered(v)) for (m,t,p,k),v in sorted(values.items())]
    write(out/'pixel_sensitivity.json',dict(complete=True,plan_sha256=plan_hash,records=records,statistics=stats,
        interpretation='Pixel sensitivity only; not a semantic success rate. Excludes fixed initial frame.'))
    print(json.dumps(dict(records=len(records),output=str(out))))


if __name__=='__main__':main()
