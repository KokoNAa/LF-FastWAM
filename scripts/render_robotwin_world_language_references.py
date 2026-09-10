#!/usr/bin/env python3
"""Expert endpoint atlas for the predetermined semantic review scenes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import textwrap
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from scripts.probe_robotwin_world_language import read,sha
from scripts.review_robotwin_world_language_videos import raw_mosaic
from scripts.run_robotwin_world_language_collection import write
from experiments.robotwin.image_io import decode_legacy_robotwin_rgb


def project_head(handle,frame,points):
    camera=handle['observation/head_camera']
    xyz=np.c_[points,np.ones(len(points))]@camera['extrinsic_cv'][frame].T
    if np.any(xyz[:,2]<=0):raise ValueError('Reference point is behind camera')
    pixels=xyz@camera['intrinsic_cv'][frame].T;pixels=pixels[:,:2]/pixels[:,2:]
    height,width=decode_legacy_robotwin_rgb(camera['rgb'][frame]).shape[:2]
    return pixels*np.array([320/width,256/height])


def atlas(scene,path):
    task=scene['task']
    labels=(['red','green','blue'] if task=='blocks_ranking_rgb' else ['red','green'] if task=='stack_blocks_two'
            else ['hamburger','fries','tray'] if task=='place_burger_fries' else ['moved object','reference object'])
    streams=[];records=[]
    for branch,first,title in [('native',True,'Shared initial observation'),('native',False,'Source expert endpoint'),
                               ('counterfactual',False,'Target expert endpoint')]:
        with h5py.File(scene['raw_paths'][branch]) as h:
            frame=0 if first else len(h['joint_action/vector'])-1
            entity=h['pgc_entity_state'];valid=entity['entity_valid'][frame].astype(bool)
            xyz=entity['entity_positions'][frame][valid];uv=project_head(h,frame,xyz)
            streams.append((title,raw_mosaic(h,frame),uv))
            records.append(dict(branch=branch,frame=frame,entity_positions=xyz.tolist(),projected_head_mosaic_pixels=uv.tolist()))
    fig,axes=plt.subplots(1,3,figsize=(15,8),dpi=140)
    for i,(ax,(title,im,uv)) in enumerate(zip(axes,streams,strict=True)):
        ax.imshow(im);ax.set_xlim(0,320);ax.set_ylim(384,0);ax.axis('off')
        for k,(x,y) in enumerate(uv):
            if 0<=x<320 and 0<=y<256:
                ax.scatter([x],[y],s=30,facecolors='none',edgecolors='yellow',linewidths=1)
                ax.annotate(str(k),(x,y),xytext=(4,-5),textcoords='offset points',fontsize=10,color='yellow',
                            bbox=dict(facecolor='black',alpha=.6,pad=1,edgecolor='none'))
        instruction=('Initial state; both goals false.' if i==0 else scene['instructions']['source' if i==1 else 'target'])
        ax.set_title(title+f' (frame {records[i]["frame"]})\n'+textwrap.fill(instruction,52),fontsize=10)
    legend='; '.join(f'{i} = {label}' for i,label in enumerate(labels))
    fig.suptitle(f'{task} | scene {scene["scene_seed"]}\nEntity centers projected using recorded camera calibration: {legend}',fontsize=12)
    fig.text(.5,.025,'Expert endpoints are from full trajectories of different lengths, not predictions or equal-duration comparison windows.\nEach column retains head, left-wrist and right-wrist views. Missing projected labels mean the center is outside the head view.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(path);plt.close(fig)
    return dict(task=task,scene_seed=scene['scene_seed'],path=str(path),sha256=sha(path),labels=labels,frames=records)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    plan=read(args.root/'probes/plan.json');selection_path=args.root/'video_review/selection.json';selection=read(selection_path)
    if selection['plan_sha256']!=sha(args.root/'probes/plan.json'):raise ValueError('Changed review plan')
    lookup={r['id']:r for r in plan['states']}
    selected={(lookup[i]['task'],lookup[i]['scene_seed']) for i in selection['states']}
    out=args.root/'video_review/references';out.mkdir(exist_ok=True)
    records=[atlas(s,out/f'{s["task"]}_{s["scene_seed"]}.png') for s in plan['scenes'] if (s['task'],s['scene_seed']) in selected]
    write(out/'references.json',dict(complete=True,selection_sha256=sha(selection_path),scenes=records,
        interpretation='Expert geometry and object-identity aid only; generated clips require separate visual inspection.'))
    print(json.dumps(dict(scenes=len(records),output=str(out))))


if __name__=='__main__':main()
