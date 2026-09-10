#!/usr/bin/env python3
"""Standalone figures for the completed fixed-state feature/cache experiments."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from scripts.probe_robotwin_world_language import read,sha,TASKS
from scripts.run_robotwin_world_language_collection import write

NAMES=['Ranking','Stacking','Place left','Place right','Burger / fries']
MODELS=['released','no_eraf']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root;report=root/'report';plan=read(root/'probes/plan.json');q=read(report/'quantitative_report.json')
    assert all(q['state_coverage'][m]['features_cache']==len(plan['states']) for m in MODELS)
    representation=read(report/'representation_summary.json');assert representation['complete']
    for name,digest in representation['output_sha256'].items():assert sha(report/name)==digest
    layers=list(csv.DictReader((report/'layer_profiles.csv').open()))
    layers=[r for r in layers if r['phase']=='initial' and r['comparison']=='target']
    statistics=[r for r in q['statistics'] if r['phase']=='initial' and r['metric'] in [
        'video_at_source_action_action_rms','video_at_target_action_action_rms',
        'action_at_source_video_action_rms','action_at_target_video_action_rms']]
    assert len(statistics)==40 and all(r['scenes']==10 for r in statistics)
    out=report/'feature_figures';out.mkdir(exist_ok=True);files=[]
    def save(fig,name):
        for ext in ['png','pdf']:
            p=out/f'{name}.{ext}';fig.savefig(p,dpi=160);files.append(dict(path=str(p),sha256=sha(p)))
        plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,5,figsize=(15,6),sharex=True,sharey=True)
    for i,model in enumerate(MODELS):
        for j,task in enumerate(TASKS):
            ax=axes[i,j]
            for kind,color,label in [('hidden','#2467a5','Post-block hidden'),('k','#cc7733','K cache'),('v','#32916b','V cache')]:
                rows=sorted((r for r in layers if r['model']==model and r['task']==task and r['kind']==kind),key=lambda r:int(r['layer']))
                assert len(rows)==30
                ax.plot(range(30),[float(r['relative_l2_mean']) for r in rows],color=color,label=label,lw=1.8)
            if i==0:ax.set_title(NAMES[j])
            if j==0:ax.set_ylabel(model.replace('_','-')+'\nRelative L2')
            if i==1:ax.set_xlabel('Video layer')
            ax.grid(alpha=.2);ax.set_xlim(0,29);ax.set_ylim(bottom=0)
    fig.suptitle('Source versus counterfactual language: video representations at the same initial observation',y=.995)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
    fig.text(.5,.025,'Means over 10 scenes per task. K/V are computed before the current block\'s text fusion; first-layer K/V can remain identical.\nRepresentation sensitivity is not evidence of correct goal execution.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.91]);save(fig,'video_language_layer_profiles')
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for i,(branch,held,title) in enumerate([('video','action','Change video language; hold action text'),('action','video','Change action text; hold video KV')]):
        ax=axes[i]
        for j,(model,condition) in enumerate([(m,c) for m in MODELS for c in ['source','target']]):
            rows=[next(r for r in statistics if r['model']==model and r['task']==task and r['metric']==f'{branch}_at_{condition}_{held}_action_rms') for task in TASKS]
            y=np.array([r['mean'] for r in rows]);ci=np.array([r['ci95'] for r in rows]);assert np.all(ci>0)
            ax.bar(np.arange(5)+(j-1.5)*.19,y,width=.18,color='#2467a5' if model=='released' else '#dc8635',
                   hatch='' if condition=='source' else '///',edgecolor='white',linewidth=.5,
                   yerr=np.stack([y-ci[:,0],ci[:,1]-y]),error_kw=dict(capsize=2,elinewidth=.8),
                   label=model.replace('_','-')+', '+condition+' held')
        ax.set_title(title);ax.set_xticks(range(5),['Ranking','Stacking','Place\nleft','Place\nright','Burger /\nfries'])
        ax.set_yscale('log');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Normalized action RMS difference (log scale)')
    fig.suptitle('Initial-state language interventions: magnitude of action changes',y=.995)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.95),ncol=4,frameon=False,fontsize=9)
    fig.text(.5,.025,'50 scenes; 10 per task; three noise draws per scene. Error bars: 95% scene-bootstrap confidence intervals.\nHeld source/target refers to the branch specified in each panel title. Action-change magnitude is not goal success.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.10,1,.87]);save(fig,'action_language_interventions')
    maps={}
    with np.load(report/'representation_maps.npz') as data:
        for entry in representation['maps']:
            if entry['phase']=='initial' and entry['comparison']=='target' and entry['kind']=='hidden':
                maps[(entry['model'],entry['task'])]=data[entry['key']][29]
    assert len(maps)==10
    positive=np.concatenate([v[v>0] for v in maps.values()]);norm=LogNorm(vmin=float(positive.min()),vmax=float(positive.max()))
    fig,axes=plt.subplots(2,5,figsize=(13,6))
    for i,model in enumerate(MODELS):
        for j,task in enumerate(TASKS):
            ax=axes[i,j];im=ax.imshow(maps[model,task],norm=norm,cmap='magma',interpolation='nearest')
            ax.axhline(7.5,color='white',lw=.8);ax.plot([4.5,4.5],[7.5,11.5],color='white',lw=.8)
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(NAMES[j])
            if j==0:ax.set_ylabel(model.replace('_','-'))
    fig.subplots_adjust(left=.07,right=.87,bottom=.17,top=.87,wspace=.10,hspace=.18)
    cax=fig.add_axes([.89,.22,.017,.58]);fig.colorbar(im,cax=cax,label='Mean token RMS difference (log colour scale)')
    fig.suptitle('Last-layer spatial language sensitivity: initial-state means',y=.97)
    fig.text(.5,.035,'12 x 10 token grid; upper region = head camera, lower regions = left / right wrist cameras.\nTen scenes per task, averaged in camera coordinates without object alignment. These are sensitivity maps, not attention or causal object-localization maps.',ha='center',fontsize=9)
    save(fig,'video_language_spatial_maps')
    data_path=out/'feature_figure_data.json'
    write(data_path,dict(plan_sha256=sha(root/'probes/plan.json'),statistics=statistics,layer_rows=layers,
                        representation_summary_sha256=sha(report/'representation_summary.json')))
    write(out/'figures.json',dict(complete=True,scope='Completed feature/cache experiments only; video-semantic and closed-loop results are still separate.',
        data_sha256=sha(data_path),files=files))
    print(json.dumps(dict(figures=3,files=len(files),output=str(out))))


if __name__=='__main__':main()
