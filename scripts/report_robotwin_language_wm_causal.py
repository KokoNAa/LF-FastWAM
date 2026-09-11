#!/usr/bin/env python3
"""Summarize audited causal WM probes; image semantics require actual review."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import textwrap
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from scripts.probe_robotwin_language_wm_causal import read,sha,write
from scripts.run_robotwin_language_wm_causal import verify_state


def scene_summary(rows):
    """Three noise draws are repeats, not three independent scene samples."""
    by_scene=defaultdict(list)
    for row in rows:by_scene[row['scene']].append(row['value'])
    means={str(k):float(np.mean(v)) for k,v in by_scene.items()}
    return dict(mean=float(np.mean(list(means.values()))),scene_means=means,
                scene_range=[min(means.values()),max(means.values())],scenes=len(means),repeats=len(rows))


def summarize(root,allow_partial=False):
    plan=read(root/'plan.json');ph=sha(root/'plan.json');records=[];pending=[];proofs={};clips=0
    for model in plan['models']:
        for seed in plan['noise_seeds']:
            for state in plan['states']:
                d=root/'results'/model/f'seed{seed}'/state['id']
                if not (d/'complete.json').exists():
                    pending.append(str(d));continue
                generated=seed in plan['generation_seeds'] and state['id'] in plan['generation_states']
                verify_state(d,ph,generated)
                fixed=read(d/'fixed.json')
                if len(fixed['rows'])!=84 or len(fixed['checks'])!=4:raise ValueError('Fixed grid incomplete')
                proofs[str(d/'complete.json')]=sha(d/'complete.json')
                for row in fixed['rows']:
                    records.append(dict(model=model,task=state['task'],scene=state['scene_seed'],**row))
                if generated:clips+=read(d/'generated/generations.json')['clips']
    if pending and not allow_partial:raise ValueError(f'{len(pending)} states remain')
    groups=defaultdict(list);conditions=defaultdict(dict)
    for row in records:
        key=tuple(row[k] for k in ['model','task','scene','seed','reference','sigma'])
        if row['condition'] in conditions[key]:raise ValueError('Duplicate statistical observation')
        conditions[key][row['condition']]=row
    for (model,task,scene,seed,reference,sigma),rows in conditions.items():
        basekey=(model,task,reference,sigma)
        wrong='target' if reference=='source' else 'source'
        for variant in ['baseline','mask_text_early','mask_text_middle','mask_text_late','mask_text_all']:
            names={b:b if variant=='baseline' else b+'_'+variant for b in ['source','target']}
            for region in ['object_mse','background_mse','full_mse']:
                value=rows[names[wrong]]['errors'][region]-rows[names[reference]]['errors'][region]
                groups[basekey+('wrong_minus_correct_'+region,variant)].append(dict(scene=scene,seed=seed,value=value))
        for name,row in rows.items():
            if '_patch_' not in name:continue
            axis=row['donor_axis']['target_axis_projection']
            # Conservative fixed numerical floor; also retain unnormalized errors in raw records.
            if axis is not None and row['natural_language_delta_mse']>=1e-10:
                groups[basekey+('donor_axis_projection',name)].append(dict(scene=scene,seed=seed,value=axis))
        for name in ['source_paraphrase','target_paraphrase']:
            value=rows[name]['donor_axis']['source_rmse']
            groups[basekey+('velocity_rms_change',name)].append(dict(scene=scene,seed=seed,value=value))
        groups[basekey+('velocity_rms_change','source_to_target')].append(
            dict(scene=scene,seed=seed,value=rows['source']['natural_language_delta_mse']**.5))
    statistics=[dict(model=k[0],task=k[1],reference=k[2],sigma=k[3],metric=k[4],condition=k[5],**scene_summary(v))
                for k,v in sorted(groups.items())]
    report=dict(format='robotwin_language_wm_causal_report_v1',complete=False,compute_complete=not pending,
        plan_sha256=ph,states_verified=len(proofs),expected_states=60,fixed_rows=len(records),
        generated_clips=clips,expected_generated_clips=270,pending=pending,statistics=statistics,
        verified_state_proofs=proofs,semantic_review_complete=False,
        uncertainty='Average paired noise within each scene. Two scenes/task: descriptive means/ranges, no population CI.',
        caveats=['New raw measurements; no success-rate weighting.',
                 'Generated latent errors are not semantic success.',
                 'A masked text stream includes prompt prefix and padding but preserves proprio.',
                 'Residual donors carry distributed context; this is not individual-word localization.',
                 'Axis is omitted if natural source/target velocity MSE is below 1e-10.'])
    out=root/('report_partial' if pending else 'report');out.mkdir(exist_ok=True)
    write(out/'quantitative.json',report)
    (out/'observations.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    return report,out


def panels(root,out):
    from PIL import Image,ImageDraw
    plan=read(root/'plan.json');items=[]
    variants=[c['name'] for c in plan['generation_conditions'] if c['name'] not in ['source','target']]
    for model in plan['models']:
        for state in [r for r in plan['states'] if r['id'] in plan['generation_states']]:
            folder=root/'results'/model/'seed42'/state['id']/'generated'
            for page,start in enumerate(range(0,len(variants),5)):
                labels=['source','target']+variants[start:start+5]+['expert_source','expert_target']
                width=5*320;height=124+len(labels)*286
                img=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(img)
                lines=[f'{model} | {state["id"]} | seed42 | page{page+1}',
                       'SOURCE: '+state['instructions']['source'],'TARGET: '+state['instructions']['target'],
                       'Head camera; frame offsets 0,8,16,24,32. Predictions are not physical execution.']
                y=4
                for line in lines:
                    for line in textwrap.wrap(line,185):draw.text((8,y),line,fill='black');y+=16
                for i,name in enumerate(labels):
                    y=124+i*286;draw.text((8,y+4),name,fill='black')
                    for j,f in enumerate([0,2,4,6,8]):
                        src=folder/name/f'{f:02d}.png'
                        with Image.open(src) as im:img.paste(im.crop((0,0,320,256)),(j*320,y+25))
                path=out/'panels'/f'{model}_{state["id"]}_p{page+1}.jpg';path.parent.mkdir(exist_ok=True)
                img.save(path,quality=94)
                items.append(dict(model=model,state=state['id'],path=str(path),sha256=sha(path),conditions=labels,
                                  frame_indices=[0,2,4,6,8],reviewed=False))
    write(out/'panels.json',dict(complete=True,panels=items,semantics_reviewed=False))


def plots(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=report['statistics'];tasks=['blocks_ranking_rgb','stack_blocks_two','place_a2b_left','place_a2b_right','place_burger_fries']
    labels=['Ranking','Stacking','Left to right','Right to left','Burger/fries'];models=['released','no_eraf']
    def lookup(model,task,ref,metric,condition):
        r=[r for r in rows if r['model']==model and r['task']==task and r['reference']==ref and r['sigma']==1.
           and r['metric']==metric and r['condition']==condition]
        return r[0]['mean'] if r else np.nan
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for mi,model in enumerate(models):
        for ri,ref in enumerate(['source','target']):
            ax=axs[mi,ri]
            for variant in ['baseline','mask_text_early','mask_text_middle','mask_text_late','mask_text_all']:
                y=[lookup(model,t,ref,'wrong_minus_correct_object_mse',variant) for t in tasks]
                ax.plot(range(5),y,'o-',label=variant.replace('mask_text_','mask '))
            ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(5),labels,rotation=15)
            ax.set_title(f'{model} / {ref} expert future');ax.set_ylabel('Wrong minus correct velocity MSE (object ROI)')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Text masking inside WM | sigma=1 | 2 scenes/task, 3 paired noise repeats\nPositive: correct language fits this expert future better; descriptive means')
    for ext in ['png','pdf']:fig.savefig(out/f'wm_text_masking.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for mi,model in enumerate(models):
        for bi,base in enumerate(['source','target']):
            ax=axs[mi,bi]
            # Show target-future noisy inputs separately; sigma1 input is shared by both references.
            for band in ['early','middle','late','all']:
                y=[lookup(model,t,'target','donor_axis_projection',f'{base}_patch_{band}') for t in tasks]
                ax.plot(range(5),y,'o-',label=band)
            ax.axhline(0,color='gray',lw=.8);ax.axhline(1,color='gray',lw=.8,ls='--')
            ax.set_xticks(range(5),labels,rotation=15);ax.set_title(f'{model} / recipient {base}')
            ax.set_ylabel('Projection toward donor velocity (not success rate)')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Cross-attention residual interchange | sigma=1 | raw descriptive means\n0=recipient baseline, 1=donor baseline; weak-denominator cases omitted')
    for ext in ['png','pdf']:fig.savefig(out/f'wm_residual_interchange.{ext}',dpi=180)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--allow-partial',action='store_true');ap.add_argument('--render',action='store_true')
    args=ap.parse_args();report,out=summarize(args.output,args.allow_partial)
    if args.render:
        if not report['compute_complete']:raise ValueError('Render final panels after all shards complete')
        plots(report,out);panels(args.output,out)
    print(json.dumps({k:report[k] for k in ['compute_complete','states_verified','fixed_rows','generated_clips']}))


if __name__=='__main__':main()
