#!/usr/bin/env python3
"""Plot completed video sensitivity and paired denoising evidence, not success."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASKS=['blocks_ranking_rgb','stack_blocks_two','place_a2b_left','place_a2b_right','place_burger_fries']
LABELS=['Ranking','Stacking','Place left','Place right','Burger / fries']
PHASES={t:'initial' if i<2 else 'shared_decision' for i,t in enumerate(TASKS)}
MODELS=['released','no_eraf']
COLORS=['#2467a5','#dd8534']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quantitative',type=Path,required=True)
    ap.add_argument('--pixel',type=Path,required=True)
    ap.add_argument('--reference-audit',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();q=read(args.quantitative);p=read(args.pixel);audit=read(args.reference_audit)
    assert p['complete'] and audit['complete'] and p['plan_sha256']==audit['plan_sha256']
    assert q['expected_states_per_model_experiment']==280
    assert all(q['state_coverage'][m]['video']==280 for m in MODELS)
    assert q['paired_video']['unpaired_metric_observations']==0
    for t in TASKS:
        cells=[r for r in audit['rows'] if r['task']==t and r['phase']==PHASES[t]]
        assert len(cells)==10 and all(not r['identical_action_window'] for r in cells)
    used={};out=args.output;out.mkdir(parents=True,exist_ok=True);files=[]
    def stat(kind,task,metric,model=None):
        rows={'pixel':p['statistics'],'fit':q['statistics'],'paired':q['paired_video']['statistics']}[kind]
        hits=[r for r in rows if r['task']==task and r['phase']==PHASES[task] and r['metric']==metric and (model is None or r['model']==model)]
        assert len(hits)==1 and hits[0]['scenes']==10
        r=hits[0];assert np.isfinite([r['mean'],*r['ci95']]).all()
        used[kind,model,task,metric]=dict(source=kind,**r)
        return r
    def bars(ax,rows,pos,width,label,color):
        y=np.array([r['mean'] for r in rows]);ci=np.array([r['ci95'] for r in rows])
        ax.bar(pos,y,width=width,color=color,label=label,yerr=[y-ci[:,0],ci[:,1]-y],error_kw=dict(capsize=3,elinewidth=.8))
    def finish(fig,name,footer,rect):
        fig.text(.5,.025,footer,ha='center',fontsize=9)
        fig.tight_layout(rect=rect)
        for ext in ['png','pdf']:
            f=out/f'{name}.{ext}';fig.savefig(f,dpi=160);files.append(dict(path=str(f),sha256=sha(f)))
        plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,model in zip(axes,MODELS):
        for j,(metric,label) in enumerate([('language_task_roi_rmse','Change language; same noise'),('noise_task_roi_rmse','Change noise; same language')]):
            rows=[stat('pixel',t,metric,model) for t in TASKS]
            bars(ax,rows,np.arange(5)+(j-.5)*.34,.32,label,COLORS[j])
        ax.set_title(model.replace('_','-'));ax.set_xticks(range(5),LABELS,rotation=15)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Generated-video task-region RGB RMSE (0–1)')
    fig.suptitle('Video sensitivity: instruction change versus sampling noise',y=.99)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,frameon=False)
    finish(fig,'video_language_vs_noise','Ranking/stacking: initial state. Placement: shared decision. 10 scenes/task; 95% scene-bootstrap intervals.\nFixed first frame excluded. Image differences measure sensitivity, not goal correctness; these are not paired significance tests.',[0,.13,1,.87])

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,metric,title in zip(axes,['correct_roi_mse','wrong_minus_correct_roi_mse'],['Change in correct-language prediction error','Change in wrong-minus-correct separation']):
        for j,ref in enumerate(['source','target']):
            rows=[stat('paired',t,f'video_ref_{ref}_{metric}_sigma1.0') for t in TASKS]
            bars(ax,rows,np.arange(5)+(j-.5)*.34,.32,ref+' expert future',COLORS[j])
        ax.axhline(0,c='#555555',lw=.8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_title(title);ax.set_xticks(range(5),LABELS,rotation=15)
        ax.set_ylabel('no-eraf minus released: task-region velocity MSE')
    fig.suptitle('Paired model differences at pure video noise (sigma = 1)',y=.99)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,frameon=False)
    finish(fig,'video_paired_fit_tradeoff','Matched state, expert reference and noise before subtraction; 10 scenes/task, 3 noise draws; 95% scene-bootstrap intervals.\nLeft: negative means better fit. Right: positive means greater language separation, which can coexist with worse correct-language fit.',[0,.13,1,.86])

    fig,axes=plt.subplots(1,5,figsize=(16,4.7),sharey=True)
    sigmas=[.2,.5,.8,1.]
    for ax,t,label in zip(axes,TASKS,LABELS):
        for mi,m in enumerate(MODELS):
            for ref,style in [('source','-'),('target','--')]:
                rows=[stat('fit',t,f'video_ref_{ref}_wrong_minus_correct_roi_mse_sigma{s}',m) for s in sigmas]
                ys=np.array([r['mean'] for r in rows]);ci=np.array([r['ci95'] for r in rows])
                ax.plot(sigmas,ys,style,color=COLORS[mi],marker='o',ms=3,label=m.replace('_','-')+' / '+ref)
                ax.fill_between(sigmas,ci[:,0],ci[:,1],color=COLORS[mi],alpha=.09)
        ax.axhline(0,c='#555555',lw=.7);ax.set_title(label);ax.set_xticks(sigmas);ax.set_xlabel('Video sigma');ax.grid(alpha=.15)
    axes[0].set_ylabel('Wrong minus correct: task-region velocity MSE')
    fig.suptitle('Language discrimination across video denoising levels',y=.99)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=4,frameon=False)
    finish(fig,'video_denoising_margins','Positive: matching language predicts the expert future better than mismatched language. Shading: 95% scene-bootstrap intervals.\nRanking/stacking: initial; placement: shared decision. 10 scenes/task, 3 noise draws. Conditional fit is not generated-video or closed-loop success.',[0,.14,1,.84])
    data=dict(plan_sha256=p['plan_sha256'],sources={str(f):sha(f) for f in [args.quantitative,args.pixel,args.reference_audit]},phases=PHASES,statistics=list(used.values()))
    dp=out/'figure_data.json';dp.write_text(json.dumps(data,indent=2)+'\n')
    manifest=dict(complete=True,figures=3,files=files,figure_data_sha256=sha(dp),scope='Completed video numerical experiments only. Qualitative review and closed-loop outcomes remain separate. No multiple-comparison correction or independent noise-seed sample interpretation.')
    (out/'figures.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(figures=3,output=str(out),statistics=len(used))))

if __name__=='__main__':main()
