#!/usr/bin/env python3
"""Report paired effects with scene-clustered uncertainty and explicit coverage."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from scripts.probe_robotwin_world_language import read,sha,TASKS
from scripts.run_robotwin_world_language_collection import write
from experiments.robotwin.world_language_probe import metrics,preference


def clustered(values,replicates=5000):
    """values=(scene_id, scalar). First average repeated seeds/states per scene."""
    groups=defaultdict(list)
    for scene,value in values:
        if not np.isfinite(value): raise ValueError('Nonfinite report statistic')
        groups[scene].append(float(value))
    means=np.array([np.mean(v) for _,v in sorted(groups.items())])
    if not len(means):return dict(scenes=0,mean=None,ci95=None)
    rng=np.random.default_rng(20260911)
    samples=rng.choice(means,size=(replicates,len(means)),replace=True).mean(axis=1)
    return dict(scenes=len(means),repeated_observations=len(values),mean=float(means.mean()),
        ci95=[float(x) for x in np.quantile(samples,[.025,.975])])


def verified_state(folder,plan_hash):
    p=folder/'complete.json'
    if not p.exists():return None
    record=read(p)
    if not record['complete'] or record['plan_sha256']!=plan_hash:raise ValueError('State proof/plan mismatch')
    if not record['output_sha256']:raise ValueError('No outputs bound to completed state')
    for name,digest in record['output_sha256'].items():
        if sha(folder/name)!=digest:raise ValueError('State artifact changed: '+str(folder/name))
    return record


def report(root):
    probe=root/'probes';plan=read(probe/'plan.json');plan_hash=sha(probe/'plan.json')
    values=defaultdict(list);counts=defaultdict(int);raw_rows=[]
    def add(model,task,phase,metric,seed,value,**extra):
        values[(model,task,phase,metric)].append((seed,value))
        raw_rows.append(dict(model=model,task=task,phase=phase,metric=metric,scene_seed=seed,value=value,**extra))
    for model in plan['models']:
        for row in plan['states']:
            task,phase,scene=row['task'],row['phase'],row['scene_seed']
            folder=probe/model/'features_cache'/row['id']
            proof=verified_state(folder,plan_hash)
            if proof:
                counts[(model,'features_cache')]+=1
                for name in ['target','source_paraphrase','target_paraphrase','empty','repeat_source']:
                    result=read(folder/(name+'-features.json'))
                    for item in result['comparisons']:
                        if item['kind']=='hidden' and item['layer']==29:
                            add(model,task,phase,name+'_last_hidden_relative_l2',scene,item['relative_l2'])
                    if name=='repeat_source':assert max(x['max_abs'] for x in result['comparisons'])==0
                cache=read(folder/'cache_actions.json')
                assert len(cache['results'])==12 and len(cache['override_crosscheck'])==2
                assert all(x['max_abs']<=1e-5 for x in cache['override_crosscheck'].values())
                assert len({(x['video_language'],x['action_language'],x['seed']) for x in cache['results']})==12
                for seed in plan['noise_seeds']:
                    assert len({x['initial_noise_sha256'] for x in cache['results'] if x['seed']==seed})==1
                with np.load(folder/'actions.npz') as data:
                    for noise in plan['noise_seeds']:
                        A=data[f'v_source_a_source_seed{noise}'];B=data[f'v_target_a_source_seed{noise}']
                        C=data[f'v_source_a_target_seed{noise}'];D=data[f'v_target_a_target_seed{noise}']
                        for name,x,y in [('video_at_source_action',A,B),('video_at_target_action',C,D),
                                         ('action_at_source_video',A,C),('action_at_target_video',B,D),('joint',A,D)]:
                            add(model,task,phase,name+'_action_rms',scene,metrics(x,y)['rms'],noise_seed=noise)
                            if row['dual_reference_valid']:
                                s,t=data['reference_source'],data['reference_target']
                                px=preference(x,s,t)['target_axis_projection'];py=preference(y,s,t)['target_axis_projection']
                                if px is not None and py is not None:
                                    add(model,task,phase,name+'_target_axis_effect',scene,py-px,noise_seed=noise)
                        add(model,task,phase,'interaction_action_rms',scene,metrics(D-B-C+A,np.zeros_like(A))['rms'],noise_seed=noise)
            folder=probe/model/'video'/row['id'];proof=verified_state(folder,plan_hash)
            if proof:
                counts[(model,'video')]+=1
                results=read(folder/'video_losses.json')
                expected=(2 if row['dual_reference_valid'] else 1)*len(plan['sigmas'])*len(plan['noise_seeds'])
                assert len(results['rows'])==expected
                keys=[(r['reference'],r['sigma'],r['seed']) for r in results['rows']]
                assert len(set(keys))==expected
                assert results['joint_equivalence']['max_abs']<=.02
                for r in results['rows']:
                    for metric in ['wrong_minus_correct_mse','wrong_minus_correct_roi_mse']:
                        add(model,task,phase,f'video_{metric}_sigma{r["sigma"]}',scene,r[metric],
                            noise_seed=r['seed'],reference=r['reference'])
                for seed in plan['noise_seeds']:
                    hashes={r['noisy_sha256'] for r in results['rows'] if r['sigma']==1 and r['seed']==seed}
                    assert len(hashes)==1,'Pure video noise must match for shared-state reference pairs'
                generated=read(folder/'generations.json')['generations'];assert len(generated)==6
                assert {(g['language'],g['seed']) for g in generated}=={(l,s) for l in ['source','target'] for s in plan['noise_seeds']}
                assert all(g['frames']==9 for g in generated)
    closed={};input_hashes={};paired_outcomes={}
    for model in plan['models']:
        for seed in plan['noise_seeds']:
            for v in ['source','target']:
                for a in ['source','target']:
                    name=f'{model}-closed-v_{v}-a_{a}-seed{seed}';folder=root/'closed_loop'/name
                    marker=folder/'world_language_complete.json'
                    if not marker.exists():continue
                    assert read(marker)==dict(complete=True,video_language=v,noise_seed=seed,
                        intervention='Video text only; action text and all other production inputs retained.')
                    inputs=[json.loads(x) for x in (folder/'world_language_inputs.jsonl').read_text().splitlines()]
                    assert len(inputs)==50
                    by_key={(x['source_task'],x['scene_seed']):x for x in inputs};assert len(by_key)==50
                    rows=[]
                    for task in TASKS:
                        cell=folder/task/'demo_randomized'/('correct' if a=='source' else 'counterfactual')
                        proof=read(cell/'complete.json');assert proof['complete'] and proof['manipulation_metrics']
                        assert proof['checkpoint_sha256']==plan['checkpoint_sha256'][model]
                        records=[json.loads(x) for x in (cell/'episodes.jsonl').read_text().splitlines()]
                        canonical=[json.loads(x) for x in (probe/'catalog'/task/'demo_randomized/correct/episodes.jsonl').read_text().splitlines()]
                        initial=read(cell/'initial_states.json');assert len(initial)==len(records)==len(canonical)==10
                        for x,c,h in zip(records,canonical,initial,strict=True):
                            for key in ['scene_seed','episode_index','source_instruction','counterfactual_instruction']:
                                assert x[key]==c[key]
                            assert h['sha256']==c['initial_physical_state_sha256']
                            key=(task,x['scene_seed']);inp=by_key[key]
                            digest=inp['initial_observation_sha256']
                            assert input_hashes.setdefault(key,digest)==digest,'Different initial observations across arms/models/noise seeds'
                            assert inp['noise_seed']==seed and inp['video_language']==v
                            assert inp['action_instruction']==c['source_instruction' if a=='source' else 'counterfactual_instruction']
                            assert inp['video_instruction']==c['source_instruction' if v=='source' else 'counterfactual_instruction']
                            for metric in ['source_goal_ever_success','counterfactual_goal_ever_success']:
                                add(model,task,f'closed_v_{v}_a_{a}',metric,x['scene_seed'],float(x[metric]),noise_seed=seed)
                                paired_outcomes[(model,task,x['scene_seed'],seed,v,a,metric)]=float(x[metric])
                            for metric in ['any_correct_object_lifted','all_instruction_objects_lifted','full_goal_after_any_lift','correct_placement_after_lift']:
                                add(model,task,f'closed_v_{v}_a_{a}',metric,x['scene_seed'],float(x['manipulation_metrics'][metric]),noise_seed=seed)
                            rows.append(x)
                    closed[name]=dict(episodes=len(rows),source_success=sum(x['source_goal_ever_success'] for x in rows),
                        cf_success=sum(x['counterfactual_goal_ever_success'] for x in rows))
    for scene in plan['scenes']:
        task,s=scene['task'],scene['scene_seed']
        for model in plan['models']:
            for seed in plan['noise_seeds']:
                for metric in ['source_goal_ever_success','counterfactual_goal_ever_success']:
                    def get(v,a):return paired_outcomes.get((model,task,s,seed,v,a,metric))
                    A,B,C,D=get('source','source'),get('target','source'),get('source','target'),get('target','target')
                    if None not in [A,B,C,D]:
                        for name,val in [('video_at_source_action',B-A),('video_at_target_action',D-C),
                                         ('action_at_source_video',C-A),('action_at_target_video',D-B),('interaction',D-B-C+A)]:
                            add(model,task,'closed_paired',name+'_'+metric,s,val,noise_seed=seed)
    summary=[dict(model=m,task=t,phase=p,metric=k,**clustered(v)) for (m,t,p,k),v in sorted(values.items())]
    complete_compute=all(counts[(m,e)]==len(plan['states']) for m in plan['models'] for e in ['features_cache','video']) and len(closed)==24
    out=root/'report';out.mkdir(exist_ok=True)
    output=dict(format='robotwin_world_language_report_v1',complete=False,compute_coverage_complete=complete_compute,
        state_coverage={m:{e:counts[(m,e)] for e in ['features_cache','video']} for m in plan['models']},
        expected_states_per_model_experiment=len(plan['states']),closed_loop=closed,
        expected_closed_loop_episodes=1200,actual_closed_loop_episodes=sum(x['episodes'] for x in closed.values()),
        statistics=summary,uncertainty='Average repeated noise seeds and states within each scene; percentile bootstrap across scenes.',
        pending='Independent input/code/process audit and generated-video semantic scoring remain required.')
    write(out/'quantitative_report.json',output)
    with (out/'statistics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['model','task','phase','metric','scenes','repeated_observations','mean','ci95'])
        w.writeheader();w.writerows(summary)
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    r=report(a.root);print(json.dumps({k:v for k,v in r.items() if k not in ['statistics','closed_loop']}))
