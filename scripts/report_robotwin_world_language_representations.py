#!/usr/bin/env python3
"""Report every video layer and spatial location after complete feature coverage."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from scripts.probe_robotwin_world_language import read,sha
from scripts.report_robotwin_world_language import verified_state
from scripts.run_robotwin_world_language_collection import write

METRICS=['rms','relative_l2','cosine_distance','max_abs']
COMPARISONS=['target','source_paraphrase','target_paraphrase','empty','repeat_source']


def feature_matrix(comparisons,kind):
    rows=sorted((x for x in comparisons if x['kind']==kind),key=lambda x:x['layer'])
    if [x['layer'] for x in rows]!=list(range(30)):raise ValueError('Incomplete video layers')
    result=np.asarray([[x[k] for k in METRICS]+x['token_delta_rms'] for x in rows],dtype=np.float64)
    # WanVideoVAE38 reduces 384x320 by16; DiT patch_size=(1,2,2).
    # pre_dit flattens (frame,height,width), hence 12x10, not VAE's24x20.
    if result.shape!=(30,4+120) or not np.isfinite(result).all():
        raise ValueError('Unexpected feature spatial shape or nonfinite values')
    return result


def report(root):
    probe=root/'probes';plan=read(probe/'plan.json');plan_hash=sha(probe/'plan.json')
    identity_path=root/'checkpoint_identity_audit.json';identity=read(identity_path)
    if (identity['plan_sha256']!=plan_hash or identity['selected_checkpoints']!=plan['checkpoints']
            or identity['selected_checkpoint_sha256']!=plan['checkpoint_sha256']
            or identity['no_eraf_base_checkpoint']!=plan['checkpoints']['released']):
        raise ValueError('Checkpoint identity audit does not match the frozen experiment')
    groups={}
    for model in plan['models']:
        for state in plan['states']:
            folder=probe/model/'features_cache'/state['id']
            proof=verified_state(folder,plan_hash)
            if not proof:raise ValueError('All-layer report requires complete feature coverage')
            # Frozen worker run_features stores model.lora_base_checkpoint here.
            # Both primary models share the released base; selected adapter identity
            # is bound by the plan hash and controller's --model/load_policy path.
            if proof['checkpoint']!=plan['checkpoints']['released']:raise ValueError('Wrong base checkpoint')
            for comparison in COMPARISONS:
                data=read(folder/(comparison+'-features.json'))['comparisons']
                for kind in ['hidden','k','v']:
                    matrix=feature_matrix(data,kind)
                    if comparison=='repeat_source' and np.max(np.abs(matrix[:,:2]))!=0:
                        raise ValueError('Identical-input control failed')
                    key=(model,state['task'],state['phase'],comparison,kind)
                    group=groups.setdefault(key,dict(scenes=set(),sum=np.zeros_like(matrix),sumsq=np.zeros((30,4))))
                    if state['scene_seed'] in group['scenes']:
                        raise ValueError('More than one feature state per scene and phase')
                    group['scenes'].add(state['scene_seed']);group['sum']+=matrix;group['sumsq']+=matrix[:,:4]**2
    rows=[];maps={};map_records=[]
    for index,(key,group) in enumerate(sorted(groups.items())):
        model,task,phase,comparison,kind=key;n=len(group['scenes']);mean=group['sum']/n
        variance=np.maximum(group['sumsq']-n*mean[:,:4]**2,0)/(n-1) if n>1 else np.zeros((30,4))
        for layer in range(30):
            row=dict(model=model,task=task,phase=phase,comparison=comparison,kind=kind,layer=layer,scenes=n)
            for j,metric in enumerate(METRICS):row[metric+'_mean']=float(mean[layer,j]);row[metric+'_scene_sd']=float(variance[layer,j]**.5)
            rows.append(row)
        ident=f'map_{index:04d}';maps[ident]=mean[:,4:].reshape(30,12,10).astype(np.float32)
        map_records.append(dict(key=ident,model=model,task=task,phase=phase,comparison=comparison,kind=kind,scenes=n))
    out=root/'report';out.mkdir(exist_ok=True)
    with (out/'layer_profiles.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/'representation_maps.npz',**maps)
    result=dict(complete=True,plan_sha256=plan_hash,states_per_model=len(plan['states']),layer_rows=len(rows),maps=map_records,
        checkpoint_identity_audit_sha256=sha(identity_path),
        selected_checkpoints=plan['checkpoints'],selected_checkpoint_sha256=plan['checkpoint_sha256'],
        feature_proof_checkpoint_semantics='checkpoint denotes the shared LoRA base; selected checkpoint identities are listed separately.',
        map_axes=['layer','height','width'],map_shape=[30,12,10],camera_rows=dict(head=[0,8],wrists=[8,12]),
        camera_columns=dict(left_wrist=[0,5],right_wrist=[5,10]),
        interpretation='Mean token RMS difference across scenes in each task/phase. Spatial magnitude is sensitivity, not causal attention or semantic correctness. CSV scene_sd is scene variability, not a confidence interval; endpoint confidence intervals are in the quantitative report.',
        output_sha256={name:sha(out/name) for name in ['layer_profiles.csv','representation_maps.npz']})
    write(out/'representation_summary.json',result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    result=report(args.root);print(json.dumps({k:v for k,v in result.items() if k!='maps'}))
