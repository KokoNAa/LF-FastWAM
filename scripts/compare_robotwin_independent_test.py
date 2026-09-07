#!/usr/bin/env python3
"""Compare complete paired 100-scene test matrices without changing CF predicates."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from experiments.robotwin.catalog_protocol import NAMESPACES, excluded_scene_records
from scripts.compare_robotwin_eraf_fg_evaluations import compare,evaluation
from scripts.compare_robotwin_ten_task_methods import GROUPS

TASKS={t for _,tasks in GROUPS.values() for t in tasks}


def build_report(config):
    """The caller must separately prove checkpoint selection preceded test access."""
    if config['target'] not in config['methods'] or len(config['methods'])<2:
        raise ValueError('Require a target and actual controls.')
    blocked,receipts=excluded_scene_records(config['exclude_records'],split='test')
    lower,upper=NAMESPACES['test']
    scores={};slots={};bindings={}
    for name,root in config['methods'].items():
        summary,evidence=evaluation(root)
        if set(evidence)!={(t,'counterfactual') for t in TASKS}:
            raise ValueError('Test requires exactly ten complete CF task cells.')
        cells={}
        all_seeds=set()
        for (task,_),(episodes,initial,meta) in evidence.items():
            seeds=[r['scene_seed'] for r in episodes]
            if len(episodes)!=10 or len(set(seeds))!=10:
                raise ValueError('Each test task requires ten unique scenes.')
            if any(type(s) is not int or not lower<=s<upper or s in blocked for s in seeds):
                raise ValueError('Test namespace or training/development exclusion violation.')
            if all_seeds.intersection(seeds):raise ValueError('Test tasks reuse a scene seed.')
            all_seeds.update(seeds)
            if any(type(r['selected_goal_success']) is not bool or r['selected_goal']!='counterfactual'
                   or r['instruction_goal']!='counterfactual' or r['condition']!='counterfactual'
                   or r['policy_instruction']!=r['counterfactual_instruction'] for r in episodes):
                raise ValueError('CF instruction or success contract changed.')
            cells[task]=dict(episodes=10,successes=sum(r['selected_goal_success'] for r in episodes))
            if task=='place_burger_fries':
                strict=simultaneous=0
                for row in episodes:
                    source={a['actor']:a for a in row['final_source_goal']['assignments']}
                    cf={a['actor']:a for a in row['final_counterfactual_goal']['assignments']}
                    if source.keys()!=cf.keys() or len(cf)!=2:raise ValueError('Missing slot geometry.')
                    strict+=bool(row['counterfactual_goal_final_success'] and all(
                        a['planar_distance']<source[actor]['planar_distance'] for actor,a in cf.items()))
                    simultaneous+=bool(row['source_goal_final_success'] and row['counterfactual_goal_final_success'])
                slots[name]=dict(episodes=10,strict_final_slot_successes=strict,simultaneous_final_goals=simultaneous)
        scores[name]=dict(ten_task_macro_cf=float(sum((Fraction(c['successes'],10) for c in cells.values()),Fraction())/10),
            cells=cells,episodes=100,successes=sum(c['successes'] for c in cells.values()))
        bindings[name]=summary['checkpoint']
    paired={};target=config['target']
    for name,root in config['methods'].items():
        if name==target:continue
        report,episodes=compare(root,config['methods'][target])
        paired[name]=dict(report=report,episodes=episodes,
            gains=sum(len(c['gained_scene_seeds']) for c in report['cells']),
            losses=sum(len(c['lost_scene_seeds']) for c in report['cells']),
            macro_delta=scores[target]['ten_task_macro_cf']-scores[name]['ten_task_macro_cf'],
            regressed_tasks=[c['task'] for c in report['cells'] if c['delta_successes']<0])
    return dict(complete=True,format='robotwin_independent_test_comparison_v1',target=target,methods=scores,
        checkpoint_paths=bindings,exclusion_receipts=receipts,paired_target_vs_controls=paired,
        supplementary_final_slots=slots,all_scene_seeds_in_reserved_test_namespace=True,
        target_strictly_exceeds_all_listed_controls=all(v['macro_delta']>0 for v in paired.values()),
        checkpoint_selection_freeze_verified=False,goal_achievement_claim=False,
        scope='Complete paired test matrices with explicit train/dev seed exclusions. Caller must separately audit pre-test checkpoint selection, catalog expert feasibility, file hashes and controller provenance. No statistical or component-causality claim.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=build_report(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({n:r['ten_task_macro_cf'] for n,r in report['methods'].items()},indent=2))


if __name__=='__main__':main()
