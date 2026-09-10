#!/usr/bin/env python3
"""Paired collection with the exact evaluation texture domain and initial goals."""
from __future__ import annotations
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]


def main():
    import scripts.collect_pgc_robotwin_pairs as collect
    from experiments.robotwin.language_interventions import (
        GoalObserver,load_intervention_manifest,select_intervention_pair)
    from experiments.robotwin.eraf_fg_collection import physical_state
    from experiments.robotwin.pgc_data import array_sha256
    original=collect._load_robotwin_args

    def load(**kwargs):
        task,args=original(**kwargs)
        args['eval_mode']=True
        pair=select_intervention_pair(load_intervention_manifest(
            REPO/'configs/eval/robotwin_cis_ten_tasks.json',robotwin_root=kwargs['robotwin_root']),
            source_task=kwargs['task_name'])
        setup=task.setup_demo
        audit=kwargs['output_root']/'initial_goal_audit.jsonl'
        audit.parent.mkdir(parents=True,exist_ok=True)

        def checked_setup(**kw):
            assert kw['eval_mode'] is True
            result=setup(**kw)
            goals=GoalObserver(task,pair).update()
            row=dict(seed=kw['seed'],episode=kw['now_ep_num'],eval_mode=task.eval_mode,
                source_initial=bool(goals.source.success),target_initial=bool(goals.counterfactual.success),
                physical_state_sha256=array_sha256(physical_state(task)))
            with audit.open('a') as f:f.write(json.dumps(row)+'\n')
            if row['source_initial'] or row['target_initial']:
                raise ValueError('Initial scene already satisfies a tested goal')
            return result

        task.setup_demo=checked_setup
        return task,args

    collect._load_robotwin_args=load
    collect.main()


if __name__=='__main__':main()
