#!/usr/bin/env python3
"""Freeze actual development-selected models before an independent test is created."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from experiments.robotwin.catalog_protocol import excluded_scene_records
from scripts.compare_robotwin_ten_task_methods import build_report,score_cells
from scripts.compare_robotwin_eraf_fg_evaluations import evaluation,file_binding

REQUIRED=('eraf_fg','no_eraf','fg_only','eraf_only','historical_strongest_no_eraf',
          'historical_best_fg_only','pre_cross_goal_eraf_only')


def select_methods(report):
    if not report.get('complete') or report.get('target')!='eraf_fg':
        raise ValueError('Require a complete ERAF+FG development comparison.')
    methods=report['methods']
    if not set(REQUIRED)<=set(methods):raise ValueError('Required matched or historical controls are missing.')
    scores={name:score_cells(row['cells']) for name,row in methods.items()}
    if any(scores['eraf_fg']<=v for name,v in scores.items() if name!='eraf_fg'):
        raise ValueError('Candidate does not strictly exceed every declared development control.')
    best=max(v for name,v in scores.items() if name!='eraf_fg')
    selected=set(REQUIRED)|{name for name,v in scores.items() if name!='eraf_fg' and v==best}
    return sorted(selected),scores


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def metadata(path):
    path=Path(path).resolve();stat=path.stat()
    return dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def freeze(source,test_root):
    source=Path(source).resolve();test_root=Path(test_root).resolve()
    if test_root.exists():raise ValueError('Create the selection before creating or opening the declared test run.')
    read=lambda p:json.loads(Path(p).read_text())
    driver=read(source/'driver.json')
    if not driver.get('complete') or not driver.get('terminal') or any(j['exit_code']!=0 for j in driver['jobs'].values()):
        raise ValueError('Development controller has not completed successfully.')
    pid=read(source.with_name(source.name+'-launch.json'))['pid']
    try:state=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()[0]
    except FileNotFoundError:state=None
    if state not in (None,'Z'):raise ValueError('Development controller is still live.')
    config=read(source/'comparison_config.json');report=build_report(config)
    if report!=read(source/'comparison.json'):raise ValueError('Actual episode evidence differs from the stored comparison.')
    selected,scores=select_methods(report);plan=read(source/'protocol.json')
    if sha(plan['manifest'])!=plan['manifest_sha256']:raise ValueError('Training manifest changed.')
    catalogs=[str(Path(group['catalog'])/task/'demo_clean/correct/episodes.jsonl')
              for group in plan['groups'].values() for task in group['tasks']]
    _,exclusions=excluded_scene_records([plan['manifest'],*catalogs],split='test')
    models={}
    for name in selected:
        checkpoint=Path(report['checkpoint_paths'][name]);before=metadata(checkpoint);digest=sha(checkpoint)
        if metadata(checkpoint)!=before:raise ValueError('Checkpoint changed during hashing.')
        signatures=set()
        for group,path in config['methods'][name].items():
            _,cells=evaluation(path)
            for _,(_,_,complete) in cells.items():
                binding=file_binding(complete,'checkpoint')
                expected=('metadata',before['path'],before['size'],before['mtime_ns']) if binding[0]=='metadata' else ('sha256',digest)
                if binding!=expected:raise ValueError('Actual selected model differs from the development-evaluated identity.')
                signatures.add((complete['eraf'],complete['policy_kind'],complete.get('memory_mode','carry'),
                                json.dumps(complete['deployment'],sort_keys=True)))
        if len(signatures)!=1:raise ValueError('Selected method has mixed deployment settings.')
        eraf,kind,memory,deployment=signatures.pop()
        models[name]=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,checkpoint_metadata=before,
            eraf=eraf,policy_kind=kind,memory_mode=memory,deployment=json.loads(deployment),
            development_macro_cf=scores[name],development_groups=config['methods'][name])
    return dict(complete=True,format='robotwin_independent_model_selection_v1',
        frozen_at=datetime.now().astimezone().isoformat(),test_root=str(test_root),test_root_absent_at_freeze=True,
        source_root=str(source),source_controller_pid=pid,source_controller_terminal=True,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_artifact_sha256={name:sha(source/name) for name in ('protocol.json','driver.json','comparison_config.json','comparison.json')},
        target='eraf_fg',models=models,all_declared_development_scores=scores,exclusion_receipts=exclusions,
        manifest=plan['manifest'],manifest_sha256=plan['manifest_sha256'],
        selection_rule='Require strict lead over every complete declared DEV method; include matched three ablations, named historical strongest controls and every globally strongest DEV control tied at the best control score.',
        independent_namespace=[91700000,91800000],episodes_per_task=10,
        no_test_outcomes_consumed_by_this_tool=True,policy_test_evaluated=False,goal_achievement_claim=False,
        scope='This receipt binds actual full models and development evidence before the declared test directory exists. It cannot prove that no test outcomes were previously accessed elsewhere; collection/controller chronology and complete exclusions still need audit. Compacted or reserialized models need an additional identity audit and are not silently accepted by this tool.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--test-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.output.resolve().is_relative_to(args.test_root.resolve()):
        ap.error('Keep the selection receipt outside the as-yet-uncreated test directory.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Commit first.')
    result=freeze(args.source_root,args.test_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,models=list(result['models']),test_started=False)))


if __name__=='__main__':main()
