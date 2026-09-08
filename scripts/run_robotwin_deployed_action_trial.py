#!/usr/bin/env python3
"""Matched four-arm final200 trial supervising the deployed ten-step action sampler."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
INITIAL_TASKS=['place_empty_cup','move_pillbottle_pad']
TASK_WEIGHTS={'place_a2b_left':.2,'blocks_ranking_rgb':.2}


def completed_source(source):
    read=lambda p:json.loads(p.read_text())
    driver=read(source/'driver.json')
    if not (driver.get('complete') and driver.get('terminal') and driver.get('stage')=='complete'):
        raise ValueError('The source experiment is not complete.')
    launch=read(source.with_name(source.name+'-launch.json'))
    for pid in [launch['pid'],*[v['pid'] for v in driver['jobs'].values()]]:
        proc=Path('/proc')/str(pid)
        try:
            if (proc/'stat').read_text().rsplit(') ',1)[1][0]!='Z' and str(source).encode() in (proc/'cmdline').read_bytes():
                raise ValueError('Source process still live.')
        except FileNotFoundError:pass
    if any(v.get('exit_code')!=0 for v in driver['jobs'].values()):raise ValueError('Source contains failed jobs.')
    if not read(source/'comparison.json')['complete']:raise ValueError('Incomplete source comparison.')
    return read(source/'protocol.json')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--deadline',required=True)
    ap.add_argument('--preflight-only',action='store_true')
    args=ap.parse_args();source=args.source_root.resolve();root=args.output.resolve()
    cutoff=datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or cutoff.timestamp()<=time.time():ap.error('Future absolute deadline required.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Commit code first.')
    from scripts.run_robotwin_expanded_fg_trial import run_action_stages
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    from experiments.robotwin.eraf_action_protocol import validate_joint_identity_audit
    from experiments.robotwin.eraf_fg_training import mixture_stream
    from experiments.robotwin.initial_anchor import source_task, effective_weight
    read=lambda p:json.loads(Path(p).read_text())
    old=completed_source(source)
    diagnostic=source.with_name('20260908-deployed-action-feasibility')
    driver=read(diagnostic/'driver.json')
    if not driver['complete'] or not driver['terminal'] or any(v['exit_code']!=0 for v in driver['jobs'].values()):
        raise ValueError('Actual production/backward feasibility is incomplete.')
    from scripts.run_robotwin_deployed_action_probe import live, compare
    if live(read(diagnostic.with_name(diagnostic.name+'-launch.json'))['pid'],diagnostic) or any(live(v['pid'],diagnostic) for v in driver['jobs'].values()):
        raise ValueError('Diagnostic still active.')
    summaries={a:read(diagnostic/a/'summary.json') for a in ('eraf_only','eraf_fg')}
    actual=compare(summaries)
    if actual!=read(diagnostic/'comparison.json'):raise ValueError('Feasibility summary differs from actual records.')
    for arm in summaries:
        expected=summaries[arm]['input_hashes']
        actual_hashes={'checkpoint':file_sha256(old['arms'][arm]['parent']),
                      'manifest':file_sha256(old['manifest']),
                      'correct_teacher':file_sha256(old['correct_teacher']),
                      'cf_teacher':file_sha256(old['strongest_checkpoint'])}
        if expected!=actual_hashes:raise ValueError('Feasibility did not test these actual inputs.')
    plan=deepcopy(old)
    plan.update(format='robotwin_deployed_action_fourarm_v1',action_objective='deployed_rollout_v1',disk_reserve_GiB=1,record_final_hashes=True,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_trial=str(source),source_action_diagnostic=str(diagnostic),initial_expert_tasks=INITIAL_TASKS,
        historical_semantic_refresh=deepcopy(old['historical_semantic_refresh']),
        semantic_parents={mode:old['arms'][arm]['parent'] for mode,arm in [('ordinary','eraf_only'),('fg','eraf_fg')]},
        semantic_steps=0,semantic_step_offset=1500,semantic_training_this_trial=False,
        correction_task_weights=TASK_WEIGHTS,comparison_history_prefix='pre_deployed_action_',
        prior_comparison_config=str(source/'comparison_config.json'),deadline=args.deadline,
        recipe_change='Replace flow plus noise-endpoint supervision by final24 executed-action fit through all10 production denoising steps. Same-noise full-rollout teacher retention and same-observation final-action conditional difference. All four arms share the objective; same initial-anchor data schedule, rates and weights as completed prior trial.',
        action_recipe='Full-gradient10-step deployed rollout objective. Fresh200 from the same strongest action policy and audited zero-output semantic1500 parents. One initial expert, two ordinary expert, two Correct retention, four CF retention, three FG or matched ordinary slots. No continuation from regressed action weights.',
        checkpoint_selection='Prespecified final200; all prior actual methods retained; no independent-test selection.')
    if file_sha256(plan['manifest'])!=plan['manifest_sha256'] or file_sha256(plan['strongest_checkpoint'])!=plan['source_policy_sha256']:
        raise ValueError('Immutable data or strongest parent changed.')
    for mode,arm in [('ordinary','eraf_only'),('fg','eraf_fg')]:
        validate_joint_identity_audit(read(Path(plan['identity_root'])/mode/'summary.json'),
            checkpoint_sha256=file_sha256(plan['arms'][arm]['parent']),manifest_sha256=plan['manifest_sha256'])
    if any(plan['arms'][a]['parent']!=plan['strongest_checkpoint'] for a in ['no_eraf','fg_only']):
        raise ValueError('Off arms do not share the strongest action initialization.')
    plan['prior_comparison_sha256']=file_sha256(plan['prior_comparison_config'])
    plan['source_comparison_sha256']=file_sha256(source/'comparison.json')
    plan['source_diagnostic_sha256']=file_sha256(diagnostic/'comparison.json')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPUs are owned by another process.')
    # Use the measured previous output sizes, even if retired source weights
    # have since been losslessly compacted on the server.
    ledger=read(source/'final_checkpoint_hashes.json')
    measured=sum(v['bytes'] for v in ledger['models'].values())+sum((source/a/'joint/optimizer_last.pt').stat().st_size for a in plan['arms'])
    plan['reserved_output_bytes']=int(measured*1.03)+128*1024**2
    if shutil.disk_usage(root.parent).free<plan['disk_reserve_GiB']*1024**3+plan['reserved_output_bytes']:
        raise ValueError('Measured final outputs and one-GiB reserve do not fit.')
    rows=read(plan['manifest'])['states'];schedule={}
    for arm,spec in plan['arms'].items():
        stream=mixture_stream(rows,42,spec['fg'],task_balanced=True,correct_count=2,cf_count=4,initial_expert_tasks=INITIAL_TASKS)
        count=Counter();weighted=Counter()
        for _ in range(200):
            batch=next(stream)
            assert sum(bool(r.get('initial_expert_anchor')) for r in batch)==1
            for r in batch:
                if r.get('initial_expert_anchor'):count[source_task(r)]+=1
                if r.get('fg_correction') or r.get('ordinary_cf_control'):
                    weighted[source_task(r)]+=effective_weight(r,1,TASK_WEIGHTS)
        assert dict(count)==dict.fromkeys(INITIAL_TASKS,100)
        schedule[arm]=dict(initial_anchor_draws=dict(count),nominal_weighted_correction_draws=dict(weighted))
    plan['simulated_schedule']=schedule
    if args.preflight_only:
        print(json.dumps(plan,indent=2));return
    root.mkdir(parents=True,exist_ok=False)
    state=dict(complete=False,terminal=False,stage='source_verified',jobs={});processes={}
    def write(name,value):
        p=root/(name+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(root/name)
    write('protocol.json',plan);write('driver.json',state)
    env=os.environ|dict(PATH='/opt/conda/bin:'+os.environ['PATH'],PYTHONPATH=str(REPO/'src')+':'+str(REPO),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
    def budget():
        if time.time()>=cutoff.timestamp():raise TimeoutError('Own process deadline reached; platform stays on.')
        if shutil.disk_usage(root).free<plan['disk_reserve_GiB']*1024**3:raise RuntimeError('One-GiB reserve reached.')
    def stop(signum,frame):raise InterruptedError('Signal'+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(name,cmd,gpus):
        budget()
        with (root/(name+'.log')).open('x') as log:
            p=subprocess.Popen(cmd,cwd=REPO,env=env|{'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus))},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[name]=p;state['jobs'][name]=dict(pid=p.pid,command=cmd,gpus=gpus,exit_code=None)
        write('driver.json',state);return p
    def group(stage,commands):
        state['stage']=stage;active={n:launch(n,c,g) for n,(c,g) in commands.items()}
        while active:
            budget()
            for name,p in list(active.items()):
                rc=p.poll()
                if rc is not None:
                    state['jobs'][name]['exit_code']=rc;del active[name];write('driver.json',state)
                    if rc:raise RuntimeError(name+' failed: '+str(rc))
            if active:time.sleep(5)
    try:
        run_action_stages(plan,root,read(plan['prior_comparison_config']),group=group,write=write,
                          launch=launch,budget=budget,processes=processes,state=state)
    except BaseException as error:
        state.update(stage='stopped',error=repr(error));raise
    finally:
        for p in processes.values():
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for name,p in processes.items():
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            state['jobs'][name]['exit_code']=p.returncode
        state.update(terminal=True,finished_at=datetime.now().astimezone().isoformat());write('driver.json',state)


if __name__=='__main__':main()
