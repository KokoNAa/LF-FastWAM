#!/usr/bin/env python3
"""Run six bounded, server-only FG collection pilots with explicit ownership."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src')]


def pilot_jobs():
    from experiments.robotwin.expanded_fg import TASKS
    return [dict(name=task, task=task, split='train', start_seed=86000000+1000*i, gpu=i)
            for i, task in enumerate(TASKS)] + [dict(name='cup_holdout', task='place_empty_cup',
                split='replay_holdout', start_seed=87000000, gpu=5)]


def expansion_jobs():
    jobs = []
    for task, offset in [('place_empty_cup', 0), ('move_pillbottle_pad', 4000)]:
        for part, split, seed, scenes in [('train_a', 'train', 86000100+offset, 12),
                ('train_b', 'train', 86000300+offset, 12),
                ('holdout', 'replay_holdout', 87000100+offset, 6)]:
            jobs.append(dict(name=task+'_'+part, task=task, split=split,
                start_seed=seed, scenes=scenes, attempts=120, gpu=len(jobs)))
    return jobs


def audit_pilot_evidence(pilot_root, mask_root, checkpoint_sha256, manifest_sha256):
    """Bind actual full-goal captures and exact mask replays before expansion."""
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    from experiments.robotwin.eraf_fg_contract import validate_correction
    driver = json.loads((pilot_root/'driver.json').read_text())
    protocol = json.loads((pilot_root/'protocol.json').read_text())
    if not driver.get('complete') or not driver.get('terminal') or any(
            j.get('exit_code') != 0 for j in driver['jobs'].values()):
        raise ValueError('Complete six-pilot collection evidence is required.')
    if (protocol['checkpoint_sha256'] != checkpoint_sha256
            or protocol['manifest_sha256'] != manifest_sha256):
        raise ValueError('Expansion inputs differ from the pilot evidence.')
    rows = []
    hashes = {str(p): file_sha256(p) for p in [pilot_root/'driver.json', pilot_root/'protocol.json']}
    for name, job in driver['jobs'].items():
        if (Path('/proc')/str(job['pid'])).exists():
            raise ValueError('Pilot worker PID still exists; inspect before expanding.')
        path = pilot_root/name/'manifest.json'; manifest = json.loads(path.read_text())
        if manifest.get('complete') is not True or len(manifest['records']) != 1:
            raise ValueError('Missing complete pilot collection.')
        hashes[str(path)] = file_sha256(path)
        rows.extend(validate_correction(row) for row in manifest['records'])
    launch = json.loads((mask_root/'launch.json').read_text())
    catalog_path = mask_root/'capture_manifest.json'
    if file_sha256(catalog_path) != launch['capture_manifest_sha256']:
        raise ValueError('Mask replay input catalog changed.')
    catalog = json.loads(catalog_path.read_text())
    selected = {r['frame_path']: r for r in catalog['states']}
    originals = {r['frame_path']: r for r in rows}
    for path, r in selected.items():
        if path not in originals or any(r.get(k) != v for k, v in originals[path].items()):
            raise ValueError('Mask pilot catalog differs from original correction.')
    masked = set()
    for name, job in launch['jobs'].items():
        if (Path('/proc')/str(job['pid'])).exists():
            raise ValueError('Mask pilot process still exists.')
        exit_path = mask_root/(name+'.exit.json')
        if json.loads(exit_path.read_text())['exit_code'] != 0:
            raise ValueError('Mask replay pilot failed.')
        path = mask_root/name/'complete.json'; report = json.loads(path.read_text())
        plan = json.loads(path.with_name('plan.json').read_text())
        if not report.get('complete') or plan['manifest_sha256'] != file_sha256(catalog_path):
            raise ValueError('Mask pilot must bind the exact capture catalog.')
        hashes[str(path)] = file_sha256(path)
        for r in report['scenes']:
            original = originals[r['frame_path']]
            if (r['frame_path'] in masked or not r['complete'] or not r['all_rgb_frames_equal']
                    or r['phase_labels_valid'] is not False or r['temporal_labels_valid'] is not False
                    or r['replay_state_max_abs'] > 1e-7):
                raise ValueError('Invalid or duplicate full mask replay evidence.')
            for key in ('pair_id','task_config','scene_seed','replay_split','frame_sha256','controls_sha256'):
                if r[key] != original[key]:raise ValueError('Mask replay identity mismatch: '+key)
            if file_sha256(r['labels']) != r['labels_sha256']:
                raise ValueError('Mask label artifact changed.')
            masked.add(r['frame_path'])
    if masked != set(selected):raise ValueError('Incomplete mask pilot coverage.')
    required = {('place_empty_cup','train'), ('place_empty_cup','replay_holdout'), ('move_pillbottle_pad','train')}
    if not required.issubset({(originals[p]['source_task'], originals[p]['replay_split']) for p in masked}):
        raise ValueError('Cup/pill expansion lacks required physical and mask pilots.')
    hashes[str(catalog_path)] = file_sha256(catalog_path)
    hashes[str(mask_root/'launch.json')] = file_sha256(mask_root/'launch.json')
    return rows, hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'checkpoint', 'output', 'robotwin-root'):
        ap.add_argument('--'+key, required=True, type=Path)
    for key in ('checkpoint-sha256', 'manifest-sha256', 'deadline'):
        ap.add_argument('--'+key, required=True)
    ap.add_argument('--mode', choices=['pilot', 'cup_pill_expansion'], default='pilot')
    ap.add_argument('--pilot-root', type=Path)
    ap.add_argument('--mask-root', type=Path)
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or not 0 < cutoff.timestamp()-time.time() <= 3600:
        ap.error('Require a future absolute pilot cutoff within one hour.')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Commit pilot code before starting.')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('GPUs are owned by another job.')
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    from experiments.robotwin.expanded_fg import validate_scope
    from scripts.collect_robotwin_eraf_fg import historical_scene_keys
    if file_sha256(args.checkpoint) != args.checkpoint_sha256 or file_sha256(args.manifest) != args.manifest_sha256:
        raise ValueError('Pilot checkpoint or input manifest identity changed.')
    manifest = json.loads(args.manifest.read_text())
    excluded = historical_scene_keys(manifest['states'])
    evidence = {}
    if args.mode == 'cup_pill_expansion':
        if args.pilot_root is None or args.mask_root is None:
            ap.error('Expansion requires completed collection and mask pilot roots.')
        pilot_rows, evidence = audit_pilot_evidence(args.pilot_root, args.mask_root,
            args.checkpoint_sha256, args.manifest_sha256)
        excluded |= historical_scene_keys(pilot_rows)
        jobs = expansion_jobs()
    else:
        jobs = [job | dict(scenes=1, attempts=3) for job in pilot_jobs()]
    for job in jobs:
        validate_scope(job['task'], job['split'], job['start_seed'], job['attempts'])
        if any((job['task'], 'demo_clean', seed) in excluded for seed in range(job['start_seed'], job['start_seed']+job['attempts'])):
            raise ValueError('Pilot scene was already used in the input manifest.')
    root = args.output.resolve()
    allocation = 4. if args.mode == 'cup_pill_expansion' else 1.5
    if shutil.disk_usage(root.parent).free < (3.+allocation)*1024**3:
        raise RuntimeError('Require 3GiB reserve plus the declared collection allocation.')
    root.mkdir(exist_ok=False)
    state = dict(complete=False, terminal=False, stage='starting', jobs={})
    processes = {}
    def write(name, data):
        temp=root/(name+'.tmp');temp.write_text(json.dumps(data, indent=2)+'\n');temp.replace(root/name)
    protocol = dict(format='robotwin_expanded_fg_pilots_v1',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        manifest=str(args.manifest.resolve()), manifest_sha256=args.manifest_sha256,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=args.checkpoint_sha256,
        jobs=jobs, mode=args.mode, requested_scenes=sum(j['scenes'] for j in jobs), candidates_per_scene=6,
        candidate_order='late_first', policy_kind='repair', eraf_mode='on', memory_mode='carry',
        deadline=args.deadline, disk_reserve_GiB=3, platform_shutdown=None,
        pilot_evidence_sha256=evidence, collection_allocation_GiB=allocation,
        scope='Real failed-policy full-goal correction collection. No model training, no policy evaluation score, no goal achievement. Expansion uses fresh seeds, excludes pilot scenes, and retains unchanged full-goal and exact replay criteria.',
        model_storage='server_only')
    write('protocol.json', protocol);write('driver.json', state)
    def stop(signum, frame):raise InterruptedError('Pilot interrupted: '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env=os.environ | dict(PATH='/opt/conda/bin:'+os.environ['PATH'], PYTHONPATH=str(REPO/'src')+':'+str(REPO),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
        DIFFSYNTH_MODEL_BASE_PATH='/root/gpufree-data/fastwam/FastWAM/checkpoints')
    try:
        for job in jobs:
            name=job['name']
            cmd=[sys.executable, str(REPO/'scripts/collect_robotwin_eraf_fg.py'),
                '--manifest',str(args.manifest),'--checkpoint',str(args.checkpoint),
                '--robotwin-root',str(args.robotwin_root),'--output',str(root/name),
                '--task',job['task'],'--gpu',str(job['gpu']),'--protocol','expanded_v2',
                '--split',job['split'],'--start-seed',str(job['start_seed']),
                '--scenes',str(job['scenes']),'--holdout-scenes','0','--max-attempts',str(job['attempts']),'--candidates','6',
                '--policy-kind','repair','--eraf','on','--deadline',args.deadline,'--reserve-gib','3']
            with (root/(name+'.log')).open('x') as log:
                process=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[name]=process
            state['jobs'][name]=dict(pid=process.pid, command=cmd, exit_code=None)
            write('driver.json',state)
        state['stage']='collecting'
        while True:
            active=[]
            for name, process in processes.items():
                state['jobs'][name]['exit_code']=process.poll()
                if process.returncode is None:active.append(name)
            write('driver.json',state)
            if not active:break
            if time.time() >= cutoff.timestamp():raise TimeoutError('Pilot cutoff; stop only owned process groups.')
            if shutil.disk_usage(root).free < 3*1024**3:raise RuntimeError('Pilot disk reserve reached.')
            time.sleep(5)
        reports={}
        for name in processes:
            p=root/name/'manifest.json'
            reports[name]=json.loads(p.read_text()) if p.exists() else {'complete':False,'records':[]}
        from experiments.robotwin.eraf_fg_contract import validate_correction
        for report in reports.values():
            for row in report['records']:
                validate_correction(row)
                for path, sha in [(row['frame_path'],row['frame_sha256']),
                                  (str(Path(row['frame_path']).with_name('controls.pkl')),row['controls_sha256'])]:
                    if file_sha256(path)!=sha:raise ValueError('Pilot recorded file identity changed.')
        if args.mode == 'cup_pill_expansion' and all(r.get('complete') for r in reports.values()):
            from collections import Counter
            rows = [r for report in reports.values() for r in report['records']]
            keys = [(r['source_task'],r['task_config'],r['scene_seed']) for r in rows]
            if len(set(keys)) != 60:raise ValueError('Formal corrections contain repeated or missing scenes.')
            counts = Counter((r['source_task'], r['replay_split']) for r in rows)
            if counts != {(t,s):n for t in ('place_empty_cup','move_pillbottle_pad')
                           for s,n in [('train',24),('replay_holdout',6)]}:
                raise ValueError('Formal FG scene budget differs from24train+6holdout per task.')
            if set(keys) & excluded:raise ValueError('Formal FG scenes overlap previous data.')
            from experiments.robotwin.expanded_fg import FORMAT
            write('collections.json',dict(format=FORMAT,complete=True,records=rows,
                source_collections=[str(root/name/'manifest.json') for name in reports],
                requested_scenes=60,collection_mode=args.mode,policy_checkpoint_sha256=args.checkpoint_sha256))
        if file_sha256(args.checkpoint)!=args.checkpoint_sha256:raise ValueError('Pilot modified its source checkpoint.')
        state.update(terminal=True, complete=all(r.get('complete') for r in reports.values())
                     and all(p.returncode == 0 for p in processes.values()),stage='finished',
                     accepted={name:len(r['records']) for name,r in reports.items()})
        write('summary.json',dict(state,checkpoint_unchanged=True,performance_gain_claim=False))
    except BaseException as error:
        state.update(stage='stopped',error=repr(error))
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
        for name, process in processes.items():
            try:process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            state['jobs'][name]['exit_code']=process.returncode
        state.update(terminal=True,finished_at=datetime.now().astimezone().isoformat())
        write('driver.json',state)


if __name__=='__main__':main()
