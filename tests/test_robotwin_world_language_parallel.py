from copy import deepcopy
import json
from pathlib import Path
import pytest
from scripts.run_robotwin_world_language_parallel import arms,name,command,check_inputs,journal_rows,recover


def test_worker_arguments_preserve_semantics_and_gpu_selection():
    assert len({name(c) for c in arms()})==24
    plan={'checkpoints':{'released':'/released.pt','no_eraf':'/no_eraf.pt'}}
    for model in plan['checkpoints']:
        cmd=command(Path('/run'),Path('/frozen'),plan,'/manifest', (model,43,'target','source'),2)
        for flag,value in [('--gpu','2'),('--world-video-language','target'),('--policy-seed','43'),('--conditions','correct'),('--checkpoint',plan['checkpoints'][model]),('--episodes','10'),('--policy-kind','legacy' if model=='released' else 'repair')]:
            assert cmd[cmd.index(flag)+1]==value
        assert cmd[1]=='/frozen/scripts/eval_robotwin_world_language.py'


def test_live_input_guard_rejects_cross_gpu_or_language_drift(tmp_path):
    r=dict(source_task='task',scene_seed=10,initial_observation_sha256='same',source_instruction='source',counterfactual_instruction='target',noise_seed=43,video_language='target',video_instruction='target',action_instruction='source')
    ref={('task',10):r};cell=('released',43,'target','source')
    check_inputs([r],ref,cell)
    for field,value in [('initial_observation_sha256','different'),('video_instruction','wrong'),('noise_seed',42)]:
        bad=r|{field:value}
        with pytest.raises(ValueError):check_inputs([bad],ref,cell)
    with pytest.raises(ValueError,match='duplicate'):check_inputs([r,r],ref,cell)
    p=tmp_path/'journal';p.write_bytes(json.dumps(r).encode()+b'\n'+b'{"partial":')
    assert journal_rows(p)==[r]


def test_partial_recovery_preserves_evidence_and_refuses_complete_group(tmp_path):
    stage='released-closed-v_target-a_source-seed43';folder=tmp_path/'closed_loop'/stage;folder.mkdir(parents=True)
    (folder/'episodes.jsonl').write_text('{}\n');(tmp_path/(stage+'.log')).write_text('partial')
    pause=dict(interrupted_stage=stage,completed_episode_records=301,terminal_audited_episodes=300)
    (tmp_path/'user_pause_4gpu.json').write_text(json.dumps(pause))
    (tmp_path/'launch_resume1.json').write_text(json.dumps(dict(pid=99999999,start_time='0')))
    status=dict(paused=True,jobs={stage:dict(pid=99999998,start_time='0',status='interrupted_by_user')})
    (folder/'world_language_complete.json').write_text('{}')
    with pytest.raises(ValueError,match='unexpectedly complete'):recover(tmp_path,deepcopy(status))
    (folder/'world_language_complete.json').unlink()
    receipt=recover(tmp_path,status);r=json.loads(receipt.read_text())
    assert r['partial_episodes']==1 and stage not in status['jobs']
    assert not folder.exists() and (receipt.parent/stage/'episodes.jsonl').read_text()=='{}\n'
    assert json.loads((receipt.parent/'status_before.json').read_text())['jobs'][stage]['status']=='interrupted_by_user'
