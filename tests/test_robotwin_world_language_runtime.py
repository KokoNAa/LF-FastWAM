from copy import deepcopy
import pytest
from scripts.audit_robotwin_world_language_runtime import expected_jobs,validate_processes


def state():
    jobs={name:dict(pid=i+100,start_time=str(i+500),status='exited',exit_code=0) for i,name in enumerate(expected_jobs())}
    launch=dict(pid=900,start_time='1234',code_commit='frozen')
    return dict(compute_complete=True,jobs=jobs,controller_pid=900,code_commit='frozen'),launch


def proc(root,pid,start,state='R'):
    path=root/str(pid);path.mkdir()
    fields=[state]+['0']*18+[start]
    (path/'stat').write_text(str(pid)+' (name with spaces) '+' '.join(fields))


def test_completed_flags_do_not_override_live_owned_process(tmp_path):
    s,l=state();validate_processes(s,l,tmp_path)
    job=s['jobs']['collection'];proc(tmp_path,job['pid'],job['start_time'])
    with pytest.raises(ValueError,match='still live'):validate_processes(s,l,tmp_path)


def test_reused_pid_is_not_mistaken_for_owned_worker(tmp_path):
    s,l=state();proc(tmp_path,s['jobs']['collection']['pid'],'different-start')
    assert len(validate_processes(s,l,tmp_path)['jobs'])==34


def test_final_audit_refuses_incomplete_jobs_or_live_controller(tmp_path):
    s,l=state();bad=deepcopy(s);bad['compute_complete']=False
    with pytest.raises(ValueError,match='not completed'):validate_processes(bad,l,tmp_path)
    bad=deepcopy(s);bad['jobs'].pop('prepare')
    with pytest.raises(ValueError,match='Missing'):validate_processes(bad,l,tmp_path)
    bad=deepcopy(s);bad['jobs']['prepare']['exit_code']=1
    with pytest.raises(ValueError,match='successful process exit'):validate_processes(bad,l,tmp_path)
    proc(tmp_path,l['pid'],l['start_time'])
    with pytest.raises(ValueError,match='controller still live'):validate_processes(s,l,tmp_path)


def test_parallel_final_audit_checks_old_controller_and_active_queue(tmp_path):
    s,l=state();l['previous_controller']=dict(pid=901,start_time='old',code_commit='frozen')
    assert len(validate_processes(s,l,tmp_path)['previous_controllers'])==1
    proc(tmp_path,901,'old')
    with pytest.raises(ValueError,match='Previous owned controller'):validate_processes(s,l,tmp_path)
    (tmp_path/'901/stat').unlink()
    s['active_jobs']=['still-running']
    with pytest.raises(ValueError,match='active jobs'):validate_processes(s,l,tmp_path)
