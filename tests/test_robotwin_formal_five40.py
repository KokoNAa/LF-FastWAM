import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

from scripts.run_robotwin_formal_five40 import summarize, TASKS, sha, write, records
from experiments.robotwin.language_interventions import GoalObserver, GoalEvaluation, GoalSnapshot


def test_observer_does_not_change_success_or_add_final_sample(monkeypatch):
    import experiments.robotwin.manipulation_metrics as mm
    samples=[]
    class Observer:
        def __init__(self, env, goal): pass
        def sample(self, snapshot): samples.append(snapshot.success)
        def report(self): return dict(physics_samples=len(samples))
    monkeypatch.setattr(mm,'ManipulationObserver',Observer)
    env=NS(check_success=lambda: 'native',_record_manipulation_metrics=True)
    observer=GoalObserver(env,NS(source_goal={},counterfactual_goal={}))
    values=iter([False,True,True])
    def update():
        result=next(values)
        observer.counterfactual_ever_success |= result
        return GoalSnapshot(GoalEvaluation(False,{}),GoalEvaluation(result,{}))
    observer.update=update
    observer.install_selected_goal('counterfactual')
    assert env.check_success() is False
    assert env.check_success() is True
    report=observer.episode_diagnostics(selected_goal='counterfactual')
    assert samples==[False,True] and report['manipulation_metrics']['physics_samples']==2
    observer.restore_native_goal()
    assert env.check_success()=='native'


@pytest.fixture
def matrix(tmp_path):
    models={a:dict(sha256=a,eraf='off' if a=='no_eraf' else 'on') for a in ['no_eraf','eraf_only','eraf_fg']}
    for task in TASKS:
        canonical=[dict(scene_seed=i,episode_index=i,source_instruction='source',counterfactual_instruction='cf',
                        initial_physical_state_sha256=str(i)) for i in range(40)]
        path=tmp_path/'catalog'/task/'demo_clean/correct/episodes.jsonl';path.parent.mkdir(parents=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in canonical))
        for arm,model in models.items():
            base=tmp_path/'evaluation'/arm/task/'demo_clean/counterfactual';base.mkdir(parents=True)
            write(base/'complete.json',dict(complete=True,checkpoint_sha256=model['sha256'],canonical_sha256=sha(path),
                                            eraf=model['eraf'],manipulation_metrics=True))
            write(base/'initial_states.json',[dict(scene_seed=i,sha256=str(i)) for i in range(40)])
            output=[]
            for i,c in enumerate(canonical):
                success=i<10;lift=i<20
                output.append(c|dict(counterfactual_goal_ever_success=success,counterfactual_goal_final_success=success,
                    manipulation_metrics=dict(physics_samples=50,any_correct_object_lifted=lift,
                        correct_placement_after_lift=success,full_goal_after_any_lift=success,
                        all_instruction_objects_lifted=lift,any_object_placed_after_lift=success,
                        final_strict_slot_assignment={'a':success} if task=='place_burger_fries' else None,
                        objects={'a':dict(correctly_lifted=lift,placed_after_lift=success)})))
            (base/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in output))
    return tmp_path,dict(models=models)


def test_complete_summary_preserves_denominators_and_pairing(matrix):
    root,plan=matrix;r=summarize(root,plan)
    assert r['complete'] and r['episodes']==600
    assert r['macro_cf']==dict(no_eraf=.25,eraf_only=.25,eraf_fg=.25)
    assert all(c['placement_given_lift_rate']==.5 for c in r['cells'])
    assert all(c['gains']==c['losses']==0 for c in r['paired_comparisons'])


@pytest.mark.parametrize('mutation',['wrong_seed','missing','no_samples'])
def test_incomplete_or_unpaired_cells_fail_closed(matrix,mutation):
    root,plan=matrix;path=root/'evaluation/no_eraf'/TASKS[0]/'demo_clean/counterfactual/episodes.jsonl'
    rows=records(path)
    if mutation=='missing': rows.pop()
    elif mutation=='wrong_seed': rows[0]['scene_seed']=9999
    else: rows[0]['manipulation_metrics']['physics_samples']=0
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):summarize(root,plan)
