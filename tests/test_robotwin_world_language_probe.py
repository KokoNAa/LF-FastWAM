import numpy as np
import pytest
from experiments.robotwin.world_language_probe import metrics, preference, replace_method


def test_normalized_distance_and_zero_controls():
    a = np.arange(12.).reshape(3,4)
    assert metrics(a,a)['relative_l2'] == 0
    assert metrics(a,a)['cosine_distance'] == pytest.approx(0)
    assert metrics(a,2*a)['relative_l2'] == pytest.approx(2/3)
    assert metrics(np.zeros(3),np.zeros(3))['cosine_distance'] == 0
    with pytest.raises(ValueError): metrics(a, np.ones((4,3)))
    with pytest.raises(ValueError): metrics(a, a*np.nan)


def test_reference_direction_and_degenerate_prefix():
    a, b = np.zeros((32,14)), np.ones((32,14))
    assert preference(a,a,b)['target_axis_projection'] == 0
    assert preference(b,a,b)['target_axis_projection'] == 1
    assert preference(b,a,b)['source_minus_target_rmse'] > 0
    assert preference(a,a,a)['target_axis_projection'] is None


def test_restore_inherited_and_preexisting_method_on_failure():
    class Model:
        def f(self): return 1
    m = Model()
    with pytest.raises(RuntimeError):
        with replace_method(m,'f',lambda:2):
            assert m.f() == 2
            raise RuntimeError()
    assert m.f() == 1 and 'f' not in vars(m)
    m.f = lambda:3
    with replace_method(m,'f',lambda:4): assert m.f() == 4
    assert m.f() == 3


@pytest.mark.parametrize('verb,replacement',[('Place','Please position'),('Position','Please place')])
def test_spatial_paraphrase_keeps_bound_object_names_and_goal(monkeypatch,verb,replacement):
    from scripts.probe_robotwin_world_language import instructions
    import experiments.robotwin.decision_language_replay as language
    source=verb+' the left-handed mouse to the right of the blue bell.'
    target=verb+' the left-handed mouse to the left of the blue bell.'
    monkeypatch.setattr(language,'bound_spatial_instruction_pairs',lambda *a:[dict(source=source,target=target)])
    result=instructions(dict(source_instruction='old',counterfactual_instruction='old',
        source_task='place_a2b_right',pair_id='place_a2b_right_to_left',scene_info=dict(info={})))
    assert result['source']==source and result['target']==target
    assert result['source_paraphrase']==replacement+source[len(verb):]
    assert result['target_paraphrase']==replacement+target[len(verb):]
