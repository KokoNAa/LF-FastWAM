import pytest
from scripts.report_robotwin_world_language import clustered,first_goal_category


def test_noise_replicates_do_not_reweight_scenes():
    x=clustered([(1,1.)]*30+[(2,0.)])
    y=clustered([(1,1.),(2,0.)])
    assert x['scenes']==2 and x['repeated_observations']==31
    assert x['mean']==.5 and x['ci95']==y['ci95']


def test_missing_and_nonfinite_are_not_reported_as_zero():
    assert clustered([])['mean'] is None
    with pytest.raises(ValueError):clustered([(1,float('nan'))])


@pytest.mark.parametrize('selected,source_ever,target_ever,source_final,target_final,expected',[
    ('source',False,False,False,False,'neither'),
    ('source',True,False,True,False,'source'),
    ('counterfactual',True,False,False,False,'source'),
    ('source',False,True,False,False,'counterfactual'),
    ('source',True,True,True,False,'counterfactual'),
    ('counterfactual',True,True,False,True,'source'),
    ('source',True,True,True,True,'ambiguous'),
    ('source',True,True,False,False,'ambiguous'),
])
def test_first_goal_preserves_opposite_goal_before_selected_termination(
        selected,source_ever,target_ever,source_final,target_final,expected):
    row=dict(selected_goal=selected,source_goal_ever_success=source_ever,counterfactual_goal_ever_success=target_ever,
        source_goal_final_success=source_final,counterfactual_goal_final_success=target_final)
    assert first_goal_category(row)==expected


def test_first_goal_rejects_inconsistent_diagnostics():
    with pytest.raises(ValueError):first_goal_category(dict(selected_goal='source',source_goal_ever_success=False,
        counterfactual_goal_ever_success=False,source_goal_final_success=True,counterfactual_goal_final_success=False))
def test_representation_grid_uses_patched_visual_tokens_and_requires_all_layers():
    from scripts.report_robotwin_world_language_representations import feature_matrix
    import pytest
    rows=[dict(layer=i,kind='hidden',rms=0.,relative_l2=0.,cosine_distance=0.,max_abs=0.,
               token_delta_rms=list(range(120))) for i in reversed(range(30))]
    matrix=feature_matrix(rows,'hidden')
    assert matrix.shape==(30,124)
    assert matrix[0,4:].reshape(12,10)[8,0]==80
    with pytest.raises(ValueError,match='Incomplete'):feature_matrix(rows[1:],'hidden')
