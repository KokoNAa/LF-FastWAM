import pytest
from scripts.report_robotwin_world_language import clustered,first_goal_category


@pytest.mark.parametrize('reference',['source','target'])
def test_video_discrimination_does_not_replace_absolute_fit(reference):
    from scripts.report_robotwin_world_language import video_error_metrics
    wrong='target' if reference=='source' else 'source'
    def record(correct_error,wrong_error):
        return dict(reference=reference,errors={
            reference:dict(mse=correct_error,object_roi_mse=2*correct_error),
            wrong:dict(mse=wrong_error,object_roi_mse=2*wrong_error)},
            wrong_minus_correct_mse=wrong_error-correct_error,
            wrong_minus_correct_roi_mse=2*(wrong_error-correct_error))
    baseline=video_error_metrics(record(1,1.1))
    adapted=video_error_metrics(record(3,5))
    assert adapted['wrong_minus_correct_mse']>baseline['wrong_minus_correct_mse']
    assert adapted['correct_mse']>baseline['correct_mse']
    assert adapted['correct_roi_mse']==6 and adapted['wrong_roi_mse']==10
    swapped=record(1,2);swapped['reference']=wrong
    with pytest.raises(ValueError,match='margin mismatch'):video_error_metrics(swapped)


def test_video_model_contrast_pairs_noise_and_reference_before_scene_aggregation():
    from scripts.report_robotwin_world_language import paired_video_statistics
    def row(model,scene,noise,value,reference='source'):
        return dict(model=model,task='task',phase='initial',metric='video_correct_mse_sigma1.0',
                    scene_seed=scene,noise_seed=noise,reference=reference,value=value)
    rows=[row('released',1,42,10),row('no_eraf',1,42,12),
          row('released',1,43,20),row('no_eraf',1,43,22),
          row('released',2,42,100),row('no_eraf',2,42,104),
          row('released',3,42,1000),row('no_eraf',3,43,9000),
          row('no_eraf',3,42,9000,reference='target')]
    result=paired_video_statistics(rows)
    assert result['matched_metric_observations']==3
    assert result['unpaired_metric_observations']==3
    stat=result['statistics'][0]
    assert stat['scenes']==2 and stat['mean']==3 and stat['repeated_observations']==3
    with pytest.raises(ValueError,match='Duplicate'):paired_video_statistics(rows+[rows[0]])


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
