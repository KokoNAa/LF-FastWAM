import numpy as np
import pytest
from scripts.review_robotwin_world_language_videos import selected_states,image_delta,validate_annotations


def test_qualitative_selection_is_catalog_order_not_outcome_or_seed_order():
    scenes=[dict(task='a',scene_seed=n) for n in [9,3,1]]
    states=[dict(task='a',scene_seed=n,phase=p,id=f'{n}_{p}') for n in [9,3,1] for p in ['initial','source_mid','source_late']]
    plan=dict(tasks=['a'],scenes=scenes,states=states)
    assert [r['id'] for r in selected_states(plan)]==['9_initial','9_source_late','3_initial','3_source_late']


def test_pixel_measure_excludes_initial_and_separates_task_region():
    a=np.zeros((3,2,2,3),dtype=np.uint8);b=a.copy();b[0]=255
    mask=np.array([[True,False],[False,False]])
    assert image_delta(a,b,mask)==dict(full_rmse=0.,task_roi_rmse=0.)
    b[1:,0,0]=255
    assert image_delta(a,b,mask)['task_roi_rmse']==1
    assert image_delta(a,b,mask)['full_rmse']==pytest.approx(.5)


def test_semantic_review_rejects_missing_unbound_and_inconsistent_ratings():
    import copy
    selection=dict(states=['s'],models=['m'],noise_seeds=[42])
    panels=dict(selection_sha256='selection',panels=[dict(state_id='s',seed=42,sha256='panel')])
    row=dict(state_id='s',model='m',seed=42,panel_sha256='panel',visible_change='Red block moves toward blue.',
        evidence='Final inspected frame: blocks remain separate; ordering is not established.',
        relation_established='no',clear_contradiction='insufficient_evidence',generation_quality='usable',
        paired_semantic_change='no_visible_change')
    annotations=dict(selection_sha256='selection',annotations=[dict(row,language=l) for l in ['source','target']])
    assert validate_annotations(selection,'selection',panels,annotations)['clips']==2
    for field,value in [('evidence',None),('panel_sha256','other'),('paired_semantic_change','semantic_change')]:
        bad=copy.deepcopy(annotations);bad['annotations'][0][field]=value
        with pytest.raises(ValueError):validate_annotations(selection,'selection',panels,bad)
    annotations['annotations'].pop()
    with pytest.raises(ValueError,match='incomplete'):validate_annotations(selection,'selection',panels,annotations)
