import numpy as np
import pytest
from scripts.review_robotwin_world_language_videos import selected_states,image_delta


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
