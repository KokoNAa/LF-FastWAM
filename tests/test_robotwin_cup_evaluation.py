import numpy as np
import pytest
from experiments.robotwin.cup_evaluation import DEPLOYMENT, require_same_initial, summarize_records


def test_initial_pairing_rejects_visual_or_state_change():
    a = {'rgb': np.zeros((2, 2, 3), dtype=np.uint8), 'qpos': np.zeros(14)}
    require_same_initial(a, {k:v.copy() for k,v in a.items()})
    for key in a:
        b = {k:v.copy() for k,v in a.items()}; b[key].flat[0] = 1
        with pytest.raises(ValueError): require_same_initial(a,b)


def test_paired_summary_rejects_partial_duplicate_and_wrong_goal():
    catalog = {'episodes': [{'scene_seed':93000000}]}
    source = dict(scene_seed=93000000, condition='source', success=True,
                  source_success=True, target_success=False,
                  initial_observations_equal=True, deployment=DEPLOYMENT)
    target = source | dict(condition='target', success=False)
    result = summarize_records(catalog, [source,target])
    assert result['source_successes']==1 and result['cf_successes']==0
    assert result['cf_source_goal_only']==1
    for rows in ([source], [source,target,target], [source,target|dict(success=True)]):
        with pytest.raises(ValueError): summarize_records(catalog,rows)
