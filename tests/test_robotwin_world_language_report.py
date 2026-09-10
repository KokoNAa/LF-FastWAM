import pytest
from scripts.report_robotwin_world_language import clustered


def test_noise_replicates_do_not_reweight_scenes():
    x=clustered([(1,1.)]*30+[(2,0.)])
    y=clustered([(1,1.),(2,0.)])
    assert x['scenes']==2 and x['repeated_observations']==31
    assert x['mean']==.5 and x['ci95']==y['ci95']


def test_missing_and_nonfinite_are_not_reported_as_zero():
    assert clustered([])['mean'] is None
    with pytest.raises(ValueError):clustered([(1,float('nan'))])
