import pytest

from scripts.run_robotwin_randomized_eval import verify_domain


def test_randomized_configuration_rejects_clean_or_changed_runtime():
    expected = dict(random_background=True, cluttered_table=True, random_light=True,
                    random_table_height=.03, random_head_camera_dis=0)
    verify_domain(expected.copy(), expected)
    with pytest.raises(ValueError, match='not active'):
        verify_domain(expected | {'random_light': False}, expected)
    with pytest.raises(ValueError, match='not active'):
        verify_domain(expected | {'random_table_height': 0}, expected)
    clean = dict(random_background=False, cluttered_table=False, random_light=False)
    with pytest.raises(ValueError, match='not active'):
        verify_domain(clean, clean)
