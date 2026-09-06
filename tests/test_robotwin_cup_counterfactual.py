from experiments.robotwin.cup_counterfactual import behind_goal


def test_behind_goal_requires_full_placement_and_excludes_native_target():
    def success(point, up=1., left=True, right=True):
        return behind_goal(point, [0., 0., .745], .74, up, left, right)
    assert success([0., .13, .74])
    assert not success([0., 0., .745])
    assert not success([.06, .13, .74])
    assert not success([0., -.13, .74])
    assert not success([0., .13, .80])
    assert not success([0., .13, .74], up=.8)
    assert not success([0., .13, .74], left=False)
    assert not success([0., .13, .74], right=False)
