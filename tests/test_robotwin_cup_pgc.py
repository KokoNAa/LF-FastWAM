from types import SimpleNamespace
import numpy as np
import pytest
from experiments.robotwin.cup_counterfactual import initialize_geometry,counterfactual_success
from experiments.robotwin.pgc_data import pair_spec_from_source_task,ROBOTWIN_ERAF_PAIR_IDS
from experiments.robotwin.pgc_task_variants import _clauses,check_variant


class Pose:
    def __init__(self,p):self.p=np.array(p,dtype=float)
    def to_transformation_matrix(self):return np.eye(4)


class Actor:
    def __init__(self,p):self.pose=Pose(p)
    def get_pose(self):return self.pose
    def get_functional_point(self,*args):return self.pose


@pytest.mark.parametrize('point,source,target',[
    ([0.,0.,.745],True,False),([0.,-.13,.74],False,True),
    ([0.,.13,.74],False,False),([0.,-.13,.80],False,False)])
def test_cup_pgc_relation_preserves_evaluation_semantics(point,source,target):
    task=SimpleNamespace(cup=Actor([.2,.1,.74]),coaster=Actor([0.,0.,.745]),
        is_left_gripper_open=lambda:True,is_right_gripper_open=lambda:True)
    initialize_geometry(task,-1)
    task.cup.pose.p=np.array(point)
    spec=pair_spec_from_source_task('place_empty_cup')
    assert check_variant(task,spec,'on_coaster')==source
    assert check_variant(task,spec,'front_coaster')==target==counterfactual_success(task)
    assert _clauses(task,spec,'front_coaster')[0]['predicate_id']==5
    assert len(ROBOTWIN_ERAF_PAIR_IDS)==5
