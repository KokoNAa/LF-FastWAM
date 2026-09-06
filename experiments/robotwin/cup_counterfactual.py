"""Native cup-on-coaster versus cup-behind-coaster in the same RoboTwin scene."""
from __future__ import annotations
import numpy as np

SOURCE_INSTRUCTION = 'Place the empty cup on the coaster.'
TARGET_INSTRUCTION = 'Place the empty cup behind the coaster on the table.'
FRONT_INSTRUCTION = 'Place the empty cup in front of the coaster on the table.'


def behind_goal(cup_point, coaster_point, floor_z, upright_dot, left_open, right_open, direction=1):
    delta = np.asarray(cup_point) - np.asarray(coaster_point)
    return bool(abs(delta[0]) < .035 and .095 < direction * delta[1] < .165
                and abs(float(cup_point[2]) - floor_z) < .015
                and upright_dot > .95 and left_open and right_open)


def initialize_geometry(task, direction=1):
    if direction not in (-1, 1):
        raise ValueError('Cup relation direction must be -1 (front) or 1 (behind).')
    task._cup_cf_direction = direction
    task._cup_cf_floor_z = float(task.cup.get_functional_point(0, 'pose').p[2])
    rotation = task.cup.get_pose().to_transformation_matrix()[:3, :3]
    task._cup_cf_local_up = rotation.T @ np.array([0., 0., 1.])


def counterfactual_success(task):
    rotation = task.cup.get_pose().to_transformation_matrix()[:3, :3]
    upright = float((rotation @ task._cup_cf_local_up)[2])
    return behind_goal(task.cup.get_functional_point(0, 'pose').p,
                       task.coaster.get_functional_point(0, 'pose').p,
                       task._cup_cf_floor_z, upright,
                       task.is_left_gripper_open(), task.is_right_gripper_open(), task._cup_cf_direction)


def play_counterfactual(task):
    from envs.utils import ArmTag
    arm = ArmTag('right' if task.cup.get_pose().p[0] > 0 else 'left')
    task.move(task.close_gripper(arm, pos=.6))
    task.move(task.grasp_actor(task.cup, arm, pre_grasp_dis=.1,
                              contact_point_id=[0, 2][int(arm == 'left')]))
    task.move(task.move_by_displacement(arm, z=.08, move_axis='arm'))
    target = np.array(task.coaster.get_functional_point(0), dtype=float, copy=True)
    target[1] += .13 * task._cup_cf_direction
    target[2] = task._cup_cf_floor_z
    task.move(task.place_actor(task.cup, arm, target_pose=target,
                              functional_point_id=0, pre_dis=.05))
    task.move(task.move_by_displacement(arm, z=.05, move_axis='arm'))
    task.delay(30)
    return task.info
