"""Measured action differences for paired original-task goal continuations."""
import numpy as np

FORMAT = 'robotwin_target_goal_branch_probe_v1'


def action_difference(source, target):
    arrays = [np.asarray(value) for value in (source, target)]
    if any(value.ndim != 2 or value.shape[1] != 14 or len(value) < 32 or
           not np.isfinite(value).all() for value in arrays):
        raise ValueError('Paired branches require at least 32 finite 14-D actions each.')
    length = min(map(len, arrays))
    delta = arrays[0][:length].astype(np.float64) - arrays[1][:length]
    different = np.flatnonzero(np.max(np.abs(delta), axis=1) > 1e-4)
    return {'shared_comparison_frames': length,
            'first_different_frame': int(different[0]) if len(different) else None,
            'action_rmse_first24': float(np.sqrt(np.mean(delta[:24] ** 2))),
            'action_rmse_first32': float(np.sqrt(np.mean(delta[:32] ** 2))),
            'informative_first24': bool(np.any(np.abs(delta[:24]) > 1e-4)),
            'difference_threshold_raw_action': 1e-4}


def require_same_observation(first, second, cameras):
    if (not np.array_equal(first['qpos'], second['qpos']) or
            any(not np.array_equal(first['images'][camera], second['images'][camera]) for camera in cameras)):
        raise ValueError('Goal branches changed the real policy observation.')


def continue_open_gripper_expert(task, spec, *, selected_goal):
    """Plan directly at an early open-gripper state, without a release prefix."""
    if selected_goal not in {'source', 'target'}:
        raise ValueError('Unknown expert continuation goal.')
    if not task.is_left_gripper_open() or not task.is_right_gripper_open():
        raise ValueError('Direct expert continuation requires both grippers open.')
    from experiments.robotwin.pgc_task_variants import play_variant
    variant = spec.source_variant if selected_goal == 'source' else spec.counterfactual_variant
    task.need_plan, task.plan_success = True, True
    play_variant(task, spec, variant)


def continue_held_placement(task, spec, *, selected_goal, arm_name, initial_object_z):
    """Try a direct placement from an elevated real policy state; verify later."""
    if spec.source_task != 'place_a2b_left' or selected_goal not in {'source', 'target'}:
        raise ValueError('Held placement is restricted to the left placement goal pair.')
    if arm_name not in {'left', 'right'}:
        raise ValueError('A held-placement probe must explicitly select its arm.')
    if float(task.object.get_pose().p[2]) - initial_object_z < .03:
        raise ValueError('Object is not elevated by 3 cm from its initial pose.')
    from envs.utils import ArmTag
    variant = spec.source_variant if selected_goal == 'source' else spec.counterfactual_variant
    task._pgc_active_variant = variant
    task.need_plan, task.plan_success = True, True
    arm = ArmTag(arm_name)
    target = np.array(task.target_object.get_pose().p, dtype=float, copy=True)
    target[0] += -.13 if variant == 'left' else .13
    task.move(task.place_actor(task.object, arm_tag=arm, target_pose=target.tolist()))
    task.move(task.move_by_displacement(arm_tag=arm, z=.08, move_axis='arm'))
