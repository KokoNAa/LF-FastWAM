"""Share an actual planned grasp/lift prefix before alternative placement goals."""
from contextlib import contextmanager
from copy import deepcopy

SUPPORTED_TASKS = {'place_a2b_left', 'place_a2b_right', 'place_burger_fries'}


@contextmanager
def grasp_prefix(task, *, replay=None):
    """Capture or replay the first two high-level moves, then plan normally.

    These three tasks share grasp and lift across goal variants. Replaying the
    exact same joint paths removes independent planner noise before the actual
    placement decision. Later camera/state equality is still checked on data.
    """
    if not task.need_plan:
        raise ValueError('Shared prefix is installed during planning only.')
    original = task.move
    existed, previous = 'move' in vars(task), vars(task).get('move')
    result = {}
    count = 0
    if replay is not None:
        if task.left_joint_path or task.right_joint_path or task.left_cnt or task.right_cnt:
            raise ValueError('Shared prefix needs a fresh planning scene.')
        task.left_joint_path = deepcopy(replay['left_joint_path'])
        task.right_joint_path = deepcopy(replay['right_joint_path'])

    def move(*args, **kwargs):
        nonlocal count
        shared = count < 2
        if shared and replay is not None:
            task.need_plan = False
        try:
            value = original(*args, **kwargs)
        finally:
            task.need_plan = True
        count += 1
        if count == 2 and task.plan_success:
            if replay is not None:
                if (task.left_cnt != len(replay['left_joint_path']) or
                        task.right_cnt != len(replay['right_joint_path'])):
                    raise ValueError('Goal variants did not consume the same grasp/lift moves.')
            result.update(left_joint_path=deepcopy(task.left_joint_path),
                          right_joint_path=deepcopy(task.right_joint_path))
        return value

    task.move = move
    try:
        yield result
        if task.plan_success and not result:
            raise ValueError('Expert did not execute both grasp and lift moves.')
    finally:
        task.need_plan = True
        if existed:
            task.move = previous
        else:
            del task.move
