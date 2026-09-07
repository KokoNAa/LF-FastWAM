"""Native pad placement versus the same object placed in front of its pad."""
from __future__ import annotations

from types import MethodType
import numpy as np

PAD_ACTORS = {
    'place_mouse_pad': ('mouse', 'target'),
    'move_stapler_pad': ('stapler', 'pad'),
    'move_pillbottle_pad': ('pillbottle', 'pad'),
}
FRONT_OFFSET = .13


def actors(task, task_name):
    return [getattr(task, key) for key in PAD_ACTORS[task_name]]


def geometry(task, task_name, variant):
    if variant not in {'native_pad', 'front_pad'}:
        raise ValueError('Unknown pad relation')
    actor, pad = actors(task, task_name)
    position = np.asarray(actor.get_pose().p, dtype=float)
    target = np.array(pad.get_pose().p, dtype=float, copy=True)
    if variant == 'front_pad':
        target[1] -= FRONT_OFFSET
    delta = np.abs(position - target)
    q = np.abs(np.asarray(actor.get_pose().q, dtype=float))
    # Preserve each native task's own pose and orientation tolerances.
    if task_name == 'place_mouse_pad':
        success = (np.all(delta[:2] < [.015, .012]) and
                   (abs(q[2] * q[3] - .49) < .015 or abs(q[0] * q[1] - .49) < .015))
    elif task_name == 'move_stapler_pad':
        success = np.all(delta < [.02, .02, .01]) and q.max() - q.min() < .02
    else:
        success = (np.all(delta[:2] < [.03, .03]) and
                   abs(position[2] - (.741 + task.table_z_bias)) < .005)
    return bool(success), target


def play_front(task, task_name):
    """Offset the expert's placement command; never move the physical pad."""
    actor, _ = actors(task, task_name)
    original = task.place_actor

    def place(bound_task, selected_actor, *args, **kwargs):
        if selected_actor is not actor or 'target_pose' not in kwargs:
            raise ValueError('Unexpected native placement command')
        target = np.array(kwargs['target_pose'], dtype=float, copy=True)
        target[1] -= FRONT_OFFSET
        return original(selected_actor, *args, **(kwargs | {'target_pose': target}))

    task.place_actor = MethodType(place, task)
    try:
        return task.play_once()
    finally:
        task.place_actor = original
