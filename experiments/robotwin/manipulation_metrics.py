"""Read-only physical pick/place events. Never changes actions or termination.

Sample once at each policy physics-step success check, not at each replan.
The original full-goal success remains the benchmark metric. These diagnostics
describe how it was reached; sorting need not lift objects already in place.
"""
from __future__ import annotations

from copy import deepcopy
import math

FORMAT = 'robotwin_manipulation_events_v1'
PROTOCOL = dict(lift_height_m=0.03, sustained_grasp_seconds=0.10,
                gripper_open_threshold=0.8, minimum_contact_links=2,
                contact_impulse_min=1e-8, table_height_tolerance_m=0.02,
                sampling='one sample per executed physics step inside selected check_success',
                placement='selected geometry with open grippers, no finger contact, after verified lift',
                termination='unchanged; no added settling or recovery steps')


def entity_id(entity):
    entity = getattr(entity, 'entity', entity)
    return int(entity.per_scene_id)


def actor_attributes(goal):
    kind = goal['type']
    if kind == 'ordered_row':
        return list(goal['actors'])
    if kind == 'functional_point_assignment':
        return [a['actor'] for a in goal['assignments']]
    if kind == 'stacked_on':
        # Include the instruction's base-block relocation as a separate object.
        return [goal['actor'], goal['reference']]
    if kind == 'relative_pose':
        return [goal['actor']]
    raise ValueError(f'No physical event definition for {kind}')


class EventAccumulator:
    """Pure state machine, independently testable without SAPIEN."""
    def __init__(self, actors, dt):
        if not 0 < dt <= 0.02:
            raise ValueError('Unexpected simulator timestep')
        self.dt, self.tick = float(dt), 0
        self.minimum_ticks = math.ceil(PROTOCOL['sustained_grasp_seconds'] / dt)
        self.objects = {a: dict(correctly_lifted=False, first_lift_tick=None,
                               placed_after_lift=False, first_placement_tick=None,
                               max_lift_m=0.0, max_contact_links=0,
                               final_placement=False, streak=0, arm=None,
                               lift_arm=None, lift_contact_links=[]) for a in actors}
        self.full_goal_after_any_lift = False
        self.first_full_goal_after_lift_tick = None
        self.correct_placement_after_lift = False
        self.first_correct_placement_tick = None

    def update(self, samples, *, full_goal, correct_placement=None):
        if set(samples) != set(self.objects):
            raise ValueError('Missing object observations')
        self.tick += 1
        for name, sample in samples.items():
            row = self.objects[name]
            arm = sample['grasp_arm']
            row['max_lift_m'] = max(row['max_lift_m'], float(sample['lift_m']))
            row['max_contact_links'] = max(row['max_contact_links'], sample['contact_links_count'])
            valid = arm is not None and sample['lift_m'] >= PROTOCOL['lift_height_m']
            row['streak'] = (row['streak'] + 1 if row['arm'] == arm else 1) if valid else 0
            row['arm'] = arm
            if not row['correctly_lifted'] and row['streak'] >= self.minimum_ticks:
                row.update(correctly_lifted=True, first_lift_tick=self.tick,
                           lift_arm=arm, lift_contact_links=sample['contact_links'])
            placed = bool(row['correctly_lifted'] and sample['placement'] and
                          self.tick > row['first_lift_tick'])
            if placed and not row['placed_after_lift']:
                row.update(placed_after_lift=True, first_placement_tick=self.tick)
            row['final_placement'] = placed
        if full_goal and any(r['correctly_lifted'] for r in self.objects.values()):
            self.full_goal_after_any_lift = True
            if self.first_full_goal_after_lift_tick is None:
                self.first_full_goal_after_lift_tick = self.tick
        if correct_placement is None:
            correct_placement = full_goal and all(s['placement'] for s in samples.values())
        if correct_placement and any(r['correctly_lifted'] for r in self.objects.values()):
            self.correct_placement_after_lift = True
            if self.first_correct_placement_tick is None:
                self.first_correct_placement_tick = self.tick

    def report(self):
        objects = {name: {k: v for k, v in row.items() if k not in ('streak', 'arm')}
                   for name, row in self.objects.items()}
        return dict(format=FORMAT, protocol=deepcopy(PROTOCOL), physics_samples=self.tick,
                    simulation_seconds=self.tick*self.dt, objects=objects,
                    any_correct_object_lifted=any(x['correctly_lifted'] for x in objects.values()),
                    all_instruction_objects_lifted=all(x['correctly_lifted'] for x in objects.values()),
                    any_object_placed_after_lift=any(x['placed_after_lift'] for x in objects.values()),
                    all_objects_placed_after_lift=all(x['placed_after_lift'] for x in objects.values()),
                    full_goal_after_any_lift=self.full_goal_after_any_lift,
                    correct_placement_after_lift=self.correct_placement_after_lift,
                    first_correct_placement_tick=self.first_correct_placement_tick,
                    first_full_goal_after_lift_tick=self.first_full_goal_after_lift_tick)


class ManipulationObserver:
    def __init__(self, env, goal):
        self.env, self.goal = env, deepcopy(goal)
        self.attributes = actor_attributes(goal)
        self.actors = {a: getattr(env, a) for a in self.attributes}
        # RoboTwin names several colored blocks "box". Match entity IDs, never names.
        self.actor_ids = {entity_id(actor.actor): a for a, actor in self.actors.items()}
        if len(self.actor_ids) != len(self.actors):
            raise ValueError('Ambiguous physical actor IDs')
        self.initial_z = {a: float(actor.get_pose().p[2]) for a, actor in self.actors.items()}
        self.fingers = {}
        for arm in ('left', 'right'):
            children = [joint[0].child_link for joint in getattr(env.robot, arm+'_gripper')]
            children.extend(getattr(env.robot, arm+'_entity').find_link_by_name(name)
                            for name in getattr(env.robot, arm+'_fix_gripper_name'))
            links = {entity_id(link): link.get_name() for link in children}
            if len(links) < 2:
                raise ValueError('Bilateral grasp needs two distinct finger links')
            self.fingers[arm] = links
        self.accumulator = EventAccumulator(self.attributes, env.scene.get_timestep())
        self.last_strict_slots = None

    def contacts(self):
        found = {a: {'left': set(), 'right': set()} for a in self.attributes}
        for contact in self.env.scene.get_contacts():
            # Ignore zero-impulse proximity contacts.
            if not any(sum(float(v)**2 for v in p.impulse) > PROTOCOL['contact_impulse_min']**2
                       for p in contact.points):
                continue
            identities = [entity_id(body) for body in contact.bodies]
            for i in (0, 1):
                if identities[i] in self.actor_ids:
                    for arm, fingers in self.fingers.items():
                        if identities[1-i] in fingers:
                            found[self.actor_ids[identities[i]]][arm].add(fingers[identities[1-i]])
        return found

    def placement_truth(self, snapshot):
        goal, detail = self.goal, snapshot.details
        kind, opened = goal['type'], detail['grippers_open']
        if kind == 'relative_pose':
            truth = {goal['actor']: snapshot.success}
        elif kind == 'stacked_on':
            # Base center is supplemental; existing stacked_on CF ignores it.
            base = self.actors[goal['reference']].get_pose().p
            center = self.env.block1_target_pose
            truth = {goal['actor']: snapshot.success,
                     goal['reference']: bool(abs(float(base[0])-center[0]) < .025 and
                                             abs(float(base[1])-center[1]) < .025 and opened)}
        elif kind == 'ordered_row':
            adjacency = [0 < d['x_delta'] < goal['max_adjacent_x_distance'] and
                         abs(d['y_delta']) < goal['max_adjacent_y_distance'] for d in detail['adjacent']]
            truth = {a: bool(opened and all(adjacency[j] for j in range(len(adjacency))
                                           if i in (j, j+1))) for i, a in enumerate(self.attributes)}
        else:
            truth = {a['actor']: bool(a['assignment_ok'] and opened) for a in detail['assignments']}
            strict = {}
            for assignment, observed in zip(goal['assignments'], detail['assignments'], strict=True):
                alternate = 1 - assignment['reference_point']
                other = getattr(self.env, assignment['reference']).get_functional_point(alternate, 'pose').p
                p = observed['actor_position']
                opposite_distance = math.hypot(p[0]-float(other[0]), p[1]-float(other[1]))
                strict[assignment['actor']] = bool(truth[assignment['actor']] and
                                                  observed['planar_distance'] < opposite_distance)
            self.last_strict_slots = strict
            truth = {a['actor']: bool(strict[a['actor']] and abs(a['actor_position'][2]-a['reference_position'][2]) < .03)
                     for a in detail['assignments']}
        # Prevent airborne/fallen table objects from being counted as placed.
        # Stack top has its own explicit vertical tolerance in the benchmark.
        for a in truth:
            if not (kind == 'stacked_on' and a == goal['actor']) and kind != 'functional_point_assignment':
                truth[a] = bool(truth[a] and abs(float(self.actors[a].get_pose().p[2]) - self.initial_z[a])
                                <= PROTOCOL['table_height_tolerance_m'])
        return truth

    def sample(self, snapshot):
        contacts = self.contacts()
        truth = self.placement_truth(snapshot)
        samples = {}
        for a, actor in self.actors.items():
            selected = next((arm for arm in ('left', 'right') if len(contacts[a][arm]) >= 2 and
                             getattr(self.env.robot, 'get_'+arm+'_gripper_val')() < .8), None)
            all_links = sorted(contacts[a]['left'] | contacts[a]['right'])
            samples[a] = dict(lift_m=float(actor.get_pose().p[2])-self.initial_z[a], grasp_arm=selected,
                              contact_links=sorted(contacts[a][selected]) if selected else [],
                              contact_links_count=len(all_links), placement=bool(truth[a] and not all_links))
        self.accumulator.update(samples, full_goal=snapshot.success,
                                correct_placement=bool(snapshot.success and all(s['placement'] for s in samples.values())))

    def report(self):
        report = self.accumulator.report()
        report.update(initial_object_z=self.initial_z,
                      gripper_contact_links={a: v for a, v in self.fingers.items()},
                      final_strict_slot_assignment=self.last_strict_slots)
        return report
