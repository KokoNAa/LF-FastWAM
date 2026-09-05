import unittest
from experiments.robotwin.shared_grasp_prefix import grasp_prefix


class FakeTask:
    def __init__(self):
        self.need_plan = self.plan_success = True
        self.left_joint_path = []
        self.right_joint_path = []
        self.left_cnt = self.right_cnt = 0
        self.executed = []

    def move(self, value):
        if self.need_plan:
            self.left_joint_path.append([value])
            self.executed.append(value)
        else:
            self.executed.append(self.left_joint_path[self.left_cnt][0])
            self.left_cnt += 1
        return True


class SharedPrefixTest(unittest.TestCase):
    def test_replays_shared_prefix_then_plans_new_goal_without_mutating_other_branch(self):
        source = FakeTask()
        with grasp_prefix(source) as prefix:
            source.move(1)
            source.move(2)
            source.move(3)
        self.assertEqual(prefix['left_joint_path'], [[1], [2]])
        target = FakeTask()
        with grasp_prefix(target, replay=prefix):
            self.assertFalse(target.need_plan)
            target.move(100)
            self.assertFalse(target.need_plan)
            target.move(200)
            self.assertTrue(target.need_plan)
            target.move(9)
        self.assertEqual(target.executed, [1, 2, 9])
        self.assertEqual(target.left_joint_path, [[1], [2], [9]])
        self.assertEqual(source.left_joint_path, [[1], [2], [3]])
        self.assertNotIn('move', vars(target))

    def test_exception_restores_planning_and_original_method(self):
        task = FakeTask()
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            with grasp_prefix(task):
                task.move(1)
                raise RuntimeError('failed')
        self.assertTrue(task.need_plan)
        self.assertNotIn('move', vars(task))


if __name__ == '__main__':
    unittest.main()
