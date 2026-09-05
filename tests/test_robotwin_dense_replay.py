import tempfile
import unittest
from pathlib import Path

from scripts.prepare_robotwin_trajectory_replay import dense_phase_frames


class DenseFramesTest(unittest.TestCase):
    def test_both_branches_reach_terminal_without_missing_a_replan_interval(self):
        frames = [(0, 0)] + dense_phase_frames(613, 271, 24)
        self.assertEqual(frames[-1], (612, 270))
        for before, after in zip(frames, frames[1:]):
            for a, b in zip(before, after):
                self.assertLessEqual(b - a, 24)
                self.assertGreaterEqual(b, a)
        self.assertEqual(len(frames), len(set(frames)))

    def test_short_episode_is_not_dropped(self):
        self.assertEqual(dense_phase_frames(2, 1, 24), [(1, 0)])
        self.assertEqual(dense_phase_frames(1, 1, 24), [])


class CompactReplayTest(unittest.TestCase):
    def test_restore_is_exact_and_does_not_mutate_initial_state(self):
        import torch
        from experiments.robotwin.compact_replay import capture_delta, restore_capture, ReplayPayloads
        parent = {'video_inputs': {'x': torch.arange(8).reshape(1, 2, 2, 2),
                                   'context': torch.arange(20.).reshape(1, 5, 4),
                                   'action': None, 'flag': True},
                  'action_inputs': {'mask': torch.ones(4, 4, dtype=torch.bool), 'length': 4}}
        new = {group: dict(values) for group, values in parent.items()}
        new['video_inputs']['x'] = parent['video_inputs']['x'] + 100
        new['video_inputs']['context'] = parent['video_inputs']['context'].clone()
        new['video_inputs']['context'][:, -1] += 30
        delta = capture_delta(new, parent)
        self.assertNotIn('action_inputs', delta)
        self.assertEqual(set(delta['video_inputs']['context']), {'last_token'})
        restored = restore_capture(parent, delta)
        torch.testing.assert_close(restored['video_inputs']['x'], new['video_inputs']['x'])
        torch.testing.assert_close(restored['video_inputs']['context'], new['video_inputs']['context'])
        torch.testing.assert_close(parent['video_inputs']['context'], torch.arange(20.).reshape(1, 5, 4))
        with tempfile.TemporaryDirectory() as tmp:
            base_path, path = Path(tmp) / 'parent.pt', Path(tmp) / 'later.pt'
            torch.save({'captured': {'source': parent, 'target': parent}}, base_path)
            torch.save({'format': 'robotwin_compact_replay_v1', 'parent_payload': str(base_path),
                        'references': {'source': torch.ones(1, 32, 14), 'target': torch.zeros(1, 32, 14)},
                        'capture_deltas': {'source': delta, 'target': {}}}, path)
            actual = ReplayPayloads([{'id': 'test', 'payload': str(path)}], 'cpu')['test']
            torch.testing.assert_close(actual['captured']['source']['video_inputs']['context'],
                                       new['video_inputs']['context'])
            torch.testing.assert_close(actual['captured']['target']['video_inputs']['context'],
                                       parent['video_inputs']['context'])


class NativeTeacherTest(unittest.TestCase):
    def test_teacher_restores_student_parameters_and_optimizer_gradient(self):
        import torch
        from experiments.robotwin.native_teacher import teacher_parameters
        student = torch.nn.Parameter(torch.tensor([2.]))
        original_pointer = student.data_ptr()
        with teacher_parameters({'a': student}, {'a': torch.tensor([7.])}):
            with torch.no_grad():
                teacher_output = student * 3
        self.assertEqual(student.data_ptr(), original_pointer)
        loss = (student * 3 - teacher_output).square().sum()
        loss.backward()
        torch.testing.assert_close(student.grad, torch.tensor([-90.]))
        with self.assertRaises(RuntimeError):
            with teacher_parameters({'a': student}, {'a': torch.tensor([9.])}):
                raise RuntimeError('teacher forward failed')
        self.assertEqual(student.data_ptr(), original_pointer)
        torch.testing.assert_close(student.detach(), torch.tensor([2.]))


if __name__ == '__main__':
    unittest.main()
