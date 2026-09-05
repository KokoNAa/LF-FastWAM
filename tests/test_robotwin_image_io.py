import unittest

from experiments.robotwin.image_io import decode_legacy_robotwin_rgb


class LegacyRGBTest(unittest.TestCase):
    def test_rebuild_keeps_actions_proprio_and_language_deltas(self):
        import torch
        from scripts.rebuild_robotwin_rgb_replay import corrected_payload
        payload = {'format': 'robotwin_compact_replay_v1', 'parent_payload': '/parent.pt',
                   'references': {'source': torch.ones(1, 32, 14), 'target': torch.zeros(1, 32, 14)},
                   'capture_deltas': {'source': {'video_inputs': {'x': torch.zeros(1),
                       'context': {'last_token': torch.ones(1, 1, 3)}}},
                       'target': {'action_inputs': {'context': {'last_token': torch.ones(1, 1, 3) * 2}}}}}
        new = corrected_payload(payload, '/old.pt', {'source': torch.ones(1) * 3, 'target': torch.ones(1) * 4})
        self.assertEqual(new['parent_payload'], '/parent.pt')
        for branch in ('source', 'target'):
            torch.testing.assert_close(new['references'][branch], payload['references'][branch])
        torch.testing.assert_close(new['capture_deltas']['source']['video_inputs']['context']['last_token'],
                                   payload['capture_deltas']['source']['video_inputs']['context']['last_token'])
        torch.testing.assert_close(new['capture_deltas']['target']['action_inputs']['context']['last_token'],
                                   payload['capture_deltas']['target']['action_inputs']['context']['last_token'])
        self.assertEqual(payload['capture_deltas']['source']['video_inputs']['x'].item(), 0)
        self.assertEqual(new['capture_deltas']['source']['video_inputs']['x'].item(), 3)
        self.assertEqual(new['capture_deltas']['target']['video_inputs']['x'].item(), 4)

    def test_reconstructs_camera_rgb_after_legacy_opencv_encoding(self):
        import cv2
        import numpy as np
        camera = np.zeros((32, 96, 3), dtype=np.uint8)
        for index, color in enumerate(([240, 20, 10], [20, 230, 30], [10, 20, 240])):
            camera[:, index * 32:(index + 1) * 32] = color
        success, encoded = cv2.imencode('.jpg', camera)
        self.assertTrue(success)
        recovered = decode_legacy_robotwin_rgb(encoded.tobytes() + b'\0' * 20)
        self.assertTrue(recovered.flags.c_contiguous)
        self.assertEqual(recovered.dtype, camera.dtype)
        self.assertEqual(recovered.shape, camera.shape)
        for center in (16, 48, 80):
            np.testing.assert_allclose(recovered[16, center], camera[16, center], atol=3)


if __name__ == '__main__':
    unittest.main()
