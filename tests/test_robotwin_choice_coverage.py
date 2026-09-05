import unittest
import numpy as np
from scripts.audit_robotwin_choice_coverage import classify, distances


class ChoiceCoverageTest(unittest.TestCase):
    def test_future_window_alignment_and_delayed_choice(self):
        source = np.zeros((120, 14))
        target = source.copy()
        target[80:] = 1
        point, future = distances(source, target)
        exact = np.array([np.sqrt(np.mean((source[i:i+32]-target[i:i+32])**2)) for i in range(89)])
        np.testing.assert_allclose(future, exact, rtol=1e-14)
        self.assertEqual(future[0], 0)
        self.assertEqual(point[60], 0)
        self.assertGreater(future[60], .1)
        masks = classify(point, future, np.zeros(len(point)))
        self.assertFalse(masks['tight'][0])
        self.assertTrue(masks['tight'][60])
        self.assertFalse(masks['loose'][80])

    def test_undecoded_rgb_cannot_be_counted(self):
        masks = classify(np.array([0.]), np.array([1.]), np.array([np.inf]))
        self.assertFalse(any(v[0] for v in masks.values()))


if __name__ == '__main__':
    unittest.main()
