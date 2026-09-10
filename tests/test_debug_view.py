import math
import unittest

import cv2
import numpy as np

from Lib.DebugView import project_landmarks, render_tracking_debug


class TrackingDebugViewTests(unittest.TestCase):
    def test_projects_valid_points_and_rejects_bad_values(self):
        points = project_landmarks([
            [0.5, 0.25, 0.0, 0.9],
            [math.nan, 0.4, 0.0, 1.0],
            [4.0, 0.5, 0.0, 1.0],
            ["bad"],
        ], 200, 100)
        self.assertEqual(points[0], (100, 25, 0.9))
        self.assertEqual(points[1:], [None, None, None])

    def test_overlay_draws_without_modifying_camera_frame(self):
        source = np.zeros((360, 640, 3), dtype=np.uint8)
        original = source.copy()
        landmarks = [[0.5, 0.5, 0.0, 0.9] for _ in range(31)]
        landmarks[11][:2] = [0.35, 0.35]
        landmarks[12][:2] = [0.65, 0.35]
        landmarks[23][:2] = [0.40, 0.60]
        landmarks[24][:2] = [0.60, 0.60]
        rendered = render_tracking_debug(source, landmarks, 0.92, 29.7, 123, "test camera", cv2)
        self.assertEqual(rendered.shape, source.shape)
        self.assertTrue(np.array_equal(source, original))
        self.assertGreater(np.count_nonzero(rendered), 1000)

    def test_low_confidence_and_short_landmark_list_are_safe(self):
        source = np.zeros((120, 160, 3), dtype=np.uint8)
        rendered = render_tracking_debug(source, [[0.5, 0.5, 0.0, 0.1]], 0.1, 2.0, 1, "phone", cv2)
        self.assertEqual(rendered.shape, source.shape)
        self.assertGreater(np.count_nonzero(rendered), 0)


if __name__ == "__main__":
    unittest.main()
