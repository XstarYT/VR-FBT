import math
import unittest

import cv2
import numpy as np

from Lib.DebugView import DebugScene3D, project_landmarks, render_camera_mosaic, render_tracking_debug
from Lib.Tracking import CameraPose, PoseResult


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

    def test_rotatable_3d_scene_draws_skeleton_and_camera(self):
        scene = DebugScene3D(cv2, 640, 480)
        landmarks = [[(index % 4) * 0.12, index * 0.04 - 0.6, (index % 3) * 0.08, 0.9] for index in range(31)]
        camera = CameraPose(
            "phone:test",
            (1.5, 0.8, -2.5),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            2.0,
        )
        rendered = scene.render(landmarks, [camera], 0.88, 28.0, "TRACKING", 1)
        self.assertEqual(rendered.shape, (480, 640, 3))
        self.assertGreater(np.count_nonzero(rendered), 10_000)
        fixed_center = scene._world_center.copy()
        shifted = [[x + 0.8, y, z + 0.4, visibility] for x, y, z, visibility in landmarks]
        scene.render(shifted, [camera], 0.88, 28.0, "TRACKING", 1)
        self.assertTrue(np.array_equal(scene._world_center, fixed_center))
        original_yaw = scene.yaw
        scene.mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 100, 0)
        scene.mouse_callback(cv2.EVENT_MOUSEMOVE, 150, 110, 0)
        scene.mouse_callback(cv2.EVENT_LBUTTONUP, 150, 110, 0)
        self.assertNotEqual(scene.yaw, original_yaw)
        scene.reset_world()
        self.assertIsNone(scene._world_center)

    def test_camera_mosaic_restores_all_annotated_views(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        landmarks = [[0.5, 0.5, 0.0, 0.9] for _ in range(31)]
        pose = PoseResult(True, 0.9, landmarks, landmarks)
        mosaic = render_camera_mosaic([(frame, pose, "front"), (frame, pose, "side")], cv2, (320, 240))
        self.assertEqual(mosaic.shape, (240, 640, 3))
        self.assertGreater(np.count_nonzero(mosaic), 1000)


if __name__ == "__main__":
    unittest.main()
