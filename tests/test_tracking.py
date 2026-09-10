from dataclasses import dataclass
import unittest

import numpy as np

from Lib.Tracking import Pose, convert_mediapipe_landmarks


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float
    visibility: float = 0.9
    presence: float = 0.8


class MediaPipeTrackingTests(unittest.TestCase):
    def test_33_landmarks_map_to_31_world_points_and_real_feet(self):
        image = [FakeLandmark(i / 100, i / 100 + 0.1, i / 100 + 0.2) for i in range(33)]
        world = [FakeLandmark(float(i), float(i + 1), float(i + 2)) for i in range(33)]
        result = convert_mediapipe_landmarks(image, world)
        self.assertTrue(result.detected)
        self.assertEqual(len(result.image_landmarks), 31)
        self.assertEqual(len(result.world_landmarks), 31)
        self.assertEqual(result.image_landmarks[29][0], image[31].x)
        self.assertEqual(result.image_landmarks[30][0], image[32].x)
        self.assertEqual(result.world_landmarks[11][:3], [-11.0, -12.0, -13.0])
        self.assertAlmostEqual(result.confidence, 0.8)

    def test_incomplete_result_is_reported_as_no_pose(self):
        result = convert_mediapipe_landmarks([], [])
        self.assertFalse(result.detected)
        self.assertEqual(result.image_landmarks, [])
        self.assertEqual(result.world_landmarks, [])

    def test_real_lite_runtime_accepts_video_frames_and_closes(self):
        tracker = Pose("lite")
        try:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            first = tracker.process(frame, 100)
            second = tracker.process(frame, 100)  # wrapper makes timestamps monotonic
            self.assertIsInstance(first.detected, bool)
            self.assertIsInstance(second.detected, bool)
            self.assertGreaterEqual(second.confidence, 0.0)
        finally:
            tracker.close()
            tracker.close()


if __name__ == "__main__":
    unittest.main()
