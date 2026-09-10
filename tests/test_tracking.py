from dataclasses import dataclass
import unittest

import numpy as np

from Lib.Tracking import CameraObservation, MultiCameraPoseFusion, Pose, PoseResult, convert_mediapipe_landmarks


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

    def test_three_camera_calibration_and_triangulation(self):
        import cv2

        width, height = 960, 720
        focal = width / (2 * np.tan(np.radians(30.0)))
        intrinsic = np.array(((focal, 0, width / 2), (0, focal, height / 2), (0, 0, 1)), dtype=float)
        points = np.array([
            ((index % 5 - 2) * 0.16, (index // 5 - 3) * 0.22, np.sin(index * 0.7) * 0.16)
            for index in range(31)
        ], dtype=float)

        def look_at(position):
            position = np.asarray(position, dtype=float)
            forward = -position / np.linalg.norm(position)
            right = np.cross(np.array((0.0, 1.0, 0.0)), forward); right /= np.linalg.norm(right)
            down = -np.cross(forward, right)
            rotation = np.stack((right, down, forward))
            return rotation, -rotation @ position

        cameras = {
            "phone:left": look_at((-1.7, 0.2, -3.8)),
            "phone:right": look_at((1.8, 0.1, -3.6)),
            "phone:rear": look_at((0.1, 1.2, -4.2)),
        }

        def observation(source, world):
            rotation, translation = cameras[source]
            camera_points = (rotation @ world.T).T + translation
            pixels = (intrinsic @ camera_points.T).T
            pixels = pixels[:, :2] / pixels[:, 2:]
            image = [[u / (width - 1), v / (height - 1), 0.0, 0.95] for u, v in pixels]
            world_landmarks = [[*point, 0.95] for point in world]
            return CameraObservation(source, PoseResult(True, 0.95, image, world_landmarks), (width, height))

        fusion = MultiCameraPoseFusion(tuple(cameras), calibration_frames=3)
        for _ in range(3):
            result = fusion.update([observation(source, points) for source in cameras], cv2)
        self.assertTrue(result.calibrated)
        self.assertEqual(len(result.camera_poses), 3)
        calibrated_camera_positions = [camera.position for camera in result.camera_poses]
        self.assertTrue(all(camera.sample_count >= 1 for camera in result.camera_poses))

        moved = points + np.array((0.12, 0.04, -0.03))
        result = fusion.update([observation(source, moved) for source in cameras], cv2)
        reconstructed = np.asarray([point[:3] for point in result.pose.world_landmarks])
        self.assertLess(float(np.linalg.norm(reconstructed - moved, axis=1).mean()), 0.04)
        self.assertEqual(result.contributing_cameras, 3)
        self.assertEqual([camera.position for camera in result.camera_poses], calibrated_camera_positions)


if __name__ == "__main__":
    unittest.main()
