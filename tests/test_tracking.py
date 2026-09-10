from dataclasses import dataclass
import unittest

import numpy as np

from Lib.Config import CameraSetup
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

    def test_manual_room_cameras_reconstruct_motion_and_reject_impossible_limb_ray(self):
        import cv2

        width, height = 960, 720
        setups = (
            CameraSetup("phone:front", (0.0, 1.4, -3.0), (0.0, 0.0, 0.0), 60.0),
            CameraSetup("phone:side", (3.0, 1.4, 0.0), (-90.0, 0.0, 0.0), 60.0),
        )
        fusion = MultiCameraPoseFusion(
            tuple(setup.source_id for setup in setups),
            manual_camera_setup=True,
            camera_setups=setups,
            room_size_m=(6.0, 3.0, 6.0),
        )
        body = np.array([
            ((index % 5 - 2) * 0.13, 1.2 + (index // 5 - 3) * 0.18, np.sin(index * 0.6) * 0.12)
            for index in range(31)
        ], dtype=float)

        def observations(projected_body):
            items = []
            for setup in setups:
                rotation, translation, _camera_to_world, _position = fusion._manual_camera_geometry(setup, np)
                intrinsic = fusion._camera_matrix((width, height), np, setup.source_id)
                camera_points = (rotation @ projected_body.T).T + translation
                pixels = (intrinsic @ camera_points.T).T
                pixels = pixels[:, :2] / pixels[:, 2:]
                image = [[u / (width - 1), v / (height - 1), 0.0, 0.95] for u, v in pixels]
                world = [[*point, 0.95] for point in body]
                items.append(CameraObservation(setup.source_id, PoseResult(True, 0.95, image, world), (width, height)))
            return items

        result = fusion.update(observations(body), cv2)
        reconstructed = np.asarray([point[:3] for point in result.pose.world_landmarks])
        self.assertLess(float(np.linalg.norm(reconstructed - body, axis=1).mean()), 0.04)
        self.assertEqual([camera.position for camera in result.camera_poses], [setup.position for setup in setups])

        corrupted = body.copy()
        corrupted[15] = (18.0, 5.0, 12.0)  # perfect ray intersection, impossible human/room location
        corrupted[27] = (-15.0, 4.0, 11.0)  # core ankle fallback must still permit calibration
        result = fusion.update(observations(corrupted), cv2)
        wrist = np.asarray(result.pose.world_landmarks[15][:3])
        self.assertLess(float(np.linalg.norm(wrist - body[15])), 0.5)
        self.assertLess(abs(float(wrist[0])), 3.5)
        self.assertGreaterEqual(result.pose.world_landmarks[27][3], 0.55)

    def test_t_pose_timer_and_secondary_camera_full_limb_takeover(self):
        import cv2

        now = [0.0]
        width, height = 960, 720
        setups = (
            CameraSetup("phone:front", (0.0, 1.4, -3.0), (0.0, 0.0, 0.0), 60.0),
            CameraSetup("phone:side", (3.0, 1.4, 0.0), (-90.0, 0.0, 0.0), 60.0),
        )
        fusion = MultiCameraPoseFusion(
            tuple(setup.source_id for setup in setups),
            calibration_frames=3,
            manual_camera_setup=True,
            camera_setups=setups,
            room_size_m=(6.0, 3.0, 6.0),
            calibration_duration_seconds=10.0,
            clock=lambda: now[0],
        )
        body = np.zeros((31, 3), dtype=float)
        body[:, 1] = 1.0
        body[11], body[12] = (-0.22, 1.4, 0.0), (0.22, 1.4, 0.0)
        body[13], body[14] = (-0.55, 1.4, 0.0), (0.55, 1.4, 0.0)
        body[15], body[16] = (-0.90, 1.4, 0.0), (0.90, 1.4, 0.0)
        body[17], body[18], body[19], body[20], body[21], body[22] = body[15], body[16], body[15], body[16], body[15], body[16]
        body[23], body[24] = (-0.14, 0.9, 0.0), (0.14, 0.9, 0.0)
        body[25], body[26] = (-0.14, 0.5, 0.0), (0.14, 0.5, 0.0)
        body[27], body[28] = (-0.14, 0.08, 0.0), (0.14, 0.08, 0.0)
        body[29], body[30] = (-0.14, 0.04, 0.18), (0.14, 0.04, 0.18)

        def make_observations(primary_left_arm_visible=True):
            items = []
            for setup in setups:
                rotation, translation, _camera_to_world, _position = fusion._manual_camera_geometry(setup, np)
                intrinsic = fusion._camera_matrix((width, height), np, setup.source_id)
                camera_points = (rotation @ body.T).T + translation
                pixels = (intrinsic @ camera_points.T).T
                pixels = pixels[:, :2] / pixels[:, 2:]
                visibility = np.full(31, 0.95)
                world_points = body.copy()
                if setup.source_id == "phone:front" and not primary_left_arm_visible:
                    for index in (11, 13, 15, 17, 19, 21):
                        visibility[index] = 0.05
                        world_points[index] = (9.0, 9.0, 9.0)
                image = [[u / (width - 1), v / (height - 1), 0.0, float(visibility[index])] for index, (u, v) in enumerate(pixels)]
                world = [[*point, float(visibility[index])] for index, point in enumerate(world_points)]
                items.append(CameraObservation(setup.source_id, PoseResult(True, 0.95, image, world), (width, height)))
            return items

        original_wrists = body[[15, 16]].copy()
        body[15, 1] = body[16, 1] = 0.8
        result = fusion.update(make_observations(), cv2)
        self.assertFalse(result.calibrated)
        self.assertIn("T-pose", result.calibration_hint)
        body[[15, 16]] = original_wrists

        for timestamp in (0.0, 4.0, 8.0):
            now[0] = timestamp
            result = fusion.update(make_observations(), cv2)
            self.assertFalse(result.calibrated)
        now[0] = 10.0
        result = fusion.update(make_observations(), cv2)
        self.assertTrue(result.calibrated)
        self.assertAlmostEqual(result.calibration_progress[0], 10.0)

        now[0] = 10.1
        result = fusion.update(make_observations(primary_left_arm_visible=False), cv2)
        for index in (11, 13, 15, 17, 19, 21):
            joint = np.asarray(result.pose.world_landmarks[index][:3])
            self.assertLess(float(np.linalg.norm(joint - body[index])), 0.2)
            self.assertGreater(result.pose.world_landmarks[index][3], 0.55)


if __name__ == "__main__":
    unittest.main()
