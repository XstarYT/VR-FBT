from dataclasses import dataclass
import math
import unittest

from Lib.VRChat import VRChatPoseSolver, rotation_from_up_forward
from Lib.Engine import TrackingController
from Lib import OSCKit


@dataclass
class Point:
    pos: tuple[float, float, float]
    vis: float = 0.95


def neutral_pose():
    return {
        "L-Ear": Point((-0.08, 0.78, 0.0)),
        "R-Ear": Point((0.08, 0.78, 0.0)),
        "L-Shoulder": Point((-0.22, 0.56, 0.0)),
        "R-Shoulder": Point((0.22, 0.56, 0.0)),
        "L-Elbow": Point((-0.34, 0.30, 0.0)),
        "R-Elbow": Point((0.34, 0.30, 0.0)),
        "L-Hip": Point((-0.14, 0.0, 0.0)),
        "R-Hip": Point((0.14, 0.0, 0.0)),
        "L-Knee": Point((-0.14, -0.44, 0.0)),
        "R-Knee": Point((0.14, -0.44, 0.0)),
        "L-Ankle": Point((-0.14, -0.88, 0.0)),
        "R-Ankle": Point((0.14, -0.88, 0.0)),
        "L-Foot": Point((-0.14, -0.91, 0.18)),
        "R-Foot": Point((0.14, -0.91, 0.18)),
    }


class VRChatPoseTests(unittest.TestCase):
    def test_identity_orientation_is_zero(self):
        rotation = rotation_from_up_forward((0, 1, 0), (0, 0, 1))
        self.assertIsNotNone(rotation)
        for angle in rotation:
            self.assertAlmostEqual(angle, 0.0, places=6)

    def test_euler_extraction_matches_vrchat_z_x_y_convention(self):
        yaw = rotation_from_up_forward((0, 1, 0), (1, 0, 0))
        pitch = rotation_from_up_forward((0, math.sqrt(0.5), math.sqrt(0.5)), (0, -math.sqrt(0.5), math.sqrt(0.5)))
        roll = rotation_from_up_forward((-math.sqrt(0.5), math.sqrt(0.5), 0), (0, 0, 1))
        self.assertAlmostEqual(yaw[1], 90.0, places=5)
        self.assertAlmostEqual(pitch[0], 45.0, places=5)
        self.assertAlmostEqual(roll[2], 45.0, places=5)
        self.assertIsNone(rotation_from_up_forward((0, 1, 0), (0, 2, 0)))

    def test_solver_builds_all_numeric_tracker_addresses(self):
        frame = VRChatPoseSolver(1.70).solve(neutral_pose())
        self.assertEqual(set(frame.trackers), {str(index) for index in range(1, 9)})
        self.assertGreaterEqual(frame.scale, 0.65)
        self.assertLessEqual(frame.scale, 1.55)
        for tracker in frame.trackers.values():
            self.assertEqual(len(tracker.position), 3)
            self.assertEqual(len(tracker.rotation), 3)
            self.assertTrue(all(math.isfinite(value) for value in tracker.position + tracker.rotation))

    def test_tracker_roles_use_expected_joints(self):
        pose = neutral_pose()
        frame = VRChatPoseSolver(1.70).solve(pose)
        scale = frame.scale
        self.assertAlmostEqual(frame.trackers["2"].position[1], 0.0, places=6)
        self.assertAlmostEqual(frame.trackers["3"].position[0], -0.31 * scale, places=6)
        self.assertAlmostEqual(frame.trackers["7"].position[1], -0.88 * scale, places=6)

    def test_low_visibility_is_propagated_for_output_filtering(self):
        pose = neutral_pose()
        solver = VRChatPoseSolver()
        initial = solver.solve(pose, timestamp=1.0, smooth=False)
        pose["L-Knee"].vis = 0.2
        held = solver.solve(pose, timestamp=1.1, smooth=False)
        self.assertEqual(held.trackers["5"].confidence, 0.5)
        self.assertEqual(held.trackers["7"].confidence, 0.5)
        frame = solver.solve(pose, timestamp=1.5, smooth=False)
        self.assertEqual(frame.trackers["5"].confidence, 0.2)
        self.assertEqual(frame.trackers["7"].confidence, 0.2)
        self.assertGreater(frame.trackers["6"].confidence, 0.5)

    def test_tracker_hold_and_reacquisition_match_actual_osc_output(self):
        solver = VRChatPoseSolver()
        pose = neutral_pose()
        solver.solve(pose, timestamp=0.0, smooth=False)

        class Server:
            def __init__(self): self.messages = []
            def Send(self, message): self.messages.append(message)

        pose["L-Foot"].vis = 0.40
        held = solver.solve(pose, timestamp=0.10, smooth=False)
        server = Server()
        TrackingController._send_vrchat_frame(held, server, OSCKit)
        self.assertTrue(held.trackers["7"].held)
        self.assertTrue(any("/7/" in path for path, _value in server.messages))

        pose["L-Foot"].vis = 0.20
        expired = solver.solve(pose, timestamp=0.50, smooth=False)
        server = Server()
        TrackingController._send_vrchat_frame(expired, server, OSCKit)
        self.assertFalse(expired.trackers["7"].output_eligible)
        self.assertFalse(any("/7/" in path for path, _value in server.messages))

        pose["L-Foot"].vis = 0.50
        not_reacquired = solver.solve(pose, timestamp=0.55, smooth=False)
        self.assertFalse(not_reacquired.trackers["7"].output_eligible)
        pose["L-Foot"].vis = 0.55
        reacquired = solver.solve(pose, timestamp=0.60, smooth=False)
        self.assertTrue(reacquired.trackers["7"].output_eligible)

    def test_multiframe_calibration_waits_for_stable_full_body(self):
        solver = VRChatPoseSolver(1.70, calibration_frames=5)
        obscured = neutral_pose()
        obscured["L-Ankle"].vis = 0.2
        self.assertIsNone(solver.solve(obscured, timestamp=0.0))
        self.assertEqual(solver.calibration_progress, (0, 5))
        for index in range(4):
            self.assertIsNone(solver.solve(neutral_pose(), timestamp=0.1 + index * 0.04))
        frame = solver.solve(neutral_pose(), timestamp=0.3)
        self.assertIsNotNone(frame)
        self.assertFalse(solver.calibrating)

    def test_calibration_rejects_unstable_turning(self):
        solver = VRChatPoseSolver(calibration_frames=3)
        for index, degrees in enumerate((-20, 0, 20)):
            angle = math.radians(degrees)
            cosine, sine = math.cos(angle), math.sin(angle)
            pose = neutral_pose()
            for point in pose.values():
                x, y, z = point.pos
                point.pos = (cosine * x + sine * z, y, -sine * x + cosine * z)
            self.assertIsNone(solver.solve(pose, timestamp=index * 0.04, smooth=False))
        self.assertTrue(solver.calibrating)

    def test_neutral_yaw_calibration_faces_tracker_space_forward(self):
        angle = math.radians(25)
        cosine, sine = math.cos(angle), math.sin(angle)
        pose = neutral_pose()
        for point in pose.values():
            x, y, z = point.pos
            point.pos = (cosine * x + sine * z, y, -sine * x + cosine * z)
        solver = VRChatPoseSolver(calibration_frames=3)
        self.assertIsNone(solver.solve(pose, timestamp=0.0, smooth=False))
        self.assertIsNone(solver.solve(pose, timestamp=0.04, smooth=False))
        frame = solver.solve(pose, timestamp=0.08, smooth=False)
        self.assertAlmostEqual(abs(solver.neutral_yaw), 25.0, places=4)
        self.assertAlmostEqual(frame.trackers["1"].rotation[1], 0.0, places=4)

    def test_one_euro_filter_reduces_single_frame_position_jitter(self):
        solver = VRChatPoseSolver()
        first = solver.solve(neutral_pose(), timestamp=1.0, smooth=True)
        noisy = neutral_pose()
        noisy["L-Knee"].pos = (-0.04, -0.44, 0.0)
        second = solver.solve(noisy, timestamp=1.0 + 1 / 30, smooth=True)
        raw_change = 0.10 * first.scale
        actual_change = second.trackers["5"].position[0] - first.trackers["5"].position[0]
        self.assertGreater(actual_change, 0.0)
        self.assertLess(actual_change, raw_change)


if __name__ == "__main__":
    unittest.main()
