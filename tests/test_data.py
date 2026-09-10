import unittest

from Lib import Data
from Lib.Config import load_joint_map


class LandmarkDataTests(unittest.TestCase):
    def test_map_updates_and_bounds_history(self):
        pose_map = Data.Map(load_joint_map("FULLMAP"))
        landmarks = [[float(i), float(i + 1), float(i + 2), 0.8] for i in range(31)]
        for _ in range(20):
            pose_map.Update(landmarks)
        self.assertEqual(pose_map.KeyPoints["Nose"].pos, [0.0, 1.0, 2.0])
        self.assertEqual(len(pose_map.KeyPoints["Nose"].his), 15)

    def test_smoothing_reduces_step_change(self):
        pose_map = Data.Map(load_joint_map("FULLMAP"))
        pose_map.Update([[0, 0, 0, 1] for _ in range(31)])
        pose_map.Update([[10, 10, 10, 1] for _ in range(31)], smooth=True)
        # Adaptive smoothing should reduce the jump without imposing the old,
        # high-latency fixed 0.35 blend on deliberate large movements.
        self.assertTrue(all(0.0 < value < 10.0 for value in pose_map.KeyPoints["Nose"].pos))
        self.assertGreater(pose_map.KeyPoints["Nose"].pos[0], 3.5)

    def test_fused_point(self):
        left = Data.keypoint([0, 2, 4], 0.4)
        right = Data.keypoint([2, 4, 6], 0.8)
        fused = Data.Map.make_fused(left, right, [1, -1, 0])
        self.assertEqual(fused.pos, [2.0, 2.0, 5.0])
        self.assertAlmostEqual(fused.vis, 0.6)

    def test_occluded_joint_retains_last_good_position(self):
        pose_map = Data.Map(load_joint_map("FULLMAP"))
        visible = [[1, 2, 3, 0.9] for _ in range(31)]
        hidden = [[99, 99, 99, 0.1] for _ in range(31)]
        pose_map.Update(visible)
        pose_map.Update(hidden, smooth=True)
        self.assertEqual(pose_map.KeyPoints["L-Knee"].pos, [1.0, 2.0, 3.0])
        self.assertEqual(pose_map.KeyPoints["L-Knee"].vis, 0.1)

    def test_filters_return_values_and_validate_cutoff(self):
        samples = [0, 1, 0, -1, 0]
        self.assertEqual(len(Data.legendre_lowpass_o2(samples, 3, 30)), len(samples))
        self.assertEqual(len(Data.legendre_highpass_o2(samples, 3, 30)), len(samples))
        with self.assertRaises(ValueError):
            Data.legendre_lowpass_o2(samples, 20, 30)


if __name__ == "__main__":
    unittest.main()
