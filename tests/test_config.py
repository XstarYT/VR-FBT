import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from Lib.Config import (
    CameraSetup,
    ConfigurationError,
    Profile,
    Settings,
    camera_corner_for_setup,
    camera_setup_for_corner,
    load_joint_map,
    load_profile,
    load_settings,
    save_settings,
    save_profile,
    validate_profile,
    validate_settings,
)


class ConfigurationTests(unittest.TestCase):
    def test_repository_configuration_loads(self):
        settings = load_settings()
        profile = load_profile(settings.default_profile)
        joint_map = load_joint_map(profile.joint_map)
        self.assertEqual(settings.fps, 30)
        self.assertEqual(profile.tracking_mode, "SINGLE")
        self.assertEqual(profile.vrchat_tracker_set, "stable")
        self.assertEqual(len(joint_map["KeyPoints"]), 31)

    def test_settings_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            expected = Settings(fps=72, default_profile="Default")
            save_settings(expected, path)
            self.assertEqual(load_settings(path), expected)

    def test_corner_preset_places_and_aims_a_fixed_camera(self):
        original = CameraSetup("phone:corner", (0.0, 1.4, -2.5), (0.0, 0.0, 0.0), 67.0, 90)
        placed = camera_setup_for_corner(original, "Back right (+X, +Z)", 1.8, (6.0, 3.0, 4.0))
        self.assertEqual(placed.position, (3.0, 1.8, 2.0))
        self.assertEqual(camera_corner_for_setup(placed, (6.0, 3.0, 4.0)), "Back right (+X, +Z)")
        self.assertAlmostEqual(placed.rotation[0], -123.6900675, places=5)
        self.assertLess(placed.rotation[1], 0.0)
        self.assertEqual(placed.horizontal_fov, 67.0)
        self.assertEqual(placed.image_rotation, 90)

    def test_multi_camera_profile_round_trip(self):
        with tempfile.TemporaryDirectory() as directory, patch("Lib.Config.PROFILES_DIR", Path(directory)):
            expected = Profile(
                name="Multi Test",
                camera_source="phone:front",
                camera_sources=("phone:front", "phone:left", "local:1"),
                manual_camera_setup=True,
                camera_setups=(
                    CameraSetup("phone:front", (0.0, 1.4, -2.5), (0.0, -8.0, 0.0), 65.0),
                    CameraSetup("phone:left", (2.5, 1.4, 0.0), (-90.0, -8.0, 0.0), 70.0, 90),
                    CameraSetup("local:1", (-2.5, 1.4, 0.0), (90.0, -8.0, 0.0), 55.0),
                ),
                room_size_m=(5.0, 3.0, 6.0),
                tracking_mode="MULTI",
            )
            save_profile(expected)
            loaded = load_profile("Multi Test")
            self.assertEqual(loaded.camera_sources, expected.camera_sources)
            self.assertEqual(loaded.camera_source, "phone:front")
            self.assertEqual(loaded.tracking_mode, "MULTI")
            self.assertTrue(loaded.manual_camera_setup)
            self.assertEqual(loaded.camera_setups, expected.camera_setups)
            self.assertEqual(loaded.room_size_m, expected.room_size_m)

    def test_invalid_ranges_are_rejected(self):
        with self.assertRaises(ConfigurationError):
            validate_settings(Settings(fps=0))
        with self.assertRaises(ConfigurationError):
            validate_profile(Profile(server_port=70000))
        with self.assertRaises(ConfigurationError):
            validate_profile(Profile(camera_index=-1))
        with self.assertRaisesRegex(ConfigurationError, "image rotation"):
            validate_profile(Profile(camera_setups=(CameraSetup("local:0", (0.0, 1.4, -2.5), (0.0, 0.0, 0.0), 60.0, 45),)))

    def test_multi_mode_accepts_two_or_three_unique_sources(self):
        validate_profile(Profile(
            camera_source="local:0",
            camera_sources=("local:0", "phone:side"),
            tracking_mode="MULTI",
        ))
        validate_profile(Profile(
            camera_source="local:0",
            camera_sources=("local:0", "phone:side", "phone:rear"),
            tracking_mode="MULTI",
        ))
        with self.assertRaisesRegex(ConfigurationError, "between one and three"):
            validate_profile(Profile(camera_sources=("local:0", "phone:a", "phone:b", "phone:c"), tracking_mode="MULTI"))
        with self.assertRaisesRegex(ConfigurationError, "unique"):
            validate_profile(Profile(camera_sources=("local:0", "local:0"), tracking_mode="MULTI"))
        with self.assertRaisesRegex(ConfigurationError, "must be MULTI"):
            validate_profile(Profile(camera_sources=("local:0", "phone:side"), tracking_mode="SINGLE"))

    def test_pose_quality_is_validated(self):
        for quality in ("lite", "full", "heavy"):
            validate_profile(Profile(pose_quality=quality))
        with self.assertRaisesRegex(ConfigurationError, "Pose quality"):
            validate_profile(Profile(pose_quality="enormous"))

    def test_vrchat_height_is_validated(self):
        validate_profile(Profile(user_height_m=1.75))
        with self.assertRaisesRegex(ConfigurationError, "height"):
            validate_profile(Profile(user_height_m=0.8))
        with self.assertRaisesRegex(ConfigurationError, "height"):
            validate_profile(Profile(user_height_m=2.8))

    def test_vrchat_tracker_set_is_validated(self):
        validate_profile(Profile(vrchat_tracker_set="stable"))
        validate_profile(Profile(vrchat_tracker_set="full"))
        with self.assertRaisesRegex(ConfigurationError, "tracker set"):
            validate_profile(Profile(vrchat_tracker_set="everything"))

    def test_camera_source_validation(self):
        validate_profile(Profile(camera_source="local:0"))
        validate_profile(Profile(camera_source="phone:abc-123"))
        with self.assertRaisesRegex(ConfigurationError, "Camera source"):
            validate_profile(Profile(camera_source="rtsp://example"))
        with self.assertRaisesRegex(ConfigurationError, "non-negative"):
            validate_profile(Profile(camera_source="local:-1"))


if __name__ == "__main__":
    unittest.main()
