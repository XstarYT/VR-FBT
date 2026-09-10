import tempfile
from pathlib import Path
import unittest

from Lib.Config import (
    ConfigurationError,
    Profile,
    Settings,
    load_joint_map,
    load_profile,
    load_settings,
    save_settings,
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

    def test_invalid_ranges_are_rejected(self):
        with self.assertRaises(ConfigurationError):
            validate_settings(Settings(fps=0))
        with self.assertRaises(ConfigurationError):
            validate_profile(Profile(server_port=70000))
        with self.assertRaises(ConfigurationError):
            validate_profile(Profile(camera_index=-1))

    def test_unimplemented_multi_mode_is_rejected(self):
        with self.assertRaisesRegex(ConfigurationError, "Only SINGLE"):
            validate_profile(Profile(tracking_mode="MULTI"))

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
