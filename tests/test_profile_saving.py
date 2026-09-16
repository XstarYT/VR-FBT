from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Lib import Config


class ProfileSavingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.profiles = root / "profiles"
        self.settings_path = root / "settings.json"
        self.paths = patch.multiple(Config, PROFILES_DIR=self.profiles, SETTINGS_PATH=self.settings_path)
        self.paths.start()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self.paths.stop)
        self.profile = Config.Profile()
        Config.save_configuration(Config.Settings(), self.profile)
        self.original_profile = (self.profiles / "Default.toml").read_bytes()
        self.original_settings = self.settings_path.read_bytes()

    def test_invalid_global_settings_do_not_modify_saved_rig(self):
        with self.assertRaises(Config.ConfigurationError):
            Config.save_configuration(Config.Settings(fps=0), replace(self.profile, user_height_m=1.95))
        self.assertEqual((self.profiles / "Default.toml").read_bytes(), self.original_profile)
        self.assertEqual(self.settings_path.read_bytes(), self.original_settings)

    def test_settings_write_failure_restores_existing_profile(self):
        with patch.object(Config, "save_settings", side_effect=PermissionError("settings locked")):
            with self.assertRaises(PermissionError):
                Config.save_configuration(Config.Settings(fps=60), replace(self.profile, user_height_m=1.95))
        self.assertEqual((self.profiles / "Default.toml").read_bytes(), self.original_profile)
        self.assertEqual(self.settings_path.read_bytes(), self.original_settings)

    def test_new_profile_is_removed_when_default_preference_cannot_be_saved(self):
        with patch.object(Config, "save_settings", side_effect=PermissionError("settings locked")):
            with self.assertRaises(PermissionError):
                Config.save_configuration(Config.Settings(default_profile="New rig"), replace(self.profile, name="New rig"), create=True)
        self.assertFalse((self.profiles / "New rig.toml").exists())
        self.assertEqual(self.settings_path.read_bytes(), self.original_settings)

    def test_profile_copy_preserves_original_and_becomes_default(self):
        copied = replace(self.profile, name="Three cameras", camera_sources=("phone:a", "phone:b", "phone:c"), tracking_mode="MULTI")
        copied = replace(copied, camera_setups=tuple(Config.CameraSetup(source, (0, 1.4, -2), (0, 0, 0), latency_ms=index * 25)
            for index, source in enumerate(copied.camera_sources)))
        Config.save_configuration(Config.Settings(fps=24, default_profile=copied.name), copied, create=True)
        loaded = Config.load_profile(copied.name)
        self.assertEqual(loaded.camera_sources, copied.camera_sources)
        self.assertEqual(loaded.tracking_mode, "MULTI")
        self.assertEqual([setup.latency_ms for setup in loaded.camera_setups], [0, 25, 50])
        self.assertEqual(Config.load_settings(self.settings_path).default_profile, copied.name)
        self.assertEqual((self.profiles / "Default.toml").read_bytes(), self.original_profile)

    def test_existing_case_variant_cannot_be_overwritten_by_save_as(self):
        with self.assertRaisesRegex(Config.ConfigurationError, "already exists"):
            Config.save_configuration(Config.Settings(default_profile="default"), replace(self.profile, name="default"), create=True)
        self.assertEqual((self.profiles / "Default.toml").read_bytes(), self.original_profile)

    def test_windows_reserved_names_are_rejected_before_writing(self):
        for name in ("CON", "nul", "COM1", "LPT9", "COM¹", "../escape", "x" * 81):
            with self.subTest(name=name), self.assertRaises(Config.ConfigurationError):
                Config.save_profile(replace(self.profile, name=name))
        self.assertEqual(list(self.profiles.iterdir()), [self.profiles / "Default.toml"])

    def test_rollback_failure_is_reported_explicitly(self):
        atomic_write = Config._atomic_write
        calls = []

        def fail_restore(path, text):
            calls.append(path)
            if len(calls) == 2:
                raise PermissionError("profile locked")
            return atomic_write(path, text)

        with patch.object(Config, "_atomic_write", side_effect=fail_restore), patch.object(Config, "save_settings", side_effect=PermissionError("settings locked")):
            with self.assertRaisesRegex(Config.ConfigurationError, "rollback also failed"):
                Config.save_configuration(Config.Settings(), replace(self.profile, user_height_m=1.95))
