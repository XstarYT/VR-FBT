import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from Lib.Config import Profile, Settings
from Lib.Support import LOG_TAIL_BYTES, Redactor, export_support_bundle


class SupportTests(unittest.TestCase):
    def test_bundle_anonymizes_rig_and_redacts_all_exported_logs_and_diagnostics(self):
        profile = Profile(name="Private rig", server_ip="rig.internal", user_height_m=1.831,
            camera_sources=("phone:private-phone", "local:1"), tracking_mode="MULTI")
        secret = "private-session-token-123"
        message = (f"https://rig.internal:123/#token={secret} phone:private-phone "
                   "192.168.0.3 C:\\Users\\someone\\Documents\\private.py "
                   "old phone:other-device 123e4567-e89b-12d3-a456-426614174000")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "vr-fbt.log"
            log.write_text(message, encoding="utf-8")
            (root / "vr-fbt.log.1").write_text(f"token='{secret}'", encoding="utf-8")
            (root / "private.key").write_text("MUST NEVER EXPORT", encoding="utf-8")

            def diagnostics(report):
                report("Profile: Private rig", True, repr(profile))
                report("Connection error", False, message)

            target = export_support_bundle(root / "support.zip", Settings(), profile,
                secrets=(secret,), activity_text=message, logs=log, diagnostics=diagnostics)
            with ZipFile(target) as archive:
                self.assertEqual(set(archive.namelist()), {"summary.json", "activity.txt", "logs/runtime-0.txt", "logs/runtime-1.txt"})
                contents = "\n".join(archive.read(name).decode() for name in archive.namelist())
                for value in (secret, "private-phone", "other-device", "192.168.0.3", "someone", "rig.internal", "Private rig", "1.831", "MUST NEVER EXPORT", "123e4567-e89b-12d3-a456-426614174000"):
                    self.assertNotIn(value, contents)
                summary = json.loads(archive.read("summary.json"))
                self.assertEqual([camera["alias"] for camera in summary["configuration_snapshot"]["cameras"]], ["camera-1", "camera-2"])
                self.assertEqual(summary["configuration_snapshot"]["tracking_mode"], "MULTI")
                self.assertFalse(summary["saved_configuration_diagnostics"][1]["passed"])

    def test_log_input_is_bounded_and_no_logs_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "vr-fbt.log"
            log.write_bytes(b"old-content" + b"x" * LOG_TAIL_BYTES)
            target = export_support_bundle(root / "support.zip", Settings(), Profile(), logs=log, diagnostics=lambda _report: None)
            with ZipFile(target) as archive:
                self.assertEqual(len(archive.read("logs/runtime-0.txt")), LOG_TAIL_BYTES)
                self.assertNotIn(b"old-content", archive.read("logs/runtime-0.txt"))
            target = export_support_bundle(root / "empty.zip", Settings(), Profile(), logs=root / "missing.log", diagnostics=lambda _report: None)
            with ZipFile(target) as archive:
                self.assertEqual(archive.namelist(), ["summary.json"])

    def test_failed_replace_retains_old_archive_and_cleans_staging_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "support.zip"
            target.write_bytes(b"original")
            with patch.object(Path, "replace", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    export_support_bundle(target, Settings(), Profile(), logs=root / "missing", diagnostics=lambda _report: None)
            self.assertEqual(target.read_bytes(), b"original")
            self.assertEqual(list(root.iterdir()), [target])

    def test_short_identifier_does_not_mangle_other_words(self):
        self.assertEqual(Redactor((("A", "<profile>"),))("Profile: A; Camera available"), "Profile: <profile>; Camera available")

    def test_cli_support_export_does_not_start_gui(self):
        from Main import main
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "support.zip"
            with patch("Lib.GUI.launch", side_effect=AssertionError("GUI must not start")):
                self.assertEqual(main(["--support-bundle", str(target)]), 0)
            with ZipFile(target) as archive:
                self.assertIn("summary.json", archive.namelist())

    def test_cli_can_export_when_saved_configuration_is_broken(self):
        from Main import main
        with tempfile.TemporaryDirectory() as directory, patch("Lib.Config.load_settings", side_effect=ValueError("Malformed settings JSON")):
            target = Path(directory) / "broken.zip"
            self.assertEqual(main(["--support-bundle", str(target)]), 0)
            with ZipFile(target) as archive:
                summary = json.loads(archive.read("summary.json"))
            self.assertIsNone(summary["configuration_snapshot"])
            self.assertIn("Malformed settings JSON", summary["configuration_snapshot_error"])
            self.assertTrue(any(not report["passed"] for report in summary["saved_configuration_diagnostics"]))
