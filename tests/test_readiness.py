"""Failure-path checks for configuration, diagnostics, and camera ownership."""

import json
import logging
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from Lib import Config
from Lib.Engine import EngineCallbacks, _LatestFrameReader, TrackingController
from Lib.Logging import _RedactToken, _RedactingFormatter
from Lib import Data, OSCKit
from Lib.Tracking import convert_mediapipe_landmarks
from Lib.VRChat import TrackerPose, VRChatFrame


class ReadinessTests(unittest.TestCase):
    def test_nonfinite_model_output_cannot_poison_pose_history(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                landmarks = [SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.9) for _ in range(33)]
                mapped = Data.Map(Config.load_joint_map("FULLMAP"))
                good = convert_mediapipe_landmarks(landmarks, landmarks)
                mapped.Update(good.world_landmarks)
                previous = mapped.KeyPoints["L-Knee"].pos.copy()
                landmarks[25].x = invalid
                bad = convert_mediapipe_landmarks(landmarks, landmarks)
                self.assertEqual(bad.world_landmarks[25][3], 0)
                mapped.Update(bad.world_landmarks, smooth=True)
                self.assertEqual(mapped.KeyPoints["L-Knee"].pos, previous)
                bad.world_landmarks[25] = [invalid, 1, 2, 0.9]
                mapped.Update(bad.world_landmarks, smooth=True)
                self.assertEqual(mapped.KeyPoints["L-Knee"].pos, previous)
                self.assertEqual(mapped.KeyPoints["L-Knee"].vis, 0)

    def test_sender_withholds_nonfinite_trackers_and_head(self):
        server = Mock()
        frame = VRChatFrame({
            "1": TrackerPose((float("nan"), 0, 0), (0, 0, 0), 0.9, True),
            "2": TrackerPose((0, 1, 0), (0, 0, 0), 0.9, True),
        }, (0, float("inf"), 0), (0, 0, 0), 1.0, 0.9)
        TrackingController._send_vrchat_frame(frame, server, OSCKit, align_head=True)
        self.assertEqual([call.args[0][0] for call in server.Send.call_args_list], [
            "/tracking/trackers/2/position", "/tracking/trackers/2/rotation",
        ])

    def test_rejects_camera_entries_before_normalizing(self):
        setup = Config.CameraSetup("local:0", (0, 1, 2), (0, 0, 0))
        for entries, message in (
            ((setup, setup), "unique"),
            ((Config.CameraSetup("phone:unknown", (0, 1, 2), (0, 0, 0)),), "unselected"),
            ((Config.CameraSetup("local:0", (0, 1), (0, 0, 0)),), "three values"),
        ):
            with self.subTest(entries=entries), self.assertRaisesRegex(Config.ConfigurationError, message):
                Config.validate_profile(Config.Profile(camera_setups=entries))

    def test_bad_joint_indices_fail_during_load(self):
        original = Config.load_joint_map("FULLMAP")
        with tempfile.TemporaryDirectory() as directory, patch.object(Config, "JOINT_MAPS_DIR", Path(directory)):
            for index in (-1, 31, 32, True, "11", 1.5):
                original["KeyPoints"]["Nose"] = index
                (Path(directory) / "Broken.json").write_text(json.dumps(original), encoding="utf-8")
                with self.subTest(index=index), self.assertRaisesRegex(Config.ConfigurationError, "indices"):
                    Config.load_joint_map("Broken")

    def test_failed_atomic_replace_preserves_original_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text("original", encoding="utf-8")
            with patch.object(Path, "replace", side_effect=PermissionError("read only")):
                with self.assertRaises(PermissionError):
                    Config._atomic_write(path, "replacement")
            self.assertEqual(path.read_text(), "original")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_diagnostics_rejects_missing_default_and_broken_binary_import(self):
        reports = []
        real_import = Config.import_module

        def import_module(name):
            if name == "cv2":
                raise ImportError("DLL load failed")
            return real_import(name)

        with patch.object(Config, "load_settings", return_value=Config.Settings(default_profile="missing-profile")), patch.object(Config, "import_module", side_effect=import_module):
            self.assertFalse(Config.run_diagnostics(lambda *report: reports.append(report)))
        outcomes = {name: (ok, detail) for name, ok, detail in reports}
        self.assertFalse(outcomes["Default profile"][0])
        self.assertEqual(outcomes["Dependency: OpenCV"], (False, "DLL load failed"))
        self.assertTrue(outcomes["Pose model: pose_landmarker_full.task"][0])

    def test_capture_exception_is_observable_and_released_by_owner(self):
        calls = []

        class Capture:
            def isOpened(self):
                return True

            def read(self):
                calls.append(("read", threading.get_ident()))
                raise RuntimeError("driver failed")

            def release(self):
                calls.append(("release", threading.get_ident()))

        reader = _LatestFrameReader(Capture(), "broken")
        reader.start()
        reader.join(1)
        self.assertEqual(str(reader.error), "driver failed")
        self.assertEqual([name for name, _ in calls], ["read", "release"])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertNotEqual(calls[0][1], threading.get_ident())

    def test_invalid_start_does_not_create_worker(self):
        controller = TrackingController()
        with self.assertRaises(Config.ConfigurationError):
            controller.start(Config.Settings(fps=0), Config.Profile())
        self.assertFalse(controller.running)

    def test_failed_camera_cleanup_still_closes_socket_and_reports_final_state(self):
        states, logs = [], []
        controller = TrackingController(EngineCallbacks(state=states.append, log=lambda *args: logs.append(args)))
        capture = Mock()
        capture.isOpened.return_value = True
        capture.release.side_effect = RuntimeError("release failed")
        server = Mock(target=("127.0.0.1", 9000))
        with patch("Lib.Workers.CameraWorker", return_value=capture), patch("Lib.OSCKit.Server", return_value=server), patch("Lib.Workers.PoseWorker", side_effect=RuntimeError("model failed")), patch("cv2.destroyAllWindows"):
            controller._run(Config.Settings(), Config.Profile())
        server.close.assert_called_once()
        capture.release.assert_called_once()
        self.assertEqual(states[-1], "error")
        self.assertTrue(any("release failed" in message for _, message in logs))

    def test_phone_tokens_are_redacted_from_persistent_messages(self):
        record = logging.LogRecord("vrfbt", logging.INFO, "", 0, "URL %s", ("https://localhost/#token=secret&device=x",), None)
        _RedactToken().filter(record)
        self.assertEqual(record.getMessage(), "URL https://localhost/#token=[REDACTED]&device=x")
        error = ValueError("https://localhost/?token=exception-secret&device=x")
        record.exc_info = (ValueError, error, None)
        formatted = _RedactingFormatter().format(record)
        self.assertNotIn("exception-secret", formatted)
        self.assertIn("token=[REDACTED]&device=x", formatted)
