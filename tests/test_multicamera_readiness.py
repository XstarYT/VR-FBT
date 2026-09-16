"""Production multi-camera scheduling and calibration failure cases."""

from dataclasses import replace
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from Lib.Config import CameraSetup
from Lib.Engine import _PoseInference, _LatestFrameReader
from Lib.RemoteCam import RemoteCameraRegistry, RemoteCapture
from Lib.Tracking import CameraObservation, MultiCameraPoseFusion, PoseResult


class MultiCameraReadinessTests(unittest.TestCase):
    def test_geometry_fault_stops_live_controller_without_resending_last_frame(self):
        from types import SimpleNamespace
        from Lib.Config import Profile, Settings
        from Lib.Engine import TrackingController, EngineCallbacks
        from Lib.Tracking import FusionResult
        from Lib.VRChat import TrackerPose, VRChatFrame
        registry=RemoteCameraRegistry();registry.connect('test','Test','synthetic')
        stop=threading.Event();calls=[];logs=[];packets=[]
        pose=PoseResult(True,.95,[[.5,.5,0,.95]]*31,[[0,1,0,.95]]*31)
        def feed():
            while not stop.is_set():
                registry.update_decoded_frame('test',np.zeros((48,64,3),np.uint8));stop.wait(.01)
        def fused(*args):
            if not args[0]:
                return FusionResult(PoseResult(False,0.,[],[]),(),True,(10,10),0)
            calls.append(time.monotonic())
            paused=len(calls)>1
            return FusionResult(pose if not paused else PoseResult(False,0.,[],[]),(),True,(10,10),1,
                                'Geometry fault' if paused else '',0.,paused)
        fusion=Mock();fusion.update.side_effect=fused
        fusion.reset_calibration.side_effect=calls.clear
        solver=Mock();solver.calibrating=False;solver.neutral_yaw=0.
        solver.solve.return_value=VRChatFrame({'2':TrackerPose((0,1,0),(0,0,0),.9)},(0,1.7,0),(0,0,0),1.,.9)
        worker=Mock();worker.process.return_value=pose
        server=Mock();server.target=('127.0.0.1',9000)
        controller=TrackingController(EngineCallbacks(log=lambda *args:logs.append(args)))
        feeder=threading.Thread(target=feed,daemon=True);feeder.start()
        try:
            with patch('Lib.Tracking.MultiCameraPoseFusion',return_value=fusion),patch('Lib.Workers.PoseWorker',return_value=worker),patch('Lib.VRChat.VRChatPoseSolver',return_value=solver),patch('Lib.Data.Map'),patch('Lib.OSCKit.Server',return_value=server),patch.object(controller,'_send_vrchat_frame',side_effect=lambda *args,**kwargs:packets.append(len(calls)) or 1):
                controller.start(Settings(),Profile(camera_sources=('phone:test',),show_output=False),SimpleNamespace(running=True,registry=registry))
                deadline=time.monotonic()+3
                while len(calls)<5 and controller.running and time.monotonic()<deadline:time.sleep(.01)
                controller.stop();self.assertTrue(controller.wait(2),logs)
                self.assertGreaterEqual(len(calls),5,logs)
                self.assertEqual(packets,[1],logs)
                self.assertEqual(solver.solve.call_count,1)
                self.assertIn(('WARN','Geometry fault'),logs)
        finally:
            stop.set();feeder.join(1);controller.stop();controller.wait(2)

    def test_slow_inference_does_not_block_healthy_camera_or_build_queue(self):
        release = threading.Event()
        slow = Mock()
        slow.process.side_effect = lambda _frame: release.wait(3)
        healthy = Mock()
        healthy.process.side_effect = lambda frame: frame
        inference = _PoseInference({"slow": slow, "healthy": healthy})
        try:
            self.assertTrue(inference.submit("slow", "Slow", 1, (1, 1), 0))
            for sequence in range(5):
                self.assertFalse(inference.submit("slow", "Slow", sequence, (1, 1), 0))
                self.assertTrue(inference.submit("healthy", "Healthy", sequence, (1, 1), 0))
                deadline = time.monotonic() + 1
                completed = []
                while not completed and time.monotonic() < deadline:
                    completed = inference.poll()
                    time.sleep(0.001)
                self.assertEqual(len(completed), 1)
                self.assertEqual(completed[0][0], "healthy")
                self.assertEqual(completed[0][5], sequence)
                self.assertIsNone(completed[0][6])
            self.assertTrue(inference.busy("slow"))
            self.assertEqual(slow.process.call_count, 1)
        finally:
            release.set()
            inference.close()

    def test_phone_frame_keeps_original_receive_time_through_capture(self):
        registry = RemoteCameraRegistry()
        registry.connect("phone", "Phone", "local")
        with patch("Lib.RemoteCam.time.monotonic", return_value=100.0):
            registry.update_decoded_frame("phone", np.zeros((2, 2, 3), dtype=np.uint8))
        capture = RemoteCapture(registry, "phone", cv2, timeout=0.01)
        reader = _LatestFrameReader(capture, "phone")
        reader.start()
        try:
            deadline = time.monotonic() + 1
            frame = None
            while frame is None and time.monotonic() < deadline:
                frame = reader.latest(0)
                time.sleep(0.001)
            self.assertIsNotNone(frame)
            self.assertEqual(frame[1], 100.0)
        finally:
            reader.stop()
            reader.join()

    def test_quarter_turn_preserves_native_focal_length(self):
        for rotation in (0, 90, 180, 270):
            setup = CameraSetup("phone:a", (0, 1, -3), (0, 0, 0), 60, rotation)
            fusion = MultiCameraPoseFusion((setup.source_id,), camera_setups=(setup,))
            size = (720, 1280) if rotation in (90, 270) else (1280, 720)
            intrinsic = fusion._camera_matrix(size, np, setup.source_id)
            self.assertAlmostEqual(intrinsic[0, 0], 1280 / (2 * np.tan(np.pi / 6)))
            self.assertAlmostEqual(intrinsic[1, 1], intrinsic[0, 0])
            self.assertEqual(intrinsic[0, 2], size[0] / 2)
            self.assertEqual(intrinsic[1, 2], size[1] / 2)

    def test_cached_camera_frames_and_short_dropout_do_not_inflate_calibration(self):
        now = [0.0]
        setups = (
            CameraSetup("a", (0, 1.4, -3), (0, 0, 0)),
            CameraSetup("b", (3, 1.4, 0), (-90, 0, 0)),
        )
        fusion = MultiCameraPoseFusion(("a", "b"), camera_setups=setups, manual_camera_setup=True,
            calibration_frames=3, calibration_duration_seconds=10, clock=lambda: now[0])
        body = [[0, 1, 0, 0.95] for _ in range(31)]
        for index, point in {
            11: (-0.22, 1.4, 0), 12: (0.22, 1.4, 0),
            13: (-0.55, 1.4, 0), 14: (0.55, 1.4, 0),
            15: (-0.9, 1.4, 0), 16: (0.9, 1.4, 0),
            23: (-0.14, 0.9, 0), 24: (0.14, 0.9, 0),
            25: (-0.14, 0.5, 0), 26: (0.14, 0.5, 0),
            27: (-0.14, 0.08, 0), 28: (0.14, 0.08, 0),
        }.items():
            body[index] = [*point, 0.95]
        pose = PoseResult(True, 0.95, [[0.5, 0.5, 0, 0.95] for _ in range(31)], body)
        observations = [CameraObservation(source, pose, (960, 720), 0) for source in ("a", "b")]

        def update(timestamp, fresh=True, missing=False):
            now[0] = timestamp
            if fresh:
                observations[:] = [replace(item, captured_at=timestamp) for item in observations]
            return fusion.update(observations[:1] if missing else observations, cv2)

        update(0)
        update(0.1)
        update(0.2, fresh=False)
        update(0.3, fresh=False)
        self.assertEqual(len(fusion._alignment_samples["a"]), 2)
        self.assertAlmostEqual(fusion.calibration_progress[0], 0.1)
        update(0.35, missing=True)
        update(0.4)
        self.assertAlmostEqual(fusion.calibration_progress[0], 0.1)
        update(0.5)
        self.assertAlmostEqual(fusion.calibration_progress[0], 0.2)
