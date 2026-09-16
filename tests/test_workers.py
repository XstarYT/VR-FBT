"""Real subprocess cancellation tests; no physical camera is opened."""

import multiprocessing
import threading
import time
import unittest
from types import SimpleNamespace

from Lib.Workers import _ProcessWorker, PoseWorker
from Lib.Config import Settings, Profile
from Lib.Engine import EngineCallbacks, TrackingController
from Lib.RemoteCam import RemoteCameraRegistry


def synthetic_worker(connection, kind, option):
    if kind == "hang-on-open":
        time.sleep(60)
        return
    connection.send(("ready", None))
    while True:
        command, payload = connection.recv()
        if command == "close":
            connection.close()
            return
        if option == "hang":
            time.sleep(60)
        else:
            connection.send(("result", payload))


class WorkerTests(unittest.TestCase):
    def assert_worker_exited(self, pid):
        self.assertNotIn(pid, [child.pid for child in multiprocessing.active_children()])

    def test_hung_camera_open_times_out_without_leaking_child(self):
        before = {child.pid for child in multiprocessing.active_children()}
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            _ProcessWorker("hang-on-open", None, startup_timeout=0.2, target=synthetic_worker)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, before)

    def test_hung_request_times_out_and_next_worker_can_start(self):
        worker = _ProcessWorker("test", "hang", target=synthetic_worker)
        pid = worker.process.pid
        try:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                worker.request("process", None, timeout=0.1)
            self.assertLess(time.monotonic() - started, 3)
            self.assert_worker_exited(pid)
        finally:
            worker.close()
        replacement = _ProcessWorker("test", "echo", target=synthetic_worker)
        try:
            self.assertEqual(replacement.request("process", [1, 2, 3]), [1, 2, 3])
        finally:
            replacement.close()

    def test_stop_interrupts_hung_request_before_request_deadline(self):
        stop = threading.Event()
        worker = _ProcessWorker("test", "hang", stop, target=synthetic_worker)
        pid = worker.process.pid
        timer = threading.Timer(0.1, stop.set)
        timer.start()
        try:
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                worker.request("process", None, timeout=30)
            self.assertLess(time.monotonic() - started, 3)
            self.assert_worker_exited(pid)
        finally:
            timer.cancel()
            worker.close()

    def test_real_pose_model_runs_in_worker_and_closes(self):
        import numpy as np

        worker = PoseWorker("lite")
        pid = worker.worker.process.pid
        try:
            result = worker.process(np.zeros((240, 320, 3), dtype=np.uint8))
            self.assertFalse(result.detected)
            self.assertEqual(result.world_landmarks, [])
        finally:
            worker.close()
        self.assert_worker_exited(pid)

    def test_multi_phone_controller_with_real_models_runs_and_stops(self):
        for count in (2, 3):
            with self.subTest(cameras=count):
                self._run_phone_controller(count)

    def _run_phone_controller(self, count):
        import numpy as np

        registry = RemoteCameraRegistry()
        sources = ("front", "side", "rear")[:count]
        for source in sources:
            registry.connect(source, source, "synthetic")
        stop_feeding = threading.Event()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        def feed():
            while not stop_feeding.is_set():
                for source in sources:
                    registry.update_decoded_frame(source, frame)
                stop_feeding.wait(0.02)

        feeder = threading.Thread(target=feed, daemon=True)
        states, errors, stats = [], [], []
        controller = TrackingController(EngineCallbacks(
            state=states.append,
            log=lambda level, message: errors.append(message) if level == "ERROR" else None,
            stats=lambda *sample: stats.append(sample),
        ))
        before = {child.pid for child in multiprocessing.active_children()}
        feeder.start()
        try:
            controller.start(Settings(), Profile(camera_sources=tuple(f"phone:{source}" for source in sources),
                tracking_mode="MULTI", pose_quality="lite", show_output=False),
                SimpleNamespace(running=True, registry=registry))
            deadline = time.monotonic() + 15
            while len(stats) < 2 and time.monotonic() < deadline and controller.running:
                time.sleep(0.02)
            self.assertIn("running", states)
            self.assertGreaterEqual(len(stats), 2, errors)
            self.assertFalse(errors)
            started = time.monotonic()
            controller.stop()
            self.assertTrue(controller.wait(3), errors)
            self.assertLess(time.monotonic() - started, 3)
            self.assertEqual(states[-1], "stopped")
            health = controller.metrics.snapshot()
            self.assertFalse(health["running"])
            self.assertEqual(len(health["sources"]), count)
            self.assertTrue(all(row["captures"] > 0 and row["inference"]["count"] > 0 for row in health["sources"]))
            self.assertEqual(health["osc_packets_sent"], 0)
            self.assertEqual({child.pid for child in multiprocessing.active_children()}, before)
        finally:
            stop_feeding.set()
            feeder.join(1)
            controller.stop()
            controller.wait(3)
