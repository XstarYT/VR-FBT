import json
from pathlib import Path
import tempfile
import threading
import unittest
from zipfile import ZipFile

from Lib.Config import Profile, Settings
from Lib.Metrics import SAMPLE_LIMIT, SessionMetrics
from Lib.Support import export_support_bundle


class SessionMetricsTests(unittest.TestCase):
    def test_rates_decay_and_status_distinguishes_pose_loss_stale_and_failure(self):
        now = [0.0]
        metrics = SessionMetrics(("phone:private-id",), clock=lambda: now[0])
        self.assertEqual(metrics.snapshot()["sources"][0]["status"], "waiting")
        for frame in range(1, 21):
            now[0] = frame / 10
            metrics.capture("phone:private-id", now[0])
            metrics.update()
        metrics.inference("phone:private-id", 20, 1.98, 2.0, False)
        row = metrics.snapshot()["sources"][0]
        self.assertEqual(row["capture_fps"], 10)
        self.assertEqual(row["status"], "no pose")
        now[0] = 2.6
        self.assertEqual(metrics.snapshot()["sources"][0]["status"], "stale")
        now[0] = 4.1
        self.assertEqual(metrics.snapshot()["update_fps"], 0)
        self.assertEqual(metrics.snapshot()["sources"][0]["capture_fps"], 0)
        metrics.fail("phone:private-id")
        self.assertEqual(metrics.snapshot()["sources"][0]["status"], "failed")
        self.assertNotIn("private-id", json.dumps(metrics.snapshot()))

    def test_latency_windows_are_bounded_and_skips_count_each_capture_once(self):
        metrics = SessionMetrics(("local:0",))
        metrics.select("local:0", 3)
        metrics.select("local:0", 3)
        metrics.select("local:0", 5)
        for index in range(SAMPLE_LIMIT * 3):
            metrics.inference("local:0", 20, 1, 1.03, True)
            metrics.stage("fusion", 4)
        snapshot = metrics.snapshot()
        row = snapshot["sources"][0]
        self.assertEqual(row["capture_skipped"], 3)
        self.assertEqual(row["inference"]["count"], SAMPLE_LIMIT)
        self.assertEqual(row["inference"]["p95_ms"], 20)
        self.assertEqual(row["capture_to_result"]["p99_ms"], 30)
        self.assertEqual(snapshot["stages"]["fusion"]["count"], SAMPLE_LIMIT)

    def test_snapshots_are_thread_safe_detached_and_freeze_on_stop(self):
        now = [10.0]
        metrics = SessionMetrics(("local:0",), clock=lambda: now[0])

        def record():
            for _ in range(1000):
                metrics.capture("local:0", 10)

        threads = [threading.Thread(target=record) for _ in range(3)]
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            json.dumps(metrics.snapshot(), allow_nan=False)
        for thread in threads:
            thread.join()
        self.assertEqual(metrics.snapshot()["sources"][0]["captures"], 3000)
        snapshot = metrics.snapshot()
        snapshot["sources"][0]["captures"] = -1
        self.assertEqual(metrics.snapshot()["sources"][0]["captures"], 3000)
        now[0] = 12
        metrics.finish()
        now[0] = 100
        self.assertEqual(metrics.snapshot()["duration_seconds"], 2)
        self.assertEqual(metrics.snapshot()["sources"][0]["status"], "stopped")

    def test_support_export_includes_numeric_metrics_without_identifier_leaks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = export_support_bundle(root / "support.zip", Settings(), Profile(name="1"),
                logs=root / "missing", diagnostics=lambda _report: None,
                runtime_metrics={"update_fps": 1, "status": "phone:private-device", "sources": []})
            with ZipFile(target) as archive:
                summary = json.loads(archive.read("summary.json"))
            self.assertEqual(summary["runtime_metrics"]["update_fps"], 1)
            self.assertNotIn("private-device", json.dumps(summary))
