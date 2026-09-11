import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from Lib.Engine import _LatestFrameReader


ROOT = Path(__file__).resolve().parents[1]


class ReviewRegressionTests(unittest.TestCase):
    def test_silent_capture_does_not_block_healthy_capture_worker(self):
        class Capture:
            def __init__(self, delay, succeeds):
                self.delay = delay
                self.succeeds = succeeds
                self.open = True

            def isOpened(self):
                return self.open

            def read(self):
                time.sleep(self.delay)
                return (True, object()) if self.succeeds else (False, None)

            def release(self):
                self.open = False

        silent_capture = Capture(0.30, False)
        healthy_capture = Capture(0.005, True)
        silent = _LatestFrameReader(silent_capture, "silent")
        healthy = _LatestFrameReader(healthy_capture, "healthy")
        silent.start(); healthy.start()
        try:
            time.sleep(0.08)
            latest = healthy.latest(0)
            self.assertIsNotNone(latest)
            self.assertGreater(latest[0], 3)
        finally:
            silent.stop(); healthy.stop()
            silent_capture.release(); healthy_capture.release()
            silent.join(0.5); healthy.join(0.5)

    def test_nested_launcher_imports_root_from_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "VR-FBT-main" / "Main.py"), "--check"],
                cwd=directory, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("[PASS] Settings", completed.stdout)


if __name__ == "__main__":
    unittest.main()
