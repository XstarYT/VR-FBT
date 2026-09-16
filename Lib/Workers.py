"""Supervised native camera/model processes with bounded cancellation.

Only one request is outstanding per worker. Images and results stay on this PC.
Keeping native calls outside the GUI process allows a stuck driver/model to be
terminated without releasing a camera concurrently with a native read.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
import traceback


def _native_worker(connection, kind, option):
    resource = None
    try:
        if kind == "camera":
            import cv2
            resource = cv2.VideoCapture(option, getattr(cv2, "CAP_DSHOW", 0))
            if not resource.isOpened():
                raise RuntimeError(f"Could not open local camera {option}")
            resource.set(getattr(cv2, "CAP_PROP_BUFFERSIZE", 38), 1)
        else:
            from Lib.Tracking import Pose
            resource = Pose(option)
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "read":
                ok, frame = resource.read()
                connection.send(("result", (ok, frame, time.monotonic())))
            elif command == "process":
                connection.send(("result", resource.process(payload)))
            else:
                raise RuntimeError(f"Unsupported worker command: {command}")
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        if resource is not None:
            try:
                resource.release() if kind == "camera" else resource.close()
            except Exception:
                pass
        connection.close()


class _ProcessWorker:
    def __init__(self, kind, option, stop_event=None, startup_timeout=15.0, target=_native_worker):
        self.stop_event = stop_event or threading.Event()
        self._closed = False
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=target, args=(child, kind, option),
            name=f"vr-fbt-{kind}-{option}", daemon=True)
        try:
            self.process.start()
            child.close()
            status, detail = self._receive(startup_timeout)
            if status != "ready":
                raise RuntimeError(f"{kind} startup failed: {detail}")
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            if self.stop_event.is_set():
                raise RuntimeError("Tracking was stopped")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.process.name} did not respond within {timeout:g} seconds")
            if self.connection.poll(min(0.05, remaining)):
                try:
                    return self.connection.recv()
                except EOFError as exc:
                    raise RuntimeError(f"{self.process.name} exited unexpectedly") from exc
            if not self.process.is_alive():
                raise RuntimeError(f"{self.process.name} exited with code {self.process.exitcode}")

    def request(self, command, payload=None, timeout=3.0):
        if self._closed:
            raise RuntimeError("Worker is closed")
        try:
            self.connection.send((command, payload))
            status, result = self._receive(timeout)
            if status != "result":
                raise RuntimeError(f"{self.process.name}: {result}")
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.process.is_alive():
                try:
                    self.connection.send(("close", None))
                except (OSError, EOFError):
                    pass
                self.process.join(0.2)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(1.0)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(1.0)
        finally:
            self.connection.close()
            if self.process.pid is not None and not self.process.is_alive():
                self.process.close()


class CameraWorker:
    def __init__(self, index, stop_event=None):
        self.worker = _ProcessWorker("camera", index, stop_event)
        self.captured_at = None

    def isOpened(self):
        return not self.worker._closed

    def read(self):
        ok, frame, self.captured_at = self.worker.request("read")
        return ok, frame

    def release(self):
        self.worker.close()


class PoseWorker:
    def __init__(self, quality, stop_event=None):
        self.worker = _ProcessWorker("pose", quality, stop_event)

    def process(self, frame):
        return self.worker.request("process", frame)

    def close(self):
        self.worker.close()
