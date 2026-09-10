"""Threaded OpenCV camera capture with explicit lifecycle management."""

import threading as thr
import time

import cv2


class AsyncCamError(Exception):
    pass


class Camera:
    class Config:
        def __init__(self):
            self.FPS = 1 / 30
            self.SharedDict = None
            self.Lock = thr.Lock()
            self.Stop = False

    class Async:
        def __init__(self, index, config):
            self._CC = config
            self._index = index
            self._error = None
            self.cap = None
            if self._CC.SharedDict is None:
                raise AsyncCamError("SharedDict is None")
            self._thr = thr.Thread(target=self._loop_, name=f"camera-{index}", daemon=True)

        def _loop_(self):
            failures = 0
            try:
                while not self._CC.Stop:
                    if not self.cap.grab():
                        failures += 1
                        if failures > 3:
                            raise AsyncCamError(f"Camera disconnected at index {self._index}")
                    else:
                        failures = 0
                    time.sleep(max(self._CC.FPS, 0))
            except Exception as exc:
                self._error = exc
            finally:
                if self.cap is not None:
                    self.cap.release()

        def start(self):
            self._CC.Stop = False
            self.cap = cv2.VideoCapture(self._index, getattr(cv2, "CAP_DSHOW", 0))
            if not self.cap.isOpened():
                self.cap.release()
                raise AsyncCamError(f"No camera at index {self._index}")
            self._thr.start()

        def retrive(self):
            if self._error is not None:
                raise AsyncCamError(str(self._error)) from self._error
            if self.cap is None:
                raise AsyncCamError("Camera has not been started")
            with self._CC.Lock:
                ok, frame = self.cap.retrieve()
                if not ok or frame is None:
                    raise AsyncCamError(f"No frame from camera index {self._index}")
                self._CC.SharedDict[str(self._index)] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return self._CC.SharedDict[str(self._index)]

        def stop(self, timeout=2.0):
            self._CC.Stop = True
            self._thr.join(timeout)


class Scale:
    @staticmethod
    def down(img, size):
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def up(img, size):
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
