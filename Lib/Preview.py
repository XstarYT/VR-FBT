"""Capture-only camera preview, with no pose model or OSC sender."""

import threading


class CapturePreview:
    def __init__(self, sources, registry, rotations=(), *, cv2_module=None, reader_factory=None):
        if cv2_module is None:
            import cv2 as cv2_module
        self.cv2 = cv2_module
        self.sources = tuple(sources)
        self.registry = registry
        self.rotations = dict(rotations)
        self.reader_factory = reader_factory or self._open_reader
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._sequence = 0
        self._thread = threading.Thread(target=self._run, name="camera-preview", daemon=True)

    def _open_reader(self, source):
        if source.startswith("phone:"):
            from Lib.RemoteCam import RemoteCapture
            return RemoteCapture(self.registry, source.split(":", 1)[1], self.cv2, timeout=0.12)
        return self.cv2.VideoCapture(int(source.split(":", 1)[1]), getattr(self.cv2, "CAP_DSHOW", 0))

    def start(self):
        self._thread.start()

    def stop(self, timeout=3.0):
        self._stop.set()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def latest(self):
        with self._lock:
            return self._latest

    def _run(self):
        import numpy as np
        from Lib.DebugView import fit_image_to_viewport, rotate_camera_frame

        readers = []
        try:
            for source in self.sources:
                try:
                    reader = self.reader_factory(source)
                    if not reader.isOpened() and not source.startswith("phone:"):
                        reader.release()
                        reader = None
                except Exception:
                    reader = None
                readers.append(reader)
            while not self._stop.is_set():
                tiles, received = [], 0
                for index, (source, reader) in enumerate(zip(self.sources, readers), 1):
                    frame = None
                    if reader is not None:
                        try:
                            ok, candidate = reader.read()
                            if ok and getattr(candidate, "ndim", 0) == 3:
                                frame = candidate
                        except Exception:
                            pass
                    if frame is not None:
                        received += 1
                        native_height, native_width = frame.shape[:2]
                        frame = rotate_camera_frame(frame, self.rotations.get(source, 0), self.cv2)
                        tile = fit_image_to_viewport(frame, (320, 240), self.cv2)
                        caption = f"CAM {index}  {native_width}x{native_height} received"
                    else:
                        tile = np.zeros((240, 320, 3), dtype=np.uint8)
                        caption = f"CAM {index}  waiting for frame"
                    self.cv2.rectangle(tile, (0, 0), (320, 28), (10, 16, 30), -1)
                    self.cv2.putText(tile, caption, (7, 19), self.cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
                    tiles.append(tile)
                mosaic = fit_image_to_viewport(np.hstack(tiles), (420, 200), self.cv2)
                with self._lock:
                    self._sequence += 1
                    self._latest = (self._sequence, mosaic, received)
                self._stop.wait(0.12)
        finally:
            for reader in readers:
                if reader is not None:
                    try:
                        reader.release()
                    except Exception:
                        pass
