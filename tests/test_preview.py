import time

import cv2
import numpy as np

from Lib.Preview import CapturePreview
from Lib.RemoteCam import RemoteCameraRegistry


class FakeReader:
    def __init__(self):
        self.released = False
        self.read_count = 0

    def isOpened(self):
        return True

    def read(self):
        self.read_count += 1
        return True, np.zeros((120, 160, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def test_capture_only_preview_receives_frames_and_releases_reader():
    reader = FakeReader()
    preview = CapturePreview(("local:0",), None, cv2_module=cv2, reader_factory=lambda _source: reader)
    preview.start()
    deadline = time.monotonic() + 2
    latest = None
    while latest is None and time.monotonic() < deadline:
        latest = preview.latest()
        time.sleep(0.01)
    assert latest is not None
    assert latest[1].shape == (200, 420, 3)
    assert latest[2] == 1
    assert preview.stop()
    assert reader.released
    assert reader.read_count >= 1


def test_phone_preview_reads_decoded_frame_and_records_native_resolution():
    registry = RemoteCameraRegistry()
    registry.connect("phone-id", "Phone", "local")
    registry.update_decoded_frame("phone-id", np.zeros((240, 320, 3), dtype=np.uint8))
    preview = CapturePreview(("phone:phone-id",), registry, cv2_module=cv2)
    preview.start()
    deadline = time.monotonic() + 2
    latest = None
    while (latest is None or latest[2] != 1) and time.monotonic() < deadline:
        latest = preview.latest()
        time.sleep(0.01)
    assert latest is not None and latest[2] == 1
    assert registry.list_cameras(False)[0].frame_size == (320, 240)
    assert preview.stop()
