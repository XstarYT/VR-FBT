r"""Regression probes for the fixes from the 2026-09-11 review.

Run: .venv\Scripts\python.exe review/bug_reproductions.py
No physical cameras, saved profiles, or OSC destinations are touched.
"""

from pathlib import Path
import asyncio
import json
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from Lib.Config import CameraSetup, Profile
from Lib.Engine import TrackingController, _LatestFrameReader
from Lib.DebugView import DebugScene3D
from Lib.GUI import VRFBTApp
from Lib.RemoteCam import LocalCamera, RemoteCameraHub, RemoteCameraRegistry, RemoteCapture
from Lib.Tracking import CameraObservation, MultiCameraPoseFusion, PoseResult, convert_mediapipe_landmarks
from Lib.VRChat import VRChatPoseSolver
from Lib import OSCKit
from tests.test_tracking import FakeLandmark
from tests.test_vrchat import neutral_pose
from vrfbt_calib.posegraph import solve_global_translations
from vrfbt_calib.types import PairwiseCalibration


def report(name, detail):
    print(f"PASS {name}: {detail}")


def converted_coordinate_probe():
    width, height = 1280, 720
    setup = CameraSetup("phone:front", (0, 1.4, -3), (0, 0, 0), 60)
    fusion = MultiCameraPoseFusion(
        (setup.source_id,), manual_camera_setup=True,
        camera_setups=(setup,), room_size_m=(6, 3, 6),
    )
    body = np.array([
        ((i % 5 - 2) * .13, (i // 5 - 3) * .18, np.sin(i * .61) * .11)
        for i in range(33)
    ])
    rotation, translation, _, _ = fusion._manual_camera_geometry(setup, np)
    camera_local = (rotation @ body.T).T
    room_body = body + np.array([0, 1.1, 0])
    camera_points = (rotation @ room_body.T).T + translation
    pixels = (fusion._camera_matrix((width, height), np) @ camera_points.T).T
    pixels = pixels[:, :2] / pixels[:, 2:]
    pose = convert_mediapipe_landmarks(
        [FakeLandmark(u / (width - 1), v / (height - 1), 0, .95, .95) for u, v in pixels],
        [FakeLandmark(*point, .95, .95) for point in camera_local],
    )
    result = fusion.update([CameraObservation(setup.source_id, pose, (width, height))], cv2)
    expected = room_body[list(range(29)) + [31, 32]]
    actual = np.array(result.pose.world_landmarks)[:, :3]
    error = np.linalg.norm(actual - expected, axis=1).mean()
    correlation = np.corrcoef(actual[:, 1], expected[:, 1])[0, 1]
    assert error < .05 and correlation > .99
    report("F01 coordinates", f"mean error={error:.3f} m; vertical correlation={correlation:.3f}")


def t_pose():
    body = np.zeros((31, 3))
    body[:, 1] = 1
    values = {
        11: (-.22, 1.4, 0), 12: (.22, 1.4, 0),
        13: (-.55, 1.4, 0), 14: (.55, 1.4, 0),
        15: (-.9, 1.4, 0), 16: (.9, 1.4, 0),
        23: (-.14, .9, 0), 24: (.14, .9, 0),
        25: (-.14, .5, 0), 26: (.14, .5, 0),
        27: (-.14, .08, 0), 28: (.14, .08, 0),
    }
    for index, point in values.items():
        body[index] = point
    return body


def manual_fixture(timed=False):
    clock = [0.0]
    setups = (
        CameraSetup("local:0", (0, 1.4, -3), (0, 0, 0)),
        CameraSetup("local:1", (3, 1.4, 0), (-90, 0, 0)),
    )
    fusion = MultiCameraPoseFusion(
        tuple(s.source_id for s in setups), calibration_frames=3,
        manual_camera_setup=True, camera_setups=setups,
        room_size_m=(6, 3, 6), calibration_duration_seconds=10 if timed else None,
        clock=lambda: clock[0],
    )
    body = t_pose()
    observations = []
    for setup in setups:
        rotation, translation, _, _ = fusion._manual_camera_geometry(setup, np)
        cam = (rotation @ body.T).T + translation
        pixels = (fusion._camera_matrix((960, 720), np) @ cam.T).T
        pixels = pixels[:, :2] / pixels[:, 2:]
        image = [[u / 959, v / 719, 0, .95] for u, v in pixels]
        world = [[*point, .95] for point in body]
        observations.append(CameraObservation(setup.source_id, PoseResult(True, .95, image, world), (960, 720)))
    return fusion, observations, clock


def calibration_gap_probe():
    fusion, observations, clock = manual_fixture(timed=True)
    for timestamp in (0, .1):
        clock[0] = timestamp
        assert not fusion.update(observations, cv2).calibrated
    for timestamp in (1, 5, 10):
        clock[0] = timestamp
        fusion.update([], cv2)
    clock[0] = 10.1
    result = fusion.update(observations, cv2)
    assert not result.calibrated and result.calibration_progress[0] < .2
    report("F02 calibration gap", "missing-pose time does not advance the valid T-pose hold")


def calibration_buffer_probe():
    fusion, observations, clock = manual_fixture(timed=True)
    fusion.manual_camera_setup = False
    fusion._solve_camera = lambda _world, observation, _cv: (
        (np.zeros(3), np.array([0, 0, 3]), 1.) if observation.source_id == "local:0" else None
    )
    for index in range(650):
        clock[0] = index / 30
        fusion.update(observations, cv2)
    count = len(fusion._alignment_samples.get("local:0", ()))
    assert count <= 600 and not fusion.calibrated
    report("F14 calibration buffer", f"failed synchronized calibration keeps alignment history bounded ({count} samples)")


def confidence_probe():
    solver = VRChatPoseSolver()
    points = neutral_pose()
    solver.solve(points, timestamp=0, smooth=False)
    points["L-Foot"].vis = .4
    frame = solver.solve(points, timestamp=.1, smooth=False)
    messages = []
    TrackingController._send_vrchat_frame(frame, SimpleNamespace(Send=messages.append), OSCKit)
    assert any("/7/" in path for path, _ in messages)
    report("F03 confidence hold", "0.95 -> 0.40 emits the bounded held foot pose after 100 ms")


def fallback_probe():
    fusion, observations, _ = manual_fixture()
    fusion.update(observations, cv2)
    for observation in observations:
        observation.pose.image_landmarks[15][3] = .05
        observation.pose.world_landmarks[15] = [9, 9, 9, .05]
    result = fusion.update(observations, cv2)
    wrist = result.pose.world_landmarks[15]
    assert np.linalg.norm(wrist[:3]) < 8 and wrist[3] < .35
    report("F04 invalid fallback", f"unseen out-of-room wrist is bounded with visibility {wrist[3]}")


class Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def fake_gui():
    # Use the real selection methods with simple variable/widget adapters.
    app = SimpleNamespace(
        loaded_profile=Profile(), camera_sources={},
        camera_choices=[Value(), Value("Off"), Value("Off")],
        camera_combos=[SimpleNamespace(configure=lambda **kwargs: None) for _ in range(3)],
        phone_hub=SimpleNamespace(registry=RemoteCameraRegistry()),
        _write_log=lambda *args: None,
    )
    app.camera_choice = app.camera_choices[0]
    app._select_sources = lambda sources: VRFBTApp._select_sources(app, sources)
    return app


def camera_selection_probe():
    app = fake_gui()
    app.loaded_profile = Profile(camera_source="local:2", camera_index=2)
    VRFBTApp._set_camera_options(app, [LocalCamera(0, "USB Camera"), LocalCamera(1, "USB Camera")])
    assert {"local:0", "local:1"}.issubset(app.camera_sources.values())
    report("F05 duplicate names", f"two identical webcams remain selectable as {list(app.camera_sources.values())}")
    desired = ("phone:saved-offline", "local:8")
    app.loaded_profile = Profile(camera_source=desired[0], camera_sources=desired, tracking_mode="MULTI")
    VRFBTApp._set_camera_options(app, [], desired)
    selected = tuple(app.camera_sources.get(choice.get()) for choice in app.camera_choices[:2])
    assert selected == desired
    report("F06 profile selection", "offline saved source IDs are rebuilt and selected exactly")


def missing_baseline_probe():
    def edge(a, b, raw_width):
        return PairwiseCalibration(a, b, "ok", 100, R=np.eye(3),
                                   t_unit=np.array([1., 0, 0]), raw_shoulder_distance=raw_width)
    pairs = [edge("A", "B", .4), edge("B", "C", None)]
    try:
        solve_global_translations(
            ["A", "B", "C"], "A", {c: np.eye(3) for c in "ABC"}, pairs, .4,
        )
    except ValueError as error:
        assert "underdetermined" in str(error)
    else:
        raise AssertionError("Unscaled edge was silently accepted")
    report("F10 missing edge scale", "an edge without shoulder scale is rejected as underdetermined")


def capture_wait_probe():
    registry = RemoteCameraRegistry()
    readers = []
    for name in ("silent-a", "silent-b"):
        registry.connect(name, name, "127.0.0.1")
        capture = RemoteCapture(registry, name, cv2, timeout=.30)
        reader = _LatestFrameReader(capture, name)
        reader.start()
        readers.append((capture, reader))
    start = time.perf_counter()
    time.sleep(.05)
    elapsed = time.perf_counter() - start
    assert elapsed < .20
    for capture, reader in readers:
        reader.stop(); capture.release(); reader.join(.5)
    report("F07 capture blocking", f"silent sources remain isolated from the tracking loop ({elapsed:.3f} s)")


def debug_floor_probe():
    scene = DebugScene3D(cv2)
    scene.render([], [], 0, 30, "NO POSE", 0)
    assert scene._floor_y is None
    body = [[0., .1, 0., .95] for _ in range(31)]
    scene.render(body, [], .95, 30, "TRACKING", 1, floor_y=0.0)
    assert scene._floor_y == 0.0
    report("F13 debug floor", "empty startup defers anchoring and manual room floor stays at 0 m")


def launcher_probe():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "VR-FBT-main" / "Main.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    report("F09 legacy launcher", "nested launcher imports and runs the maintained root application")


async def reservation_probe():
    from aiohttp import ClientSession, WSServerHandshakeError
    from aiortc import RTCConfiguration, RTCPeerConnection, VideoStreamTrack

    hub = RemoteCameraHub(host="127.0.0.1", port=0)
    hub.start()
    peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    peer.addTrack(VideoStreamTrack())
    socket = None
    try:
        await peer.setLocalDescription(await peer.createOffer())
        payload = json.dumps({"sdp": peer.localDescription.sdp, "type": "offer"}).encode()
        release_body = asyncio.Event()

        async def slow_body():
            yield payload[:1]
            await release_body.wait()
            yield payload[1:]

        # Instrument the authorization boundary to ensure the pending offer
        # has entered its handler before making the competing connection.
        entered = asyncio.Event()
        client_loop = asyncio.get_running_loop()
        original = hub._authorized

        def authorized(value):
            client_loop.call_soon_threadsafe(entered.set)
            return original(value)

        hub._authorized = authorized
        query = f"token={hub.token}&device_id=same-phone"
        async with ClientSession() as session:
            pending = asyncio.create_task(session.post(
                f"http://127.0.0.1:{hub.port}/api/webrtc/offer?{query}",
                data=slow_body(), headers={"Content-Type": "application/json"},
            ))
            await asyncio.wait_for(entered.wait(), 3)
            try:
                socket = await session.ws_connect(f"http://127.0.0.1:{hub.port}/ws?{query}")
            except WSServerHandshakeError as error:
                assert error.status == 409
            else:
                raise AssertionError("Competing WebSocket bypassed the pending WebRTC reservation")
            release_body.set()
            response = await asyncio.wait_for(pending, 10)
            await response.read()
            assert response.status == 200
            assert "same-phone" not in hub._sockets and "same-phone" in hub._peers
            report("F08 connection race", "pending WebRTC offer exclusively reserves its device ID and capacity")
    finally:
        await peer.close()
        await asyncio.to_thread(hub.stop)


if __name__ == "__main__":
    converted_coordinate_probe()
    calibration_gap_probe()
    calibration_buffer_probe()
    confidence_probe()
    fallback_probe()
    camera_selection_probe()
    capture_wait_probe()
    debug_floor_probe()
    launcher_probe()
    missing_baseline_probe()
    asyncio.run(reservation_probe())
