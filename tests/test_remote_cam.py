import asyncio
import importlib.util
import socket
import ssl
import tempfile
import time
import unittest
import urllib.request
from urllib.parse import urlsplit

from pathlib import Path

from Lib.RemoteCam import OPENSSL_MARKER, LocalCamera, MAX_FRAME_BYTES, RemoteCameraHub, RemoteCameraRegistry, RemoteCapture, ensure_local_certificates, find_openssl


class RemoteCameraRegistryTests(unittest.TestCase):
    def test_friendly_local_camera_label(self):
        camera = LocalCamera(2, "Logitech BRIO")
        self.assertEqual(camera.source_id, "local:2")
        self.assertEqual(camera.display_name, "Local · Logitech BRIO")

    def test_multiple_phones_route_frames_independently(self):
        registry = RemoteCameraRegistry()
        registry.connect("phone-a", "Pixel rear", "192.168.1.10")
        registry.connect("phone-b", "iPhone wide", "192.168.1.11")
        self.assertTrue(registry.update_frame("phone-a", b"jpeg-a"))
        self.assertTrue(registry.update_frame("phone-b", b"jpeg-b"))
        self.assertEqual(registry.wait_for_frame("phone-a", timeout=0.01), (1, b"jpeg-a"))
        self.assertEqual(registry.wait_for_frame("phone-b", timeout=0.01), (1, b"jpeg-b"))
        self.assertEqual(len(registry.list_cameras(False)), 2)

    def test_disconnect_and_reconnect_preserve_identity(self):
        registry = RemoteCameraRegistry()
        registry.connect("stable-id", "Old name", "one")
        registry.disconnect("stable-id")
        self.assertFalse(registry.list_cameras()[0].connected)
        registry.connect("stable-id", "New name", "two")
        info = registry.list_cameras()[0]
        self.assertTrue(info.connected)
        self.assertEqual(info.name, "New name")

    def test_invalid_and_oversized_frames_are_rejected(self):
        registry = RemoteCameraRegistry()
        registry.connect("phone", "Phone", "local")
        self.assertFalse(registry.update_frame("phone", b""))
        self.assertFalse(registry.update_frame("phone", b"x" * (MAX_FRAME_BYTES + 1)))

    def test_wait_returns_none_after_disconnect(self):
        registry = RemoteCameraRegistry()
        registry.connect("phone", "Phone", "local")
        registry.disconnect("phone")
        started = time.monotonic()
        self.assertIsNone(registry.wait_for_frame("phone", timeout=1))
        self.assertLess(time.monotonic() - started, 0.1)

    def test_remote_capture_consumes_new_sequences(self):
        class FakeCV2:
            IMREAD_COLOR = 1
            @staticmethod
            def imdecode(encoded, mode):
                return bytes(encoded)
        registry = RemoteCameraRegistry(); registry.connect("phone", "Phone", "local")
        capture = RemoteCapture(registry, "phone", FakeCV2)
        registry.update_frame("phone", b"jpeg")
        self.assertEqual(capture.read(), (True, b"jpeg"))
        capture.release(); self.assertFalse(capture.isOpened())


@unittest.skipUnless(importlib.util.find_spec("aiohttp"), "aiohttp is not installed")
class RemoteCameraServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        self.hub = RemoteCameraHub(host="127.0.0.1", port=port)
        await asyncio.to_thread(self.hub.start)

    async def asyncTearDown(self):
        await asyncio.to_thread(self.hub.stop)

    async def test_page_health_auth_and_two_websockets(self):
        from aiohttp import ClientSession, WSServerHandshakeError

        base = f"http://127.0.0.1:{self.hub.port}"
        with urllib.request.urlopen(f"{base}/health", timeout=2) as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b'"ok": true', response.read())
        with urllib.request.urlopen(f"{base}/?token={self.hub.token}", timeout=2) as response:
            page = response.read()
            self.assertIn(b"VR-FBT Phone Camera", page)

        async with ClientSession() as session:
            with self.assertRaises(WSServerHandshakeError):
                await session.ws_connect(f"{base}/ws?token=wrong&device_id=x")
            first = await session.ws_connect(f"{base}/ws?token={self.hub.token}&device_id=one&name=Pixel")
            second = await session.ws_connect(f"{base}/ws?token={self.hub.token}&device_id=two&name=iPhone")
            await first.send_bytes(b"frame-one")
            await second.send_bytes(b"frame-two")
            one = await asyncio.to_thread(self.hub.registry.wait_for_frame, "one", 0, 2)
            two = await asyncio.to_thread(self.hub.registry.wait_for_frame, "two", 0, 2)
            self.assertEqual(one, (1, b"frame-one"))
            self.assertEqual(two, (1, b"frame-two"))
            self.assertEqual(len(self.hub.registry.list_cameras(False)), 2)
            await first.close(); await second.close()

    async def test_duplicate_camera_and_active_shutdown(self):
        from aiohttp import ClientSession, WSServerHandshakeError
        base = f'http://127.0.0.1:{self.hub.port}'
        query = {'token': self.hub.token, 'device_id': 'phone'}
        async with ClientSession() as session:
            ws = await session.ws_connect(base + '/ws', params=query)
            with self.assertRaises(WSServerHandshakeError) as error:
                await session.ws_connect(base + '/ws', params=query)
            self.assertEqual(error.exception.status, 409)
            await asyncio.to_thread(self.hub.stop)
            self.assertFalse(self.hub.running)
            await ws.close()
            self.assertEqual(self.hub.registry.list_cameras(False), [])
            self.hub.stop()  # Idempotent even after the event loop closes.

    async def test_phone_hub_enforces_three_camera_limit(self):
        from aiohttp import ClientSession, WSServerHandshakeError
        base = f'http://127.0.0.1:{self.hub.port}'
        sockets = []
        async with ClientSession() as session:
            try:
                for index in range(3):
                    sockets.append(await session.ws_connect(
                        base + '/ws', params={'token': self.hub.token, 'device_id': f'phone-{index}'},
                    ))
                with self.assertRaises(WSServerHandshakeError) as error:
                    await session.ws_connect(
                        base + '/ws', params={'token': self.hub.token, 'device_id': 'phone-4'},
                    )
                self.assertEqual(error.exception.status, 503)
            finally:
                for websocket in sockets:
                    await websocket.close()

    @unittest.skipUnless(importlib.util.find_spec("aiortc"), "WebRTC dependencies are not installed")
    async def test_pending_webrtc_offer_reserves_identity_before_body_arrives(self):
        import json
        from aiohttp import ClientSession, WSServerHandshakeError
        from aiortc import RTCConfiguration, RTCPeerConnection, VideoStreamTrack

        client = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        client.addTrack(VideoStreamTrack())
        await client.setLocalDescription(await client.createOffer())
        payload = json.dumps({
            "sdp": client.localDescription.sdp, "type": "offer",
        }).encode()
        release_body = asyncio.Event()
        entered = asyncio.Event()
        caller_loop = asyncio.get_running_loop()
        original_authorized = self.hub._authorized

        def authorized(value):
            caller_loop.call_soon_threadsafe(entered.set)
            return original_authorized(value)

        self.hub._authorized = authorized

        async def slow_body():
            yield payload[:1]
            await release_body.wait()
            yield payload[1:]

        query = f"token={self.hub.token}&device_id=same-phone"
        try:
            async with ClientSession() as session:
                pending = asyncio.create_task(session.post(
                    f"http://127.0.0.1:{self.hub.port}/api/webrtc/offer?{query}",
                    data=slow_body(), headers={"Content-Type": "application/json"},
                ))
                await asyncio.wait_for(entered.wait(), 3)
                with self.assertRaises(WSServerHandshakeError) as error:
                    await session.ws_connect(f"http://127.0.0.1:{self.hub.port}/ws?{query}")
                self.assertEqual(error.exception.status, 409)
                release_body.set()
                response = await asyncio.wait_for(pending, 10)
                self.assertEqual(response.status, 200)
                await response.read()
                self.assertIn("same-phone", self.hub._peers)
                self.assertNotIn("same-phone", self.hub._sockets)
        finally:
            release_body.set()
            await client.close()

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is not installed")
    async def test_real_jpeg_websocket_to_capture_decode(self):
        from aiohttp import ClientSession
        import cv2
        import numpy as np

        base = f"http://127.0.0.1:{self.hub.port}"
        expected = np.zeros((24, 32, 3), dtype=np.uint8)
        expected[:, :] = (20, 90, 210)
        ok, encoded = cv2.imencode(".jpg", expected, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.assertTrue(ok)
        async with ClientSession() as session:
            ws = await session.ws_connect(f"{base}/ws?token={self.hub.token}&device_id=jpeg-phone&name=Test")
            await ws.send_bytes(encoded.tobytes())
            capture = RemoteCapture(self.hub.registry, "jpeg-phone", cv2)
            success, decoded = await asyncio.to_thread(capture.read)
            self.assertTrue(success)
            self.assertEqual(decoded.shape, expected.shape)
            self.assertLess(float(np.abs(decoded.astype(int) - expected.astype(int)).mean()), 5.0)
            await ws.close()

    async def test_generated_certificate_serves_https_without_download_endpoint(self):
        from aiohttp import ClientSession

        await asyncio.to_thread(self.hub.stop)
        with tempfile.TemporaryDirectory() as directory:
            certificate, key, ca_certificate = ensure_local_certificates(Path(directory))
            self.assertTrue(certificate.exists() and key.exists() and ca_certificate.exists())
            self.assertTrue((Path(directory) / OPENSSL_MARKER).exists())
            probe = socket.socket(); probe.bind(("127.0.0.1", 0)); port = probe.getsockname()[1]; probe.close()
            self.hub = RemoteCameraHub(host="127.0.0.1", port=port)
            await asyncio.to_thread(self.hub.start, certificate, key)
            trust = ssl.create_default_context(cafile=str(ca_certificate))
            async with ClientSession() as session:
                async with session.get(f"https://127.0.0.1:{port}/health", ssl=trust) as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue((await response.json())["tls"])
                async with session.get(f"https://127.0.0.1:{port}/vrfbt-ca.crt", ssl=trust) as response:
                    self.assertEqual(response.status, 404)

    async def test_openssl_generator_reuses_private_ca(self):
        await asyncio.to_thread(self.hub.stop)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _certificate, _key, ca_certificate = ensure_local_certificates(root)
            first_ca = ca_certificate.read_bytes()
            (root / OPENSSL_MARKER).unlink()
            certificate, key, same_ca = ensure_local_certificates(root)
            self.assertEqual(same_ca.read_bytes(), first_ca)
            self.assertTrue(certificate.read_text(encoding="ascii").startswith("-----BEGIN CERTIFICATE-----"))
            self.assertTrue(key.read_text(encoding="ascii").startswith("-----BEGIN PRIVATE KEY-----"))
            self.assertIn("openssl", (root / OPENSSL_MARKER).read_text(encoding="utf-8").casefold())
            self.assertTrue(find_openssl().is_file())

    async def test_active_lan_address_accepts_strict_https(self):
        from aiohttp import ClientSession

        await asyncio.to_thread(self.hub.stop)
        with tempfile.TemporaryDirectory() as directory:
            certificate, key, ca_certificate = ensure_local_certificates(Path(directory))
            self.hub = RemoteCameraHub(host="0.0.0.0", port=0)
            await asyncio.to_thread(self.hub.start, certificate, key)
            target = urlsplit(self.hub.local_urls()[0])
            self.assertNotEqual(target.hostname, "127.0.0.1")
            trust = ssl.create_default_context(cafile=str(ca_certificate))
            async with ClientSession() as session:
                async with session.get(f"https://{target.hostname}:{self.hub.port}/health", ssl=trust) as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue((await response.json())["tls"])

    @unittest.skipUnless(importlib.util.find_spec("aiortc") and importlib.util.find_spec("av"), "WebRTC dependencies are not installed")
    async def test_direct_webrtc_video_reaches_remote_capture(self):
        from aiohttp import ClientSession
        from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
        from aiortc import VideoStreamTrack
        from av import VideoFrame
        import numpy as np

        class SyntheticVideo(VideoStreamTrack):
            async def recv(self):
                pts, time_base = await self.next_timestamp()
                array = np.zeros((48, 64, 3), dtype=np.uint8)
                array[:, :] = (17, 91, 203)
                frame = VideoFrame.from_ndarray(array, format="bgr24")
                frame.pts, frame.time_base = pts, time_base
                return frame

        client = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        client.addTrack(SyntheticVideo())
        try:
            offer = await client.createOffer()
            await client.setLocalDescription(offer)
            query = f"token={self.hub.token}&device_id=webrtc-phone&name=Direct"
            async with ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{self.hub.port}/api/webrtc/offer?{query}",
                    json={"sdp": client.localDescription.sdp, "type": client.localDescription.type},
                ) as response:
                    self.assertEqual(response.status, 200)
                    answer = await response.json()
            await client.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                cameras = self.hub.registry.list_cameras(False)
                if cameras and cameras[0].sequence:
                    break
                await asyncio.sleep(0.05)
            capture = RemoteCapture(self.hub.registry, "webrtc-phone", None)
            success, decoded = await asyncio.to_thread(capture.read)
            self.assertTrue(success)
            self.assertEqual(decoded.shape, (48, 64, 3))
            self.assertLess(float(np.abs(decoded.astype(int) - np.array([17, 91, 203])).mean()), 8.0)
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
