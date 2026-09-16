"""Friendly local-camera discovery and a multi-phone camera web hub."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import asyncio
import hmac
import ipaddress
import math
import json
import struct
import os
import secrets
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable
from Lib.Synchronization import MediaTimeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_PAGE = PROJECT_ROOT / "Content" / "Website" / "index.html"
MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_PHONE_CAMERAS = 3
MAX_OFFLINE_PHONE_HISTORY = 32
WEBRTC_OFFER_TIMEOUT = 10.0
WEBRTC_FRAME_TIMEOUT = 15.0
CERTIFICATE_DIR = Path(os.environ.get("LOCALAPPDATA", PROJECT_ROOT)) / "VR-FBT" / "certificates"
OPENSSL_MARKER = ".generated-by-openssl"


@dataclass(frozen=True, slots=True)
class LocalCamera:
    index: int
    name: str

    @property
    def source_id(self) -> str:
        return f"local:{self.index}"

    @property
    def display_name(self) -> str:
        return f"Local · {self.name}"


@dataclass(frozen=True, slots=True)
class PhoneCameraInfo:
    device_id: str
    name: str
    address: str
    connected: bool
    last_frame_at: float | None
    sequence: int

    @property
    def source_id(self) -> str:
        return f"phone:{self.device_id}"

    @property
    def display_name(self) -> str:
        status = "online" if self.connected else "offline"
        return f"Phone · {self.name} ({status})"


@dataclass(slots=True)
class _PhoneCameraState:
    device_id: str
    name: str
    address: str
    connected: bool = True
    last_frame_at: float | None = None
    sequence: int = 0
    frame: object | None = None


def discover_local_cameras(max_probe: int = 8) -> list[LocalCamera]:
    """Return DirectShow-friendly names, with a safe OpenCV probe fallback."""
    try:
        from pygrabber.dshow_graph import FilterGraph

        names = FilterGraph().get_input_devices()
        if names:
            return [LocalCamera(index, str(name)) for index, name in enumerate(names)]
    except Exception:
        pass

    found: list[LocalCamera] = []
    try:
        import cv2

        backend = getattr(cv2, "CAP_DSHOW", 0)
        for index in range(max_probe):
            capture = cv2.VideoCapture(index, backend)
            try:
                if capture.isOpened():
                    found.append(LocalCamera(index, f"Camera {index}"))
            finally:
                capture.release()
    except Exception:
        pass
    return found


def local_ipv4_addresses() -> list[str]:
    addresses = {"127.0.0.1"}
    try:
        addresses.update(info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET) if info[4][0])
    except OSError:
        pass
    preferred = None
    try:
        route_probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # UDP connect selects the active route without transmitting a packet.
            route_probe.connect(("192.0.2.1", 9))
            preferred = route_probe.getsockname()[0]
            addresses.add(preferred)
        finally:
            route_probe.close()
    except OSError:
        pass

    def priority(value: str) -> tuple[int, str]:
        address = ipaddress.ip_address(value)
        if value == preferred:
            return 0, value
        if address.is_private and not address.is_loopback and not address.is_link_local:
            return 1, value
        if not address.is_loopback:
            return 2, value
        return 3, value

    return sorted(addresses, key=priority)


def find_openssl() -> Path:
    """Find a usable OpenSSL executable without requiring it to be on PATH."""
    override = os.environ.get("VRFBT_OPENSSL")
    candidates = [
        override,
        shutil.which("openssl"),
        PROJECT_ROOT / "Bin" / "openssl" / "openssl.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "usr" / "bin" / "openssl.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "mingw64" / "bin" / "openssl.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "FireDaemon OpenSSL 3" / "bin" / "openssl.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "OpenSSL-Win64" / "bin" / "openssl.exe",
    ]
    checked: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        normalized = os.path.normcase(str(path))
        if normalized in checked or not path.is_file():
            continue
        checked.add(normalized)
        try:
            result = subprocess.run(
                [str(path), "version"], capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0 and "OpenSSL" in result.stdout:
            return path
    raise RuntimeError(
        "OpenSSL was not found. Install Git for Windows or OpenSSL, or set "
        "VRFBT_OPENSSL to the full path of openssl.exe."
    )


def _run_openssl(executable: Path, arguments: list[str], working_directory: Path) -> None:
    result = subprocess.run(
        [str(executable), *arguments],
        cwd=str(working_directory),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"OpenSSL certificate generation failed: {detail or f'exit code {result.returncode}'}")


def ensure_local_certificates(directory: Path = CERTIFICATE_DIR) -> tuple[Path, Path, Path]:
    """Use OpenSSL to create a persistent CA and LAN-address-aware HTTPS certificate."""
    ca_cert_path, ca_key_path = directory / "vrfbt-ca.crt", directory / "vrfbt-ca.key"
    server_cert_path, server_key_path = directory / "server.crt", directory / "server.key"
    marker_path = directory / OPENSSL_MARKER
    from cryptography import x509

    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    current_ips = set(local_ipv4_addresses())
    if all(path.exists() for path in (ca_cert_path, ca_key_path, server_cert_path, server_key_path)):
        try:
            existing = x509.load_pem_x509_certificate(server_cert_path.read_bytes())
            existing_ips = {str(value) for value in existing.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(x509.IPAddress)}
            if marker_path.exists() and current_ips <= existing_ips and existing.not_valid_after_utc > now + timedelta(days=30):
                return server_cert_path, server_key_path, ca_cert_path
        except (ValueError, OSError, x509.ExtensionNotFound):
            pass

    openssl = find_openssl()
    existing_ca_is_usable = False
    if ca_cert_path.is_file() and ca_key_path.is_file():
        try:
            x509.load_pem_x509_certificate(ca_cert_path.read_bytes())
            check = subprocess.run(
                [str(openssl), "pkey", "-in", str(ca_key_path), "-noout"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            existing_ca_is_usable = check.returncode == 0
        except (ValueError, OSError, subprocess.SubprocessError):
            pass

    with tempfile.TemporaryDirectory(prefix="openssl-", dir=directory) as temporary:
        work = Path(temporary)
        generated_ca_cert = work / "vrfbt-ca.crt"
        generated_ca_key = work / "vrfbt-ca.key"
        signing_cert = ca_cert_path if existing_ca_is_usable else generated_ca_cert
        signing_key = ca_key_path if existing_ca_is_usable else generated_ca_key
        if not existing_ca_is_usable:
            _run_openssl(openssl, [
                "req", "-x509", "-newkey", "rsa:3072", "-nodes", "-sha256", "-days", "3650",
                "-keyout", str(generated_ca_key), "-out", str(generated_ca_cert),
                "-subj", "/CN=VR-FBT Local Camera CA",
                "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0",
                "-addext", "keyUsage=critical,digitalSignature,keyCertSign,cRLSign",
                "-addext", "subjectKeyIdentifier=hash",
            ], work)

        server_csr = work / "server.csr"
        generated_server_key = work / "server.key"
        generated_server_cert = work / "server.crt"
        _run_openssl(openssl, [
            "req", "-new", "-newkey", "rsa:2048", "-nodes", "-sha256",
            "-keyout", str(generated_server_key), "-out", str(server_csr),
            "-subj", "/CN=VR-FBT Phone Camera",
        ], work)

        hostname = socket.gethostname()
        safe_hostname = hostname if all(ch.isalnum() or ch in ".-" for ch in hostname) else "vrfbt-pc"
        config_lines = [
            "[v3_server]",
            "basicConstraints=critical,CA:FALSE",
            "keyUsage=critical,digitalSignature,keyEncipherment",
            "extendedKeyUsage=serverAuth",
            "subjectKeyIdentifier=hash",
            "authorityKeyIdentifier=keyid,issuer",
            "subjectAltName=@alt_names",
            "[alt_names]",
            "DNS.1=localhost",
            f"DNS.2={safe_hostname}",
        ]
        for index, address in enumerate(sorted(current_ips, key=lambda value: ipaddress.ip_address(value)), start=1):
            config_lines.append(f"IP.{index}={address}")
        extension_config = work / "server-ext.cnf"
        extension_config.write_text("\n".join(config_lines) + "\n", encoding="ascii")
        serial = secrets.token_hex(19)
        _run_openssl(openssl, [
            "x509", "-req", "-in", str(server_csr), "-CA", str(signing_cert),
            "-CAkey", str(signing_key), "-set_serial", f"0x{serial}", "-days", "825", "-sha256",
            "-extfile", str(extension_config), "-extensions", "v3_server", "-out", str(generated_server_cert),
        ], work)

        # Replace complete files only after every OpenSSL command has succeeded.
        if not existing_ca_is_usable:
            os.replace(generated_ca_cert, ca_cert_path)
            os.replace(generated_ca_key, ca_key_path)
        os.replace(generated_server_cert, server_cert_path)
        os.replace(generated_server_key, server_key_path)
    marker_path.write_text(f"{openssl}\n", encoding="utf-8")
    return server_cert_path, server_key_path, ca_cert_path


class RemoteCameraRegistry:
    """Thread-safe phone registry and latest-frame exchange."""

    def __init__(self, on_change: Callable[[], None] | None = None):
        self._states: dict[str, _PhoneCameraState] = {}
        self._offline_order: dict[str, None] = {}
        self._retired_sequence = 0
        self._condition = threading.Condition()
        self._on_change = on_change or (lambda: None)

    @staticmethod
    def _clean(value: str, fallback: str, max_length: int) -> str:
        cleaned = "".join(ch for ch in value.strip() if ch.isprintable())[:max_length]
        return cleaned or fallback

    def connect(self, device_id: str, name: str, address: str) -> PhoneCameraInfo:
        device_id = self._clean(device_id, secrets.token_hex(8), 80)
        name = self._clean(name, "Phone camera", 80)
        address = self._clean(address, "unknown", 80)
        with self._condition:
            self._offline_order.pop(device_id, None)
            state = self._states.get(device_id)
            if state is None:
                state = _PhoneCameraState(device_id, name, address, sequence=self._retired_sequence)
                self._states[device_id] = state
            else:
                state.name, state.address, state.connected = name, address, True
                state.frame, state.last_frame_at = None, None
            self._condition.notify_all()
            info = self._snapshot(state)
        self._on_change()
        return info

    def disconnect(self, device_id: str) -> None:
        with self._condition:
            state = self._states.get(device_id)
            if state is not None:
                state.connected = False
                state.frame = None
                self._offline_order.pop(device_id, None)
                self._offline_order[device_id] = None
                while len(self._offline_order) > MAX_OFFLINE_PHONE_HISTORY:
                    oldest = next(iter(self._offline_order))
                    self._offline_order.pop(oldest)
                    retired = self._states.pop(oldest)
                    # A capture may still hold a cursor for an evicted device.
                    # Its first frame after reconnect must exceed that cursor.
                    self._retired_sequence = max(self._retired_sequence, retired.sequence)
                self._condition.notify_all()
        self._on_change()

    def update_frame(self, device_id: str, frame: bytes, *, captured_at=None) -> bool:
        if not frame or len(frame) > MAX_FRAME_BYTES:
            return False
        with self._condition:
            state = self._states.get(device_id)
            if state is None or not state.connected:
                return False
            state.frame = bytes(frame)
            state.sequence += 1
            state.last_frame_at = time.monotonic() if captured_at is None else captured_at
            self._condition.notify_all()
        return True

    def update_decoded_frame(self, device_id: str, frame, *, captured_at=None) -> bool:
        """Publish a decoded BGR frame received through WebRTC."""
        if frame is None or getattr(frame, "ndim", 0) != 3 or getattr(frame, "size", 0) == 0:
            return False
        with self._condition:
            state = self._states.get(device_id)
            if state is None or not state.connected:
                return False
            state.frame = frame.copy()
            state.sequence += 1
            state.last_frame_at = time.monotonic() if captured_at is None else captured_at
            self._condition.notify_all()
        return True

    def wait_for_frame(self, device_id: str, after_sequence: int = 0, timeout: float = 3.0, *, include_timestamp: bool = False):
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                state = self._states.get(device_id)
                if state is not None and state.connected and state.frame is not None and state.sequence > after_sequence:
                    if include_timestamp:
                        return state.sequence, state.frame, state.last_frame_at
                    return state.sequence, state.frame
                if state is not None and not state.connected:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def list_cameras(self, include_offline: bool = True) -> list[PhoneCameraInfo]:
        with self._condition:
            items = [self._snapshot(state) for state in self._states.values() if include_offline or state.connected]
        return sorted(items, key=lambda item: (not item.connected, item.name.casefold(), item.device_id))

    @staticmethod
    def _snapshot(state: _PhoneCameraState) -> PhoneCameraInfo:
        return PhoneCameraInfo(state.device_id, state.name, state.address, state.connected, state.last_frame_at, state.sequence)


class RemoteCameraHub:
    """Host the phone page and receive direct WebRTC or JPEG fallback frames."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9100, on_change: Callable[[], None] | None = None):
        self.host, self.port = host, port
        self.token = secrets.token_urlsafe(24)
        self.registry = RemoteCameraRegistry(on_change)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._ssl_context: ssl.SSLContext | None = None
        self._sockets = {}
        self._peers = {}
        self._peer_tasks = {}
        self._reservations = {}
        self._connection_owners = {}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._ready.is_set() and self._error is None

    @property
    def scheme(self) -> str:
        return "https" if self._ssl_context else "http"

    def start(self, certificate: Path | None = None, private_key: Path | None = None, timeout: float = 5.0) -> None:
        if self.running:
            return
        if bool(certificate) != bool(private_key):
            raise ValueError("Both TLS certificate and private key are required")
        self._ssl_context = None
        if certificate and private_key:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(certificate, private_key)
            self._ssl_context = context
        self._ready.clear(); self._error = None
        self._thread = threading.Thread(target=self._thread_main, name="phone-camera-hub", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("Phone camera server did not start in time")
        if self._error is not None:
            raise RuntimeError(f"Phone camera server failed: {self._error}") from self._error

    def stop(self, timeout: float = 5.0) -> None:
        if self._loop is not None and not self._loop.is_closed() and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout)

    def local_urls(self) -> list[str]:
        return [f"{self.scheme}://{address}:{self.port}/?token={self.token}" for address in local_ipv4_addresses()]

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception as exc:
            self._error = exc
            self._ready.set()

    async def _serve(self) -> None:
        from aiohttp import WSMsgType, web

        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()

        async def index(_request):
            return web.FileResponse(WEB_PAGE, headers={"Cache-Control": "no-store", "Permissions-Policy": "camera=(self)", "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff"})

        async def health(_request):
            return web.json_response({"ok": True, "cameras": len(self.registry.list_cameras(False)), "tls": bool(self._ssl_context)})

        async def cameras(request):
            if not self._authorized(request.query.get("token", "")):
                raise web.HTTPUnauthorized()
            return web.json_response([{"id": item.device_id, "name": item.name, "connected": item.connected, "sequence": item.sequence} for item in self.registry.list_cameras()])

        def reserve_camera(request):
            device_id = request.query.get("device_id", "")
            name = request.query.get("name", "Phone camera")
            if not device_id or len(device_id) > 80 or any(not c.isalnum() and c not in '-_' for c in device_id):
                raise web.HTTPBadRequest(text="invalid device_id")
            if device_id in self._sockets or device_id in self._peers or device_id in self._reservations:
                raise web.HTTPConflict(text="This phone is already connected in another tab")
            if len(self._sockets) + len(self._peers) + len(self._reservations) >= MAX_PHONE_CAMERAS:
                raise web.HTTPServiceUnavailable(text=f"Maximum of {MAX_PHONE_CAMERAS} cameras reached")
            owner = object()
            self._reservations[device_id] = owner
            return device_id, name, owner

        def release_reservation(device_id, owner):
            if self._reservations.get(device_id) is owner:
                self._reservations.pop(device_id, None)

        async def webrtc_offer(request):
            if not self._authorized(request.query.get("token", "")):
                raise web.HTTPUnauthorized()
            device_id, name, owner = reserve_camera(request)
            try:
                async with asyncio.timeout(WEBRTC_OFFER_TIMEOUT):
                    parameters = await request.json()
                sdp, description_type = parameters["sdp"], parameters["type"]
                if description_type != "offer" or not isinstance(sdp, str) or len(sdp) > 100_000:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                release_reservation(device_id, owner)
                raise web.HTTPBadRequest(text="invalid WebRTC offer")
            except TimeoutError:
                release_reservation(device_id, owner)
                raise web.HTTPRequestTimeout(text="WebRTC offer body timed out")
            except BaseException:
                release_reservation(device_id, owner)
                raise

            try:
                from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
                from aiortc.mediastreams import MediaStreamError
                peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            except Exception:
                release_reservation(device_id, owner)
                raise
            release_reservation(device_id, owner)
            self._peers[device_id] = peer
            self._connection_owners[device_id] = owner

            async def close_peer():
                owns_connection = (
                    self._connection_owners.get(device_id) is owner
                    and self._peers.get(device_id) is peer
                )
                if owns_connection:
                    task = self._peer_tasks.pop(device_id, None)
                    if task and task is not asyncio.current_task():
                        task.cancel()
                    self._peers.pop(device_id, None)
                    self._connection_owners.pop(device_id, None)
                    self.registry.disconnect(device_id)
                if peer.connectionState != "closed":
                    await peer.close()

            async def receive_video(track):
                if self._connection_owners.get(device_id) is not owner:
                    return
                self.registry.connect(device_id, name, request.remote or "unknown")
                timeline = MediaTimeline()
                try:
                    while True:
                        frame = await asyncio.wait_for(track.recv(), timeout=WEBRTC_FRAME_TIMEOUT)
                        if self._connection_owners.get(device_id) is not owner:
                            return
                        arrived_at = time.monotonic()
                        media_time = float(frame.pts * frame.time_base) if frame.pts is not None and frame.time_base is not None else None
                        captured_at = timeline.timestamp(media_time, arrived_at)
                        if captured_at is not None:
                            self.registry.update_decoded_frame(device_id, frame.to_ndarray(format="bgr24"), captured_at=captured_at)
                except (MediaStreamError, asyncio.CancelledError, TimeoutError):
                    pass
                finally:
                    await close_peer()

            @peer.on("track")
            def on_track(track):
                if track.kind == "video":
                    if self._connection_owners.get(device_id) is owner and device_id not in self._peer_tasks:
                        self._peer_tasks[device_id] = asyncio.create_task(receive_video(track))

            @peer.on("connectionstatechange")
            async def on_connectionstatechange():
                if peer.connectionState in {"failed", "closed"}:
                    await close_peer()

            try:
                async with asyncio.timeout(WEBRTC_OFFER_TIMEOUT):
                    await peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=description_type))
                    if device_id not in self._peer_tasks:
                        raise web.HTTPBadRequest(text="WebRTC offer must include video")
                    answer = await peer.createAnswer()
                    await peer.setLocalDescription(answer)
                return web.json_response({"sdp": peer.localDescription.sdp, "type": peer.localDescription.type})
            except BaseException:
                await close_peer()
                raise

        async def websocket(request):
            if not self._authorized(request.query.get("token", "")):
                raise web.HTTPUnauthorized()
            device_id, name, owner = reserve_camera(request)
            try:
                ws = web.WebSocketResponse(heartbeat=20, max_msg_size=MAX_FRAME_BYTES + 20, compress=False)
            except Exception:
                release_reservation(device_id, owner)
                raise
            release_reservation(device_id, owner)
            self._sockets[device_id] = ws
            self._connection_owners[device_id] = owner
            try:
                await ws.prepare(request)
                self.registry.connect(device_id, name, request.remote or "unknown")
                async for message in ws:
                    if message.type is WSMsgType.BINARY:
                        data = message.data
                        captured_at = None
                        if data.startswith(b"VFT1"):
                            if len(data) <= 20:
                                await ws.close(code=1007, message=b"invalid timing header")
                                break
                            captured_at, uncertainty = struct.unpack("!dd", data[4:20])
                            arrived_at = time.monotonic()
                            if not all(math.isfinite(value) for value in (captured_at, uncertainty)) or not 0 <= uncertainty <= 0.025:
                                await ws.close(code=1007, message=b"invalid clock estimate")
                                break
                            if captured_at > arrived_at + uncertainty or arrived_at - captured_at > 0.250:
                                continue  # stale or impossible capture time; never relabel as fresh
                            data = data[20:]
                        if not self.registry.update_frame(device_id, data, captured_at=captured_at):
                            await ws.close(code=1009, message=b"invalid frame")
                    elif message.type is WSMsgType.TEXT:
                        try:
                            value = json.loads(message.data)
                            client = value["client"]
                            if value.get("type") != "clock" or not isinstance(client, (int, float)) or not math.isfinite(client):
                                raise ValueError
                            await ws.send_json({"type": "clock", "client": client, "server": time.monotonic()})
                        except (ValueError, TypeError, KeyError):
                            await ws.close(code=1007, message=b"invalid clock request")
                            break
                    elif message.type is WSMsgType.ERROR:
                        break
            finally:
                if self._connection_owners.get(device_id) is owner and self._sockets.get(device_id) is ws:
                    self._sockets.pop(device_id, None)
                    self._connection_owners.pop(device_id, None)
                    self.registry.disconnect(device_id)
            return ws

        app = web.Application(client_max_size=MAX_FRAME_BYTES)
        async def shutdown(_app):
            await asyncio.gather(*(ws.close(code=1001, message=b'Server stopped') for ws in list(self._sockets.values())))
            for task in list(self._peer_tasks.values()):
                task.cancel()
            await asyncio.gather(*list(self._peer_tasks.values()), return_exceptions=True)
            await asyncio.gather(*(peer.close() for peer in list(self._peers.values())), return_exceptions=True)
            self._peer_tasks.clear(); self._peers.clear(); self._sockets.clear()
            self._reservations.clear(); self._connection_owners.clear()
        app.on_shutdown.append(shutdown)
        app.add_routes([web.get("/", index), web.get("/health", health), web.get("/api/cameras", cameras), web.post("/api/webrtc/offer", webrtc_offer), web.get("/ws", websocket)])
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port, ssl_context=self._ssl_context)
        try:
            await site.start()
            if self.port == 0:
                self.port = runner.addresses[0][1]
            self._ready.set()
            await self._stop_event.wait()
        finally:
            await runner.cleanup()

    def _authorized(self, supplied: str) -> bool:
        return bool(supplied) and supplied.isascii() and hmac.compare_digest(supplied, self.token)


class RemoteCapture:
    """OpenCV-compatible reader for one registered phone camera."""

    def __init__(self, registry: RemoteCameraRegistry, device_id: str, cv2_module, timeout: float = 3.0):
        self.registry, self.device_id, self.cv2 = registry, device_id, cv2_module
        self.timeout = max(0.01, float(timeout))
        self.sequence = 0
        self.closed = False
        self.captured_at = None

    def isOpened(self) -> bool:
        return not self.closed and any(item.device_id == self.device_id and item.connected for item in self.registry.list_cameras())

    def read(self):
        import numpy as np

        result = self.registry.wait_for_frame(self.device_id, self.sequence, self.timeout, include_timestamp=True)
        if result is None:
            return False, None
        self.sequence, encoded, self.captured_at = result
        if isinstance(encoded, (bytes, bytearray, memoryview)):
            frame = self.cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), self.cv2.IMREAD_COLOR)
        else:
            frame = encoded.copy()
        return frame is not None, frame

    def release(self) -> None:
        self.closed = True
