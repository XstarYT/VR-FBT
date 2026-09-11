"""Configuration loading, validation, and project diagnostics for VR-FBT."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.util import find_spec
import json
import math
from pathlib import Path
import sys
import tempfile
import tomllib
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = PROJECT_ROOT / "Content"
SETTINGS_PATH = CONTENT_DIR / "settings.json"
PROFILES_DIR = CONTENT_DIR / "Profiles"
JOINT_MAPS_DIR = CONTENT_DIR / "Joint-Maps"
MODEL_DIR = PROJECT_ROOT / "Model"
POSE_MODEL_HASHES = {
    "pose_landmarker_lite.task": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    "pose_landmarker_full.task": "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
    "pose_landmarker_heavy.task": "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b",
}

CAMERA_CORNER_SIGNS = {
    "Front left (-X, -Z)": (-1.0, -1.0),
    "Front right (+X, -Z)": (1.0, -1.0),
    "Back left (-X, +Z)": (-1.0, 1.0),
    "Back right (+X, +Z)": (1.0, 1.0),
}


class ConfigurationError(ValueError):
    """Raised when a settings, profile, or joint-map file is invalid."""


def _secure_equal(left: str, right: str) -> bool:
    import hmac
    return hmac.compare_digest(left, right)


@dataclass(slots=True)
class Settings:
    fps: int = 30
    default_profile: str = "Default"
    tcp_server: bool = False
    udp_server: bool = False
    live_switch: bool = True


@dataclass(frozen=True, slots=True)
class CameraSetup:
    source_id: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]  # yaw, pitch, roll in degrees
    horizontal_fov: float = 60.0
    image_rotation: int = 0  # clockwise correction before tracking/display


@dataclass(slots=True)
class Profile:
    name: str = "Default"
    server_ip: str = "127.0.0.1"
    server_port: int = 9000
    camera_index: int = 0
    camera_source: str = "local:0"
    camera_sources: tuple[str, ...] = ()
    manual_camera_setup: bool = False
    camera_setups: tuple[CameraSetup, ...] = ()
    room_size_m: tuple[float, float, float] = (4.0, 2.7, 4.0)
    show_output: bool = True
    tracking_mode: str = "SINGLE"
    smooth: bool = True
    pose_quality: str = "full"
    user_height_m: float = 1.70
    vrchat_tracker_set: str = "stable"
    joint_map: str = "FULLMAP"
    joy_con_remote: bool = False


def camera_setup_for_corner(
    setup: CameraSetup,
    corner: str,
    height_m: float,
    room_size_m: tuple[float, float, float],
) -> CameraSetup:
    """Place a fixed camera in a room corner and aim it at body-center height."""
    if corner not in CAMERA_CORNER_SIGNS:
        raise ConfigurationError(f"Unknown camera corner: {corner}")
    width, room_height, depth = (float(value) for value in room_size_m)
    height = float(height_m)
    if not 0.2 <= height <= room_height:
        raise ConfigurationError(f"Camera height must be between 0.2 m and {room_height:.2f} m")
    x_sign, z_sign = CAMERA_CORNER_SIGNS[corner]
    x, z = x_sign * width / 2.0, z_sign * depth / 2.0
    target_height = min(1.0, room_height * 0.45)
    yaw = math.degrees(math.atan2(-x, -z))
    pitch = math.degrees(math.atan2(target_height - height, math.hypot(x, z)))
    return CameraSetup(
        setup.source_id,
        (x, height, z),
        (yaw, pitch, 0.0),
        setup.horizontal_fov,
        setup.image_rotation,
    )


def camera_corner_for_setup(setup: CameraSetup, room_size_m: tuple[float, float, float]) -> str:
    """Recover the quick-placement corner for a saved transform, if it has one."""
    width, _height, depth = (float(value) for value in room_size_m)
    for name, (x_sign, z_sign) in CAMERA_CORNER_SIGNS.items():
        expected_x, expected_z = x_sign * width / 2.0, z_sign * depth / 2.0
        if abs(setup.position[0] - expected_x) <= 0.01 and abs(setup.position[2] - expected_z) <= 0.01:
            return name
    return "Custom coordinates"


def _require_mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be an object/table")
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def list_profiles() -> list[str]:
    return sorted((p.stem for p in PROFILES_DIR.glob("*.toml")), key=str.casefold)


def list_joint_maps() -> list[str]:
    return sorted((p.stem for p in JOINT_MAPS_DIR.glob("*.json")), key=str.casefold)


def _named_file(directory: Path, name: str, suffix: str) -> Path:
    wanted = name.strip().casefold()
    for candidate in directory.glob(f"*{suffix}"):
        if candidate.stem.casefold() == wanted:
            return candidate
    raise ConfigurationError(f"No {suffix[1:].upper()} file named {name!r} in {directory}")


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    try:
        data = _require_mapping(json.loads(path.read_text(encoding="utf-8")), "settings")
        settings = Settings(
            fps=int(data.get("fps", 30)),
            default_profile=str(data.get("default_profile", "Default")),
            tcp_server=bool(data.get("TCP_server", False)),
            udp_server=bool(data.get("UDP_server", False)),
            live_switch=bool(data.get("Live_switch", True)),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Could not load {path}: {exc}") from exc
    validate_settings(settings)
    return settings


def save_settings(settings: Settings, path: Path = SETTINGS_PATH) -> None:
    validate_settings(settings)
    payload = {
        "fps": settings.fps,
        "TCP_server": settings.tcp_server,
        "UDP_server": settings.udp_server,
        "Live_switch": settings.live_switch,
        "default_profile": settings.default_profile,
    }
    _atomic_write(path, json.dumps(payload, indent=2) + "\n")


def load_profile(name: str) -> Profile:
    path = _named_file(PROFILES_DIR, name, ".toml")
    try:
        with path.open("rb") as handle:
            data = _require_mapping(tomllib.load(handle), "profile")
        server = _require_mapping(data.get("server"), "server")
        camera = _require_mapping(data.get("camera"), "camera")
        tracking = _require_mapping(data.get("tracking"), "tracking")
        misc = _require_mapping(data.get("misc", {}), "misc")
        indices = camera.get("cam-index", [0])
        if not isinstance(indices, list) or not indices:
            raise ConfigurationError("camera.cam-index must contain at least one index")
        legacy_source = str(camera.get("source", f"local:{int(indices[0])}"))
        raw_sources = camera.get("sources", [legacy_source])
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ConfigurationError("camera.sources must contain between one and three sources")
        sources = tuple(str(value) for value in raw_sources)
        raw_room_size = camera.get("room-size-m", [4.0, 2.7, 4.0])
        if not isinstance(raw_room_size, list) or len(raw_room_size) != 3:
            raise ConfigurationError("camera.room-size-m must contain width, height, and depth")
        raw_setups = camera.get("setup", [])
        if not isinstance(raw_setups, list):
            raise ConfigurationError("camera.setup must be an array of camera tables")
        setups = []
        for entry in raw_setups:
            entry = _require_mapping(entry, "camera.setup entry")
            position = entry.get("position", [0.0, 1.4, -2.5])
            rotation = entry.get("rotation", [0.0, 0.0, 0.0])
            if not isinstance(position, list) or len(position) != 3 or not isinstance(rotation, list) or len(rotation) != 3:
                raise ConfigurationError("Each camera setup needs three-value position and rotation arrays")
            setups.append(CameraSetup(
                str(entry.get("source", "")),
                tuple(float(value) for value in position),
                tuple(float(value) for value in rotation),
                float(entry.get("horizontal-fov", 60.0)),
                int(entry.get("image-rotation", 0)),
            ))
        profile = Profile(
            name=path.stem,
            server_ip=str(server.get("ip", "127.0.0.1")),
            server_port=int(server.get("port", 9000)),
            camera_index=int(indices[0]),
            camera_source=sources[0],
            camera_sources=sources,
            manual_camera_setup=bool(camera.get("manual-setup", False)),
            camera_setups=tuple(setups),
            room_size_m=tuple(float(value) for value in raw_room_size),
            show_output=bool(camera.get("show-output", True)),
            tracking_mode=str(tracking.get("mode", "single")).upper(),
            smooth=bool(tracking.get("smooth", True)),
            pose_quality=str(tracking.get("pose-quality", "full")).casefold(),
            user_height_m=float(tracking.get("user-height-m", 1.70)),
            vrchat_tracker_set=str(tracking.get("vrchat-tracker-set", "stable")).casefold(),
            joint_map=str(tracking.get("joint-map", "FULLMAP")),
            joy_con_remote=bool(misc.get("joy-con-remote", False)),
        )
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Could not load {path}: {exc}") from exc
    validate_profile(profile)
    return profile


def save_profile(profile: Profile) -> None:
    validate_profile(profile)
    safe_name = "".join(c for c in profile.name.strip() if c.isalnum() or c in "-_ ")
    if not safe_name or safe_name != profile.name.strip():
        raise ConfigurationError("Profile name may only contain letters, numbers, spaces, - and _")
    sources = profile_camera_sources(profile)
    setups = profile_camera_setups(profile)
    local_indices = [int(source.split(":", 1)[1]) for source in sources if source.startswith("local:")]
    camera_text = (
        "[server]\n"
        f"ip = {json.dumps(profile.server_ip)}\n"
        f"port = {profile.server_port}\n\n"
        "[camera]\n"
        f"cam-index = {json.dumps(local_indices or [profile.camera_index])}\n"
        f"source = {json.dumps(sources[0])}\n"
        f"sources = {json.dumps(list(sources))}\n"
        f"manual-setup = {str(profile.manual_camera_setup).lower()}\n"
        f"room-size-m = {json.dumps(list(profile.room_size_m))}\n"
        f"show-output = {str(profile.show_output).lower()}\n\n"
    )
    setup_text = "".join(
            "[[camera.setup]]\n"
            f"source = {json.dumps(setup.source_id)}\n"
            f"position = {json.dumps(list(setup.position))}\n"
            f"rotation = {json.dumps(list(setup.rotation))}\n"
            f"horizontal-fov = {setup.horizontal_fov:.3f}\n"
            f"image-rotation = {setup.image_rotation}\n\n"
            for setup in setups
    )
    tracking_text = (
        "[tracking]\n"
        f"mode = {json.dumps(profile.tracking_mode.lower())}\n"
        f"smooth = {str(profile.smooth).lower()}\n"
        f"pose-quality = {json.dumps(profile.pose_quality)}\n"
        f"user-height-m = {profile.user_height_m:.3f}\n"
        f"vrchat-tracker-set = {json.dumps(profile.vrchat_tracker_set)}\n"
        f"joint-map = {json.dumps(profile.joint_map)}\n"
        "\n[misc]\n"
        f"joy-con-remote = {str(profile.joy_con_remote).lower()}\n"
    )
    text = camera_text + setup_text + tracking_text
    _atomic_write(PROFILES_DIR / f"{safe_name}.toml", text)


def validate_settings(settings: Settings) -> None:
    if not 1 <= settings.fps <= 240:
        raise ConfigurationError("FPS must be between 1 and 240")
    if not settings.default_profile.strip():
        raise ConfigurationError("A default profile is required")


def validate_profile(profile: Profile) -> None:
    if not profile.name.strip():
        raise ConfigurationError("Profile name is required")
    if not profile.server_ip.strip():
        raise ConfigurationError("OSC host is required")
    if not 1 <= profile.server_port <= 65535:
        raise ConfigurationError("OSC port must be between 1 and 65535")
    if profile.camera_index < 0:
        raise ConfigurationError("Camera index cannot be negative")
    sources = profile_camera_sources(profile)
    if not 1 <= len(sources) <= 3:
        raise ConfigurationError("Select between one and three camera sources")
    if len(set(sources)) != len(sources):
        raise ConfigurationError("Camera sources must be unique")
    for source in sources:
        if not (source.startswith("local:") or source.startswith("phone:")):
            raise ConfigurationError("Camera source must start with local: or phone:")
        if source.startswith("local:"):
            try:
                if int(source.split(":", 1)[1]) < 0:
                    raise ValueError
            except ValueError as exc:
                raise ConfigurationError("Local camera source must contain a non-negative index") from exc
        elif not source.split(":", 1)[1].strip():
            raise ConfigurationError("Phone camera source must contain a device id")
    expected_mode = "MULTI" if len(sources) > 1 else "SINGLE"
    if profile.tracking_mode != expected_mode:
        raise ConfigurationError(f"Tracking mode must be {expected_mode} for {len(sources)} selected camera(s)")
    try:
        room_width, room_height, room_depth = (float(value) for value in profile.room_size_m)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Room size must contain width, height, and depth") from exc
    if not all(math.isfinite(value) for value in (room_width, room_height, room_depth)):
        raise ConfigurationError("Room size values must be finite")
    if not (1.0 <= room_width <= 20.0 and 1.8 <= room_height <= 6.0 and 1.0 <= room_depth <= 20.0):
        raise ConfigurationError("Room size must be 1–20 m wide/deep and 1.8–6 m high")
    setups = profile_camera_setups(profile)
    if profile.manual_camera_setup and {setup.source_id for setup in setups} != set(sources):
        raise ConfigurationError("Manual setup must define every selected camera")
    for setup in setups:
        if setup.source_id not in sources:
            raise ConfigurationError(f"Camera setup references an unselected source: {setup.source_id}")
        values = (*setup.position, *setup.rotation, setup.horizontal_fov)
        if not all(math.isfinite(float(value)) for value in values):
            raise ConfigurationError("Camera setup values must be finite")
        if not 25.0 <= setup.horizontal_fov <= 120.0:
            raise ConfigurationError("Camera horizontal FOV must be between 25 and 120 degrees")
        if setup.image_rotation not in {0, 90, 180, 270}:
            raise ConfigurationError("Camera image rotation must be 0, 90, 180, or 270 degrees")
    if profile.pose_quality not in {"lite", "full", "heavy"}:
        raise ConfigurationError("Pose quality must be lite, full, or heavy")
    if not 1.0 <= profile.user_height_m <= 2.5:
        raise ConfigurationError("VRChat user height must be between 1.0 and 2.5 meters")
    if profile.vrchat_tracker_set not in {"stable", "full"}:
        raise ConfigurationError("VRChat tracker set must be stable or full")
    _named_file(JOINT_MAPS_DIR, profile.joint_map, ".json")


def profile_camera_sources(profile: Profile) -> tuple[str, ...]:
    """Return the ordered sources while remaining compatible with old profiles."""
    return tuple(profile.camera_sources) if profile.camera_sources else (profile.camera_source,)


def profile_camera_setups(profile: Profile) -> tuple[CameraSetup, ...]:
    """Return one editable setup per selected camera, filling sensible room defaults."""
    sources = profile_camera_sources(profile)
    existing = {setup.source_id: setup for setup in profile.camera_setups}
    defaults = (
        ((0.0, 1.4, -2.5), (0.0, -8.0, 0.0)),
        ((2.5, 1.4, 0.0), (-90.0, -8.0, 0.0)),
        ((-2.5, 1.4, 0.0), (90.0, -8.0, 0.0)),
    )
    return tuple(
        existing.get(source, CameraSetup(source, defaults[index][0], defaults[index][1], 60.0))
        for index, source in enumerate(sources)
    )


def load_joint_map(name: str) -> dict:
    path = _named_file(JOINT_MAPS_DIR, name, ".json")
    try:
        data = _require_mapping(json.loads(path.read_text(encoding="utf-8")), "joint map")
        keypoints = _require_mapping(data.get("KeyPoints"), "KeyPoints")
        positions = _require_mapping(data.get("Position"), "Position")
        settings = _require_mapping(data.get("Settings"), "Settings")
        if not keypoints or not positions:
            raise ConfigurationError("Joint map must define KeyPoints and Position entries")
        available = set(keypoints) | {"Chest", "Mid-Hip"}
        missing = sorted(set(positions.values()) - available)
        if missing:
            raise ConfigurationError(f"Position map references unknown joints: {', '.join(missing)}")
        _require_mapping(settings.get("Correct", {}), "Settings.Correct")
        return data
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"Could not load {path}: {exc}") from exc


def run_diagnostics(report: Callable[[str, bool, str], None]) -> bool:
    """Run non-invasive checks and report (name, success, detail)."""
    all_ok = True

    def check(name: str, function: Callable[[], str]) -> None:
        nonlocal all_ok
        try:
            report(name, True, function())
        except Exception as exc:  # diagnostics must continue after individual failures
            all_ok = False
            report(name, False, str(exc))

    def python_check() -> str:
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info[:2] != (3, 12):
            raise ConfigurationError(f"running {version}; project requires Python 3.12")
        return version

    check("Python version", python_check)
    check("Settings", lambda: str(load_settings()))
    for profile_name in list_profiles():
        check(f"Profile: {profile_name}", lambda n=profile_name: str(load_profile(n)))
    for map_name in list_joint_maps():
        check(f"Joint map: {map_name}", lambda n=map_name: f"{len(load_joint_map(n)['KeyPoints'])} keypoints")
    for filename, expected_hash in POSE_MODEL_HASHES.items():
        def model_check(name=filename, expected=expected_hash) -> str:
            path = MODEL_DIR / "mediapipe" / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if not _secure_equal(digest, expected):
                raise ConfigurationError(f"checksum mismatch for {path}")
            return f"{path.stat().st_size:,} bytes; SHA-256 verified"
        check(f"Pose model: {filename}", model_check)

    modules = {
        "OpenCV": "cv2",
        "MediaPipe pose tracking": "mediapipe",
        "OSC": "pythonosc",
        "Phone server": "aiohttp",
        "Direct WebRTC video": "aiortc",
        "Phone TLS": "cryptography",
        "Friendly camera names": "pygrabber",
        "Phone QR codes": "qrcode",
    }
    for label, module in modules.items():
        def module_check(m=module) -> str:
            if not find_spec(m):
                raise ConfigurationError("not installed")
            return "available"
        check(f"Dependency: {label}", module_check)
    def openssl_check() -> str:
        from Lib.RemoteCam import find_openssl
        return str(find_openssl())
    check('OpenSSL certificate generator', openssl_check)
    return all_ok
