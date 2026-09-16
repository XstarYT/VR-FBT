"""Explicit local support exports with bounded input and redacted identifiers."""

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

from Lib.Config import PROJECT_ROOT, profile_camera_setups, profile_camera_sources, run_diagnostics, validate_profile, validate_settings
from Lib.Logging import log_path

LOG_TAIL_BYTES = 256 * 1024
PACKAGES = ("numpy", "scipy", "opencv-contrib-python", "mediapipe", "python-osc",
            "aiohttp", "aiortc", "av", "cryptography", "pygrabber", "Pillow", "pywin32")


class Redactor:
    def __init__(self, replacements=()):
        # Apply longest matches first so a repository under the user's home
        # becomes <project>, rather than retaining its relative private path.
        pairs = [(str(value), label) for value, label in replacements if value]
        self.replacements = sorted(set(pairs), key=lambda pair: len(pair[0]), reverse=True)

    def __call__(self, text):
        text = str(text)
        for value, replacement in self.replacements:
            variants = {value, value.replace("\\", "/"), value.replace("/", "\\")}
            for variant in variants:
                pattern = re.escape(variant)
                if re.fullmatch(r"[\w-]+", variant):
                    pattern = r"(?<!\w)" + pattern + r"(?!\w)"
                text = re.sub(pattern, lambda _match: replacement, text, flags=re.IGNORECASE)
        text = re.sub(r"(?i)([?#&]token=)[^\s&#\"'<>]+", r"\1<redacted>", text)
        text = re.sub(r'''(?i)(["']?token["']?\s*[:=]\s*["']?)[^\s,}\]"']+''', r"\1<redacted>", text)
        text = re.sub(r"(?i)phone:[\w-]+", "phone:<device>", text)
        text = re.sub(r"(?i)\b(https?|wss?)://[^/\s?#]+", r"\1://<host>", text)
        text = re.sub(r"(?i)\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", "<device-id>", text)
        text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<ip-address>", text)
        # Older logs can come from another Windows account or checkout.
        text = re.sub(r"(?i)[a-z]:[\\/]Users[\\/][^\\/\s\"']+", "<user-home>", text)
        return text


def _profile_summary(settings, profile):
    sources = profile_camera_sources(profile)
    aliases = {source: f"camera-{index + 1}" for index, source in enumerate(sources)}
    return {
        "target_fps": settings.fps,
        "tracking_mode": profile.tracking_mode,
        "pose_quality": profile.pose_quality,
        "tracker_set": profile.vrchat_tracker_set,
        "smoothing": profile.smooth,
        "debug_windows": profile.show_output,
        "manual_camera_setup": profile.manual_camera_setup,
        "room_size_m": profile.room_size_m,
        "cameras": [{"alias": aliases[item.source_id],
                     "kind": "phone" if item.source_id.startswith("phone:") else "local",
                     "position_m": item.position, "rotation_degrees": item.rotation,
                     "horizontal_fov": item.horizontal_fov, "image_rotation": item.image_rotation,
                     "latency_ms": item.latency_ms}
                    for item in profile_camera_setups(profile)],
    }


def _redact_structure(value, redact):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {redact(key): _redact_structure(item, redact) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_structure(item, redact) for item in value]
    return value


def export_support_bundle(destination, settings=None, profile=None, *, secrets=(), activity_text="",
                          logs=None, diagnostics=run_diagnostics, snapshot_error=None, runtime_metrics=None):
    """Write a ZIP locally; no video, certificates, environment dump or upload.

    Uses the supplied settings snapshot, which may include unsaved GUI edits.
    Diagnostics validate saved files separately. Log tails are bounded to 1 MiB.
    """
    if settings is not None and profile is not None:
        validate_settings(settings)
        validate_profile(profile)
    destination = Path(destination)
    replacements = [(PROJECT_ROOT, "<project>"), (Path.home(), "<user-home>")]
    if profile is not None:
        replacements += [(profile.server_ip, "<osc-host>"), (profile.name, "<profile>")]
        replacements += [(source, f"camera-{index + 1}")
                         for index, source in enumerate(profile_camera_sources(profile))]
    replacements += [(secret, "<redacted>") for secret in secrets if secret]
    # Profile names occur in diagnostic labels and reprs, not just filenames.
    redact = Redactor(replacements)
    reports = []
    def report(name, passed, detail):
        if passed and (name == "Settings" or name == "Default profile" or name.startswith("Profile: ")):
            detail = "Saved configuration validated"
        reports.append({"check": redact(name), "passed": bool(passed), "detail": redact(detail)})
    diagnostics(report)
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "system": platform.system(),
                    "release": platform.release(), "machine": platform.machine()},
        "packages": packages,
        "configuration_snapshot": _profile_summary(settings, profile) if settings is not None and profile is not None else None,
        "configuration_snapshot_error": redact(snapshot_error) if snapshot_error else None,
        "runtime_metrics": _redact_structure(runtime_metrics, redact),
        "saved_configuration_diagnostics": reports,
        "notes": ["Configuration snapshot can include unsaved GUI edits.",
                  "Diagnostics inspect saved configuration.",
                  "No frames, body landmarks, certificates or private keys are included.",
                  "Review the ZIP before sharing; custom error messages can contain personal text."],
    }
    current_log = Path(logs) if logs is not None else log_path()
    entries = {"summary.json": json.dumps(summary, indent=2, allow_nan=False) + "\n"}
    if activity_text:
        tail = activity_text.encode("utf-8")[-LOG_TAIL_BYTES:].decode("utf-8", errors="replace")
        entries["activity.txt"] = redact(tail)
    for index in range(4):
        source = current_log if index == 0 else current_log.with_name(f"{current_log.name}.{index}")
        try:
            with source.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                length = handle.tell()
                handle.seek(max(0, length - LOG_TAIL_BYTES))
                entries[f"logs/runtime-{index}.txt"] = redact(handle.read(LOG_TAIL_BYTES).decode("utf-8", errors="replace"))
        except FileNotFoundError:
            continue
        except OSError as exc:
            entries[f"logs/runtime-{index}-unavailable.txt"] = redact(f"Could not read log: {exc}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            for name, content in entries.items():
                archive.writestr(name, content)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
