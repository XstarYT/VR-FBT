"""Bounded, persistent diagnostics for console and pythonw launches."""

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re


def log_path() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VR-FBT" / "logs" / "vr-fbt.log"


class _RedactToken(logging.Filter):
    def filter(self, record):
        record.msg = re.sub(r"([?#&]token=)[^\s&#]+", r"\1[REDACTED]", record.getMessage())
        record.args = ()
        return True


class _RedactingFormatter(logging.Formatter):
    def format(self, record):
        return re.sub(r"([?#&]token=)[^\s&#]+", r"\1[REDACTED]", super().format(record))


def configure_logging() -> Path | None:
    logger = logging.getLogger("vrfbt")
    path = log_path()
    if logger.handlers:
        return path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    except OSError:
        return None  # A read-only log directory must not prevent tracking.
    handler.addFilter(_RedactToken())
    handler.setFormatter(_RedactingFormatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return path
