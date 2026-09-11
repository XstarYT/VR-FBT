"""Automatic two/three-camera T-pose extrinsic calibration."""

from .calibrator import TposeMultiCamCalibrator
from .camera import CameraStream, KeypointSource, intrinsics_from_fov
from .types import CalibrationError, CalibrationPrior, CalibrationResult, CameraIntrinsics

__all__ = [
    "CalibrationError",
    "CalibrationPrior",
    "CalibrationResult",
    "CameraIntrinsics",
    "CameraStream",
    "KeypointSource",
    "TposeMultiCamCalibrator",
    "intrinsics_from_fov",
]
