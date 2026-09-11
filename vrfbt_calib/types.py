"""Public data types for multi-camera T-pose calibration."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels for an image of ``width`` x ``height``."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> FloatArray:
        """Return the 3x3 pixel calibration matrix."""

        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CalibrationPrior:
    """Optional approximate directed pair yaws in degrees, keyed as ``A_B``."""

    pair_yaw_deg: dict[str, float] = field(default_factory=dict)


@dataclass
class PairwiseCalibration:
    """Relative pose mapping camera-A coordinates into camera-B coordinates.

    ``R`` is dimensionless, ``t_unit`` has unit norm, image errors are pixels,
    and triangulated points use the arbitrary coordinate scale induced by
    ``t_unit``. An ``insufficient_data`` record carries no pose arrays.
    """

    camera_a: str
    camera_b: str
    status: Literal["ok", "insufficient_data"]
    correspondence_count: int
    R: FloatArray | None = None
    t_unit: FloatArray | None = None
    inlier_mask: NDArray[np.bool_] | None = None
    cheirality_fraction: float = 0.0
    mean_reprojection_error_px: float = float("inf")
    measured_yaw_deg: float | None = None
    triangulated_points: FloatArray | None = None
    correspondence_metadata: list[tuple[float, str]] = field(default_factory=list)
    raw_shoulder_distance: float | None = None
    raw_hip_distance: float | None = None
    points_a_px: FloatArray | None = None
    points_b_px: FloatArray | None = None
    intrinsics_a: CameraIntrinsics | None = None
    intrinsics_b: CameraIntrinsics | None = None

    @property
    def pair_key(self) -> str:
        """Return the stable directed pair key ``camera_a_camera_b``."""

        return f"{self.camera_a}_{self.camera_b}"


@dataclass
class CalibrationResult:
    """Calibrated camera poses relative to the selected reference camera.

    Each pose follows OpenCV coordinates: ``X_camera = R[camera] @ X_ref +
    t_scaled[camera]``. Rotations are 3x3 matrices, translations are metres,
    reprojection errors are pixels, and angles are degrees. Pair dictionaries
    use directed keys of the form ``A_B``.
    """

    reference_camera: str
    R: dict[str, FloatArray]
    t_scaled: dict[str, FloatArray]
    scale_factor: float
    scale_anchor_pair: str
    mean_reprojection_error_px: dict[str, float]
    measured_yaw_deg: dict[str, float]
    consistency_residual_deg: float | None
    warnings: list[str]
    effective_confidence_threshold: float
    pair_status: dict[str, Literal["ok", "insufficient_data"]]
    valid_correspondence_counts: dict[str, int]


class CalibrationError(RuntimeError):
    """Calibration failure with machine-readable diagnostic ``payload``."""

    def __init__(self, message: str, payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.payload = payload
        offending = payload.get("offending_pairs", [])
        self.offending_pairs = list(offending) if isinstance(offending, list) else []
        measured = payload.get("measured_values", {})
        self.measured_values = dict(measured) if isinstance(measured, dict) else {}
        residual = payload.get("residual_deg")
        self.residual_deg = float(residual) if isinstance(residual, int | float) else None
