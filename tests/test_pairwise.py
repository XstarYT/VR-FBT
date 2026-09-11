"""Pairwise dropout and structured consistency failure tests."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vrfbt_calib.pairwise import estimate_pairwise_pose
from vrfbt_calib.types import CalibrationError, CameraIntrinsics, PairwiseCalibration
from vrfbt_calib.validation import validate_consistency


def test_insufficient_data_returns_status_without_opencv_failure() -> None:
    intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
    points = np.zeros((29, 2), dtype=np.float64)
    pair = estimate_pairwise_pose(
        "A", "B", points, points, [(float(index), "nose") for index in range(29)], intrinsics, intrinsics, 1.0
    )
    assert pair.status == "insufficient_data"
    assert pair.correspondence_count == 29
    assert pair.R is None


def _pair(a: str, b: str, yaw: float, quality: float = 0.9) -> PairwiseCalibration:
    return PairwiseCalibration(
        a,
        b,
        "ok",
        100,
        R=Rotation.from_euler("y", yaw, degrees=True).as_matrix(),
        t_unit=np.array([1.0, 0.0, 0.0]),
        cheirality_fraction=quality,
        mean_reprojection_error_px=1.0 / quality,
        measured_yaw_deg=yaw,
    )


def test_broken_triangle_raises_structured_calibration_error() -> None:
    pairs = [_pair("A", "B", 70.0), _pair("B", "C", 70.0), _pair("A", "C", -70.0, 0.4)]
    with pytest.raises(CalibrationError) as captured:
        validate_consistency(["A", "B", "C"], pairs, None)
    assert captured.value.offending_pairs == ["A_C"]
    assert captured.value.residual_deg is not None
    assert captured.value.residual_deg > 20.0
