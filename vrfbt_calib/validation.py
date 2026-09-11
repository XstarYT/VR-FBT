"""Geometry consistency checks and structured calibration failures."""

import numpy as np

from .pairwise import measured_yaw_deg
from .types import CalibrationError, CalibrationPrior, FloatArray, PairwiseCalibration


def _directed_rotation(
    pair: PairwiseCalibration, source: str, target: str
) -> FloatArray:
    if pair.R is None:
        raise ValueError("Pair has no rotation")
    if pair.camera_a == source and pair.camera_b == target:
        return pair.R
    if pair.camera_b == source and pair.camera_a == target:
        return pair.R.T
    raise ValueError("Requested cameras do not match pair")


def _angular_difference_deg(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def validate_consistency(
    camera_ids: list[str],
    pairs: list[PairwiseCalibration],
    prior: CalibrationPrior | None,
) -> float | None:
    """Validate yaw closure or a two-camera prior and return degrees residual.

    Directed pair yaw uses extrinsic ``xyz`` Euler extraction about OpenCV's
    y-axis. For a triangle A→B→C→A, each yaw is mapped to [0, 360); distance
    of their sum to the nearest multiple of 360 is the reported residual.
    Small pitch/roll make Euler yaw only approximately additive, hence the
    specified 20-degree tolerance.
    """

    usable = [pair for pair in pairs if pair.status == "ok"]
    if len(camera_ids) == 2:
        if not usable:
            return None
        pair = usable[0]
        measured = float(pair.measured_yaw_deg or 0.0)
        if prior is None:
            return None
        direct_key = pair.pair_key
        reverse_key = f"{pair.camera_b}_{pair.camera_a}"
        if direct_key in prior.pair_yaw_deg:
            expected = prior.pair_yaw_deg[direct_key]
        elif reverse_key in prior.pair_yaw_deg:
            expected = -prior.pair_yaw_deg[reverse_key]
        else:
            return None
        residual = _angular_difference_deg(measured, expected)
        if residual > 30.0:
            raise CalibrationError(
                f"Measured yaw for {direct_key} differs from its prior by {residual:.1f}°",
                {
                    "offending_pairs": [direct_key],
                    "measured_values": {direct_key: measured, "expected_yaw_deg": expected},
                    "residual_deg": residual,
                },
            )
        return residual

    if len(camera_ids) != 3 or len(usable) < 3:
        return None
    ordered = [(camera_ids[0], camera_ids[1]), (camera_ids[1], camera_ids[2]), (camera_ids[2], camera_ids[0])]
    directed_yaws: dict[str, float] = {}
    for source, target in ordered:
        pair = next(
            candidate
            for candidate in usable
            if {candidate.camera_a, candidate.camera_b} == {source, target}
        )
        yaw = measured_yaw_deg(_directed_rotation(pair, source, target)) % 360.0
        directed_yaws[f"{source}_{target}"] = yaw
    yaw_sum = sum(directed_yaws.values())
    residual = abs((yaw_sum + 180.0) % 360.0 - 180.0)
    if residual > 20.0:
        worst = min(
            usable,
            key=lambda item: (item.cheirality_fraction, -item.mean_reprojection_error_px),
        )
        raise CalibrationError(
            f"Three-camera yaw closure residual is {residual:.1f}°; {worst.pair_key} is least reliable",
            {
                "offending_pairs": [worst.pair_key],
                "measured_values": directed_yaws,
                "residual_deg": residual,
            },
        )
    return residual
