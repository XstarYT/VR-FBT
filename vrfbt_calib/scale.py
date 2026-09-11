"""Robust anthropometric scale measurements from triangulated joints."""

from collections import defaultdict

import numpy as np

from .types import PairwiseCalibration


def attach_body_widths(pair: PairwiseCalibration) -> PairwiseCalibration:
    """Attach median shoulder/hip widths in the pair's arbitrary 3-D units."""

    if pair.triangulated_points is None or pair.inlier_mask is None:
        return pair
    by_frame: dict[float, dict[str, np.ndarray]] = defaultdict(dict)
    for point, valid, (timestamp, joint) in zip(
        pair.triangulated_points,
        pair.inlier_mask,
        pair.correspondence_metadata,
        strict=True,
    ):
        if valid and np.all(np.isfinite(point)):
            by_frame[timestamp][joint] = point

    shoulders: list[float] = []
    hips: list[float] = []
    for joints in by_frame.values():
        if "left_shoulder" in joints and "right_shoulder" in joints:
            shoulders.append(
                float(np.linalg.norm(joints["left_shoulder"] - joints["right_shoulder"]))
            )
        if "left_hip" in joints and "right_hip" in joints:
            hips.append(float(np.linalg.norm(joints["left_hip"] - joints["right_hip"])))
    pair.raw_shoulder_distance = float(np.median(shoulders)) if shoulders else None
    pair.raw_hip_distance = float(np.median(hips)) if hips else None
    return pair


def relative_translation_factor(pair: PairwiseCalibration, anchor_raw: float) -> float:
    """Convert an edge's unit baseline into the anchor's arbitrary 3-D units.

    Essential matrices normalize every baseline independently. Since a known
    rigid shoulder has raw size inversely proportional to baseline, this ratio
    reconciles *relative* edge lengths before one global metric scale is used.
    """

    if pair.raw_shoulder_distance is None or pair.raw_shoulder_distance <= 0.0:
        raise ValueError(
            f"Pair {pair.pair_key} has no simultaneous shoulder measurement; "
            "its metric baseline is underdetermined"
        )
    return anchor_raw / pair.raw_shoulder_distance
