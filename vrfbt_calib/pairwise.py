"""Pairwise Essential-matrix pose recovery and triangulation."""

from collections.abc import Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .scale import attach_body_widths
from .types import CameraIntrinsics, FloatArray, PairwiseCalibration


MIN_CORRESPONDENCES = 30


def measured_yaw_deg(rotation: FloatArray) -> float:
    """Return yaw in degrees using SciPy extrinsic ``xyz`` Euler convention.

    With OpenCV camera axes (x right, y down, z forward), yaw is the second
    returned component (rotation about y). Keeping this extraction centralized
    prevents accidental mixing with intrinsic Euler conventions.
    """

    euler = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
    yaw = float(euler[1])
    # scipy's canonical xyz middle angle is limited to [-90, 90]. For camera
    # separations beyond 90 degrees the same matrix is represented with x/z
    # near 180 degrees; unfold that equivalent branch to retain directed yaw.
    if abs(float(euler[0])) > 90.0 and abs(float(euler[2])) > 90.0:
        yaw = 180.0 - yaw if yaw >= 0.0 else -180.0 - yaw
    return yaw


def estimate_jitter_threshold_px(
    points_by_track: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
) -> float:
    """Estimate a 1–5 pixel RANSAC threshold from consecutive-point jitter."""

    samples: list[float] = []
    for positions in points_by_track.values():
        values = np.asarray(positions, dtype=np.float64)
        if len(values) < 3:
            continue
        deltas = np.diff(values, axis=0)
        centered = deltas - np.median(deltas, axis=0)
        radial = np.linalg.norm(centered, axis=1) / np.sqrt(2.0)
        samples.extend(radial.tolist())
    noise = float(np.std(samples)) if samples else 0.5
    return float(np.clip(2.0 * noise, 1.0, 5.0))


def _points_b_in_a_pixels(
    points_b: FloatArray, intrinsics_a: CameraIntrinsics, intrinsics_b: CameraIntrinsics
) -> FloatArray:
    normalized = cv2.undistortPoints(points_b.reshape(-1, 1, 2), intrinsics_b.matrix, None)
    normalized = normalized.reshape(-1, 2)
    return np.column_stack(
        (
            normalized[:, 0] * intrinsics_a.fx + intrinsics_a.cx,
            normalized[:, 1] * intrinsics_a.fy + intrinsics_a.cy,
        )
    ).astype(np.float64)


def _normalized_homogeneous(points: FloatArray, intrinsics: CameraIntrinsics) -> FloatArray:
    normalized = cv2.undistortPoints(points.reshape(-1, 1, 2), intrinsics.matrix, None)
    return np.column_stack((normalized.reshape(-1, 2), np.ones(len(points))))


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _refine_essential_pose(
    rotation: FloatArray,
    translation: FloatArray,
    points_a: FloatArray,
    points_b: FloatArray,
    intrinsics_a: CameraIntrinsics,
    intrinsics_b: CameraIntrinsics,
    inliers: NDArray[np.bool_],
    threshold_px: float,
) -> tuple[FloatArray, FloatArray]:
    """Robustly refine five-DOF pose using geometric Sampson residuals."""

    normalized_a = _normalized_homogeneous(points_a[inliers], intrinsics_a)
    normalized_b = _normalized_homogeneous(points_b[inliers], intrinsics_b)
    focal_scale = float(
        np.mean([intrinsics_a.fx, intrinsics_a.fy, intrinsics_b.fx, intrinsics_b.fy])
    )
    initial = np.concatenate((Rotation.from_matrix(rotation).as_rotvec(), translation))

    def residuals(values: FloatArray) -> FloatArray:
        candidate_r = Rotation.from_rotvec(values[:3]).as_matrix()
        candidate_t = values[3:]
        norm = np.linalg.norm(candidate_t)
        if norm < 1e-9:
            candidate_t = translation
        else:
            candidate_t = candidate_t / norm
        essential = _skew(candidate_t) @ candidate_r
        ex1 = (essential @ normalized_a.T).T
        etx2 = (essential.T @ normalized_b.T).T
        numerator = np.sum(normalized_b * ex1, axis=1)
        denominator = np.sqrt(
            ex1[:, 0] ** 2
            + ex1[:, 1] ** 2
            + etx2[:, 0] ** 2
            + etx2[:, 1] ** 2
            + 1e-15
        )
        return numerator / denominator * focal_scale

    optimized = least_squares(
        residuals,
        initial,
        method="trf",
        loss="soft_l1",
        f_scale=max(threshold_px, 1.0),
        max_nfev=150,
    )
    refined_r = Rotation.from_rotvec(optimized.x[:3]).as_matrix()
    refined_t = optimized.x[3:]
    refined_t /= np.linalg.norm(refined_t)
    if float(np.dot(refined_t, translation)) < 0.0:
        refined_t = -refined_t
    return refined_r, refined_t


def _project(points: FloatArray, intrinsics: CameraIntrinsics) -> FloatArray:
    safe_z = np.where(np.abs(points[:, 2]) > 1e-12, points[:, 2], np.nan)
    return np.column_stack(
        (
            intrinsics.fx * points[:, 0] / safe_z + intrinsics.cx,
            intrinsics.fy * points[:, 1] / safe_z + intrinsics.cy,
        )
    )


def reprojection_residuals(
    pair: PairwiseCalibration, rotation: FloatArray | None = None
) -> FloatArray:
    """Return signed two-view pixel residuals for a pair and candidate rotation.

    Translation direction remains the Essential-matrix estimate. Points are
    re-triangulated for the candidate, so this is a genuine image reprojection
    objective used by the three-camera joint refinement.
    """

    if (
        pair.t_unit is None
        or pair.points_a_px is None
        or pair.points_b_px is None
        or pair.intrinsics_a is None
        or pair.intrinsics_b is None
    ):
        raise ValueError("Pair does not contain reprojection data")
    candidate = pair.R if rotation is None else rotation
    if candidate is None:
        raise ValueError("Pair has no rotation")
    pair_t = pair.t_unit
    points_a = pair.points_a_px
    points_b = pair.points_b_px
    intrinsics_a = pair.intrinsics_a
    intrinsics_b = pair.intrinsics_b
    projection_a = intrinsics_a.matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    projection_b = intrinsics_b.matrix @ np.hstack(
        (candidate, pair_t.reshape(3, 1))
    )
    homogeneous = np.asarray(
        cv2.triangulatePoints(projection_a, projection_b, points_a.T, points_b.T),
        dtype=np.float64,
    )
    xyz = np.asarray((homogeneous[:3] / homogeneous[3:4]).T, dtype=np.float64)
    xyz_b = np.asarray((candidate @ xyz.T).T + pair_t, dtype=np.float64)
    residual_a = _project(xyz, intrinsics_a) - points_a
    residual_b = _project(xyz_b, intrinsics_b) - points_b
    residual = np.hstack((residual_a, residual_b)).reshape(-1)
    return np.nan_to_num(residual, nan=1e3, posinf=1e3, neginf=-1e3)


def mean_reprojection_for_rotation(
    pair: PairwiseCalibration, rotation: FloatArray
) -> float:
    """Return root-mean-square image reprojection residual in pixels."""

    values = reprojection_residuals(pair, rotation)
    return float(np.sqrt(np.mean(values**2)))


def estimate_pairwise_pose(
    camera_a: str,
    camera_b: str,
    points_a: FloatArray,
    points_b: FloatArray,
    metadata: list[tuple[float, str]],
    intrinsics_a: CameraIntrinsics,
    intrinsics_b: CameraIntrinsics,
    ransac_threshold_px: float,
) -> PairwiseCalibration:
    """Estimate ``X_B = R @ X_A + t`` from pixel correspondences.

    All per-frame joint observations are supplied individually—never averaged
    before RANSAC—because repeated noisy capture samples are essential to its
    outlier robustness. Fewer than 30 correspondences return an
    ``insufficient_data`` record. OpenCV failures raise ``ValueError``.
    """

    count = len(points_a)
    if count < MIN_CORRESPONDENCES:
        return PairwiseCalibration(camera_a, camera_b, "insufficient_data", count)
    if points_a.shape != points_b.shape or points_a.shape != (count, 2):
        raise ValueError("Pairwise point arrays must have equal Nx2 shape")

    points_b_fit = _points_b_in_a_pixels(points_b, intrinsics_a, intrinsics_b)
    # OpenCV's RANSAC has process-global RNG state; reset it so an identical
    # capture produces identical calibration regardless of earlier calls.
    cv2.setRNGSeed(1337)
    essential, ransac_mask = cv2.findEssentialMat(
        points_a,
        points_b_fit,
        intrinsics_a.matrix,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=float(ransac_threshold_px),
    )
    if essential is None:
        raise ValueError(f"Essential-matrix estimation failed for {camera_a}_{camera_b}")
    if essential.shape != (3, 3):
        candidates = essential.reshape(-1, 3, 3)
    else:
        candidates = essential.reshape(1, 3, 3)

    best: tuple[int, FloatArray, FloatArray, np.ndarray] | None = None
    for candidate in candidates:
        passing, rotation, translation, pose_mask = cv2.recoverPose(
            candidate,
            points_a,
            points_b_fit,
            intrinsics_a.matrix,
            mask=ransac_mask.copy() if ransac_mask is not None else None,
        )
        if best is None or passing > best[0]:
            best = (
                int(passing),
                np.asarray(rotation, dtype=np.float64),
                np.asarray(translation, dtype=np.float64).reshape(3),
                np.asarray(pose_mask),
            )
    if best is None:
        raise ValueError(f"Pose recovery failed for {camera_a}_{camera_b}")
    passing, rotation, translation, pose_mask = best
    valid = pose_mask.reshape(-1).astype(bool)
    # Refine over the full accumulated set with a robust loss. RANSAC supplies
    # the basin; retaining every sample here recovers the precision benefit of
    # the long capture without pre-averaging joints or trusting gross outliers.
    refinement_valid = np.ones(count, dtype=bool)
    if np.count_nonzero(refinement_valid) >= 8:
        rotation, translation = _refine_essential_pose(
            rotation,
            translation,
            points_a,
            points_b,
            intrinsics_a,
            intrinsics_b,
            refinement_valid,
            ransac_threshold_px,
        )

    projection_a = intrinsics_a.matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    projection_b = intrinsics_b.matrix @ np.hstack((rotation, translation.reshape(3, 1)))
    homogeneous = np.asarray(
        cv2.triangulatePoints(
            projection_a,
            projection_b,
            points_a.T,
            points_b.T,
        ),
        dtype=np.float64,
    )
    xyz = np.asarray((homogeneous[:3] / homogeneous[3:4]).T, dtype=np.float64)
    xyz_b = np.asarray((rotation @ xyz.T).T + translation, dtype=np.float64)
    finite = np.all(np.isfinite(xyz), axis=1)
    positive_depth = (xyz[:, 2] > 0.0) & (xyz_b[:, 2] > 0.0)
    valid &= finite & positive_depth

    if np.any(valid):
        error_a = np.linalg.norm(_project(xyz, intrinsics_a) - points_a, axis=1)
        error_b = np.linalg.norm(_project(xyz_b, intrinsics_b) - points_b, axis=1)
        reprojection = float(np.mean(np.concatenate((error_a[valid], error_b[valid]))))
    else:
        reprojection = float("inf")
    pair = PairwiseCalibration(
        camera_a=camera_a,
        camera_b=camera_b,
        status="ok",
        correspondence_count=count,
        R=rotation.astype(np.float64),
        t_unit=translation.astype(np.float64),
        inlier_mask=valid,
        cheirality_fraction=passing / count,
        mean_reprojection_error_px=reprojection,
        measured_yaw_deg=measured_yaw_deg(rotation),
        triangulated_points=xyz.astype(np.float64),
        correspondence_metadata=metadata,
        points_a_px=points_a.copy(),
        points_b_px=points_b.copy(),
        intrinsics_a=intrinsics_a,
        intrinsics_b=intrinsics_b,
    )
    return attach_body_widths(pair)
