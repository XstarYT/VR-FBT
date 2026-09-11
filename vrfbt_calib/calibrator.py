"""Top-level multi-camera T-pose calibration pipeline."""

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np

from .camera import intrinsics_from_fov
from .keypoints import JOINT_NAMES, filter_named_keypoints
from .pairwise import estimate_jitter_threshold_px, estimate_pairwise_pose
from .posegraph import graph_is_connected, refine_global_rotations, solve_global_translations
from .types import (
    CalibrationError,
    CalibrationPrior,
    CalibrationResult,
    CameraIntrinsics,
    FloatArray,
    PairwiseCalibration,
)
from .validation import validate_consistency


@dataclass
class _Frame:
    timestamp: float
    keypoints: dict[str, dict[str, tuple[float, float]]]
    confidences: dict[str, dict[str, float]]


class TposeMultiCamCalibrator:
    """Accumulate a static T-pose capture and calibrate two or three cameras.

    Input keypoints are pixel coordinates and confidence is in [0, 1]. Output
    translations are metres in the chosen reference camera's OpenCV frame.
    ``calibrate`` raises :class:`CalibrationError` for disconnected data,
    unrecoverable pair geometry, scale failure, or inconsistent yaw geometry.
    """

    def __init__(
        self,
        cameras: dict[str, CameraIntrinsics | None],
        fov_priors_deg: dict[str, float] | None = None,
        reference_camera: str | None = None,
        priors: CalibrationPrior | None = None,
        min_confidence: float = 0.6,
        d_real_shoulder_m: float = 0.4,
        d_real_hip_m: float = 0.3,
        fov_convention: Literal["horizontal", "diagonal"] = "horizontal",
        image_sizes: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """Configure cameras, FOV fallback, priors, and body widths.

        Intrinsics and image sizes are pixels; FOV and optional yaw priors are
        degrees; shoulder/hip widths are metres. A camera with ``None``
        intrinsics needs both a FOV prior and ``image_sizes[camera]``. The
        default reference is selected from observed pair counts at calibration.
        Raises ``ValueError`` for invalid static configuration.
        """

        if len(cameras) not in (2, 3):
            raise ValueError("Exactly two or three cameras are required")
        if reference_camera is not None and reference_camera not in cameras:
            raise ValueError("reference_camera must name a configured camera")
        if not 0.4 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0.4, 1.0]")
        if d_real_shoulder_m <= 0.0 or d_real_hip_m <= 0.0:
            raise ValueError("Body-width priors must be positive")

        resolved: dict[str, CameraIntrinsics] = {}
        for camera_id, intrinsics in cameras.items():
            if intrinsics is not None:
                resolved[camera_id] = intrinsics
                continue
            if fov_priors_deg is None or camera_id not in fov_priors_deg:
                raise ValueError(f"Missing FOV prior for camera {camera_id}")
            if image_sizes is None or camera_id not in image_sizes:
                raise ValueError(f"Missing image size for camera {camera_id}")
            width, height = image_sizes[camera_id]
            resolved[camera_id] = intrinsics_from_fov(
                width, height, fov_priors_deg[camera_id], fov_convention
            )
        self.cameras = resolved
        self.requested_reference = reference_camera
        self.priors = priors
        self.min_confidence = min_confidence
        self.d_real_shoulder_m = d_real_shoulder_m
        self.d_real_hip_m = d_real_hip_m
        self._frames: list[_Frame] = []
        self._last_pairs: list[PairwiseCalibration] = []

    def collect_frame(
        self,
        keypoints: dict[str, dict[str, tuple[float, float]]],
        confidences: dict[str, dict[str, float]],
        timestamp: float,
    ) -> None:
        """Buffer one synchronized timestamp of pixel keypoints, up to 300.

        Unknown cameras/joints are ignored. A low-confidence joint is retained
        until the session-adaptive threshold is known, then skipped only for
        that joint and timestamp. This method does not perform geometry and
        does not raise :class:`CalibrationError`.
        """

        frame_points: dict[str, dict[str, tuple[float, float]]] = {}
        frame_confidences: dict[str, dict[str, float]] = {}
        for camera_id in self.cameras:
            points, scores = filter_named_keypoints(
                keypoints.get(camera_id, {}), confidences.get(camera_id, {})
            )
            frame_points[camera_id] = points
            frame_confidences[camera_id] = scores
        self._frames.append(_Frame(float(timestamp), frame_points, frame_confidences))
        if len(self._frames) > 300:
            del self._frames[0]

    def _observed_confidences(self) -> FloatArray:
        values = [
            score
            for frame in self._frames
            for camera_scores in frame.confidences.values()
            for score in camera_scores.values()
            if np.isfinite(score)
        ]
        return np.asarray(values, dtype=np.float64)

    def _pair_data(
        self, camera_a: str, camera_b: str, threshold: float
    ) -> tuple[FloatArray, FloatArray, list[tuple[float, str]]]:
        points_a: list[tuple[float, float]] = []
        points_b: list[tuple[float, float]] = []
        metadata: list[tuple[float, str]] = []
        for frame in self._frames:
            for joint in JOINT_NAMES:
                if (
                    frame.confidences[camera_a].get(joint, 0.0) >= threshold
                    and frame.confidences[camera_b].get(joint, 0.0) >= threshold
                    and joint in frame.keypoints[camera_a]
                    and joint in frame.keypoints[camera_b]
                ):
                    points_a.append(frame.keypoints[camera_a][joint])
                    points_b.append(frame.keypoints[camera_b][joint])
                    metadata.append((frame.timestamp, joint))
        return (
            np.asarray(points_a, dtype=np.float64).reshape(-1, 2),
            np.asarray(points_b, dtype=np.float64).reshape(-1, 2),
            metadata,
        )

    def _counts_at(self, threshold: float) -> dict[str, int]:
        return {
            f"{a}_{b}": len(self._pair_data(a, b, threshold)[0])
            for a, b in combinations(self.cameras, 2)
        }

    def _effective_threshold(self) -> tuple[float, bool]:
        observed = self._observed_confidences()
        if len(observed) == 0:
            return self.min_confidence, False
        percentile = float(np.percentile(observed, 40.0))
        proposed = max(self.min_confidence, percentile)
        camera_ids = list(self.cameras)
        relaxed = False
        for threshold in np.arange(proposed, 0.39, -0.05):
            current_threshold = float(threshold)
            synthetic_pairs = [
                PairwiseCalibration(a, b, "ok" if count >= 30 else "insufficient_data", count)
                for (a, b), count in zip(combinations(camera_ids, 2), self._counts_at(current_threshold).values(), strict=True)
            ]
            if graph_is_connected(camera_ids, synthetic_pairs):
                relaxed = bool(current_threshold < proposed - 1e-9)
                return max(current_threshold, 0.4), relaxed
        return 0.4, proposed > 0.4

    def _pick_reference(self, pairs: list[PairwiseCalibration]) -> str:
        if self.requested_reference is not None:
            return self.requested_reference
        totals = {camera: 0 for camera in self.cameras}
        for pair in pairs:
            if pair.status == "ok":
                totals[pair.camera_a] += pair.correspondence_count
                totals[pair.camera_b] += pair.correspondence_count
        return max(totals, key=totals.__getitem__)

    def calibrate(self) -> CalibrationResult:
        """Recover all camera poses from accumulated synchronized pixels.

        Returns rotations and metric translations relative to the reference,
        plus pixel errors and degree diagnostics. Raises
        :class:`CalibrationError` with structured payload if observations do
        not connect the rig, OpenCV cannot recover a pair, scale cannot be
        measured, or consistency limits fail.
        """

        camera_ids = list(self.cameras)
        threshold, relaxed = self._effective_threshold()
        warnings = [f"Effective confidence threshold: {threshold:.3f}"]
        if relaxed:
            warnings.append("Confidence threshold relaxed to preserve camera connectivity")

        tracks: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for camera in camera_ids:
            for joint in JOINT_NAMES:
                tracks[(camera, joint)] = [
                    frame.keypoints[camera][joint]
                    for frame in self._frames
                    if joint in frame.keypoints[camera]
                    and frame.confidences[camera].get(joint, 0.0) >= threshold
                ]
        ransac_threshold = estimate_jitter_threshold_px(tracks)
        pairs: list[PairwiseCalibration] = []
        for camera_a, camera_b in combinations(camera_ids, 2):
            points_a, points_b, metadata = self._pair_data(camera_a, camera_b, threshold)
            try:
                pair = estimate_pairwise_pose(
                    camera_a,
                    camera_b,
                    points_a,
                    points_b,
                    metadata,
                    self.cameras[camera_a],
                    self.cameras[camera_b],
                    ransac_threshold,
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                raise CalibrationError(
                    f"Pairwise calibration failed for {camera_a}_{camera_b}: {error}",
                    {
                        "offending_pairs": [f"{camera_a}_{camera_b}"],
                        "measured_values": {"correspondence_count": len(points_a)},
                        "residual_deg": None,
                    },
                ) from error
            pairs.append(pair)
            if pair.status == "ok" and pair.cheirality_fraction < 0.5:
                warnings.append(
                    f"{pair.pair_key}: low cheirality pass fraction {pair.cheirality_fraction:.1%}"
                )

        self._last_pairs = pairs

        if not graph_is_connected(camera_ids, pairs):
            insufficient = [pair.pair_key for pair in pairs if pair.status == "insufficient_data"]
            raise CalibrationError(
                "Insufficient observations to connect all cameras",
                {
                    "offending_pairs": insufficient,
                    "measured_values": {pair.pair_key: pair.correspondence_count for pair in pairs},
                    "residual_deg": None,
                },
            )

        reference = self._pick_reference(pairs)
        anchor_candidates = [
            pair
            for pair in pairs
            if pair.status == "ok"
            and reference in (pair.camera_a, pair.camera_b)
            and pair.raw_shoulder_distance is not None
            and pair.raw_shoulder_distance > 0.0
        ]
        if not anchor_candidates:
            raise CalibrationError(
                "Could not measure shoulder width for metric scale",
                {
                    "offending_pairs": [pair.pair_key for pair in pairs],
                    "measured_values": {},
                    "residual_deg": None,
                },
            )
        anchor = min(anchor_candidates, key=lambda item: item.mean_reprojection_error_px)
        assert anchor.raw_shoulder_distance is not None
        anchor_raw = float(anchor.raw_shoulder_distance)
        scale_factor = self.d_real_shoulder_m / anchor_raw
        if anchor.raw_hip_distance is not None and anchor.raw_hip_distance > 0.0:
            hip_scale = self.d_real_hip_m / anchor.raw_hip_distance
            disagreement = abs(hip_scale - scale_factor) / scale_factor
            if disagreement > 0.15:
                warnings.append(
                    f"Shoulder/hip scale estimates disagree by {disagreement:.1%}; using shoulders"
                )

        rotations, before, after = refine_global_rotations(camera_ids, reference, pairs)
        if len([pair for pair in pairs if pair.status == "ok"]) == 3:
            warnings.append(
                f"Three-camera joint rotation refinement residual: {before:.6g} -> {after:.6g}"
            )
        metric_pairs = [
            pair for pair in pairs
            if pair.status == "ok"
            and pair.raw_shoulder_distance is not None
            and pair.raw_shoulder_distance > 0.0
        ]
        excluded_scale_pairs = [
            pair.pair_key for pair in pairs
            if pair.status == "ok"
            and (pair.raw_shoulder_distance is None or pair.raw_shoulder_distance <= 0.0)
        ]
        if excluded_scale_pairs:
            warnings.append(
                "Excluded edge(s) without metric shoulder scale: "
                + ", ".join(excluded_scale_pairs)
            )
        if not graph_is_connected(camera_ids, metric_pairs):
            raise CalibrationError(
                "Metric translation is underdetermined after excluding unscaled edges",
                {
                    "offending_pairs": excluded_scale_pairs,
                    "measured_values": {
                        pair.pair_key: pair.raw_shoulder_distance for pair in pairs
                    },
                    "residual_deg": None,
                },
            )
        translations_raw = solve_global_translations(
            camera_ids, reference, rotations, metric_pairs, anchor_raw
        )
        translations = {camera: value * scale_factor for camera, value in translations_raw.items()}
        residual = validate_consistency(camera_ids, pairs, self.priors)

        for pair in pairs:
            if pair.status == "ok" and pair.mean_reprojection_error_px > 5.0:
                warnings.append(
                    f"{pair.pair_key}: high reprojection error {pair.mean_reprojection_error_px:.2f}px"
                )
        return CalibrationResult(
            reference_camera=reference,
            R=rotations,
            t_scaled=translations,
            scale_factor=scale_factor,
            scale_anchor_pair=anchor.pair_key,
            mean_reprojection_error_px={
                pair.pair_key: pair.mean_reprojection_error_px for pair in pairs
            },
            measured_yaw_deg={
                pair.pair_key: float(pair.measured_yaw_deg)
                for pair in pairs
                if pair.measured_yaw_deg is not None
            },
            consistency_residual_deg=residual,
            warnings=warnings,
            effective_confidence_threshold=threshold,
            pair_status={pair.pair_key: pair.status for pair in pairs},
            valid_correspondence_counts={
                pair.pair_key: pair.correspondence_count for pair in pairs
            },
        )
