"""Fast, detector-backed 3D pose tracking using MediaPipe Tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "Model" / "mediapipe"
MODEL_PATHS = {
    "lite": MODEL_DIR / "pose_landmarker_lite.task",
    "full": MODEL_DIR / "pose_landmarker_full.task",
    "heavy": MODEL_DIR / "pose_landmarker_heavy.task",
}

# VR-FBT's historical map contains 31 points. MediaPipe has two extra heel
# landmarks at 29/30; use its foot-index landmarks 31/32 for the map's L/R-Foot.
MEDIAPIPE_TO_VRFBT = (*range(29), 31, 32)
CORE_LANDMARKS = (11, 12, 23, 24, 25, 26, 27, 28)


@dataclass(frozen=True, slots=True)
class PoseResult:
    detected: bool
    confidence: float
    image_landmarks: list[list[float]]
    world_landmarks: list[list[float]]


@dataclass(frozen=True, slots=True)
class CameraObservation:
    """A pose observation and the image size that produced it."""

    source_id: str
    pose: PoseResult
    frame_size: tuple[int, int]  # width, height


@dataclass(frozen=True, slots=True)
class CameraPose:
    """Approximate camera pose in the calibrated tracking coordinate system."""

    source_id: str
    position: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]
    reprojection_error: float
    calibrated: bool = True
    sample_count: int = 0
    position_std: float = 0.0


@dataclass(frozen=True, slots=True)
class FusionResult:
    pose: PoseResult
    camera_poses: tuple[CameraPose, ...]
    calibrated: bool
    calibration_progress: tuple[float, float]
    contributing_cameras: int
    calibration_hint: str = ""


class MultiCameraPoseFusion:
    """Calibrate up to three views and triangulate their 2D pose landmarks.

    Phones do not report lens or extrinsic calibration.  We therefore assume a
    typical horizontal field of view and use the neutral MediaPipe metric
    skeleton as a temporary calibration target.  Camera positions shown in the
    debug scene are deliberately labelled approximate; the triangulated joints
    still gain useful occlusion resistance and multi-view depth constraints.
    """

    def __init__(
        self,
        source_ids: Sequence[str],
        calibration_frames: int = 12,
        horizontal_fov_degrees: float = 60.0,
        manual_camera_setup: bool = False,
        camera_setups: Sequence[object] = (),
        room_size_m: Sequence[float] = (4.0, 2.7, 4.0),
        calibration_duration_seconds: float | None = None,
        clock=None,
    ):
        source_ids = tuple(source_ids)
        if not 1 <= len(source_ids) <= 3 or len(set(source_ids)) != len(source_ids):
            raise ValueError("Multi-camera fusion needs one to three unique sources")
        if calibration_frames < 1:
            raise ValueError("Camera calibration must use at least one frame")
        if not 25.0 <= horizontal_fov_degrees <= 120.0:
            raise ValueError("Horizontal field of view must be between 25 and 120 degrees")
        self.source_ids = source_ids
        self.calibration_frames = int(calibration_frames)
        self.horizontal_fov_degrees = float(horizontal_fov_degrees)
        self.manual_camera_setup = bool(manual_camera_setup)
        self.camera_setups = {setup.source_id: setup for setup in camera_setups}
        if self.manual_camera_setup and set(self.camera_setups) != set(source_ids):
            raise ValueError("Manual setup must define every selected camera")
        self.room_size_m = tuple(float(value) for value in room_size_m)
        if len(self.room_size_m) != 3:
            raise ValueError("Room size needs width, height, and depth")
        if calibration_duration_seconds is not None and calibration_duration_seconds <= 0:
            raise ValueError("T-pose calibration duration must be positive")
        self.calibration_duration_seconds = calibration_duration_seconds
        self._clock = clock or time.monotonic
        self.reset_calibration()

    @property
    def calibrated(self) -> bool:
        if len(self.source_ids) == 1:
            return True
        geometry_ready = self.manual_camera_setup or all(source in self._extrinsics for source in self.source_ids)
        return geometry_ready and (self.calibration_duration_seconds is None or self._session_calibrated)

    @property
    def calibration_progress(self) -> tuple[float, float]:
        if len(self.source_ids) == 1:
            return self.calibration_frames, self.calibration_frames
        if self.calibration_duration_seconds is not None:
            elapsed = 0.0 if self._calibration_started_at is None else max(0.0, self._clock() - self._calibration_started_at)
            return min(elapsed, self.calibration_duration_seconds), self.calibration_duration_seconds
        if self.manual_camera_setup:
            return self.calibration_frames, self.calibration_frames
        count = min((len(self._pose_samples.get(source, ())) for source in self.source_ids), default=0)
        return min(count, self.calibration_frames), self.calibration_frames

    def reset_calibration(self) -> None:
        self._pose_samples: dict[str, list[tuple[object, object, float]]] = {}
        self._alignment_samples: dict[str, list[tuple[object, float, object]]] = {}
        self._extrinsics: dict[str, tuple[object, object, float]] = {}
        self._calibration_stats: dict[str, tuple[int, float]] = {}
        self._alignments: dict[str, tuple[object, float, object]] = {}
        self._last_points: dict[int, tuple[float, float, float]] = {}
        self._last_body_transform = None
        self._room_floor = 0.0 if self.manual_camera_setup else None
        self._session_calibrated = False
        self._calibration_started_at = None
        self._last_valid_calibration_at = None
        self._calibration_hint = "Stand in a T-pose where every camera sees your full body"

    def update(self, observations: Sequence[CameraObservation], cv2_module) -> FusionResult:
        usable = [
            item for item in observations
            if item.source_id in self.source_ids and item.pose.detected
            and len(item.pose.image_landmarks) >= 31 and len(item.pose.world_landmarks) >= 31
        ]
        usable.sort(key=lambda item: self.source_ids.index(item.source_id))
        if not usable:
            empty = PoseResult(False, 0.0, [], [])
            return FusionResult(empty, self._camera_poses(), self.calibrated, self.calibration_progress, 0, self._calibration_hint)

        if len(self.source_ids) == 1:
            if self.manual_camera_setup:
                camera_poses = self._camera_poses()
            else:
                solved = self._solve_camera(usable[0].pose.world_landmarks, usable[0], cv2_module)
                camera_poses = self._camera_poses({usable[0].source_id: solved} if solved else None)
            return FusionResult(usable[0].pose, camera_poses, True, self.calibration_progress, 1, "")

        if not self.calibrated:
            self._observe_calibration(usable, cv2_module)
        camera_poses = self._camera_poses()
        if not self.calibrated:
            # Preserve the proven single-view path while neutral-pose camera
            # calibration gathers enough stable samples.
            return FusionResult(usable[0].pose, camera_poses, False, self.calibration_progress, 1, self._calibration_hint)

        fused = self._triangulate(usable, cv2_module)
        return FusionResult(fused, camera_poses, True, self.calibration_progress, len(usable), "")

    def _camera_matrix(self, size, np, source_id: str | None = None):
        width, height = size
        field_of_view = self.horizontal_fov_degrees
        if source_id in self.camera_setups:
            field_of_view = float(self.camera_setups[source_id].horizontal_fov)
        focal = width / (2.0 * math.tan(math.radians(field_of_view) / 2.0))
        return np.array(((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0), (0.0, 0.0, 1.0)), dtype=np.float64)

    @staticmethod
    def _valid_correspondences(reference_world, observation, np):
        object_points, image_points = [], []
        width, height = observation.frame_size
        for world, image in zip(reference_world, observation.pose.image_landmarks):
            try:
                xyz = tuple(float(value) for value in world[:3])
                x, y, visibility = float(image[0]), float(image[1]), float(image[3])
            except (IndexError, TypeError, ValueError):
                continue
            if visibility < 0.55 or not all(math.isfinite(value) for value in (*xyz, x, y)):
                continue
            if not (-0.1 <= x <= 1.1 and -0.1 <= y <= 1.1):
                continue
            object_points.append(xyz)
            image_points.append((x * (width - 1), y * (height - 1)))
        return np.asarray(object_points, dtype=np.float64), np.asarray(image_points, dtype=np.float64)

    def _solve_camera(self, reference_world, observation, cv2_module):
        import numpy as np

        object_points, image_points = self._valid_correspondences(reference_world, observation, np)
        if len(object_points) < 8:
            return None
        matrix = self._camera_matrix(observation.frame_size, np, observation.source_id)
        distortion = np.zeros((4, 1), dtype=np.float64)
        try:
            ok, rvec, tvec, inliers = cv2_module.solvePnPRansac(
                object_points,
                image_points,
                matrix,
                distortion,
                iterationsCount=100,
                reprojectionError=12.0,
                confidence=0.99,
                flags=getattr(cv2_module, "SOLVEPNP_EPNP", 1),
            )
            if not ok or inliers is None or len(inliers) < 6:
                return None
            selected = inliers.reshape(-1)
            if hasattr(cv2_module, "solvePnPRefineLM"):
                rvec, tvec = cv2_module.solvePnPRefineLM(
                    object_points[selected], image_points[selected], matrix, distortion, rvec, tvec,
                )
            projected, _ = cv2_module.projectPoints(object_points[selected], rvec, tvec, matrix, distortion)
            error = float(np.linalg.norm(projected.reshape(-1, 2) - image_points[selected], axis=1).mean())
        except Exception:
            return None
        if not math.isfinite(error) or error > 18.0:
            return None
        return rvec.reshape(3), tvec.reshape(3), error

    @staticmethod
    def _similarity(source, target):
        import numpy as np

        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.shape != target.shape or source.ndim != 2 or len(source) < 3:
            return None
        source_center, target_center = source.mean(axis=0), target.mean(axis=0)
        left, right = source - source_center, target - target_center
        covariance = left.T @ right
        u, _singular, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        denominator = float((left * left).sum())
        if denominator < 1e-9:
            return None
        scale = float(((left @ rotation.T) * right).sum() / denominator)
        scale = min(max(scale, 0.65), 1.55)
        translation = target_center - scale * (rotation @ source_center)
        return rotation, scale, translation

    @staticmethod
    def _is_t_pose(observation) -> bool:
        import numpy as np

        world = observation.pose.world_landmarks
        required = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
        if any(float(world[index][3]) < 0.55 for index in required):
            return False
        points = np.asarray([landmark[:3] for landmark in world], dtype=np.float64)
        shoulder_width = float(np.linalg.norm(points[12] - points[11]))
        if not 0.15 <= shoulder_width <= 0.8:
            return False
        for shoulder, elbow, wrist in ((11, 13, 15), (12, 14, 16)):
            upper = float(np.linalg.norm(points[elbow] - points[shoulder]))
            lower = float(np.linalg.norm(points[wrist] - points[elbow]))
            reach = float(np.linalg.norm(points[wrist] - points[shoulder]))
            if upper + lower < 0.18 or reach / (upper + lower) < 0.78:
                return False
            if abs(float(points[wrist][1] - points[shoulder][1])) > max(0.22, shoulder_width * 0.65):
                return False
        return True

    def _clear_calibration_samples(self) -> None:
        self._pose_samples.clear()
        self._alignment_samples.clear()
        if not self.manual_camera_setup:
            self._extrinsics.clear()
            self._calibration_stats.clear()
        self._calibration_started_at = None

    def _finalize_alignments(self, np) -> None:
        for source in self.source_ids:
            alignments = self._alignment_samples.get(source, [])
            if not alignments:
                continue
            rotation = np.median(np.stack([item[0] for item in alignments]), axis=0)
            u, _singular, vt = np.linalg.svd(rotation)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0:
                u[:, -1] *= -1
                rotation = u @ vt
            scale = float(np.median([item[1] for item in alignments]))
            translation = np.median(np.stack([item[2] for item in alignments]), axis=0)
            self._alignments[source] = rotation, scale, translation

    def _observe_calibration(self, observations, cv2_module) -> None:
        import numpy as np

        now = self._clock()
        if self.calibration_duration_seconds is not None:
            present = {observation.source_id for observation in observations}
            if present != set(self.source_ids):
                missing = len(self.source_ids) - len(present)
                self._calibration_hint = f"Waiting for {missing} camera(s) to see a pose"
                if self._last_valid_calibration_at is None or now - self._last_valid_calibration_at > 0.75:
                    self._clear_calibration_samples()
                return
            if not all(self._is_t_pose(observation) for observation in observations):
                self._calibration_hint = "Raise both arms into a T-pose and keep every limb visible"
                if self._last_valid_calibration_at is None or now - self._last_valid_calibration_at > 0.75:
                    self._clear_calibration_samples()
                return
            if self._calibration_started_at is None:
                self._calibration_started_at = now
            self._last_valid_calibration_at = now
            remaining = max(0.0, self.calibration_duration_seconds - (now - self._calibration_started_at))
            self._calibration_hint = f"Hold the T-pose for {remaining:.1f} more seconds"

        reference = observations[0]
        reference_world = reference.pose.world_landmarks
        solved_this_frame: dict[str, tuple[object, object, float] | None] = {}
        for observation in observations:
            solved = None if self.manual_camera_setup else self._solve_camera(reference_world, observation, cv2_module)
            if not self.manual_camera_setup and solved is None:
                continue
            solved_this_frame[observation.source_id] = solved
            source_points, target_points = [], []
            for source, target in zip(observation.pose.world_landmarks, reference_world):
                if min(float(source[3]), float(target[3])) >= 0.55:
                    source_points.append(source[:3]); target_points.append(target[:3])
            alignment = self._similarity(source_points, target_points)
            if alignment is not None:
                self._alignment_samples.setdefault(observation.source_id, []).append(alignment)
        if any(source not in solved_this_frame for source in self.source_ids):
            self._calibration_hint = "Could not localize every camera; check framing and FOV"
            return
        sample_limit = max(
            self.calibration_frames,
            round((self.calibration_duration_seconds or 0.0) * 60.0),
        )
        if not self.manual_camera_setup:
            for source, sample in solved_this_frame.items():
                samples = self._pose_samples.setdefault(source, [])
                samples.append(sample)
                del samples[:-sample_limit]
        enough_samples = all(
            len(self._alignment_samples.get(source, ())) >= self.calibration_frames
            and (self.manual_camera_setup or len(self._pose_samples.get(source, ())) >= self.calibration_frames)
            for source in self.source_ids
        )
        duration_complete = (
            self.calibration_duration_seconds is None
            or self._calibration_started_at is not None
            and now - self._calibration_started_at >= self.calibration_duration_seconds
        )
        if not enough_samples or not duration_complete:
            return
        if not self.manual_camera_setup:
            for source in self.source_ids:
                samples = self._pose_samples[source]
                errors = np.asarray([item[2] for item in samples], dtype=np.float64)
                median_error = float(np.median(errors))
                mad = float(np.median(np.abs(errors - median_error)))
                threshold = median_error + 3.0 * max(mad, 0.25)
                robust_samples = [item for item in samples if item[2] <= threshold] or samples
                weights = np.asarray([1.0 / max(item[2], 0.5) for item in robust_samples], dtype=np.float64)
                weights /= weights.sum()
                rotations = [cv2_module.Rodrigues(np.asarray(item[0], dtype=np.float64))[0] for item in robust_samples]
                mean_rotation = sum(weight * rotation for weight, rotation in zip(weights, rotations))
                u, _singular, vt = np.linalg.svd(mean_rotation)
                mean_rotation = u @ vt
                if np.linalg.det(mean_rotation) < 0:
                    u[:, -1] *= -1
                    mean_rotation = u @ vt
                rvec, _ = cv2_module.Rodrigues(mean_rotation)
                tvec = np.average(np.stack([item[1] for item in robust_samples]), axis=0, weights=weights)
                error = float(np.average([item[2] for item in robust_samples], weights=weights))
                camera_positions = np.stack([
                    -(rotation.T @ np.asarray(item[1], dtype=np.float64).reshape(3))
                    for rotation, item in zip(rotations, robust_samples)
                ])
                mean_position = -(mean_rotation.T @ np.asarray(tvec, dtype=np.float64).reshape(3))
                position_std = float(np.sqrt(np.mean(np.sum((camera_positions - mean_position) ** 2, axis=1))))
                self._extrinsics[source] = (rvec, tvec, error)
                self._calibration_stats[source] = (len(robust_samples), position_std)
        self._finalize_alignments(np)
        self._session_calibrated = True
        self._calibration_hint = "T-pose calibration complete"

    @staticmethod
    def _manual_camera_geometry(setup, np):
        """Return world-to-camera R/t and camera-to-world axes for a manual pose."""
        yaw, pitch, roll = (math.radians(float(value)) for value in setup.rotation)
        forward = np.array((math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)))
        forward /= np.linalg.norm(forward)
        right = np.cross(np.array((0.0, 1.0, 0.0)), forward)
        if np.linalg.norm(right) < 1e-8:
            right = np.array((1.0, 0.0, 0.0))
        right /= np.linalg.norm(right)
        up = np.cross(forward, right); up /= np.linalg.norm(up)
        rolled_right = math.cos(roll) * right + math.sin(roll) * up
        rolled_up = -math.sin(roll) * right + math.cos(roll) * up
        # OpenCV image coordinates use +Y downward.
        camera_to_world = np.column_stack((rolled_right, -rolled_up, forward))
        world_to_camera = camera_to_world.T
        position = np.asarray(setup.position, dtype=np.float64)
        translation = -world_to_camera @ position
        return world_to_camera, translation, camera_to_world, position

    def _camera_poses(self, temporary=None) -> tuple[CameraPose, ...]:
        import numpy as np
        try:
            import cv2
        except ImportError:
            return ()

        if self.manual_camera_setup and temporary is None:
            poses = []
            for source in self.source_ids:
                setup = self.camera_setups[source]
                _world_to_camera, _translation, camera_to_world, position = self._manual_camera_geometry(setup, np)
                poses.append(CameraPose(
                    source,
                    tuple(float(value) for value in position),
                    tuple(tuple(float(value) for value in row) for row in camera_to_world),
                    0.0,
                    True,
                    1,
                    0.0,
                ))
            return tuple(poses)

        extrinsics = self._extrinsics if temporary is None else {
            source: value for source, value in temporary.items() if value is not None
        }
        poses = []
        for source in self.source_ids:
            solved = extrinsics.get(source)
            if solved is None:
                continue
            rvec, tvec, error = solved
            rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
            camera_to_world = rotation.T
            position = -camera_to_world @ np.asarray(tvec, dtype=np.float64).reshape(3)
            poses.append(CameraPose(
                source,
                tuple(float(value) for value in position),
                tuple(tuple(float(value) for value in row) for row in camera_to_world),
                float(error),
                source in self._extrinsics or len(self.source_ids) == 1,
                self._calibration_stats.get(source, (1, 0.0))[0],
                self._calibration_stats.get(source, (1, 0.0))[1],
            ))
        return tuple(poses)

    def _triangulate(self, observations, cv2_module) -> PoseResult:
        import numpy as np

        by_source = {item.source_id: item for item in observations}
        projections, camera_positions = {}, {}
        for source, observation in by_source.items():
            if self.manual_camera_setup:
                rotation, tvec, _camera_to_world, position = self._manual_camera_geometry(self.camera_setups[source], np)
            else:
                if source not in self._extrinsics:
                    continue
                rvec, tvec, _error = self._extrinsics[source]
                rotation, _ = cv2_module.Rodrigues(np.asarray(rvec, dtype=np.float64))
                position = -rotation.T @ np.asarray(tvec).reshape(3)
            matrix = self._camera_matrix(observation.frame_size, np, source)
            projections[source] = matrix @ np.column_stack((rotation, np.asarray(tvec).reshape(3)))
            camera_positions[source] = np.asarray(position, dtype=np.float64)

        primary = observations[0].pose
        raw_candidates: list[tuple[object, float] | None] = []
        for index in range(31):
            rows, views = [], []
            for source, projection in projections.items():
                image = by_source[source].pose.image_landmarks[index]
                try:
                    x, y, visibility = float(image[0]), float(image[1]), float(image[3])
                except (IndexError, TypeError, ValueError):
                    continue
                if visibility < 0.35 or not all(math.isfinite(value) for value in (x, y)):
                    continue
                width, height = by_source[source].frame_size
                u, v = x * (width - 1), y * (height - 1)
                weight = math.sqrt(visibility)
                rows.extend((weight * (u * projection[2] - projection[0]), weight * (v * projection[2] - projection[1])))
                views.append((source, u, v, visibility))

            point = None
            confidence = 0.0
            if len(views) >= 2:
                try:
                    _u, _s, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
                    homogeneous = vt[-1]
                    if abs(homogeneous[3]) > 1e-8:
                        candidate = homogeneous[:3] / homogeneous[3]
                        errors, positive_depth = [], True
                        for source, u, v, _visibility in views:
                            projected = projections[source] @ np.append(candidate, 1.0)
                            positive_depth &= projected[2] > 0
                            if abs(projected[2]) > 1e-8:
                                errors.append(math.hypot(projected[0] / projected[2] - u, projected[1] / projected[2] - v))
                        error = sum(errors) / len(errors) if errors else float("inf")
                        directions = [candidate - camera_positions[item[0]] for item in views]
                        geometry = 0.15
                        for first in range(len(directions)):
                            for second in range(first + 1, len(directions)):
                                a, b = directions[first], directions[second]
                                denominator = np.linalg.norm(a) * np.linalg.norm(b)
                                if denominator > 1e-8:
                                    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
                                    geometry = max(geometry, min(1.0, math.sin(math.acos(cosine)) / math.sin(math.radians(20))))
                        if positive_depth and math.isfinite(error) and error <= 40.0:
                            point = candidate
                            confidence = (sum(item[3] for item in views) / len(views)) * geometry * max(0.25, 1.0 - error / 50.0)
                except (ValueError, np.linalg.LinAlgError):
                    pass
            raw_candidates.append((point, min(max(float(confidence), 0.0), 1.0)) if point is not None else None)

        # A low reprojection error does not by itself guarantee a physically
        # valid intersection when approximate phone intrinsics are used. Fit a
        # rigid body transform from the stable torso, then reject any limb ray
        # that would stretch the MediaPipe skeleton beyond anatomical limits.
        aligned_world = {}
        for observation in observations:
            alignment = self._alignments.get(observation.source_id)
            if alignment is None and observation is observations[0]:
                alignment = (np.eye(3), 1.0, np.zeros(3))
            if alignment is None:
                continue
            alignment_rotation, alignment_scale, alignment_translation = alignment
            aligned_world[observation.source_id] = np.asarray([
                alignment_scale * (alignment_rotation @ np.asarray(landmark[:3], dtype=np.float64)) + alignment_translation
                for landmark in observation.pose.world_landmarks
            ])

        limb_groups = (
            (11, 13, 15, 17, 19, 21),
            (12, 14, 16, 18, 20, 22),
            (23, 25, 27, 29),
            (24, 26, 28, 30),
        )
        preferred_source = {}
        for group in limb_groups:
            candidates = [observation for observation in observations if observation.source_id in aligned_world]
            if not candidates:
                continue
            winner = max(
                candidates,
                key=lambda observation: sum(float(observation.pose.world_landmarks[index][3]) for index in group) / len(group),
            )
            for index in group:
                preferred_source[index] = winner.source_id

        local_points, local_visibilities = [], []
        for index in range(31):
            selected = None
            selected_source = preferred_source.get(index)
            if selected_source is not None:
                selected = next((item for item in observations if item.source_id == selected_source), None)
            if selected is None:
                candidates = [item for item in observations if item.source_id in aligned_world]
                selected = max(candidates, key=lambda item: float(item.pose.world_landmarks[index][3])) if candidates else observations[0]
            coordinates = aligned_world.get(selected.source_id)
            local_points.append(coordinates[index] if coordinates is not None else selected.pose.world_landmarks[index][:3])
            local_visibilities.append(float(selected.pose.world_landmarks[index][3]))
        local_points = np.asarray(local_points, dtype=np.float64)
        torso_fit = (11, 12, 23, 24)
        fit_order = torso_fit
        source_fit, target_fit = [], []
        for index in fit_order:
            candidate = raw_candidates[index]
            if candidate is not None and candidate[1] >= 0.18 and local_visibilities[index] >= 0.45:
                source_fit.append(local_points[index]); target_fit.append(candidate[0])
        if len(source_fit) < 3:
            for index in (25, 26, 27, 28):
                candidate = raw_candidates[index]
                if candidate is not None and candidate[1] >= 0.18 and local_visibilities[index] >= 0.45:
                    source_fit.append(local_points[index]); target_fit.append(candidate[0])
        transform = self._similarity(source_fit, target_fit) if len(source_fit) >= 3 else None
        room_width, room_height, room_depth = self.room_size_m
        if transform is not None:
            rotation, scale, translation = transform
            expected = np.asarray([scale * (rotation @ point) + translation for point in local_points])
            center = np.median(expected[[11, 12, 23, 24]], axis=0)
            plausible_center = abs(center[0]) <= room_width / 2 + 0.75 and abs(center[2]) <= room_depth / 2 + 0.75
            if not plausible_center or not np.isfinite(expected).all():
                transform = None
        if transform is None:
            transform = self._last_body_transform
        if transform is None:
            rotation, scale = np.eye(3), 1.0
            translation = np.array((0.0, -float(local_points[:, 1].min()) if self.manual_camera_setup else 0.0, 0.0))
            transform = rotation, scale, translation
        self._last_body_transform = transform
        rotation, scale, translation = transform
        expected_points = np.asarray([scale * (rotation @ point) + translation for point in local_points])
        if self._room_floor is None:
            self._room_floor = float(expected_points[:, 1].min())

        fused_world, fused_visibilities = [], []
        torso = {11, 12, 23, 24}
        for index, expected in enumerate(expected_points):
            candidate = raw_candidates[index]
            tolerance = 0.24 if index in torso else (0.34 if index in CORE_LANDMARKS else 0.46)
            accepted = False
            if candidate is not None:
                point, confidence = candidate
                inside_room = (
                    abs(point[0]) <= room_width / 2 + 0.5
                    and self._room_floor - 0.35 <= point[1] <= self._room_floor + room_height + 0.5
                    and abs(point[2]) <= room_depth / 2 + 0.5
                )
                accepted = inside_room and float(np.linalg.norm(point - expected)) <= tolerance
            if accepted:
                output = np.asarray(point, dtype=np.float64)
                output_confidence = confidence
            else:
                output = expected
                visibility = local_visibilities[index]
                # This is an anatomically constrained model fallback, not an
                # arbitrary held ray. Preserve enough of a clearly visible
                # primary landmark's confidence for VRChat calibration to
                # complete even when one secondary ray is rejected.
                output_confidence = min(0.74, max(0.36, visibility * 0.78))
            output_tuple = tuple(float(value) for value in output)
            self._last_points[index] = output_tuple
            fused_world.append([*output_tuple, output_confidence])
            fused_visibilities.append(output_confidence)

        core = [fused_visibilities[index] for index in CORE_LANDMARKS]
        overall = sum(core) / len(core)
        return PoseResult(overall >= 0.25, overall, primary.image_landmarks, fused_world)


def _landmark_values(landmark) -> list[float]:
    return [
        float(landmark.x),
        float(landmark.y),
        float(landmark.z),
        min(float(landmark.visibility), float(getattr(landmark, "presence", 1.0))),
    ]


def convert_mediapipe_landmarks(
    image_landmarks: Sequence,
    world_landmarks: Sequence,
) -> PoseResult:
    """Convert 33 MediaPipe landmarks to VR-FBT's 31-point, Y-up coordinates."""
    if len(image_landmarks) < 33 or len(world_landmarks) < 33:
        return PoseResult(False, 0.0, [], [])
    image = [_landmark_values(image_landmarks[index]) for index in MEDIAPIPE_TO_VRFBT]
    world_raw = [_landmark_values(world_landmarks[index]) for index in MEDIAPIPE_TO_VRFBT]
    # Convert a front-facing camera to VRChat Unity space: +X user-right,
    # +Y up, +Z user-forward, with one unit per meter.
    world = [[-x, -y, -z, visibility] for x, y, z, visibility in world_raw]
    confidence = sum(image[index][3] for index in CORE_LANDMARKS) / len(CORE_LANDMARKS)
    return PoseResult(True, min(max(confidence, 0.0), 1.0), image, world)


class Pose:
    """One-person MediaPipe detector + temporal landmark tracker."""

    def __init__(self, quality: str = "full"):
        normalized = quality.strip().casefold()
        if normalized not in MODEL_PATHS:
            raise ValueError(f"Unknown pose model {quality!r}; choose lite, full, or heavy")
        model_path = MODEL_PATHS[normalized]
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe pose model is missing: {model_path}")

        # Suppress native informational logging without hiding errors.
        os.environ.setdefault("GLOG_minloglevel", "2")
        import mediapipe as mp

        self.quality = normalized
        self._mp = mp
        self._last_timestamp_ms = -1
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.45,
            min_pose_presence_confidence=0.45,
            min_tracking_confidence=0.45,
            output_segmentation_masks=False,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def process(self, bgr_frame, timestamp_ms: int | None = None) -> PoseResult:
        """Detect/track a frame; video mode reuses its ROI between frames."""
        if timestamp_ms is None:
            timestamp_ms = time.monotonic_ns() // 1_000_000
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        rgb = bgr_frame[:, :, ::-1].copy()
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks or not result.pose_world_landmarks:
            return PoseResult(False, 0.0, [], [])
        return convert_mediapipe_landmarks(result.pose_landmarks[0], result.pose_world_landmarks[0])

    def close(self) -> None:
        landmarker = getattr(self, "_landmarker", None)
        if landmarker is not None:
            landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()
