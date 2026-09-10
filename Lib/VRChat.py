"""Convert a smoothed MediaPipe skeleton into VRChat OSC tracker poses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
import time
from typing import Mapping


Vector = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TrackerPose:
    position: Vector
    rotation: Vector
    confidence: float


@dataclass(frozen=True, slots=True)
class VRChatFrame:
    trackers: dict[str, TrackerPose]
    head_position: Vector
    head_rotation: Vector
    scale: float
    head_confidence: float = 0.0


def _add(a: Vector, b: Vector) -> Vector:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _sub(a: Vector, b: Vector) -> Vector:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _mul(a: Vector, value: float) -> Vector:
    return a[0] * value, a[1] * value, a[2] * value


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(a: Vector) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: Vector) -> Vector | None:
    length = _length(a)
    if not math.isfinite(length) or length < 1e-6:
        return None
    return _mul(a, 1.0 / length)


def _midpoint(a: Vector, b: Vector) -> Vector:
    return _mul(_add(a, b), 0.5)


def _lerp(a: Vector, b: Vector, amount: float) -> Vector:
    return _add(a, _mul(_sub(b, a), amount))


def _distance(a: Vector, b: Vector) -> float:
    return _length(_sub(a, b))


def _rotate_y(vector: Vector, degrees: float) -> Vector:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        cosine * vector[0] + sine * vector[2],
        vector[1],
        -sine * vector[0] + cosine * vector[2],
    )


def _wrap_angle(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


class _OneEuroVector:
    """Adaptive low-pass filter: stable while still, responsive in motion."""

    def __init__(self, min_cutoff: float, beta: float, derivative_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.timestamp: float | None = None
        self.value: Vector | None = None
        self.derivative: Vector = (0.0, 0.0, 0.0)

    @staticmethod
    def _alpha(delta_time: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / delta_time)

    def update(self, value: Vector, timestamp: float, angular: bool = False) -> Vector:
        if self.value is None or self.timestamp is None:
            self.value, self.timestamp = value, timestamp
            return value
        delta_time = min(max(timestamp - self.timestamp, 1.0 / 240.0), 0.2)
        sample = value
        if angular:
            sample = tuple(previous + _wrap_angle(current - previous) for current, previous in zip(value, self.value))
        raw_derivative = tuple((current - previous) / delta_time for current, previous in zip(sample, self.value))
        derivative_alpha = self._alpha(delta_time, self.derivative_cutoff)
        self.derivative = tuple(
            derivative_alpha * current + (1.0 - derivative_alpha) * previous
            for current, previous in zip(raw_derivative, self.derivative)
        )
        speed = _length(self.derivative)
        alpha = self._alpha(delta_time, self.min_cutoff + self.beta * speed)
        filtered = tuple(alpha * current + (1.0 - alpha) * previous for current, previous in zip(sample, self.value))
        self.value, self.timestamp = filtered, timestamp
        return tuple(_wrap_angle(component) for component in filtered) if angular else filtered


def rotation_from_up_forward(up_hint: Vector, forward_hint: Vector) -> Vector | None:
    """Return Unity Euler XYZ degrees (internally applied Z, X, Y)."""
    up = _normalize(up_hint)
    if up is None:
        return None
    projected_forward = _sub(forward_hint, _mul(up, _dot(forward_hint, up)))
    forward = _normalize(projected_forward)
    if forward is None:
        return None
    right = _normalize(_cross(up, forward))
    if right is None:
        return None
    forward = _normalize(_cross(right, up))
    if forward is None:
        return None
    matrix = (
        (right[0], up[0], forward[0]),
        (right[1], up[1], forward[1]),
        (right[2], up[2], forward[2]),
    )
    x = math.asin(min(max(-matrix[1][2], -1.0), 1.0))
    if abs(math.cos(x)) > 1e-6:
        z = math.atan2(matrix[1][0], matrix[1][1])
        y = math.atan2(matrix[0][2], matrix[2][2])
    else:
        z = 0.0
        y = math.atan2(-matrix[2][0], matrix[0][0])
    return tuple(math.degrees(value) for value in (x, y, z))


class VRChatPoseSolver:
    """Build stable virtual tracker transforms and a session body scale."""

    def __init__(self, user_height_m: float = 1.70, calibration_frames: int = 1):
        if not 1.0 <= user_height_m <= 2.5:
            raise ValueError("VRChat user height must be between 1.0 and 2.5 meters")
        if calibration_frames < 1:
            raise ValueError("Calibration must use at least one frame")
        self.user_height_m = float(user_height_m)
        self.calibration_frames = int(calibration_frames)
        self.scale: float | None = None
        self.neutral_yaw = 0.0
        self._calibration_scales: list[float] = []
        self._calibration_yaws: list[float] = []
        self._calibrated = False
        self._last_rotations: dict[str, Vector] = {}
        self._position_filters: dict[str, _OneEuroVector] = {}
        self._rotation_filters: dict[str, _OneEuroVector] = {}
        self._tracker_active: dict[str, bool] = {}
        self._last_good_trackers: dict[str, TrackerPose] = {}
        self._last_good_at: dict[str, float] = {}

    @property
    def calibrating(self) -> bool:
        return not self._calibrated

    @property
    def calibration_progress(self) -> tuple[int, int]:
        return len(self._calibration_scales), self.calibration_frames

    def begin_calibration(self) -> None:
        self.scale = None
        self.neutral_yaw = 0.0
        self._calibration_scales.clear()
        self._calibration_yaws.clear()
        self._calibrated = False
        self._last_rotations.clear()
        self._position_filters.clear()
        self._rotation_filters.clear()
        self._tracker_active.clear()
        self._last_good_trackers.clear()
        self._last_good_at.clear()

    @staticmethod
    def _point(points: Mapping[str, object], name: str) -> tuple[Vector, float]:
        point = points[name]
        return tuple(float(value) for value in point.pos), float(point.vis)

    def _calibrate_scale(self, points: Mapping[str, object]) -> float:
        left_hip, _ = self._point(points, "L-Hip")
        right_hip, _ = self._point(points, "R-Hip")
        left_knee, _ = self._point(points, "L-Knee")
        right_knee, _ = self._point(points, "R-Knee")
        left_ankle, _ = self._point(points, "L-Ankle")
        right_ankle, _ = self._point(points, "R-Ankle")
        left_shoulder, _ = self._point(points, "L-Shoulder")
        right_shoulder, _ = self._point(points, "R-Shoulder")
        left_ear, _ = self._point(points, "L-Ear")
        right_ear, _ = self._point(points, "R-Ear")
        hip = _midpoint(left_hip, right_hip)
        shoulder = _midpoint(left_shoulder, right_shoulder)
        head = _midpoint(left_ear, right_ear)
        legs = (
            _distance(left_hip, left_knee) + _distance(left_knee, left_ankle)
            + _distance(right_hip, right_knee) + _distance(right_knee, right_ankle)
        ) * 0.5
        estimated_height = legs + _distance(hip, shoulder) + _distance(shoulder, head) + 0.13
        if not math.isfinite(estimated_height) or estimated_height < 0.5:
            return 1.0
        return min(max(self.user_height_m / estimated_height, 0.65), 1.55)

    def _observe_calibration(self, points: Mapping[str, object]) -> bool:
        required = (
            "L-Ear", "R-Ear", "L-Shoulder", "R-Shoulder", "L-Hip", "R-Hip",
            "L-Knee", "R-Knee", "L-Ankle", "R-Ankle",
        )
        if any(self._point(points, name)[1] < 0.55 for name in required):
            return False
        left_shoulder, _ = self._point(points, "L-Shoulder")
        right_shoulder, _ = self._point(points, "R-Shoulder")
        left_hip, _ = self._point(points, "L-Hip")
        right_hip, _ = self._point(points, "R-Hip")
        torso_up = _sub(_midpoint(left_shoulder, right_shoulder), _midpoint(left_hip, right_hip))
        forward = _normalize(_cross(_sub(right_shoulder, left_shoulder), torso_up))
        if forward is None:
            return False
        scale = self._calibrate_scale(points)
        yaw = math.degrees(math.atan2(forward[0], forward[2]))
        if not math.isfinite(scale) or not math.isfinite(yaw):
            return False
        self._calibration_scales.append(scale)
        self._calibration_yaws.append(yaw)
        del self._calibration_scales[:-self.calibration_frames]
        del self._calibration_yaws[:-self.calibration_frames]
        if len(self._calibration_scales) == self.calibration_frames:
            candidate_scale = median(self._calibration_scales)
            sine = sum(math.sin(math.radians(value)) for value in self._calibration_yaws)
            cosine = sum(math.cos(math.radians(value)) for value in self._calibration_yaws)
            candidate_yaw = math.degrees(math.atan2(sine, cosine))
            scale_spread = max(abs(value - candidate_scale) for value in self._calibration_scales)
            yaw_spread = max(abs(_wrap_angle(value - candidate_yaw)) for value in self._calibration_yaws)
            if scale_spread <= 0.08 and yaw_spread <= 8.0:
                self.scale = candidate_scale
                self.neutral_yaw = candidate_yaw
                self._calibrated = True
        return True

    def _stabilize_tracker(
        self,
        tracker_id: str,
        position: Vector,
        rotation: Vector,
        confidence: float,
        timestamp: float,
        smooth: bool,
    ) -> TrackerPose:
        active = self._tracker_active.get(tracker_id, False)
        threshold = 0.30 if active else 0.55
        if confidence >= threshold:
            self._tracker_active[tracker_id] = True
            if smooth:
                position_filter = self._position_filters.setdefault(tracker_id, _OneEuroVector(2.0, 1.5))
                rotation_filter = self._rotation_filters.setdefault(tracker_id, _OneEuroVector(2.5, 0.02))
                position = position_filter.update(position, timestamp)
                rotation = rotation_filter.update(rotation, timestamp, angular=True)
            tracker = TrackerPose(position, rotation, confidence)
            self._last_good_trackers[tracker_id] = tracker
            self._last_good_at[tracker_id] = timestamp
            return tracker
        last = self._last_good_trackers.get(tracker_id)
        age = timestamp - self._last_good_at.get(tracker_id, float("-inf"))
        if active and last is not None and age <= 0.35:
            return TrackerPose(last.position, last.rotation, 0.50)
        self._tracker_active[tracker_id] = False
        self._position_filters.pop(tracker_id, None)
        self._rotation_filters.pop(tracker_id, None)
        return TrackerPose(position, rotation, confidence)

    def solve(self, points: Mapping[str, object], timestamp: float | None = None, smooth: bool = True) -> VRChatFrame | None:
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        if self.calibrating:
            self._observe_calibration(points)
            if self.calibrating:
                return None
        if self.scale is None:
            return None
        scale = self.scale
        positions = {name: self._point(points, name) for name in points}
        left_shoulder, l_shoulder_vis = positions["L-Shoulder"]
        right_shoulder, r_shoulder_vis = positions["R-Shoulder"]
        left_hip, l_hip_vis = positions["L-Hip"]
        right_hip, r_hip_vis = positions["R-Hip"]
        shoulder_mid = _midpoint(left_shoulder, right_shoulder)
        hip_mid = _midpoint(left_hip, right_hip)
        torso_up = _sub(shoulder_mid, hip_mid)
        torso_right = _sub(right_shoulder, left_shoulder)
        torso_forward = _normalize(_cross(torso_right, torso_up)) or (0.0, 0.0, 1.0)
        torso_rotation = rotation_from_up_forward(torso_up, torso_forward) or (0.0, 0.0, 0.0)
        hip_forward = _cross(_sub(right_hip, left_hip), torso_up)
        hip_rotation = rotation_from_up_forward(torso_up, hip_forward) or torso_rotation
        chest_position = _add(shoulder_mid, _mul(_sub(hip_mid, shoulder_mid), 0.25))

        left_ear, l_ear_vis = positions["L-Ear"]
        right_ear, r_ear_vis = positions["R-Ear"]
        head_position = _midpoint(left_ear, right_ear)
        head_up = _sub(head_position, shoulder_mid)
        head_forward = _cross(_sub(right_ear, left_ear), head_up)
        head_rotation = rotation_from_up_forward(head_up, head_forward) or torso_rotation

        raw: dict[str, tuple[Vector, Vector | None, float]] = {
            "1": (chest_position, torso_rotation, min(l_shoulder_vis, r_shoulder_vis)),
            "2": (hip_mid, hip_rotation, min(l_hip_vis, r_hip_vis)),
        }
        for tracker_id, side in (("3", "L"), ("4", "R")):
            shoulder, shoulder_vis = positions[f"{side}-Shoulder"]
            elbow, elbow_vis = positions[f"{side}-Elbow"]
            upper_arm = _lerp(shoulder, elbow, 0.75)
            raw[tracker_id] = upper_arm, rotation_from_up_forward(_sub(shoulder, elbow), torso_forward), min(shoulder_vis, elbow_vis)
        for tracker_id, side in (("5", "L"), ("6", "R")):
            hip, hip_vis = positions[f"{side}-Hip"]
            knee, knee_vis = positions[f"{side}-Knee"]
            raw[tracker_id] = knee, rotation_from_up_forward(_sub(hip, knee), torso_forward), min(hip_vis, knee_vis)
        for tracker_id, side in (("7", "L"), ("8", "R")):
            knee, knee_vis = positions[f"{side}-Knee"]
            ankle, ankle_vis = positions[f"{side}-Ankle"]
            foot, foot_vis = positions[f"{side}-Foot"]
            raw[tracker_id] = ankle, rotation_from_up_forward(_sub(knee, ankle), _sub(foot, ankle)), min(knee_vis, ankle_vis, foot_vis)

        trackers: dict[str, TrackerPose] = {}
        for tracker_id, (position, rotation, confidence) in raw.items():
            if rotation is None:
                rotation = self._last_rotations.get(tracker_id, (0.0, 0.0, 0.0))
            else:
                self._last_rotations[tracker_id] = rotation
            position = _rotate_y(_mul(position, scale), -self.neutral_yaw)
            rotation = (rotation[0], _wrap_angle(rotation[1] - self.neutral_yaw), rotation[2])
            trackers[tracker_id] = self._stabilize_tracker(
                tracker_id, position, rotation, confidence, timestamp, smooth,
            )
        head_confidence = min(l_ear_vis, r_ear_vis, l_shoulder_vis, r_shoulder_vis)
        head_position = _rotate_y(_mul(head_position, scale), -self.neutral_yaw)
        head_rotation = (head_rotation[0], _wrap_angle(head_rotation[1] - self.neutral_yaw), head_rotation[2])
        if smooth and head_confidence >= 0.35:
            head_position = self._position_filters.setdefault("head", _OneEuroVector(2.0, 1.5)).update(head_position, timestamp)
            head_rotation = self._rotation_filters.setdefault("head", _OneEuroVector(2.5, 0.02)).update(head_rotation, timestamp, angular=True)
        return VRChatFrame(trackers, head_position, head_rotation, scale, head_confidence)
