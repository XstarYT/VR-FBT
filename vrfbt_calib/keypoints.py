"""Thin adapter between MediaPipe BlazePose indices and calibration joints."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


MEDIAPIPE_POSE_INDICES: dict[str, int] = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
}
JOINT_NAMES: tuple[str, ...] = tuple(MEDIAPIPE_POSE_INDICES)


@dataclass(frozen=True)
class LandmarkValue:
    """Minimal MediaPipe-compatible landmark value in pixels and confidence."""

    x: float
    y: float
    confidence: float


def adapt_mediapipe_landmarks(
    landmarks: Sequence[object],
    image_width: int,
    image_height: int,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Convert normalized BlazePose objects to pixel dictionaries.

    Objects only need ``x``, ``y`` and either ``visibility`` or ``presence``
    attributes, so importing MediaPipe itself is never required. Missing or
    malformed entries are omitted rather than failing the entire frame.
    """

    points: dict[str, tuple[float, float]] = {}
    confidences: dict[str, float] = {}
    for name, index in MEDIAPIPE_POSE_INDICES.items():
        if index >= len(landmarks):
            continue
        landmark = landmarks[index]
        try:
            x = float(getattr(landmark, "x")) * image_width
            y = float(getattr(landmark, "y")) * image_height
            confidence = float(
                getattr(landmark, "visibility", getattr(landmark, "presence", 1.0))
            )
        except (AttributeError, TypeError, ValueError):
            continue
        points[name] = (x, y)
        confidences[name] = confidence
    return points, confidences


def filter_named_keypoints(
    keypoints: Mapping[str, tuple[float, float]],
    confidences: Mapping[str, float],
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Return only the five supported joints, preserving pixel units."""

    points = {name: keypoints[name] for name in JOINT_NAMES if name in keypoints}
    scores = {name: float(confidences.get(name, 0.0)) for name in points}
    return points, scores
