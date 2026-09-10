"""Fast, detector-backed 3D pose tracking using MediaPipe Tasks."""

from __future__ import annotations

from dataclasses import dataclass
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
