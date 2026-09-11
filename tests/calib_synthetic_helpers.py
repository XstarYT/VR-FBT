"""Deterministic synthetic rigs shared by calibration tests."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from vrfbt_calib import TposeMultiCamCalibrator, intrinsics_from_fov
from vrfbt_calib.types import CalibrationResult


FloatArray = NDArray[np.float64]


@dataclass
class SyntheticRun:
    result: CalibrationResult
    expected_R: dict[str, FloatArray]
    expected_t: dict[str, FloatArray]
    camera_points: dict[str, tuple[FloatArray, FloatArray]]
    calibrator: TposeMultiCamCalibrator


SKELETON = {
    "nose": np.array([0.0, 1.72, 0.035]),
    "left_shoulder": np.array([-0.2, 1.43, 0.0]),
    "right_shoulder": np.array([0.2, 1.43, 0.0]),
    "left_hip": np.array([-0.15, 0.96, 0.025]),
    "right_hip": np.array([0.15, 0.96, 0.025]),
}


def rotation_error_deg(actual: FloatArray, expected: FloatArray) -> float:
    delta = actual @ expected.T
    value = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def _look_at(camera: FloatArray, target: FloatArray, rng: np.random.Generator) -> FloatArray:
    forward = target - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    ideal = np.vstack((right, down, forward))
    jitter_deg = rng.uniform(-1.2, 1.2, size=3)
    jitter, _ = cv2.Rodrigues(np.radians(jitter_deg))
    return jitter @ ideal


def run_synthetic(
    seed: int,
    camera_count: int,
    noise_px: float = 1.2,
    dropout: float = 0.10,
    frame_count: int = 300,
) -> SyntheticRun:
    rng = np.random.default_rng(seed)
    ids = ["A", "B", "C"][:camera_count]
    if camera_count == 2:
        angles = [0.0, float(rng.uniform(60.0, 150.0))]
    else:
        # Three gaps (including C->A around 360 degrees) all land in the
        # requested 60–150 degree adjacent-camera range.
        first_gap, second_gap = rng.uniform(105.0, 125.0, size=2)
        angles = [0.0, float(first_gap), float(first_gap + second_gap)]
    target = np.array([0.0, 1.25, 0.0])
    centres: dict[str, FloatArray] = {}
    world_to_camera: dict[str, FloatArray] = {}
    world_translations: dict[str, FloatArray] = {}
    for camera_id, angle in zip(ids, angles, strict=True):
        radius = float(rng.uniform(1.75, 2.35))
        radians = np.radians(angle)
        centre = target + np.array([radius * np.sin(radians), rng.uniform(-0.12, 0.12), -radius * np.cos(radians)])
        rotation = _look_at(centre, target, rng)
        centres[camera_id] = centre
        world_to_camera[camera_id] = rotation
        world_translations[camera_id] = -rotation @ centre

    width, height, fov = 1920, 1080, 72.0
    intrinsics = intrinsics_from_fov(width, height, fov)
    calibrator = TposeMultiCamCalibrator(
        {camera_id: None for camera_id in ids},
        fov_priors_deg={camera_id: fov for camera_id in ids},
        image_sizes={camera_id: (width, height) for camera_id in ids},
        reference_camera="A",
    )
    saved_points: dict[str, tuple[list[tuple[float, float]], list[tuple[float, float]]]] = {
        camera_id: ([], []) for camera_id in ids
    }
    for frame_index in range(frame_count):
        phase = 2.0 * np.pi * frame_index / 47.0
        body_sway = np.array([0.025 * np.sin(phase), 0.008 * np.cos(phase * 0.7), 0.018 * np.cos(phase)])
        frame_keypoints: dict[str, dict[str, tuple[float, float]]] = {}
        frame_confidences: dict[str, dict[str, float]] = {}
        world_joints = {
            name: point + body_sway + rng.normal(0.0, 0.0015, size=3)
            for name, point in SKELETON.items()
        }
        for camera_id in ids:
            points: dict[str, tuple[float, float]] = {}
            scores: dict[str, float] = {}
            rotation = world_to_camera[camera_id]
            translation = world_translations[camera_id]
            for joint, world_point in world_joints.items():
                camera_point = rotation @ world_point + translation
                pixel = np.array(
                    [
                        intrinsics.fx * camera_point[0] / camera_point[2] + intrinsics.cx,
                        intrinsics.fy * camera_point[1] / camera_point[2] + intrinsics.cy,
                    ]
                )
                pixel += rng.normal(0.0, noise_px, size=2)
                points[joint] = (float(pixel[0]), float(pixel[1]))
                scores[joint] = 0.25 if rng.random() < dropout else float(rng.uniform(0.82, 0.99))
            frame_keypoints[camera_id] = points
            frame_confidences[camera_id] = scores
        calibrator.collect_frame(frame_keypoints, frame_confidences, frame_index / 30.0)

    result = calibrator.calibrate()
    reference_rotation = world_to_camera["A"]
    reference_translation = world_translations["A"]
    expected_R: dict[str, FloatArray] = {}
    expected_t: dict[str, FloatArray] = {}
    for camera_id in ids:
        relative_rotation = world_to_camera[camera_id] @ reference_rotation.T
        expected_R[camera_id] = relative_rotation
        expected_t[camera_id] = world_translations[camera_id] - relative_rotation @ reference_translation
    return SyntheticRun(
        result=result,
        expected_R=expected_R,
        expected_t=expected_t,
        camera_points={camera_id: (np.empty((0, 2)), np.empty((0, 2))) for camera_id in ids},
        calibrator=calibrator,
    )
