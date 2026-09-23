"""Placement checks use parallax at the subject, not camera yaw alone."""

from Lib.CalibrationQuality import layout_warning
from Lib.Tracking import CameraObservation, MultiCameraPoseFusion, PoseResult
import numpy as np
import cv2


def test_side_view_has_useful_parallax():
    assert not layout_warning(((0, 1.4, -3), (3, 1.4, 0)), (0, 1, 0))


def test_adjacent_phones_have_weak_parallax():
    assert "move one phone" in layout_warning(((0, 1.4, -3), (0.2, 1.4, -3)), (0, 1, 0))


def test_parallel_forward_cameras_can_still_have_useful_baseline():
    assert not layout_warning(((-1, 1.4, -3), (1, 1.4, -3)), (0, 1, 0))


def test_three_camera_rig_accepts_one_strong_pair():
    assert not layout_warning(((0, 1.4, -3), (0.2, 1.4, -3), (3, 1.4, 0)), (0, 1, 0))


def test_automatic_calibration_with_weak_pair_stays_unready():
    sources = ("phone:front", "phone:neighbor")
    fusion = MultiCameraPoseFusion(sources, calibration_frames=1)
    points = [[(index % 5) * 0.1, (index // 5) * 0.1, (index % 3) * 0.1, 0.95]
              for index in range(31)]
    pose = PoseResult(True, 0.95, [[0.5, 0.5, 0, 0.95] for _ in points], points)
    observations = [CameraObservation(source, pose, (960, 720)) for source in sources]
    positions = {sources[0]: (0, 0, -3), sources[1]: (0.2, 0, -3)}
    fusion._solve_camera = lambda _reference, observation, _cv2: (
        np.zeros(3), -np.asarray(positions[observation.source_id], dtype=float), 1.0,
    )
    fusion._observe_calibration(observations, cv2)
    assert not fusion.calibrated
    assert "move one phone" in fusion._calibration_hint
    assert not fusion._pose_samples


def test_high_reprojection_prompts_lens_import_after_calibration():
    sources = ("phone:front", "phone:side")
    fusion = MultiCameraPoseFusion(sources, calibration_frames=1)
    points = [[(index % 5) * 0.1, (index // 5) * 0.1, (index % 3) * 0.1, 0.95]
              for index in range(31)]
    pose = PoseResult(True, 0.95, [[0.5, 0.5, 0, 0.95] for _ in points], points)
    observations = [CameraObservation(source, pose, (960, 720)) for source in sources]
    positions = {sources[0]: (0, 0, -3), sources[1]: (3, 0, 0)}
    fusion._solve_camera = lambda _reference, observation, _cv2: (
        np.zeros(3), -np.asarray(positions[observation.source_id], dtype=float), 9.0,
    )
    fusion._observe_calibration(observations, cv2)
    assert fusion.calibrated
    assert "Import lens calibration" in fusion._calibration_hint
    assert "CAMERA_CALIBRATION.md" in fusion._calibration_hint


def test_localization_failure_reports_visible_joint_count():
    sources = ("phone:front", "phone:side")
    fusion = MultiCameraPoseFusion(sources, calibration_frames=1)
    points = [[(index % 5) * 0.1, (index // 5) * 0.1, (index % 3) * 0.1, 0.95]
              for index in range(31)]
    pose = PoseResult(True, 0.95, [[0.5, 0.5, 0, 0.95] for _ in points], points)
    observations = [CameraObservation(source, pose, (960, 720)) for source in sources]
    fusion._solve_camera = lambda _reference, observation, _cv2: (
        (np.zeros(3), np.array((0, 0, 3)), 2.0) if observation.source_id == sources[0] else None
    )
    fusion._observe_calibration(observations, cv2)
    assert not fusion.calibrated
    assert "phone:side: 31 visible joints" in fusion._calibration_hint
