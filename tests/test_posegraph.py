"""Joint three-camera optimization must beat naive chained rotations."""

import numpy as np
from scipy.spatial.transform import Rotation

from vrfbt_calib.pairwise import reprojection_residuals
from vrfbt_calib.posegraph import refine_global_rotations, solve_global_translations
from vrfbt_calib.types import CameraIntrinsics, PairwiseCalibration


def _project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, k: np.ndarray) -> np.ndarray:
    camera = (rotation @ points.T).T + translation
    pixels = (k @ camera.T).T
    return pixels[:, :2] / pixels[:, 2:3]


def _synthetic_pair(
    a: str,
    b: str,
    global_r: dict[str, np.ndarray],
    global_t: dict[str, np.ndarray],
    points: np.ndarray,
    intrinsics: CameraIntrinsics,
    perturbation_deg: np.ndarray,
) -> PairwiseCalibration:
    measured_r = Rotation.from_rotvec(np.radians(perturbation_deg)).as_matrix() @ (global_r[b] @ global_r[a].T)
    actual_t = global_t[b] - (global_r[b] @ global_r[a].T) @ global_t[a]
    t_unit = actual_t / np.linalg.norm(actual_t)
    points_a = _project(points, global_r[a], global_t[a], intrinsics.matrix)
    points_b = _project(points, global_r[b], global_t[b], intrinsics.matrix)
    return PairwiseCalibration(
        a,
        b,
        "ok",
        len(points),
        R=measured_r,
        t_unit=t_unit,
        cheirality_fraction=0.9,
        mean_reprojection_error_px=2.0,
        points_a_px=points_a,
        points_b_px=points_b,
        intrinsics_a=intrinsics,
        intrinsics_b=intrinsics,
    )


def test_joint_refinement_lowers_total_reprojection_error() -> None:
    rng = np.random.default_rng(991)
    intrinsics = CameraIntrinsics(1280, 720, 900.0, 900.0, 640.0, 360.0)
    global_r = {
        "A": np.eye(3),
        "B": Rotation.from_euler("y", 72.0, degrees=True).as_matrix(),
        "C": Rotation.from_euler("y", 143.0, degrees=True).as_matrix(),
    }
    global_t = {
        "A": np.zeros(3),
        "B": np.array([1.8, 0.05, 1.1]),
        "C": np.array([0.4, -0.03, 2.2]),
    }
    points = rng.uniform([-0.4, -0.5, 3.5], [0.4, 0.7, 5.0], size=(120, 3))
    pairs = [
        _synthetic_pair("A", "B", global_r, global_t, points, intrinsics, np.array([0.7, -0.8, 0.3])),
        _synthetic_pair("A", "C", global_r, global_t, points, intrinsics, np.array([-0.6, 0.9, -0.4])),
        _synthetic_pair("B", "C", global_r, global_t, points, intrinsics, np.array([0.4, -0.5, 0.6])),
    ]
    naive = {"A": np.eye(3), "B": pairs[0].R, "C": pairs[2].R @ pairs[0].R}
    naive_error = sum(
        float(np.dot(values, values))
        for pair in pairs
        for values in [reprojection_residuals(pair, naive[pair.camera_b] @ naive[pair.camera_a].T)]
    )
    refined, _, _ = refine_global_rotations(["A", "B", "C"], "A", pairs)
    refined_error = sum(
        float(np.dot(values, values))
        for pair in pairs
        for values in [reprojection_residuals(pair, refined[pair.camera_b] @ refined[pair.camera_a].T)]
    )
    assert refined_error < naive_error


def test_translation_solver_rejects_an_edge_without_metric_scale() -> None:
    pairs = [
        PairwiseCalibration(
            "A", "B", "ok", 100, R=np.eye(3), t_unit=np.array([1.0, 0.0, 0.0]),
            raw_shoulder_distance=0.4,
        ),
        PairwiseCalibration(
            "B", "C", "ok", 100, R=np.eye(3), t_unit=np.array([1.0, 0.0, 0.0]),
            raw_shoulder_distance=None,
        ),
    ]
    try:
        solve_global_translations(
            ["A", "B", "C"], "A", {name: np.eye(3) for name in "ABC"}, pairs, 0.4,
        )
    except ValueError as error:
        assert "underdetermined" in str(error)
    else:
        raise AssertionError("Unscaled edge was silently assigned a unit baseline")
