"""Primary deterministic acceptance gate for the full calibration pipeline."""

import numpy as np
import pytest

from tests.calib_synthetic_helpers import rotation_error_deg, run_synthetic


@pytest.mark.parametrize("camera_count", [2, 3])
def test_five_randomized_rigs_recover_metric_extrinsics(camera_count: int) -> None:
    for configuration in range(5):
        seed = 100 + configuration + camera_count * 1000
        synthetic = run_synthetic(
            seed,
            camera_count,
            noise_px=1.0 + 0.15 * configuration,
            dropout=0.05 + 0.025 * configuration,
        )
        result = synthetic.result
        for camera_id in result.R:
            rotation_error = rotation_error_deg(result.R[camera_id], synthetic.expected_R[camera_id])
            assert rotation_error < 2.0, f"seed={seed}, camera={camera_id}, rotation={rotation_error}"
            expected_baseline = np.linalg.norm(synthetic.expected_t[camera_id])
            if expected_baseline > 1e-9:
                baseline_error = abs(np.linalg.norm(result.t_scaled[camera_id]) - expected_baseline) / expected_baseline
                assert baseline_error < 0.05, f"seed={seed}, camera={camera_id}, translation={baseline_error}"
        # The anchor Essential translation has norm one, so its single global
        # metres-per-raw-unit factor equals that pair's physical baseline.
        anchor_a, anchor_b = result.scale_anchor_pair.split("_")
        anchor_rotation = synthetic.expected_R[anchor_b] @ synthetic.expected_R[anchor_a].T
        anchor_translation = synthetic.expected_t[anchor_b] - anchor_rotation @ synthetic.expected_t[anchor_a]
        expected_scale = np.linalg.norm(anchor_translation)
        assert abs(result.scale_factor - expected_scale) / expected_scale < 0.05, f"seed={seed}"


def test_fov_fallback_uses_square_pixel_horizontal_formula() -> None:
    from vrfbt_calib import intrinsics_from_fov

    intrinsics = intrinsics_from_fov(1280, 720, 90.0)
    assert intrinsics.fx == pytest.approx(640.0)
    assert intrinsics.fy == pytest.approx(640.0)
    assert intrinsics.cx == 640.0
    assert intrinsics.cy == 360.0

    diagonal = intrinsics_from_fov(1280, 720, 90.0, "diagonal")
    expected_horizontal = 2.0 * np.arctan(np.tan(np.pi / 4.0) / np.sqrt(1.0 + (720 / 1280) ** 2))
    assert diagonal.fx == pytest.approx(1280 / (2.0 * np.tan(expected_horizontal / 2.0)))
