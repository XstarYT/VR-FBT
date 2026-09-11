"""Camera intrinsics and source abstractions."""

from collections.abc import Mapping
from math import atan, radians, tan
from typing import Literal, Protocol

from .types import CameraIntrinsics


def intrinsics_from_fov(
    width: int,
    height: int,
    fov_deg: float,
    fov_convention: Literal["horizontal", "diagonal"] = "horizontal",
) -> CameraIntrinsics:
    """Create pixel intrinsics from horizontal or diagonal field of view.

    Diagonal FOV is converted using the image aspect ratio. The mandated
    fallback then uses ``fx = fy = width / (2*tan(horizontal_fov/2))`` and
    the image centre as principal point. Raises ``ValueError`` for invalid
    dimensions, convention, or a FOV outside (0, 180) degrees.
    """

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if not 0.0 < fov_deg < 180.0:
        raise ValueError("FOV must be between 0 and 180 degrees")
    if fov_convention not in ("horizontal", "diagonal"):
        raise ValueError(f"Unsupported FOV convention: {fov_convention}")

    fov_rad = radians(fov_deg)
    if fov_convention == "diagonal":
        # tan(theta_d/2)^2 = tan(theta_h/2)^2 + tan(theta_v/2)^2,
        # with tan(theta_v/2) = tan(theta_h/2) * H/W.
        aspect_term = (1.0 + (height / width) ** 2) ** 0.5
        fov_rad = 2.0 * atan(tan(fov_rad / 2.0) / aspect_term)
    focal = width / (2.0 * tan(fov_rad / 2.0))
    return CameraIntrinsics(width, height, focal, focal, width / 2.0, height / 2.0)


class CameraStream(Protocol):
    """Optional upstream frame source; image dimensions and pixels are explicit."""

    def read(self) -> tuple[bool, object]:
        """Return ``(available, image)`` where image coordinates are pixels."""


class KeypointSource(Protocol):
    """Optional upstream pose source independent of MediaPipe installation."""

    def extract(
        self, image: object
    ) -> tuple[Mapping[str, tuple[float, float]], Mapping[str, float]]:
        """Return named pixel coordinates and unit-interval confidences."""


__all__ = ["CameraIntrinsics", "CameraStream", "KeypointSource", "intrinsics_from_fov"]
