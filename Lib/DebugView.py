"""Visual tracking diagnostics rendered on top of an OpenCV frame."""

from __future__ import annotations

import math
from typing import Sequence


# The checkpoint returns the 31 landmarks defined by FULLMAP.json.
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29),
    (24, 26), (26, 28), (28, 30),
)

MAJOR_JOINTS = {
    11: "L shoulder", 12: "R shoulder",
    13: "L elbow", 14: "R elbow",
    15: "L wrist", 16: "R wrist",
    23: "L hip", 24: "R hip",
    25: "L knee", 26: "R knee",
    27: "L ankle", 28: "R ankle",
    29: "L foot", 30: "R foot",
}


def project_landmarks(landmarks: Sequence[Sequence[float]], width: int, height: int):
    """Convert normalized model coordinates into optional image points."""
    projected: list[tuple[int, int, float] | None] = []
    for landmark in landmarks:
        try:
            x, y = float(landmark[0]), float(landmark[1])
            visibility = float(landmark[3]) if len(landmark) > 3 else 1.0
        except (IndexError, TypeError, ValueError):
            projected.append(None)
            continue
        if not all(math.isfinite(value) for value in (x, y, visibility)):
            projected.append(None)
            continue
        # Allow a small margin so limbs at the edge do not flicker on/off.
        if not (-0.1 <= x <= 1.1 and -0.1 <= y <= 1.1):
            projected.append(None)
            continue
        px = min(max(round(x * (width - 1)), 0), width - 1)
        py = min(max(round(y * (height - 1)), 0), height - 1)
        projected.append((px, py, visibility))
    return projected


def render_tracking_debug(
    frame,
    landmarks: Sequence[Sequence[float]],
    confidence: float,
    fps: float,
    frame_count: int,
    source_label: str,
    cv2,
    detected: bool = True,
    tracking_status: str | None = None,
):
    """Return a copy of frame with a skeleton and readable tracking telemetry."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    points = project_landmarks(landmarks, width, height)
    confidence = min(max(float(confidence), 0.0), 1.0)
    tracking = detected and confidence >= 0.5
    strong = (70, 230, 120)       # BGR green
    weak = (40, 170, 255)         # BGR amber
    lost = (90, 90, 255)          # BGR red

    for first, second in POSE_CONNECTIONS:
        if first >= len(points) or second >= len(points):
            continue
        left, right = points[first], points[second]
        if left is None or right is None:
            continue
        color = strong if min(left[2], right[2]) >= 0.5 else weak
        cv2.line(canvas, left[:2], right[:2], color, 2, cv2.LINE_AA)

    visible_points = 0
    for index, point in enumerate(points):
        if point is None:
            continue
        color = strong if point[2] >= 0.5 else weak
        if point[2] >= 0.5:
            visible_points += 1
        cv2.circle(canvas, point[:2], 4, (12, 18, 30), -1, cv2.LINE_AA)
        cv2.circle(canvas, point[:2], 3, color, -1, cv2.LINE_AA)
        if index in MAJOR_JOINTS and point[2] >= 0.5:
            cv2.putText(canvas, MAJOR_JOINTS[index], (point[0] + 6, point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    panel_height = 92
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, min(panel_height, height)), (8, 13, 25), -1)
    canvas = cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0)
    status = tracking_status or ("TRACKING" if tracking else ("LOW CONFIDENCE" if detected else "NO POSE"))
    status_color = weak if status.startswith(("CALIBRATING", "HOLDING")) else (strong if tracking else lost)
    cv2.putText(canvas, status, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, status_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Pose {confidence * 100:5.1f}%   FPS {fps:5.1f}   Visible {visible_points:02d}/{len(points):02d}",
                (16, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 240, 250), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{source_label}   Frame {frame_count:,}",
                (16, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 185, 210), 1, cv2.LINE_AA)
    if height >= 130:
        cv2.putText(canvas, "Green = visible   Amber = uncertain   Q = stop", (12, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 235, 245), 1, cv2.LINE_AA)
    return canvas


class DebugScene3D:
    """Rotatable OpenCV 3D scene for fused landmarks and camera poses."""

    def __init__(self, cv2_module, width: int = 960, height: int = 640):
        self.cv2 = cv2_module
        self.width, self.height = int(width), int(height)
        self.yaw, self.pitch, self.zoom = -28.0, -16.0, 155.0
        self._dragging = False
        self._last_mouse = (0, 0)
        self._world_center = None
        self._floor_y = None

    def mouse_callback(self, event, x, y, flags, _parameter=None) -> None:
        cv2 = self.cv2
        if event == getattr(cv2, "EVENT_LBUTTONDOWN", 1):
            self._dragging = True
            self._last_mouse = (x, y)
        elif event == getattr(cv2, "EVENT_LBUTTONUP", 4):
            self._dragging = False
        elif event == getattr(cv2, "EVENT_MOUSEMOVE", 0) and self._dragging:
            dx, dy = x - self._last_mouse[0], y - self._last_mouse[1]
            self.yaw = (self.yaw + dx * 0.45) % 360.0
            self.pitch = min(max(self.pitch + dy * 0.35, -85.0), 85.0)
            self._last_mouse = (x, y)
        elif event == getattr(cv2, "EVENT_MOUSEWHEEL", 10):
            self.zoom = min(max(self.zoom * (1.12 if flags > 0 else 1.0 / 1.12), 45.0), 420.0)

    def _rotation(self):
        import numpy as np

        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
        rotate_y = np.array(((math.cos(yaw), 0.0, math.sin(yaw)), (0.0, 1.0, 0.0), (-math.sin(yaw), 0.0, math.cos(yaw))))
        rotate_x = np.array(((1.0, 0.0, 0.0), (0.0, math.cos(pitch), -math.sin(pitch)), (0.0, math.sin(pitch), math.cos(pitch))))
        return rotate_x @ rotate_y

    def _project(self, point, center, rotation):
        import numpy as np

        transformed = rotation @ (np.asarray(point, dtype=float) - center)
        return (
            int(round(self.width * 0.5 + transformed[0] * self.zoom)),
            int(round(self.height * 0.55 - transformed[1] * self.zoom)),
            float(transformed[2]),
        )

    def render(
        self,
        landmarks: Sequence[Sequence[float]],
        camera_poses: Sequence,
        confidence: float,
        fps: float,
        status: str,
        active_cameras: int,
    ):
        import numpy as np

        cv2 = self.cv2
        canvas = np.full((self.height, self.width, 3), (22, 16, 10), dtype=np.uint8)
        valid_points = []
        for landmark in landmarks:
            try:
                point = np.asarray(landmark[:3], dtype=float)
                visibility = float(landmark[3])
            except (IndexError, TypeError, ValueError):
                valid_points.append(None)
                continue
            valid_points.append((point, visibility) if np.isfinite(point).all() and math.isfinite(visibility) else None)

        body_points = [item[0] for item in valid_points if item is not None]
        if self._world_center is None:
            # Lock the viewport to the calibration coordinate system. Re-centering
            # on every live pose makes stationary cameras appear to move whenever
            # the person walks, which is exactly the opposite of a useful world view.
            self._floor_y = min((point[1] for point in body_points), default=-0.9)
            self._world_center = np.array((0.0, self._floor_y + 0.9, 0.0), dtype=float)
        center = self._world_center
        rotation = self._rotation()

        floor_y = self._floor_y
        grid_color = (48, 43, 37)
        for step in range(-6, 7):
            for first, second in (
                ((step * 0.5, floor_y, -3.0), (step * 0.5, floor_y, 3.0)),
                ((-3.0, floor_y, step * 0.5), (3.0, floor_y, step * 0.5)),
            ):
                a, b = self._project(first, center, rotation), self._project(second, center, rotation)
                cv2.line(canvas, a[:2], b[:2], grid_color, 1, cv2.LINE_AA)

        axis_origin = (0.0, floor_y, 0.0)
        for endpoint, color, label in (
            ((0.65, floor_y, 0.0), (90, 110, 255), "X"),
            ((0.0, floor_y + 0.65, 0.0), (100, 235, 130), "Y"),
            ((0.0, floor_y, 0.65), (255, 175, 75), "Z"),
        ):
            a, b = self._project(axis_origin, center, rotation), self._project(endpoint, center, rotation)
            cv2.arrowedLine(canvas, a[:2], b[:2], color, 2, cv2.LINE_AA, tipLength=0.12)
            cv2.putText(canvas, label, (b[0] + 4, b[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        camera_colors = ((255, 180, 60), (220, 100, 255), (80, 220, 255))
        for index, camera in enumerate(camera_poses):
            try:
                origin = np.asarray(camera.position, dtype=float)
                camera_rotation = np.asarray(camera.rotation, dtype=float)
                label = camera.source_id
            except (AttributeError, TypeError, ValueError):
                continue
            color = camera_colors[index % len(camera_colors)]
            local_corners = (
                (-0.22, -0.14, 0.38), (0.22, -0.14, 0.38),
                (0.22, 0.14, 0.38), (-0.22, 0.14, 0.38),
            )
            corners = [origin + camera_rotation @ np.asarray(corner) for corner in local_corners]
            projected_origin = self._project(origin, center, rotation)
            projected_corners = [self._project(corner, center, rotation) for corner in corners]
            for corner in projected_corners:
                cv2.line(canvas, projected_origin[:2], corner[:2], color, 2, cv2.LINE_AA)
            for first, second in zip(projected_corners, projected_corners[1:] + projected_corners[:1]):
                cv2.line(canvas, first[:2], second[:2], color, 2, cv2.LINE_AA)
            short_label = label.split(":", 1)[-1][:18]
            approximate = "" if getattr(camera, "calibrated", True) else " (calibrating)"
            sample_count = getattr(camera, "sample_count", 0)
            spread_cm = getattr(camera, "position_std", 0.0) * 100.0
            error_px = getattr(camera, "reprojection_error", 0.0)
            stats = f"  n={sample_count} ±{spread_cm:.1f}cm {error_px:.1f}px" if sample_count else ""
            cv2.putText(canvas, f"CAM {index + 1}: {short_label}{approximate}{stats}", (projected_origin[0] + 7, projected_origin[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

        strong, weak = (70, 230, 120), (40, 170, 255)
        projected_body = [self._project(item[0], center, rotation) if item is not None else None for item in valid_points]
        for first, second in POSE_CONNECTIONS:
            if first >= len(valid_points) or second >= len(valid_points):
                continue
            left, right = valid_points[first], valid_points[second]
            if left is None or right is None:
                continue
            color = strong if min(left[1], right[1]) >= 0.5 else weak
            cv2.line(canvas, projected_body[first][:2], projected_body[second][:2], color, 3, cv2.LINE_AA)
        for index, item in enumerate(valid_points):
            if item is None:
                continue
            point = projected_body[index]
            color = strong if item[1] >= 0.5 else weak
            cv2.circle(canvas, point[:2], 5 if index in MAJOR_JOINTS else 3, (12, 18, 30), -1, cv2.LINE_AA)
            cv2.circle(canvas, point[:2], 3 if index in MAJOR_JOINTS else 2, color, -1, cv2.LINE_AA)

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, 76), (8, 13, 25), -1)
        canvas = cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0)
        cv2.putText(canvas, status, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, strong if confidence >= 0.5 else weak, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Fused pose {confidence * 100:5.1f}%   FPS {fps:5.1f}   Cameras {active_cameras}", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 240, 250), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Drag to orbit   Mouse wheel to zoom   R reset view   Q stop", (14, self.height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 200, 220), 1, cv2.LINE_AA)
        return canvas

    def reset_view(self) -> None:
        self.yaw, self.pitch, self.zoom = -28.0, -16.0, 155.0

    def reset_world(self) -> None:
        """Re-anchor the floor on the next frame after camera recalibration."""
        self._world_center = None
        self._floor_y = None
