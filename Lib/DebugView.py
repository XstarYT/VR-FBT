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
