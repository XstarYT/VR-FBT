"""Landmark data structures and small filtering helpers."""

from __future__ import annotations

import math


class keypoint:
    """A tracked 3D point with visibility and bounded position history."""

    def __init__(self, pos, vis):
        self.pos = [float(value) for value in pos]
        self.vis = float(vis)
        self.his: list[list[float]] = []

    def append_history(self) -> None:
        self.his.append(self.pos.copy())
        del self.his[:-15]

    def update(self, pos, vis, smoothing: float | None = None) -> None:
        values = [float(value) for value in pos]
        if smoothing is not None and self.his:
            base_alpha = min(max(float(smoothing), 0.0), 1.0)
            movement = math.sqrt(sum((new - old) ** 2 for new, old in zip(values, self.pos)))
            # Small changes are stabilized; deliberate movement catches up fast.
            alpha = min(0.88, base_alpha + movement * 3.0)
            values = [alpha * new + (1.0 - alpha) * old for new, old in zip(values, self.pos)]
        self.pos = values
        self.vis = float(vis)


def get_Joint_Map(name, Json_lib):
    return Json_lib.File.load(f"Content/Joint-Maps/{name}.json")


def midpoint(p1, p2):
    return [(left + right) / 2.0 for left, right in zip(p1.pos, p2.pos)]


def add(vec1, vec2):
    return [left + right for left, right in zip(vec1, vec2)]


def _validate_filter(x, cutoff, fs):
    if fs <= 0:
        raise ValueError("sample rate must be positive")
    if not 0 < cutoff < fs / 2:
        raise ValueError("cutoff must be between zero and the Nyquist frequency")
    return [float(value) for value in x]


def legendre_lowpass_o2(x, cutoff, fs):
    values = _validate_filter(x, cutoff, fs)
    omega = math.tan(math.pi * cutoff / fs)
    c = omega**2
    norm = 1 + 1.618 * omega + c
    a0, a1 = c / norm, 2 * c / norm
    b1, b2 = 2 * (c - 1) / norm, (1 - 1.618 * omega + c) / norm
    out = [0.0] * len(values)
    for i, value in enumerate(values):
        if i == 0:
            out[i] = a0 * value
        elif i == 1:
            out[i] = a0 * value + a1 * values[i - 1] - b1 * out[i - 1]
        else:
            out[i] = a0 * value + a1 * values[i - 1] + a0 * values[i - 2] - b1 * out[i - 1] - b2 * out[i - 2]
    return out


def legendre_highpass_o2(x, cutoff, fs):
    values = _validate_filter(x, cutoff, fs)
    omega = math.tan(math.pi * cutoff / fs)
    c = omega**2
    norm = 1 + 1.618 * omega + c
    a0, a1 = 1 / norm, -2 / norm
    b1, b2 = 2 * (c - 1) / norm, (1 - 1.618 * omega + c) / norm
    out = [0.0] * len(values)
    for i, value in enumerate(values):
        if i == 0:
            out[i] = a0 * value
        elif i == 1:
            out[i] = a0 * value + a1 * values[i - 1] - b1 * out[i - 1]
        else:
            out[i] = a0 * value + a1 * values[i - 1] + a0 * values[i - 2] - b1 * out[i - 1] - b2 * out[i - 2]
    return out


class Map:
    def __init__(self, joint_map):
        self.JMap = joint_map
        self.Fused: dict[str, keypoint] = {}
        self.KeyPoints = {name: keypoint([0, 0, 0], 0.0) for name in joint_map["KeyPoints"]}

    def Update(self, landmark_list, smooth: bool = False):
        required_index = max(self.JMap["KeyPoints"].values())
        if len(landmark_list) <= required_index:
            raise ValueError(f"Model returned {len(landmark_list)} landmarks; map needs index {required_index}")
        alpha = 0.28 if smooth else None
        for name, index in self.JMap["KeyPoints"].items():
            x, y, z, visibility = landmark_list[index]
            point = self.KeyPoints[name]
            if visibility >= 0.35 or not point.his:
                point.update([x, y, z], visibility, alpha)
                point.append_history()
            else:
                # Do not let an occluded joint's noisy estimate poison the
                # next visible frame; retain position but expose confidence.
                point.vis = float(visibility)

    @staticmethod
    def make_fused(left: keypoint, right: keypoint, correction) -> keypoint:
        return keypoint(add(midpoint(left, right), correction), (left.vis + right.vis) / 2.0)
