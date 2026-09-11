"""Small rotation/translation pose graph for two or three cameras."""

from collections import deque

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .pairwise import reprojection_residuals
from .scale import relative_translation_factor
from .types import FloatArray, PairwiseCalibration


def _edge_transform(
    pair: PairwiseCalibration, source: str, target: str
) -> tuple[FloatArray, FloatArray]:
    if pair.R is None or pair.t_unit is None:
        raise ValueError("Pose graph edge has no recovered transform")
    if pair.camera_a == source and pair.camera_b == target:
        return pair.R, pair.t_unit
    if pair.camera_a == target and pair.camera_b == source:
        inverse_r = pair.R.T
        return inverse_r, -inverse_r @ pair.t_unit
    raise ValueError(f"Pair {pair.pair_key} does not join {source} and {target}")


def graph_is_connected(camera_ids: list[str], pairs: list[PairwiseCalibration]) -> bool:
    """Return whether usable pair edges connect all camera IDs."""

    if not camera_ids:
        return False
    adjacency: dict[str, set[str]] = {camera: set() for camera in camera_ids}
    for pair in pairs:
        if pair.status == "ok":
            adjacency[pair.camera_a].add(pair.camera_b)
            adjacency[pair.camera_b].add(pair.camera_a)
    reached = {camera_ids[0]}
    queue: deque[str] = deque(reached)
    while queue:
        camera = queue.popleft()
        for neighbor in adjacency[camera] - reached:
            reached.add(neighbor)
            queue.append(neighbor)
    return len(reached) == len(camera_ids)


def _initial_rotations(
    camera_ids: list[str], reference: str, pairs: list[PairwiseCalibration]
) -> dict[str, FloatArray]:
    rotations: dict[str, FloatArray] = {reference: np.eye(3)}
    while len(rotations) < len(camera_ids):
        crossing = [
            pair
            for pair in pairs
            if pair.status == "ok"
            and ((pair.camera_a in rotations) != (pair.camera_b in rotations))
        ]
        if not crossing:
            raise ValueError("Pose graph is disconnected")
        pair = min(
            crossing,
            key=lambda item: (item.mean_reprojection_error_px, -item.cheirality_fraction),
        )
        if pair.camera_a in rotations:
            edge_r, _ = _edge_transform(pair, pair.camera_a, pair.camera_b)
            rotations[pair.camera_b] = edge_r @ rotations[pair.camera_a]
        else:
            edge_r, _ = _edge_transform(pair, pair.camera_b, pair.camera_a)
            rotations[pair.camera_a] = edge_r @ rotations[pair.camera_b]
    return rotations


def refine_global_rotations(
    camera_ids: list[str],
    reference: str,
    pairs: list[PairwiseCalibration],
) -> tuple[dict[str, FloatArray], float, float]:
    """Jointly refine global rotations and return before/after edge residuals.

    The reference is fixed. Every observed pair contributes a robust SO(3)
    residual weighted by its reprojection quality and cheirality. With all
    three edges this softens, rather than hardcodes, triangle closure and makes
    the three-camera estimate an over-determined joint solution.
    """

    initial = _initial_rotations(camera_ids, reference, pairs)
    variable_ids = [camera for camera in camera_ids if camera != reference]
    x0 = np.concatenate([Rotation.from_matrix(initial[camera]).as_rotvec() for camera in variable_ids])

    def unpack(values: FloatArray) -> dict[str, FloatArray]:
        result: dict[str, FloatArray] = {reference: np.eye(3)}
        for index, camera in enumerate(variable_ids):
            result[camera] = Rotation.from_rotvec(values[index * 3 : index * 3 + 3]).as_matrix()
        return result

    usable = [pair for pair in pairs if pair.status == "ok"]

    def rotation_residuals(values: FloatArray) -> FloatArray:
        rotations = unpack(values)
        chunks: list[FloatArray] = []
        for pair in usable:
            measured_r, _ = _edge_transform(pair, pair.camera_a, pair.camera_b)
            predicted_r = rotations[pair.camera_b] @ rotations[pair.camera_a].T
            delta = measured_r.T @ predicted_r
            error = Rotation.from_matrix(delta).as_rotvec()
            quality = max(pair.cheirality_fraction, 0.1) / max(
                pair.mean_reprojection_error_px, 0.25
            )
            chunks.append(error * np.sqrt(quality))
        return np.concatenate(chunks)

    def reprojection_objective(values: FloatArray) -> FloatArray:
        rotations = unpack(values)
        chunks: list[FloatArray] = []
        for pair in usable:
            predicted_r = rotations[pair.camera_b] @ rotations[pair.camera_a].T
            # Equalize pair influence despite dropout and scale pixels to keep
            # robust-loss tuning stable across resolutions.
            image_error = reprojection_residuals(pair, predicted_r)
            chunks.append(image_error / np.sqrt(max(len(image_error), 1)))
        # A light measured-rotation term regularizes near-planar captures.
        chunks.append(rotation_residuals(values) * 0.05)
        # Human landmarks occupy a narrow, nearly planar image region. Anchor
        # the optimizer to the best-quality spanning tree so one weak long-
        # baseline edge cannot drag the joint solution into a low-reprojection
        # but geometrically wrong local minimum.
        chunks.append((values - x0) * 50.0)
        return np.concatenate(chunks)

    before_values = reprojection_objective(x0)
    before = float(np.dot(before_values, before_values))
    if len(usable) >= len(camera_ids):
        optimized = least_squares(
            reprojection_objective,
            x0,
            method="trf",
            loss="soft_l1",
            f_scale=0.5,
            max_nfev=100,
        )
        values = optimized.x
    else:
        values = x0
    after_values = reprojection_objective(values)
    after = float(np.dot(after_values, after_values))
    return unpack(values), before, after


def solve_global_translations(
    camera_ids: list[str],
    reference: str,
    rotations: dict[str, FloatArray],
    pairs: list[PairwiseCalibration],
    anchor_raw_shoulder: float,
) -> dict[str, FloatArray]:
    """Least-squares reconcile edge baselines in anchor arbitrary units."""

    unresolved = [
        pair.pair_key for pair in pairs
        if pair.status == "ok"
        and (pair.raw_shoulder_distance is None or pair.raw_shoulder_distance <= 0.0)
    ]
    if unresolved:
        raise ValueError(
            "Metric translation is underdetermined for pair(s): " + ", ".join(unresolved)
        )
    if not graph_is_connected(camera_ids, pairs):
        raise ValueError("Scaled pose graph is disconnected")
    variable_ids = [camera for camera in camera_ids if camera != reference]
    offsets = {camera: index * 3 for index, camera in enumerate(variable_ids)}
    rows: list[FloatArray] = []
    targets: list[FloatArray] = []
    for pair in pairs:
        if pair.status != "ok":
            continue
        _, edge_t = _edge_transform(pair, pair.camera_a, pair.camera_b)
        edge_r = rotations[pair.camera_b] @ rotations[pair.camera_a].T
        edge_t = edge_t * relative_translation_factor(pair, anchor_raw_shoulder)
        block = np.zeros((3, len(variable_ids) * 3), dtype=np.float64)
        if pair.camera_b != reference:
            start = offsets[pair.camera_b]
            block[:, start : start + 3] += np.eye(3)
        if pair.camera_a != reference:
            start = offsets[pair.camera_a]
            block[:, start : start + 3] -= edge_r
        rows.append(block)
        targets.append(edge_t)
    matrix = np.vstack(rows)
    target = np.concatenate(targets)
    solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    translations: dict[str, FloatArray] = {reference: np.zeros(3)}
    for camera, start in offsets.items():
        translations[camera] = np.asarray(solution[start : start + 3], dtype=np.float64)
    return translations
