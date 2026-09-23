"""Guard against camera layouts that give weak depth triangulation."""

import math
from itertools import combinations


def layout_warning(camera_positions, subject_position, *, min_baseline_m=0.5, min_ray_angle_degrees=12.0):
    """Return a placement hint when no camera pair gives useful parallax.

    Forward-vector differences alone are misleading: two parallel cameras can
    still triangulate when spaced apart. Measure the angle of their rays at the
    observed subject instead.
    """
    subject = tuple(float(value) for value in subject_position)
    for first, second in combinations(camera_positions, 2):
        a = tuple(float(x) - y for x, y in zip(first, subject))
        b = tuple(float(x) - y for x, y in zip(second, subject))
        distance = math.dist(first, second)
        length_a, length_b = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b))
        if min(distance, length_a, length_b) <= 1e-6:
            continue
        cosine = max(-1.0, min(1.0, sum(x*y for x, y in zip(a, b)) / (length_a * length_b)))
        angle = math.degrees(math.acos(cosine))
        if distance >= min_baseline_m and angle >= min_ray_angle_degrees:
            return ""
    return "Camera views have too little depth separation; move one phone farther to the side and recalibrate"
