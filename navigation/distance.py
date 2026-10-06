import math

from .coordinates import LocalPoint


def distance(point_a: LocalPoint, point_b: LocalPoint) -> float:
    """
    Calculate straight-line horizontal distance between
    two local Cartesian points.

    Returns:
        Distance in metres.
    """

    dx = point_b.x - point_a.x
    dy = point_b.y - point_a.y

    return math.hypot(dx, dy)