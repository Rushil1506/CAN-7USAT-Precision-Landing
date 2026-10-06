import math

from .coordinates import LocalPoint


def bearing(from_point: LocalPoint, to_point: LocalPoint) -> float:
    """
    Calculate navigation bearing from one local point to another.

    Bearing convention:
        0°   = North
        90°  = East
        180° = South
        270° = West

    Returns:
        Bearing in degrees [0, 360).
    """

    dx = to_point.x - from_point.x
    dy = to_point.y - from_point.y

    angle_rad = math.atan2(dx, dy)

    angle_deg = math.degrees(angle_rad)

    # Normalize to [0, 360)
    angle_deg = angle_deg % 360.0

    return angle_deg