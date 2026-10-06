from typing import List

from .coordinates import LocalPoint


def generate_waypoints(
    start: LocalPoint,
    target: LocalPoint,
    num_waypoints: int = 10
) -> List[LocalPoint]:
    """
    Generate equally spaced waypoints between start and target.

    The target itself is included as the final waypoint.

    Example:
        num_waypoints = 10

        WP1 ... WP9 = intermediate waypoints
        WP10         = target
    """

    if num_waypoints < 1:
        raise ValueError("num_waypoints must be at least 1")

    waypoints = []

    for i in range(1, num_waypoints + 1):

        fraction = i / num_waypoints

        x = start.x + fraction * (target.x - start.x)
        y = start.y + fraction * (target.y - start.y)

        waypoints.append(
            LocalPoint(x=x, y=y)
        )

    return waypoints