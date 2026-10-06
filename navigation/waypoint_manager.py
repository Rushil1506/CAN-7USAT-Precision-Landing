from dataclasses import dataclass
from typing import List

from .coordinates import LocalPoint
from .distance import distance


@dataclass
class WaypointManager:
    """
    Manage sequential waypoint navigation.

    Waypoints are stored in order. The manager advances to the next
    waypoint when the CanSat enters the acceptance radius of the
    current waypoint.
    """

    waypoints: List[LocalPoint]
    acceptance_radius: float = 20.0

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise ValueError("waypoints must contain at least one point")
        if self.acceptance_radius <= 0.0:
            raise ValueError("acceptance_radius must be greater than zero")

        # Keep our own list so callers cannot accidentally alter navigation
        # while the mission is running.
        self.waypoints = list(self.waypoints)
        self._current_index = 0

    @property
    def current_index(self) -> int:
        """Zero-based index of the active waypoint."""
        return self._current_index

    @property
    def waypoint_number(self) -> int:
        """One-based waypoint number for logging/display."""
        return self._current_index + 1

    @property
    def current_waypoint(self) -> LocalPoint:
        """Return the active waypoint."""
        if self.is_complete:
            return self.waypoints[-1]
        return self.waypoints[self._current_index]

    @property
    def target(self) -> LocalPoint:
        """Return the final landing target (last waypoint)."""
        return self.waypoints[-1]

    @property
    def is_complete(self) -> bool:
        """True after the final waypoint has been accepted."""
        return self._current_index >= len(self.waypoints)

    @property
    def total_waypoints(self) -> int:
        return len(self.waypoints)

    def distance_to_current(self, position: LocalPoint) -> float:
        """Distance from the current position to the active waypoint."""
        return distance(position, self.current_waypoint)

    def update(self, position: LocalPoint) -> bool:
        """
        Check the current waypoint and advance when reached.

        Returns:
            True if the active waypoint changed during this call.
        """
        if self.is_complete:
            return False

        changed = False

        # Use a while-loop so that very large simulation steps cannot leave
        # several already-reached waypoints active.
        while not self.is_complete:
            current = self.current_waypoint

            if distance(position, current) > self.acceptance_radius:
                break

            self._current_index += 1
            changed = True

        return changed

    def reset(self) -> None:
        """Return navigation to waypoint 1."""
        self._current_index = 0
