from dataclasses import dataclass

from control.pid import PIDController
from navigation.bearing import bearing
from navigation.coordinates import LocalPoint
from navigation.heading import heading_error
from navigation.waypoint_manager import WaypointManager


@dataclass(frozen=True)
class GuidanceOutput:
    """Output of one guidance-controller update."""

    steering: float
    desired_bearing: float
    heading_error: float
    distance_to_waypoint: float
    waypoint_number: int
    waypoint_reached: bool
    target_reached: bool
    final_approach: bool


class GuidanceController:
    """
    Closed-loop waypoint guidance for the steerable parafoil.

    Pipeline:

        Current position + heading
                    |
                    v
            Active waypoint
                    |
                    v
              Desired bearing
                    |
                    v
              Heading error
                    |
                    v
                 PID
                    |
                    v
             Steering [-1, 1]

    The controller intentionally does not know anything about servo angles.
    That conversion belongs in control/steering.py and control/servo_mapping.py.
    """

    def __init__(
        self,
        waypoint_manager: WaypointManager,
        pid: PIDController,
        final_approach_altitude: float = 120.0,
        final_approach_radius: float = 100.0,
    ) -> None:
        if final_approach_altitude < 0.0:
            raise ValueError("final_approach_altitude cannot be negative")
        if final_approach_radius <= 0.0:
            raise ValueError("final_approach_radius must be greater than zero")

        self.waypoint_manager = waypoint_manager
        self.pid = pid
        self.final_approach_altitude = final_approach_altitude
        self.final_approach_radius = final_approach_radius

    def reset(self) -> None:
        """Reset both waypoint navigation and PID memory."""
        self.waypoint_manager.reset()
        self.pid.reset()

    def update(
        self,
        position: LocalPoint,
        heading: float,
        altitude: float,
        dt: float,
    ) -> GuidanceOutput:
        """
        Calculate one closed-loop steering command.

        Args:
            position: Current local EN position in metres.
            heading: Current navigation heading in degrees, where
                     0 = North and 90 = East.
            altitude: Current altitude above ground in metres.
            dt: Controller update period in seconds.
        """
        if dt <= 0.0:
            raise ValueError("dt must be greater than zero")

        # Check whether the active waypoint has been reached before
        # calculating the new command.
        waypoint_reached = self.waypoint_manager.update(position)

        if self.waypoint_manager.is_complete:
            # The final target is special: reaching its acceptance radius
            # does not mean the CanSat should stop steering immediately.
            # The parafoil still has horizontal velocity, so we keep the
            # target as the control objective until the vehicle is in the
            # terminal landing phase.
            waypoint = self.waypoint_manager.target
            target_distance = _distance_xy(position, waypoint)
            desired_bearing = bearing(position, waypoint)
            error = heading_error(heading, desired_bearing)
            steering = self.pid.update(error=error, dt=dt)

            final_approach = (
                altitude <= self.final_approach_altitude
                or target_distance <= self.final_approach_radius
            )

            return GuidanceOutput(
                steering=steering,
                desired_bearing=desired_bearing,
                heading_error=error,
                distance_to_waypoint=target_distance,
                waypoint_number=self.waypoint_manager.total_waypoints,
                waypoint_reached=waypoint_reached,
                target_reached=True,
                final_approach=final_approach,
            )

        # A waypoint switch creates a new control objective, so clear the
        # integral/derivative memory instead of carrying error from the old
        # waypoint into the new one.
        if waypoint_reached:
            self.pid.reset()

        waypoint = self.waypoint_manager.current_waypoint
        desired_bearing = bearing(position, waypoint)
        error = heading_error(heading, desired_bearing)
        steering = self.pid.update(error=error, dt=dt)

        target = self.waypoint_manager.target
        target_distance = _distance_xy(position, target)

        final_approach = (
            altitude <= self.final_approach_altitude
            and target_distance <= self.final_approach_radius
        )

        return GuidanceOutput(
            steering=steering,
            desired_bearing=desired_bearing,
            heading_error=error,
            distance_to_waypoint=_distance_xy(position, waypoint),
            waypoint_number=self.waypoint_manager.waypoint_number,
            waypoint_reached=waypoint_reached,
            target_reached=False,
            final_approach=final_approach,
        )


def _distance_xy(a: LocalPoint, b: LocalPoint) -> float:
    """Local import-free Euclidean distance helper."""
    dx = b.x - a.x
    dy = b.y - a.y
    return (dx * dx + dy * dy) ** 0.5
