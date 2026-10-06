from dataclasses import dataclass
from enum import Enum
import math

from navigation.bearing import bearing
from navigation.coordinates import LocalPoint
from navigation.heading import heading_error


class LoiterPhase(Enum):
    """Phase of loiter engagement."""

    ORBIT_CAPTURE = "ORBIT_CAPTURE"
    ORBIT_TRACKING = "ORBIT_TRACKING"


class LoiterDirection(Enum):
    """Direction of circular holding orbit."""

    CLOCKWISE = "CLOCKWISE"
    COUNTER_CLOCKWISE = "COUNTER_CLOCKWISE"


@dataclass(frozen=True)
class LoiterGuidanceOutput:
    """
    Guidance output for a circular holding orbit.

    Attributes:
        desired_heading: Commanded heading in degrees [0, 360).
        heading_error: Shortest signed angular error in degrees [-180, 180).
        radial_distance: Euclidean distance from orbit center [m].
        radial_error: Distance error relative to target radius (r - R) [m].
        phase: Current loiter phase (ORBIT_CAPTURE or ORBIT_TRACKING).
        direction: Orbit rotation direction (CLOCKWISE or COUNTER_CLOCKWISE).
        desired_turn_direction: Signed turn direction (+1.0 for CW/right, -1.0 for CCW/left).
        nominal_turn_rate: Theoretical steady orbital rate omega = V/R [deg/s].
        steering_command: Recommended steering command normalized to [-1.0, 1.0].
    """

    desired_heading: float
    heading_error: float
    radial_distance: float
    radial_error: float
    phase: LoiterPhase
    direction: LoiterDirection
    desired_turn_direction: float
    nominal_turn_rate: float
    steering_command: float


def calculate_radial_distance(center: LocalPoint, position: LocalPoint) -> float:
    """
    Calculate Euclidean distance from orbit center to vehicle position.

    Formula:
        radial_distance = sqrt((x - cx)^2 + (y - cy)^2)
    """
    dx = position.x - center.x
    dy = position.y - center.y
    return math.hypot(dx, dy)


def calculate_radial_error(
    center: LocalPoint, position: LocalPoint, loiter_radius: float
) -> float:
    """
    Calculate radial error relative to desired loiter orbit radius.

    Formula:
        radial_error = radial_distance - loiter_radius
    """
    if loiter_radius <= 0.0:
        raise ValueError("loiter_radius must be greater than zero")
    return calculate_radial_distance(center, position) - loiter_radius


def calculate_nominal_turn_rate(
    forward_speed: float, loiter_radius: float
) -> float:
    """
    Calculate theoretical steady orbital yaw rate in degrees per second.

    Physics relationship:
        omega = V / R  [rad/s]
        omega_deg = (V / R) * (180 / pi)  [deg/s]

    For baseline parameters:
        V = 8.0 m/s
        R = 30.0 m
        omega ≈ 0.2667 rad/s ≈ 15.28 deg/s (well below max turn rate of 30.0 deg/s)
    """
    if loiter_radius <= 0.0:
        raise ValueError("loiter_radius must be greater than zero")
    if forward_speed <= 0.0:
        raise ValueError("forward_speed must be greater than zero")

    omega_rad = forward_speed / loiter_radius
    return math.degrees(omega_rad)


def calculate_tangent_heading(
    center: LocalPoint,
    position: LocalPoint,
    direction: LoiterDirection,
) -> float:
    """
    Calculate the exact tangent heading at position P on the circle.

    Coordinate convention:
        0°   = North (+Y)
        90°  = East (+X)
        180° = South (-Y)
        270° = West (-X)

    Tangential directions:
        CLOCKWISE:
            tangent = (radial_bearing + 90°) mod 360°
        COUNTER_CLOCKWISE:
            tangent = (radial_bearing - 90°) mod 360°
    """
    dx = position.x - center.x
    dy = position.y - center.y
    if math.hypot(dx, dy) < 1e-9:
        # At the exact center, radial bearing is undefined; default to North tangent
        return 90.0 if direction == LoiterDirection.CLOCKWISE else 270.0

    radial_bearing = bearing(center, position)

    if direction == LoiterDirection.CLOCKWISE:
        tangent = (radial_bearing + 90.0) % 360.0
    else:
        tangent = (radial_bearing - 90.0) % 360.0

    return tangent


class LoiterGuidance:
    """
    Dedicated guidance component for circular holding orbits.

    Physics and Kinematic Model:
        Theoretical steady orbital turn rate:
            omega = V / R

        For baseline simulation parameters:
            V = 8.0 m/s  (forward air-relative speed)
            R = 30.0 m   (loiter radius)

            omega = 8.0 / 30.0 = 0.2667 rad/s ≈ 15.28 deg/s

        This is well within the parafoil model's maximum turn rate of 30.0 deg/s.

    Assumptions and Limitations:
        - The current model assumes constant forward speed (8.0 m/s) and constant
          vertical descent rate (2.5 m/s).
        - It does NOT model turn-induced descent rate increases (spiral dive effect).
        - Currently assumes zero wind and symmetric aerodynamic behavior.
        - Loiter radius (30.0 m) is a simulation parameter, NOT a hardware-validated value.
    """

    def __init__(
        self,
        center: LocalPoint,
        loiter_radius: float = 30.0,
        direction: LoiterDirection = LoiterDirection.CLOCKWISE,
        capture_tolerance: float = 5.0,
        forward_speed: float = 8.0,
        max_turn_rate: float = 30.0,
    ) -> None:
        if loiter_radius <= 0.0:
            raise ValueError("loiter_radius must be greater than zero")
        if capture_tolerance <= 0.0:
            raise ValueError("capture_tolerance must be greater than zero")
        if forward_speed <= 0.0:
            raise ValueError("forward_speed must be greater than zero")
        if max_turn_rate <= 0.0:
            raise ValueError("max_turn_rate must be greater than zero")

        self.center = center
        self.loiter_radius = loiter_radius
        self.direction = direction
        self.capture_tolerance = capture_tolerance
        self.forward_speed = forward_speed
        self.max_turn_rate = max_turn_rate

        self.nominal_turn_rate = calculate_nominal_turn_rate(
            forward_speed=forward_speed, loiter_radius=loiter_radius
        )

    def update(
        self,
        position: LocalPoint,
        current_heading: float,
    ) -> LoiterGuidanceOutput:
        """
        Calculate one loiter guidance command.

        Args:
            position: Current vehicle local position [m].
            current_heading: Current navigation heading in degrees [0, 360).

        Returns:
            LoiterGuidanceOutput containing target heading, errors, and commands.
        """
        radial_distance = calculate_radial_distance(self.center, position)
        radial_error = radial_distance - self.loiter_radius

        dir_sign = 1.0 if self.direction == LoiterDirection.CLOCKWISE else -1.0

        # Classify phase based on capture tolerance
        is_tracking = abs(radial_error) <= self.capture_tolerance
        phase = LoiterPhase.ORBIT_TRACKING if is_tracking else LoiterPhase.ORBIT_CAPTURE

        pure_tangent = calculate_tangent_heading(
            self.center, position, self.direction
        )

        if is_tracking:
            # ORBIT_TRACKING:
            # Baseline heading is the tangent. A bounded proportional correction
            # corrects slight radial errors (inward when outside, outward when inside).
            # When radial_error == 0, desired_heading == pure_tangent exactly.
            k_radial = 1.0 / self.loiter_radius
            correction_deg = math.degrees(math.atan(k_radial * radial_error))
            correction_deg = max(-30.0, min(30.0, correction_deg))

            if self.direction == LoiterDirection.CLOCKWISE:
                desired_heading = (pure_tangent + correction_deg) % 360.0
            else:
                desired_heading = (pure_tangent - correction_deg) % 360.0

            # Nominal steady orbital feedforward command
            feedforward_steering = dir_sign * (
                self.nominal_turn_rate / self.max_turn_rate
            )

            err = heading_error(current_heading, desired_heading)
            # Gentle feedback on top of feedforward (45° error = full offset)
            feedback_steering = err / 45.0
            steering_cmd = max(-1.0, min(1.0, feedforward_steering + feedback_steering))

        else:
            # ORBIT_CAPTURE:
            if radial_distance > self.loiter_radius + self.capture_tolerance:
                # Outside the orbit: tangent-to-circle intercept geometry
                theta_to_center = bearing(position, self.center)
                ratio = min(1.0, self.loiter_radius / radial_distance)
                alpha_deg = math.degrees(math.asin(ratio))

                if self.direction == LoiterDirection.CLOCKWISE:
                    desired_heading = (theta_to_center - alpha_deg) % 360.0
                else:
                    desired_heading = (theta_to_center + alpha_deg) % 360.0

            else:
                # Inside the orbit: expand outward toward orbit radius
                if radial_distance < 1e-6:
                    desired_heading = current_heading % 360.0
                else:
                    radial_bearing = bearing(self.center, position)
                    ratio = min(1.0, radial_distance / self.loiter_radius)
                    beta_deg = math.degrees(math.acos(ratio))

                    if self.direction == LoiterDirection.CLOCKWISE:
                        desired_heading = (radial_bearing + 90.0 - beta_deg) % 360.0
                    else:
                        desired_heading = (radial_bearing - 90.0 + beta_deg) % 360.0

            err = heading_error(current_heading, desired_heading)
            # In capture mode, steer toward target heading without pure feedforward
            steering_cmd = max(-1.0, min(1.0, err / 45.0))

        return LoiterGuidanceOutput(
            desired_heading=desired_heading,
            heading_error=heading_error(current_heading, desired_heading),
            radial_distance=radial_distance,
            radial_error=radial_error,
            phase=phase,
            direction=self.direction,
            desired_turn_direction=dir_sign,
            nominal_turn_rate=self.nominal_turn_rate,
            steering_command=steering_cmd,
        )
