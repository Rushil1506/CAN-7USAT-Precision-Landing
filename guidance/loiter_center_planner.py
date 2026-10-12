"""
Wind-biased loiter center planner for terminal guidance.

This module provides physically motivated planning for the terminal loiter
orbit center based on in-flight estimated wind velocity.

Signal Flow and Architecture:
    1. Landing Target (P_target): Fixed geographic landing point (e.g., (1200.0, 0.0)).
       NEVER modified or shifted.
    2. Estimated Wind (W_est): Sourced strictly from the onboard WindEstimator
       (flight software state), never from simulation truth.
    3. Center Offset (Delta_r): Offset vector calculated as:
           offset_dist = min(k_bias * ||W_est||, max_offset)
           Delta_r = sign * offset_dist * W_hat_est
       where sign = -1.0 for upwind bias, +1.0 for downwind bias.
    4. Planned Loiter Center (P_center):
           P_center = P_target + Delta_r
    5. LoiterGuidance: Tracks circular orbit of nominal radius R around P_center.

Safety and Robustness:
    - Opt-in: Default config is enabled=False (pass-through, 0 offset).
    - Finite numerical checks: NaN/Inf inputs return zero offset.
    - Deadband: Winds below min_wind_speed return zero offset.
    - Clamping: Maximum offset capped at max_offset.
    - Latching: Latches center upon entry to prevent noise/gust chatter in orbit.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from navigation.coordinates import LocalPoint


@dataclass(frozen=True)
class LoiterCenterPlannerConfig:
    """
    Configuration parameters for the wind-biased loiter center planner.

    Attributes:
        enabled: Master toggle. If False, offset is always (0.0, 0.0).
        bias_gain: Time-scale proportionality gain k_bias [seconds].
            Specifies distance offset per unit wind speed (e.g. 2.5 s * 4 m/s = 10 m).
        max_offset: Absolute upper limit on offset distance [meters].
        direction: Direction of bias relative to wind vector:
            - "upwind": offset against wind (-k * W), biasing orbit upwind.
            - "downwind": offset with wind (+k * W), biasing orbit downwind.
            - "none": zero offset.
        min_wind_speed: Minimum wind threshold [m/s] below which bias is zero.
        latch_on_entry: Whether to freeze/latch the planned center once calculated.
    """

    enabled: bool = False
    bias_gain: float = 2.5
    max_offset: float = 30.0
    direction: str = "upwind"
    min_wind_speed: float = 0.5
    latch_on_entry: bool = True

    def __post_init__(self) -> None:
        if self.bias_gain < 0.0:
            raise ValueError(f"bias_gain must be non-negative, got {self.bias_gain}")
        if self.max_offset < 0.0:
            raise ValueError(f"max_offset must be non-negative, got {self.max_offset}")
        if self.min_wind_speed < 0.0:
            raise ValueError(f"min_wind_speed must be non-negative, got {self.min_wind_speed}")
        valid_directions = {"upwind", "downwind", "none"}
        if self.direction not in valid_directions:
            raise ValueError(
                f"direction must be one of {valid_directions}, got '{self.direction}'"
            )


class LoiterCenterPlanner:
    """
    Plans wind-biased loiter orbit centers for terminal energy management.
    """

    def __init__(self, config: Optional[LoiterCenterPlannerConfig] = None) -> None:
        self.config = config if config is not None else LoiterCenterPlannerConfig()
        self._latched_center: Optional[LocalPoint] = None
        self._latched_offset: Tuple[float, float] = (0.0, 0.0)

    @property
    def latched_center(self) -> Optional[LocalPoint]:
        """Return the currently latched loiter center, or None if unlatched."""
        return self._latched_center

    @property
    def latched_offset(self) -> Tuple[float, float]:
        """Return the currently latched offset tuple (dx, dy)."""
        return self._latched_offset

    def reset(self) -> None:
        """Clear latched center and offset state."""
        self._latched_center = None
        self._latched_offset = (0.0, 0.0)

    def calculate_offset(self, wind_x: float, wind_y: float) -> Tuple[float, float]:
        """
        Calculate the spatial offset (dx, dy) from target to loiter center.

        Args:
            wind_x: Estimated wind x component [m/s] (+East).
            wind_y: Estimated wind y component [m/s] (+North).

        Returns:
            Tuple[float, float] representing (dx, dy) in local coordinates [m].
        """
        if not self.config.enabled or self.config.direction == "none":
            return (0.0, 0.0)

        # Numerical safety: NaN or Inf protection
        if not (math.isfinite(wind_x) and math.isfinite(wind_y)):
            return (0.0, 0.0)

        wind_speed = math.hypot(wind_x, wind_y)
        if wind_speed < self.config.min_wind_speed or wind_speed < 1e-6:
            return (0.0, 0.0)

        # Unit vector along wind velocity direction
        w_hat_x = wind_x / wind_speed
        w_hat_y = wind_y / wind_speed

        # Bounded offset distance
        offset_dist = min(self.config.bias_gain * wind_speed, self.config.max_offset)

        # Direction sign factor:
        # Upwind opposes wind direction (-1.0)
        # Downwind aligns with wind direction (+1.0)
        sign = -1.0 if self.config.direction == "upwind" else (+1.0 if self.config.direction == "downwind" else 0.0)

        dx = sign * offset_dist * w_hat_x
        dy = sign * offset_dist * w_hat_y
        return (dx, dy)

    def plan_center(
        self,
        target: LocalPoint,
        wind_x: float,
        wind_y: float,
    ) -> Tuple[LocalPoint, Tuple[float, float]]:
        """
        Compute or retrieve the planned loiter center for a given landing target and wind.

        Args:
            target: Geographic landing target (LocalPoint).
            wind_x: Estimated wind x component [m/s].
            wind_y: Estimated wind y component [m/s].

        Returns:
            Tuple of (planned_center LocalPoint, (offset_x, offset_y)).
        """
        if self.config.latch_on_entry and self._latched_center is not None:
            return self._latched_center, self._latched_offset

        dx, dy = self.calculate_offset(wind_x, wind_y)
        planned_center = LocalPoint(target.x + dx, target.y + dy)

        if self.config.latch_on_entry:
            self._latched_center = planned_center
            self._latched_offset = (dx, dy)

        return planned_center, (dx, dy)
