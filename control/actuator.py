"""
Actuator Dynamics Model for CAN-7USAT Precision Landing Guidance.

This module provides a realistic yet computationally lightweight model of
actuation dynamics between the guidance/control layer (normalized steering command)
and the aerodynamic flight model (applied parafoil steering).

Features supported:
1. Output Command Saturation (e.g. [-1.0, +1.0])
2. Transport Delay (latency between command generation and servo execution)
3. Finite Slew-Rate Limiting (maximum servo movement speed [units/s])
4. Asymmetric Deadband (deadband_positive, deadband_negative with linear rescaling)
5. State Reset (for phase transitions and testing)
6. True Pass-Through Configuration (exact baseline preservation)

Signal Flow:
    command (guidance)
           ↓
    [Transport Delay Buffer] (FIFO queue)
           ↓
    [Deadband Mapping] (asymmetric thresholding & linear rescaling)
           ↓
    [Slew-Rate Limiter] (finite rate of position change)
           ↓
    [Command Saturation] (clamp to [min_command, max_command])
           ↓
    actual_steering (to ParafoilModel)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple
import math


class ActuatorConfig:
    """
    Configuration parameters for the actuator dynamics model.

    Supports standard and alias parameter names:
    - min_command / steering_min: Minimum output clamp (default -1.0)
    - max_command / steering_max: Maximum output clamp (default +1.0)
    - slew_rate_limit / max_slew_rate: Maximum rate of change [units/s] (default None = unconstrained)
    - delay_seconds / transport_delay: Transport delay [s] (default 0.0)
    - deadband_positive: Positive command threshold [0, max_command) (default 0.0)
    - deadband_negative: Negative command threshold [0, abs(min_command)) (default 0.0)
    - initial_position: Initial internal position (default 0.0)
    """

    def __init__(
        self,
        min_command: Optional[float] = None,
        max_command: Optional[float] = None,
        slew_rate_limit: Optional[float] = None,
        delay_seconds: Optional[float] = None,
        deadband_positive: float = 0.0,
        deadband_negative: float = 0.0,
        initial_position: float = 0.0,
        # Keyword aliases
        steering_min: Optional[float] = None,
        steering_max: Optional[float] = None,
        max_slew_rate: Optional[float] = None,
        transport_delay: Optional[float] = None,
    ) -> None:
        # Resolve aliases
        self.min_command = min_command if min_command is not None else (steering_min if steering_min is not None else -1.0)
        self.max_command = max_command if max_command is not None else (steering_max if steering_max is not None else 1.0)

        self.slew_rate_limit = slew_rate_limit if slew_rate_limit is not None else max_slew_rate
        self.delay_seconds = delay_seconds if delay_seconds is not None else (transport_delay if transport_delay is not None else 0.0)

        self.deadband_positive = deadband_positive
        self.deadband_negative = deadband_negative
        self.initial_position = initial_position

        self._validate()

    def _validate(self) -> None:
        if not math.isfinite(self.min_command) or not math.isfinite(self.max_command):
            raise ValueError("Command limits must be finite numbers")
        if self.min_command >= self.max_command:
            raise ValueError(f"steering_min must be less than steering_max: {self.min_command} >= {self.max_command}")

        if self.slew_rate_limit is not None:
            if math.isnan(self.slew_rate_limit):
                raise ValueError("slew_rate_limit cannot be NaN")
            if self.slew_rate_limit <= 0.0:
                raise ValueError(f"max_slew_rate must be positive, got {self.slew_rate_limit}")

        if self.delay_seconds < 0.0 or not math.isfinite(self.delay_seconds):
            raise ValueError(f"transport_delay cannot be negative, got {self.delay_seconds}")

        if self.deadband_positive < 0.0 or not math.isfinite(self.deadband_positive):
            raise ValueError(f"deadband_positive cannot be negative, got {self.deadband_positive}")
        if self.deadband_positive >= self.max_command:
            raise ValueError(f"deadband_positive must be strictly less than steering_max ({self.max_command}), got {self.deadband_positive}")

        if self.deadband_negative < 0.0 or not math.isfinite(self.deadband_negative):
            raise ValueError(f"deadband_negative cannot be negative, got {self.deadband_negative}")
        if self.deadband_negative >= abs(self.min_command):
            raise ValueError(f"deadband_negative must be strictly less than abs(steering_min) ({abs(self.min_command)}), got {self.deadband_negative}")

        if not math.isfinite(self.initial_position):
            raise ValueError(f"initial_position must be finite, got {self.initial_position}")

    @property
    def steering_min(self) -> float:
        return self.min_command

    @property
    def steering_max(self) -> float:
        return self.max_command

    @property
    def max_slew_rate(self) -> Optional[float]:
        return self.slew_rate_limit

    @property
    def transport_delay(self) -> float:
        return self.delay_seconds

    @property
    def is_passthrough(self) -> bool:
        """True if the configuration performs pure pass-through without delay, rate limit, or deadband."""
        return (
            self.delay_seconds == 0.0
            and (self.slew_rate_limit is None or self.slew_rate_limit <= 0.0)
            and self.deadband_positive == 0.0
            and self.deadband_negative == 0.0
            and self.min_command <= -1.0
            and self.max_command >= 1.0
        )


class ActuatorModel:
    """
    Discrete-time actuator dynamics model supporting:
      1. Transport delay buffer (FIFO queue based on simulation time)
      2. Asymmetric positive and negative deadbands with linear rescaling
      3. Finite slew-rate limiting (rate of position change)
      4. Output command saturation
    """

    def __init__(self, config: Optional[ActuatorConfig] = None) -> None:
        self.config = config if config is not None else ActuatorConfig()
        self.current_position: float = self.config.initial_position
        self.last_slew_rate: float = 0.0
        self.is_rate_saturated: bool = False
        self.is_command_saturated: bool = False

        # Queue storing (timestamp, commanded_value)
        self._delay_buffer: Deque[Tuple[float, float]] = deque()
        self._current_time: float = 0.0

    @property
    def current_output(self) -> float:
        """Alias for current_position."""
        return self.current_position

    @property
    def is_passthrough(self) -> bool:
        """True if underlying configuration is pure pass-through."""
        return self.config.is_passthrough

    def reset(self, initial_position: Optional[float] = None) -> None:
        """Reset internal actuator position, buffers, and diagnostic flags."""
        pos = initial_position if initial_position is not None else self.config.initial_position
        if not math.isfinite(pos):
            raise ValueError(f"initial_position must be finite, got {pos}")
        self.current_position = max(self.config.min_command, min(self.config.max_command, pos))
        self.last_slew_rate = 0.0
        self.is_rate_saturated = False
        self.is_command_saturated = False
        self._delay_buffer.clear()
        self._current_time = 0.0

    def update(self, command: float, dt: float) -> float:
        """
        Advance actuator state by one simulation timestep dt.

        Args:
            command: Requested steering command (typically [-1, +1]).
            dt: Timestep duration in seconds. Must be > 0.

        Returns:
            Actual steering output applied to the physical system.
        """
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError(f"dt must be positive and finite, got {dt}")
        if not math.isfinite(command):
            raise ValueError(f"Actuator input command must be finite, got {command}")

        self._current_time += dt

        # 1. Transport Delay Handling
        if self.config.delay_seconds > 0.0:
            self._delay_buffer.append((self._current_time, command))
            # Find the value commanded at least delay_seconds ago
            target_time = self._current_time - self.config.delay_seconds
            delayed_command = self.config.initial_position
            while len(self._delay_buffer) > 1 and self._delay_buffer[1][0] <= target_time:
                self._delay_buffer.popleft()
            if self._delay_buffer[0][0] <= target_time:
                delayed_command = self._delay_buffer[0][1]
            else:
                delayed_command = self.config.initial_position
        else:
            delayed_command = command

        # 2. Asymmetric Deadband
        if delayed_command > 0.0:
            if delayed_command <= self.config.deadband_positive:
                deadbanded = 0.0
            else:
                # Linear rescaling beyond deadband
                span = self.config.max_command - self.config.deadband_positive
                if span > 1e-9:
                    deadbanded = (delayed_command - self.config.deadband_positive) / span * self.config.max_command
                else:
                    deadbanded = self.config.max_command
        elif delayed_command < 0.0:
            if abs(delayed_command) <= self.config.deadband_negative:
                deadbanded = 0.0
            else:
                span = abs(self.config.min_command) - self.config.deadband_negative
                if span > 1e-9:
                    deadbanded = (delayed_command + self.config.deadband_negative) / span * abs(self.config.min_command)
                else:
                    deadbanded = self.config.min_command
        else:
            deadbanded = 0.0

        # 3. Slew-Rate Limiting
        target_pos = deadbanded
        delta_pos = target_pos - self.current_position

        if self.config.slew_rate_limit is not None and self.config.slew_rate_limit > 0.0:
            max_delta = self.config.slew_rate_limit * dt
            if abs(delta_pos) > max_delta:
                self.is_rate_saturated = True
                actual_delta = math.copysign(max_delta, delta_pos)
            else:
                self.is_rate_saturated = False
                actual_delta = delta_pos
        else:
            self.is_rate_saturated = False
            actual_delta = delta_pos

        self.last_slew_rate = actual_delta / dt
        self.current_position += actual_delta

        # 4. Output Saturation
        pre_clamp = self.current_position
        self.current_position = max(
            self.config.min_command,
            min(self.config.max_command, self.current_position),
        )
        self.is_command_saturated = (
            (self.current_position != pre_clamp)
            or (self.current_position >= self.config.max_command - 1e-9)
            or (self.current_position <= self.config.min_command + 1e-9)
        )

        return self.current_position
