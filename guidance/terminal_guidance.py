from dataclasses import dataclass
from enum import Enum
from typing import Optional

from flight.parafoil import ParafoilConfig
from guidance.glide_range import calculate_glide_range


class GuidanceMode(Enum):
    """
    Operational guidance modes for energy management.

    DIRECT_GUIDANCE:
        The vehicle does not have excess glide range to reach the target with
        the required margin; steer directly toward waypoints / landing target.

    TERMINAL_GUIDANCE:
        The vehicle has more remaining glide range than needed to reach the
        target (Rg >= Dt + M); terminal energy management is required to avoid
        overshooting at high altitude.
    """

    DIRECT_GUIDANCE = "DIRECT_GUIDANCE"
    TERMINAL_GUIDANCE = "TERMINAL_GUIDANCE"


@dataclass(frozen=True)
class TerminalDecision:
    """
    Outcome of the terminal guidance decision rule evaluation.

    Attributes:
        mode: Selected guidance mode (DIRECT_GUIDANCE or TERMINAL_GUIDANCE).
        should_enter_terminal: True if both spatial and energy criteria are met.
        glide_range: Estimated remaining kinematic glide range Rg [m].
        target_distance: Remaining horizontal distance to target Dt [m].
        margin: Configurable terminal margin M [m].
        excess_range: Excess glide distance Rg - (Dt + M) [m].
        capture_radius: Configurable terminal capture radius R_capture [m] (optional).
        spatial_capture: True if Dt <= R_capture (or no spatial gate configured).
        energy_condition: True if Rg >= Dt + M.
    """

    mode: GuidanceMode
    should_enter_terminal: bool
    glide_range: float
    target_distance: float
    margin: float
    excess_range: float
    capture_radius: Optional[float] = None
    spatial_capture: bool = True
    energy_condition: bool = False


def evaluate_terminal_guidance(
    altitude: float,
    horizontal_speed: float,
    descent_rate: float,
    target_distance: float,
    terminal_margin: float = 0.0,
    terminal_capture_radius: Optional[float] = None,
    wind_x: float = 0.0,
    wind_y: float = 0.0,
    desired_course_deg: float = 90.0,
    wind_field: Optional['WindField'] = None,
) -> TerminalDecision:
    """
    Evaluate whether direct guidance should continue or terminal guidance should begin.
    """
    if altitude < 0.0:
        raise ValueError("altitude cannot be negative")
    if horizontal_speed < 0.0:
        raise ValueError("horizontal_speed cannot be negative")
    if descent_rate <= 0.0:
        raise ValueError("descent_rate must be greater than zero")
    if target_distance < 0.0:
        raise ValueError("target_distance cannot be negative")
    if terminal_margin < 0.0:
        raise ValueError("terminal_margin cannot be negative")
    if terminal_capture_radius is not None and terminal_capture_radius <= 0.0:
        raise ValueError("terminal_capture_radius must be greater than zero")

    rg = calculate_glide_range(
        altitude=altitude,
        descent_rate=descent_rate,
        horizontal_speed=horizontal_speed,
        wind_x=wind_x,
        wind_y=wind_y,
        desired_course_deg=desired_course_deg,
        wind_field=wind_field,
    )
    threshold = target_distance + terminal_margin
    excess = rg - threshold
    energy_condition = rg >= threshold

    spatial_capture = (
        True
        if terminal_capture_radius is None
        else target_distance <= terminal_capture_radius
    )

    should_enter = energy_condition and spatial_capture

    mode = (
        GuidanceMode.TERMINAL_GUIDANCE
        if should_enter
        else GuidanceMode.DIRECT_GUIDANCE
    )

    return TerminalDecision(
        mode=mode,
        should_enter_terminal=should_enter,
        glide_range=rg,
        target_distance=target_distance,
        margin=terminal_margin,
        excess_range=excess,
        capture_radius=terminal_capture_radius,
        spatial_capture=spatial_capture,
        energy_condition=energy_condition,
    )

def should_enter_terminal_guidance(
    altitude: float,
    horizontal_speed: float,
    descent_rate: float,
    target_distance: float,
    terminal_margin: float = 0.0,
    terminal_capture_radius: Optional[float] = None,
    wind_x: float = 0.0,
    wind_y: float = 0.0,
    desired_course_deg: float = 90.0,
    wind_field: Optional['WindField'] = None,
) -> bool:
    """
    Boolean helper for terminal guidance decision.
    """
    decision = evaluate_terminal_guidance(
        altitude=altitude,
        horizontal_speed=horizontal_speed,
        descent_rate=descent_rate,
        target_distance=target_distance,
        terminal_margin=terminal_margin,
        terminal_capture_radius=terminal_capture_radius,
        wind_x=wind_x,
        wind_y=wind_y,
        desired_course_deg=desired_course_deg,
        wind_field=wind_field,
    )
    return decision.should_enter_terminal

def evaluate_terminal_guidance_from_config(
    altitude: float,
    target_distance: float,
    config: ParafoilConfig,
    terminal_margin: float = 0.0,
    terminal_capture_radius: Optional[float] = None,
    wind_x: float = 0.0,
    wind_y: float = 0.0,
    desired_course_deg: float = 90.0,
) -> TerminalDecision:
    """
    Convenience wrapper that extracts parameters from a ParafoilConfig instance.
    """
    return evaluate_terminal_guidance(
        altitude=altitude,
        horizontal_speed=config.forward_speed,
        descent_rate=config.descent_rate,
        target_distance=target_distance,
        terminal_margin=terminal_margin,
        terminal_capture_radius=terminal_capture_radius,
        wind_x=wind_x,
        wind_y=wind_y,
        desired_course_deg=desired_course_deg,
    )
