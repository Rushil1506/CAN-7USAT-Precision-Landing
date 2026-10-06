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
) -> TerminalDecision:
    """
    Evaluate whether direct guidance should continue or terminal guidance should begin.

    Kinematic decision relationship:
        Rg = horizontal_speed * (altitude / descent_rate)

        Energy condition:
            Rg >= Dt + M

        Spatial condition:
            Dt <= R_capture   (if R_capture is provided)

        Combined terminal-entry rule:
            should_enter_terminal = (Dt <= R_capture) AND (Rg >= Dt + M)

            True  -> TERMINAL_GUIDANCE
            False -> DIRECT_GUIDANCE

    where:
        Rg        = Remaining estimated glide range [m]
        Dt        = Remaining horizontal distance to target [m]
        M         = Configurable terminal margin [m]
        R_capture = Configurable terminal capture radius [m]

    Boundary behavior:
        When Dt == R_capture and Rg == Dt + M exactly, both conditions evaluate
        to True, selecting TERMINAL_GUIDANCE.

    Assumptions and Limitations:
        - Reuses the simplified kinematic glide range model from guidance/glide_range.py.
        - Uses air-relative forward speed and assumes zero wind.
        - Does NOT yet alter flight state machine or actuator outputs.

    Args:
        altitude: Current altitude above ground [m] (must be >= 0).
        horizontal_speed: Air-relative forward speed [m/s] (must be >= 0).
        descent_rate: Vertical descent rate [m/s] (must be > 0).
        target_distance: Horizontal distance to landing target [m] (must be >= 0).
        terminal_margin: Configurable margin buffer [m] (must be >= 0, default 0.0).
        terminal_capture_radius: Configurable spatial capture radius [m]
            (must be > 0 if provided, default None).

    Returns:
        TerminalDecision dataclass containing mode, flags, and range metrics.

    Raises:
        ValueError: If altitude < 0, horizontal_speed < 0, descent_rate <= 0,
            target_distance < 0, terminal_margin < 0, or
            terminal_capture_radius <= 0.
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
) -> bool:
    """
    Boolean helper for terminal guidance decision.

    Returns:
        True if spatial condition (Dt <= R_capture or R_capture is None) AND
        energy condition (Rg >= Dt + M) are both satisfied, False otherwise.
    """
    decision = evaluate_terminal_guidance(
        altitude=altitude,
        horizontal_speed=horizontal_speed,
        descent_rate=descent_rate,
        target_distance=target_distance,
        terminal_margin=terminal_margin,
        terminal_capture_radius=terminal_capture_radius,
    )
    return decision.should_enter_terminal


def evaluate_terminal_guidance_from_config(
    altitude: float,
    target_distance: float,
    config: ParafoilConfig,
    terminal_margin: float = 0.0,
    terminal_capture_radius: Optional[float] = None,
) -> TerminalDecision:
    """
    Convenience wrapper that extracts parameters from a ParafoilConfig instance.

    Args:
        altitude: Altitude above ground [m] (must be >= 0).
        target_distance: Distance to landing target [m] (must be >= 0).
        config: ParafoilConfig instance providing forward_speed and descent_rate.
        terminal_margin: Configurable margin buffer [m] (must be >= 0, default 0.0).
        terminal_capture_radius: Configurable spatial capture radius [m]
            (must be > 0 if provided, default None).

    Returns:
        TerminalDecision dataclass.
    """
    return evaluate_terminal_guidance(
        altitude=altitude,
        horizontal_speed=config.forward_speed,
        descent_rate=config.descent_rate,
        target_distance=target_distance,
        terminal_margin=terminal_margin,
        terminal_capture_radius=terminal_capture_radius,
    )
