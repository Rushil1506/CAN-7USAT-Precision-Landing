from dataclasses import dataclass

from flight.parafoil import ParafoilConfig


@dataclass(frozen=True)
class GlideRangeEstimate:
    """
    Simplified kinematic estimate of remaining glide capability.

    Model:
        t_ground = altitude / descent_rate
        glide_range = horizontal_speed * t_ground

    Assumptions and Limitations:
        - This is a simplified kinematic estimate.
        - Based on air-relative forward speed.
        - Currently assumes no wind.
        - NOT a high-fidelity aerodynamic prediction.

    Attributes:
        time_to_ground: Estimated time until ground level [s].
        glide_range: Estimated horizontal distance reachable [m].
    """

    time_to_ground: float
    glide_range: float


def calculate_time_to_ground(altitude: float, descent_rate: float) -> float:
    """
    Calculate estimated time to reach ground level.

    Simplified kinematic formula:
        t_ground = altitude / descent_rate

    Args:
        altitude: Altitude above ground in metres (must be >= 0).
        descent_rate: Vertical descent rate in metres per second (must be > 0).

    Returns:
        Estimated time to ground in seconds.

    Raises:
        ValueError: If altitude < 0 or descent_rate <= 0.
    """
    if altitude < 0.0:
        raise ValueError("altitude cannot be negative")
    if descent_rate <= 0.0:
        raise ValueError("descent_rate must be greater than zero")

    return altitude / descent_rate


def calculate_glide_range(
    altitude: float,
    descent_rate: float,
    horizontal_speed: float,
) -> float:
    """
    Calculate estimated remaining horizontal glide range.

    Simplified kinematic formula:
        t_ground = altitude / descent_rate
        glide_range = horizontal_speed * t_ground

    Assumptions and Limitations:
        - Simplified kinematic estimate.
        - Based on air-relative forward speed (horizontal_speed).
        - Currently assumes no wind.
        - NOT a high-fidelity aerodynamic prediction.

    Args:
        altitude: Altitude above ground in metres (must be >= 0).
        descent_rate: Vertical descent rate in metres per second (must be > 0).
        horizontal_speed: Parafoil air-relative forward speed in metres per
            second (must be >= 0).

    Returns:
        Estimated horizontal glide range in metres.

    Raises:
        ValueError: If altitude < 0, horizontal_speed < 0, or descent_rate <= 0.
    """
    if horizontal_speed < 0.0:
        raise ValueError("horizontal_speed cannot be negative")

    t_ground = calculate_time_to_ground(altitude, descent_rate)
    return horizontal_speed * t_ground


def estimate_glide_range(
    altitude: float,
    descent_rate: float,
    horizontal_speed: float,
) -> GlideRangeEstimate:
    """
    Estimate remaining time to ground and horizontal glide range.

    Simplified kinematic formula:
        t_ground = altitude / descent_rate
        glide_range = horizontal_speed * t_ground

    Assumptions and Limitations:
        - Simplified kinematic estimate.
        - Based on air-relative forward speed (horizontal_speed).
        - Currently assumes no wind.
        - NOT a high-fidelity aerodynamic prediction.

    Args:
        altitude: Altitude above ground in metres (must be >= 0).
        descent_rate: Vertical descent rate in metres per second (must be > 0).
        horizontal_speed: Parafoil air-relative forward speed in metres per
            second (must be >= 0).

    Returns:
        GlideRangeEstimate dataclass with time_to_ground and glide_range.

    Raises:
        ValueError: If altitude < 0, horizontal_speed < 0, or descent_rate <= 0.
    """
    if horizontal_speed < 0.0:
        raise ValueError("horizontal_speed cannot be negative")

    t_ground = calculate_time_to_ground(altitude, descent_rate)
    glide_dist = horizontal_speed * t_ground
    return GlideRangeEstimate(time_to_ground=t_ground, glide_range=glide_dist)


def estimate_glide_range_from_config(
    altitude: float,
    config: ParafoilConfig,
) -> GlideRangeEstimate:
    """
    Estimate glide range using parameters from a ParafoilConfig instance.

    Convenience wrapper that maps config.descent_rate and config.forward_speed
    into estimate_glide_range().

    Args:
        altitude: Altitude above ground in metres (must be >= 0).
        config: ParafoilConfig instance providing descent_rate and forward_speed.

    Returns:
        GlideRangeEstimate dataclass with time_to_ground and glide_range.
    """
    return estimate_glide_range(
        altitude=altitude,
        descent_rate=config.descent_rate,
        horizontal_speed=config.forward_speed,
    )
