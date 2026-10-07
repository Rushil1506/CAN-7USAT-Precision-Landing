from dataclasses import dataclass
from typing import Optional
from flight.parafoil import ParafoilConfig
from guidance.wind_compensation import calculate_wind_corrected_heading

@dataclass(frozen=True)
class GlideRangeEstimate:
    """
    Kinematic estimate of remaining glide capability, upgraded to be wind-aware.

    Model:
        t_ground = altitude / descent_rate
        glide_range = V_ground_along * t_ground

    Attributes:
        time_to_ground: Estimated time until ground level [s].
        glide_range: Estimated horizontal distance reachable along track [m].
                     Set to 0.0 if feasible is False.
        feasible: True if the vehicle can make forward progress along the track.
        projected_ground_speed: Calculated ground speed along the track [m/s].
        crosswind_component: Cross-track wind component [m/s].
        along_track_wind: Along-track wind component [m/s].
    """
    time_to_ground: float
    glide_range: float
    feasible: bool = True
    projected_ground_speed: float = 0.0
    crosswind_component: float = 0.0
    along_track_wind: float = 0.0

def calculate_time_to_ground(altitude: float, descent_rate: float) -> float:
    """Calculate estimated time to reach ground level."""
    if altitude < 0.0:
        raise ValueError("altitude cannot be negative")
    if descent_rate <= 0.0:
        raise ValueError("descent_rate must be greater than zero")
    return altitude / descent_rate

def calculate_glide_range(
    altitude: float,
    descent_rate: float,
    horizontal_speed: float,
    wind_x: float = 0.0,
    wind_y: float = 0.0,
    desired_course_deg: float = 90.0,
) -> float:
    """
    Calculate estimated remaining horizontal glide range along the desired track.

    Args:
        altitude: Altitude above ground [m] (must be >= 0).
        descent_rate: Vertical descent rate [m/s] (must be > 0).
        horizontal_speed: Parafoil airspeed [m/s] (must be >= 0).
        wind_x: Eastward wind component [m/s].
        wind_y: Northward wind component [m/s].
        desired_course_deg: Target ground course [0-360 deg].

    Returns:
        Estimated horizontal glide range [m]. Returns 0.0 if infeasible.
    """
    if horizontal_speed < 0.0:
        raise ValueError("horizontal_speed cannot be negative")

    t_ground = calculate_time_to_ground(altitude, descent_rate)
    
    comp = calculate_wind_corrected_heading(
        desired_course_deg=desired_course_deg,
        airspeed=horizontal_speed,
        wind_x=wind_x,
        wind_y=wind_y,
    )
    
    if not comp.feasible or comp.ground_speed_along_track <= 0:
        return 0.0
        
    return comp.ground_speed_along_track * t_ground

def estimate_glide_range(
    altitude: float,
    descent_rate: float,
    horizontal_speed: float,
    wind_x: float = 0.0,
    wind_y: float = 0.0,
    desired_course_deg: float = 90.0,
) -> GlideRangeEstimate:
    """
    Estimate remaining time to ground and wind-aware glide range.

    Args:
        altitude: Altitude above ground [m] (must be >= 0).
        descent_rate: Vertical descent rate [m/s] (must be > 0).
        horizontal_speed: Parafoil airspeed [m/s] (must be >= 0).
        wind_x: Eastward wind component [m/s].
        wind_y: Northward wind component [m/s].
        desired_course_deg: Target ground course [0-360 deg].

    Returns:
        GlideRangeEstimate containing range, feasibility, and wind metrics.
    """
    if horizontal_speed < 0.0:
        raise ValueError("horizontal_speed cannot be negative")

    t_ground = calculate_time_to_ground(altitude, descent_rate)
    
    comp = calculate_wind_corrected_heading(
        desired_course_deg=desired_course_deg,
        airspeed=horizontal_speed,
        wind_x=wind_x,
        wind_y=wind_y,
    )
    
    feasible = comp.feasible and (comp.ground_speed_along_track > 0)
    glide_dist = comp.ground_speed_along_track * t_ground if feasible else 0.0
    
    return GlideRangeEstimate(
        time_to_ground=t_ground,
        glide_range=glide_dist,
        feasible=feasible,
        projected_ground_speed=comp.ground_speed_along_track,
        crosswind_component=comp.crosswind_component,
        along_track_wind=comp.along_track_wind,
    )

def estimate_glide_range_from_config(
    altitude: float,
    config: ParafoilConfig,
    wind_x: float = 0.0,
    wind_y: float = 0.0,
    desired_course_deg: float = 90.0,
) -> GlideRangeEstimate:
    """
    Estimate glide range using parameters from a ParafoilConfig instance.
    """
    return estimate_glide_range(
        altitude=altitude,
        descent_rate=config.descent_rate,
        horizontal_speed=config.forward_speed,
        wind_x=wind_x,
        wind_y=wind_y,
        desired_course_deg=desired_course_deg,
    )
