import math
from dataclasses import dataclass
from typing import Optional
from flight.parafoil import ParafoilConfig
from environment.wind import WindField, ConstantWindField
from guidance.wind_compensation import calculate_wind_corrected_heading

@dataclass(frozen=True)
class GlideRangeEstimate:
    """
    Kinematic estimate of remaining glide capability, upgraded to be wind-aware.

    Model:
        t_ground = altitude / descent_rate
        glide_range = integral of V_ground_along(h) / descent_rate dh

    Attributes:
        time_to_ground: Estimated time until ground level [s].
        glide_range: Estimated horizontal distance reachable along track [m].
                     Set to 0.0 if feasible is False.
        feasible: True if the vehicle can make forward progress along the track
                  at all altitudes during the predicted descent.
        projected_ground_speed: Average ground speed along the track over descent [m/s].
        crosswind_component: Cross-track wind component at current altitude [m/s].
        along_track_wind: Along-track wind component at current altitude [m/s].
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

def _integrate_glide_range(
    altitude: float,
    descent_rate: float,
    horizontal_speed: float,
    desired_course_deg: float,
    wind_field: WindField
) -> tuple[float, bool]:
    """
    Numerically integrate the wind-aware glide range over altitude.
    Uses Trapezoidal rule.
    Returns (integrated_range, feasible).
    """
    dh = 1.0 # 1 meter integration step
    
    if altitude < 1e-3:
        return 0.0, True
        
    num_steps = int(math.ceil(altitude / dh))
    actual_dh = altitude / num_steps
    
    integrated_range = 0.0
    
    # Evaluate at h_0
    h_prev = altitude
    wx_prev, wy_prev = wind_field.wind_at_altitude(h_prev)
    comp_prev = calculate_wind_corrected_heading(desired_course_deg, horizontal_speed, wx_prev, wy_prev)
    
    if not comp_prev.feasible or comp_prev.ground_speed_along_track <= 0:
        return 0.0, False
        
    v_prev = comp_prev.ground_speed_along_track
    
    for i in range(1, num_steps + 1):
        h_curr = altitude - i * actual_dh
        if h_curr < 0:
            h_curr = 0.0
            
        wx_curr, wy_curr = wind_field.wind_at_altitude(h_curr)
        comp_curr = calculate_wind_corrected_heading(desired_course_deg, horizontal_speed, wx_curr, wy_curr)
        
        if not comp_curr.feasible or comp_curr.ground_speed_along_track <= 0:
            return 0.0, False
            
        v_curr = comp_curr.ground_speed_along_track
        
        # Trapezoidal step
        # dz = actual_dh
        # dt = dz / descent_rate
        # dx = 0.5 * (v_prev + v_curr) * dt
        dt = actual_dh / descent_rate
        integrated_range += 0.5 * (v_prev + v_curr) * dt
        
        v_prev = v_curr
        
    return integrated_range, True

def calculate_glide_range(
    altitude: float,
    descent_rate: float,
    horizontal_speed: float,
    wind_x: float = 0.0,
    wind_y: float = 0.0,
    desired_course_deg: float = 90.0,
    wind_field: Optional[WindField] = None,
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
        wind_field: Optional altitude-dependent wind profile.

    Returns:
        Estimated horizontal glide range [m]. Returns 0.0 if infeasible.
    """
    if horizontal_speed < 0.0:
        raise ValueError("horizontal_speed cannot be negative")
    
    # Validation
    calculate_time_to_ground(altitude, descent_rate)

    if wind_field is None:
        wind_field = ConstantWindField(wind_x, wind_y)

    r, feasible = _integrate_glide_range(altitude, descent_rate, horizontal_speed, desired_course_deg, wind_field)
    return r if feasible else 0.0

def estimate_glide_range(
    altitude: float,
    descent_rate: float,
    horizontal_speed: float,
    wind_x: float = 0.0,
    wind_y: float = 0.0,
    desired_course_deg: float = 90.0,
    wind_field: Optional[WindField] = None,
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
        wind_field: Optional altitude-dependent wind profile.

    Returns:
        GlideRangeEstimate containing range, feasibility, and wind metrics.
    """
    if horizontal_speed < 0.0:
        raise ValueError("horizontal_speed cannot be negative")

    t_ground = calculate_time_to_ground(altitude, descent_rate)
    
    if wind_field is None:
        wind_field = ConstantWindField(wind_x, wind_y)
        
    glide_dist, feasible = _integrate_glide_range(altitude, descent_rate, horizontal_speed, desired_course_deg, wind_field)
    
    # Calculate representative instantaneous components at current altitude
    curr_wx, curr_wy = wind_field.wind_at_altitude(altitude)
    comp_curr = calculate_wind_corrected_heading(desired_course_deg, horizontal_speed, curr_wx, curr_wy)
    
    if feasible:
        avg_speed = (glide_dist / t_ground) if t_ground > 1e-6 else comp_curr.ground_speed_along_track
    else:
        avg_speed = comp_curr.ground_speed_along_track
    
    return GlideRangeEstimate(
        time_to_ground=t_ground,
        glide_range=glide_dist,
        feasible=feasible,
        projected_ground_speed=avg_speed,
        crosswind_component=comp_curr.crosswind_component,
        along_track_wind=comp_curr.along_track_wind,
    )

def estimate_glide_range_from_config(
    altitude: float,
    config: ParafoilConfig,
    wind_x: float = 0.0,
    wind_y: float = 0.0,
    desired_course_deg: float = 90.0,
    wind_field: Optional[WindField] = None,
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
        wind_field=wind_field
    )
