import math
from dataclasses import dataclass

@dataclass(frozen=True)
class WindCompensationResult:
    heading_deg: float
    crab_angle_deg: float
    crosswind_component: float
    along_track_wind: float
    ground_speed_along_track: float
    feasible: bool

def calculate_wind_corrected_heading(
    desired_course_deg: float,
    airspeed: float,
    wind_x: float,
    wind_y: float,
) -> WindCompensationResult:
    """
    Calculate the required aerodynamic heading to maintain a straight ground track.

    Coordinate convention:
        +X = East, +Y = North
        Heading/Course: 0=North, 90=East, 180=South, 270=West

    Args:
        desired_course_deg: The required course over ground (0-360)
        airspeed: The vehicle's forward air speed (m/s)
        wind_x: Eastward wind component (m/s)
        wind_y: Northward wind component (m/s)
    """
    chi_rad = math.radians(desired_course_deg)
    
    # Wind perpendicular (crosswind) and parallel (along-track wind)
    # n_d = [cos(chi), -sin(chi)]
    w_perp = wind_x * math.cos(chi_rad) - wind_y * math.sin(chi_rad)
    
    # u_d = [sin(chi), cos(chi)]
    w_parallel = wind_x * math.sin(chi_rad) + wind_y * math.cos(chi_rad)
    
    # Feasibility check
    # We use a small epsilon for floating point tolerance at the boundary
    feasible = abs(w_perp) <= (airspeed + 1e-6)
    
    if feasible:
        # Clamp mathematically to handle floating point epsilon at boundaries
        ratio = 0.0 if airspeed == 0.0 else max(-1.0, min(1.0, -w_perp / airspeed))
        delta_rad = math.asin(ratio)
        delta_deg = math.degrees(delta_rad)
        heading_deg = (desired_course_deg + delta_deg) % 360.0
        
        # Ground speed
        # v_ground_along = v_air * cos(delta) + w_parallel
        ground_speed_along = airspeed * math.cos(delta_rad) + w_parallel
    else:
        # Infeasible: crosswind exceeds airspeed. 
        # Crab angle caps at 90 degrees pointing into the wind cross-track.
        delta_deg = 90.0 if w_perp < 0 else -90.0
        heading_deg = (desired_course_deg + delta_deg) % 360.0
        # Ground speed is effectively just the wind parallel component since air velocity is entirely cross-track
        ground_speed_along = w_parallel

    return WindCompensationResult(
        heading_deg=heading_deg,
        crab_angle_deg=delta_deg,
        crosswind_component=w_perp,
        along_track_wind=w_parallel,
        ground_speed_along_track=ground_speed_along,
        feasible=feasible
    )
